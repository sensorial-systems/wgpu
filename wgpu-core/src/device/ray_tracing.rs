use alloc::{string::ToString, sync::Arc, vec::Vec};
use core::mem::{size_of, ManuallyDrop};

use crate::device::stage_compiler::StageCompiler;
#[cfg(feature = "trace")]
use crate::device::trace::{Action, IntoTrace};
use crate::device::{DeviceError, ENTRYPOINT_FAILURE_ERROR};
use crate::pipeline::CreatePipelineError;
use crate::ray_tracing_pipeline;
use crate::resource::{ParentDevice, ResourceState};
use crate::{
    api_log,
    device::Device,
    hal_label,
    lock::RwLock,
    lock::{rank, Mutex},
    ray_tracing::{CreateBlasError, CreateTlasError},
    resource,
    resource::{BlasCompactState, TrackingData},
    snatch::Snatchable,
    LabelHelpers,
};
use hal::AccelerationStructureTriangleIndices;
use wgt::{Features, AABB_GEOMETRY_MIN_STRIDE};

/// Compile a stage of a hit group, with the hit attribute it reads or
/// writes and whether it reads a triangle's barycentrics.
fn compile_hit_group_stage<'a, 'd: 'a>(
    compiler: &mut StageCompiler<'d>,
    desc: &'a crate::pipeline::ProgrammableStageDescriptor<'_>,
    stage: crate::validation::ShaderStageForValidation,
) -> Result<
    (
        crate::device::stage_compiler::CompiledStage<'a>,
        Option<crate::validation::RayTracingPayloadType>,
        bool,
    ),
    CreatePipelineError,
> {
    compiler.io.hit_attribute = None;
    compiler.io.reads_barycentrics = false;
    let compiled = compiler.compile(desc, stage)?;
    Ok((
        compiled,
        compiler.io.hit_attribute.take(),
        compiler.io.reads_barycentrics,
    ))
}

impl Device {
    /// Requires any feature that allows acceleration structures as they can be provided by multiple extensions.
    pub fn require_acceleration_structures(&self) -> Result<(), crate::device::MissingFeatures> {
        self.require_features(Features::EXPERIMENTAL_RAY_QUERY)
            .or_else(|_| self.require_features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES))
    }

    pub fn create_blas(
        self: &Arc<Self>,
        blas_desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
    ) -> (Arc<resource::Blas>, Option<CreateBlasError>) {
        #[cfg(feature = "trace")]
        let trace_sizes = sizes.clone();

        let (blas, error) = match self.create_blas_inner(blas_desc, sizes) {
            Ok(blas) => (blas, None),
            Err(err) => (resource::Blas::invalid(self.clone(), blas_desc), Some(err)),
        };

        #[cfg(feature = "trace")]
        if let Some(trace) = self.trace.lock().as_mut() {
            trace.add(Action::CreateBlas {
                id: blas.to_trace(),
                desc: blas_desc.clone(),
                sizes: trace_sizes,
            });
        }

        api_log!("Device::create_blas -> {:?}", Arc::as_ptr(&blas));
        (blas, error)
    }
    pub(crate) fn create_blas_inner(
        self: &Arc<Self>,
        blas_desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
    ) -> Result<Arc<resource::Blas>, CreateBlasError> {
        self.check_is_valid()?;
        self.require_acceleration_structures()?;

        if blas_desc
            .flags
            .contains(wgt::AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN)
        {
            self.require_features(Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN)?;
        }
        let moves = blas_desc
            .flags
            .contains(wgt::AccelerationStructureFlags::ALLOW_MOTION);
        if moves {
            self.require_features(Features::EXPERIMENTAL_RAY_TRACING_MOTION_BLUR)?;
        }
        // The opacity micromaps' usage, of the geometries that have one: the
        // structure keeps a micromap for each, which a compacted copy would
        // not.
        let opacity_micromaps: Vec<&[wgt::OpacityMicromapUsage]> = match sizes {
            wgt::BlasGeometrySizeDescriptors::Triangles { ref descriptors } => descriptors
                .iter()
                .filter(|desc| !desc.opacity_micromap.is_empty())
                .map(|desc| &desc.opacity_micromap[..])
                .collect(),
            wgt::BlasGeometrySizeDescriptors::AABBs { .. } => Vec::new(),
        };
        if !opacity_micromaps.is_empty() {
            self.require_features(Features::EXPERIMENTAL_RAY_TRACING_OPACITY_MICROMAP)?;
            if blas_desc
                .flags
                .contains(wgt::AccelerationStructureFlags::ALLOW_COMPACTION)
            {
                return Err(CreateBlasError::OpacityMicromapCompaction);
            }
        }

        let size_info = match &sizes {
            wgt::BlasGeometrySizeDescriptors::Triangles { descriptors } => {
                if descriptors.len() as u32 > self.limits.max_blas_geometry_count {
                    return Err(CreateBlasError::TooManyGeometries(
                        self.limits.max_blas_geometry_count,
                        descriptors.len() as u32,
                    ));
                }

                let mut entries =
                    Vec::<hal::AccelerationStructureTriangles<dyn hal::DynBuffer>>::with_capacity(
                        descriptors.len(),
                    );
                for desc in descriptors {
                    if desc.index_count.is_some() != desc.index_format.is_some() {
                        return Err(CreateBlasError::MissingIndexData);
                    }
                    let indices =
                        desc.index_count
                            .map(|count| AccelerationStructureTriangleIndices::<
                                dyn hal::DynBuffer,
                            > {
                                format: desc.index_format.unwrap(),
                                buffer: Some(self.zero_buffer.as_ref()),
                                offset: 0,
                                count,
                            });
                    if !self
                        .features
                        .allowed_vertex_formats_for_blas()
                        .contains(&desc.vertex_format)
                    {
                        return Err(CreateBlasError::InvalidVertexFormat(
                            desc.vertex_format,
                            self.features.allowed_vertex_formats_for_blas(),
                        ));
                    }

                    let mut transform = None;

                    if blas_desc
                        .flags
                        .contains(wgt::AccelerationStructureFlags::USE_TRANSFORM)
                    {
                        transform = Some(wgpu_hal::AccelerationStructureTriangleTransform {
                            buffer: self.zero_buffer.as_ref(),
                            offset: 0,
                        })
                    }

                    if desc.vertex_count > self.limits.max_blas_primitive_count {
                        return Err(CreateBlasError::TooManyPrimitives(
                            self.limits.max_blas_primitive_count,
                            desc.vertex_count,
                        ));
                    }

                    let opacity_micromap = (!desc.opacity_micromap.is_empty()).then(|| {
                        hal::AccelerationStructureOpacityMicromap {
                            data: None,
                            data_offset: 0,
                            triangles: None,
                            triangles_offset: 0,
                            indices: None,
                            usage: &desc.opacity_micromap[..],
                        }
                    });

                    entries.push(hal::AccelerationStructureTriangles::<dyn hal::DynBuffer> {
                        vertex_buffer: Some(self.zero_buffer.as_ref()),
                        vertex_format: desc.vertex_format,
                        first_vertex: 0,
                        vertex_count: desc.vertex_count,
                        vertex_stride: 0,
                        indices,
                        transform,
                        flags: desc.flags,
                        end_vertex_buffer: moves.then_some(self.zero_buffer.as_ref()),
                        opacity_micromap,
                    });
                }
                unsafe {
                    self.raw().get_acceleration_structure_build_sizes(
                        &hal::GetAccelerationStructureBuildSizesDescriptor {
                            entries: &hal::AccelerationStructureEntries::Triangles(entries),
                            flags: blas_desc.flags,
                        },
                    )
                }
            }
            wgt::BlasGeometrySizeDescriptors::AABBs { descriptors } => {
                if descriptors.len() as u32 > self.limits.max_blas_geometry_count {
                    return Err(CreateBlasError::TooManyGeometries(
                        self.limits.max_blas_geometry_count,
                        descriptors.len() as u32,
                    ));
                }

                let mut entries =
                    Vec::<hal::AccelerationStructureAABBs<dyn hal::DynBuffer>>::with_capacity(
                        descriptors.len(),
                    );
                for desc in descriptors {
                    if desc.primitive_count > self.limits.max_blas_primitive_count {
                        return Err(CreateBlasError::TooManyPrimitives(
                            self.limits.max_blas_primitive_count,
                            desc.primitive_count,
                        ));
                    }

                    entries.push(hal::AccelerationStructureAABBs::<dyn hal::DynBuffer> {
                        buffer: Some(self.zero_buffer.as_ref()),
                        offset: 0,
                        count: desc.primitive_count,
                        stride: AABB_GEOMETRY_MIN_STRIDE,
                        flags: desc.flags,
                    });
                }
                unsafe {
                    self.raw().get_acceleration_structure_build_sizes(
                        &hal::GetAccelerationStructureBuildSizesDescriptor {
                            entries: &hal::AccelerationStructureEntries::AABBs(entries),
                            flags: blas_desc.flags,
                        },
                    )
                }
            }
        };

        let raw = unsafe {
            self.raw()
                .create_acceleration_structure(&hal::AccelerationStructureDescriptor {
                    label: hal_label(blas_desc.label.as_deref(), self.instance_flags),
                    size: size_info.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::BottomLevel,
                    allow_compaction: blas_desc
                        .flags
                        .contains(wgpu_types::AccelerationStructureFlags::ALLOW_COMPACTION),
                    allow_motion: moves,
                    max_instances: 0,
                    opacity_micromaps: &opacity_micromaps,
                })
        }
        .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        let compaction_buffer = if blas_desc
            .flags
            .contains(wgpu_types::AccelerationStructureFlags::ALLOW_COMPACTION)
        {
            Some(ManuallyDrop::new(unsafe {
                self.raw()
                    .create_buffer(&hal::BufferDescriptor {
                        label: hal_label(
                            Some("(wgpu internal) compaction read-back buffer"),
                            self.instance_flags,
                        ),
                        size: size_of::<wgpu_types::BufferAddress>() as wgpu_types::BufferAddress,
                        usage: wgpu_types::BufferUses::ACCELERATION_STRUCTURE_QUERY
                            | wgpu_types::BufferUses::MAP_READ,
                        memory_flags: hal::MemoryFlags::PREFER_COHERENT,
                    })
                    .map_err(DeviceError::from_hal)?
                    .0
            }))
        } else {
            None
        };

        let handle = unsafe {
            self.raw()
                .get_acceleration_structure_device_address(raw.as_ref())
        };

        Ok(Arc::new(resource::Blas {
            state: ResourceState::Valid(resource::BlasState {
                raw: Snatchable::new(raw),
            }),
            device: self.clone(),
            size_info,
            sizes,
            flags: blas_desc.flags,
            update_mode: blas_desc.update_mode,
            handle,
            label: blas_desc.label.to_string(),
            built_index: RwLock::new(rank::BLAS_BUILT_INDEX, None),
            tracking_data: TrackingData::new(self.tracker_indices.blas_s.clone()),
            compaction_buffer,
            compacted_state: Mutex::new(rank::BLAS_COMPACTION_STATE, BlasCompactState::Idle),
        }))
    }

    pub fn create_tlas(
        self: &Arc<Self>,
        desc: &resource::TlasDescriptor,
    ) -> (Arc<resource::Tlas>, Option<CreateTlasError>) {
        let (tlas, error) = match self.create_tlas_inner(desc) {
            Ok(tlas) => (tlas, None),
            Err(e) => (resource::Tlas::invalid(Arc::clone(self), desc), Some(e)),
        };
        #[cfg(feature = "trace")]
        if let Some(trace) = self.trace.lock().as_mut() {
            trace.add(Action::CreateTlas {
                id: tlas.to_trace(),
                desc: desc.clone(),
            });
        }

        api_log!("Device::create_tlas -> {:?}", Arc::as_ptr(&tlas));

        (tlas, error)
    }

    pub(crate) fn create_tlas_inner(
        self: &Arc<Self>,
        desc: &resource::TlasDescriptor,
    ) -> Result<Arc<resource::Tlas>, CreateTlasError> {
        self.check_is_valid()?;
        self.require_acceleration_structures()?;

        if desc.max_instances > self.limits.max_tlas_instance_count {
            return Err(CreateTlasError::TooManyInstances(
                self.limits.max_tlas_instance_count,
                desc.max_instances,
            ));
        }

        if desc
            .flags
            .contains(wgt::AccelerationStructureFlags::USE_TRANSFORM)
        {
            return Err(CreateTlasError::DisallowedFlag(
                wgt::AccelerationStructureFlags::USE_TRANSFORM,
            ));
        }

        if desc
            .flags
            .contains(wgt::AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN)
        {
            self.require_features(Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN)?;
        }
        let moves = desc
            .flags
            .contains(wgt::AccelerationStructureFlags::ALLOW_MOTION);
        if moves {
            self.require_features(Features::EXPERIMENTAL_RAY_TRACING_MOTION_BLUR)?;
        }

        let size_info = unsafe {
            self.raw().get_acceleration_structure_build_sizes(
                &hal::GetAccelerationStructureBuildSizesDescriptor {
                    entries: &hal::AccelerationStructureEntries::Instances(
                        hal::AccelerationStructureInstances {
                            buffer: Some(self.zero_buffer.as_ref()),
                            offset: 0,
                            count: desc.max_instances,
                        },
                    ),
                    flags: desc.flags,
                },
            )
        };

        let raw = unsafe {
            self.raw()
                .create_acceleration_structure(&hal::AccelerationStructureDescriptor {
                    label: hal_label(desc.label.as_deref(), self.instance_flags),
                    size: size_info.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::TopLevel,
                    allow_compaction: false,
                    allow_motion: moves,
                    max_instances: desc.max_instances,
                    opacity_micromaps: &[],
                })
        }
        .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        let raw_instance_size = match moves {
            true => self.alignments.raw_tlas_motion_instance_size,
            false => self.alignments.raw_tlas_instance_size,
        };
        let instance_buffer_size = raw_instance_size
            .checked_mul(desc.max_instances.max(1))
            .expect("max_tlas_instance_count should not allow excessive buffer size");
        let (instance_buffer, _) = unsafe {
            self.raw().create_buffer(&hal::BufferDescriptor {
                label: hal_label(Some("(wgpu-core) instances_buffer"), self.instance_flags),
                size: u64::from(instance_buffer_size),
                usage: wgt::BufferUses::COPY_DST
                    | wgt::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                memory_flags: hal::MemoryFlags::PREFER_COHERENT,
            })
        }
        .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        Ok(Arc::new(resource::Tlas {
            state: ResourceState::Valid(resource::TlasState {
                raw: Snatchable::new(raw),
                instance_buffer,
            }),
            device: self.clone(),
            size_info,
            flags: desc.flags,
            update_mode: desc.update_mode,
            built_index: RwLock::new(rank::TLAS_BUILT_INDEX, None),
            dependencies: RwLock::new(rank::TLAS_DEPENDENCIES, Vec::new()),
            label: desc.label.to_string(),
            max_instance_count: desc.max_instances,
            tracking_data: TrackingData::new(self.tracker_indices.tlas_s.clone()),
            max_intersection_index: RwLock::new(rank::TLAS_MAX_INTERSECTION_IDX, None),
            required_intersection_types: RwLock::new(
                rank::TLAS_REQUIRED_INTERSECTION_TYPES,
                Vec::new(),
            ),
        }))
    }

    pub fn create_ray_tracing_pipeline(
        self: &Arc<Self>,
        desc: ray_tracing_pipeline::RayTracingPipelineDescriptor,
    ) -> (
        Arc<ray_tracing_pipeline::RayTracingPipeline>,
        Option<CreatePipelineError>,
    ) {
        let (ray_tracing_pipeline, error) =
            match self.create_ray_tracing_pipeline_inner(desc.clone()) {
                Ok(ray_tracing_pipeline) => (ray_tracing_pipeline, None),
                Err(error) => (
                    ray_tracing_pipeline::RayTracingPipeline::invalid(
                        self.clone(),
                        desc.label.to_string(),
                    ),
                    Some(error),
                ),
            };
        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *self.trace.lock() {
            use crate::device::trace::IntoTrace;
            trace.add(Action::CreateRayTracingPipeline {
                id: ray_tracing_pipeline.to_trace(),
                desc: desc.to_trace(),
            });
        }
        (ray_tracing_pipeline, error)
    }

    pub fn create_ray_tracing_pipeline_inner(
        self: &Arc<Self>,
        desc: ray_tracing_pipeline::RayTracingPipelineDescriptor,
    ) -> Result<Arc<ray_tracing_pipeline::RayTracingPipeline>, CreatePipelineError> {
        use crate::validation;

        self.check_is_valid()?;
        self.require_features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)?;

        let io = validation::StageIo::default();

        let is_auto_layout = desc.layout.is_none();

        // Get the pipeline layout from the desc if it is provided.
        let pipeline_layout = match desc.layout {
            Some(pipeline_layout) => {
                pipeline_layout.same_device(self)?;
                Some(pipeline_layout)
            }
            None => None,
        };

        let mut compiler = StageCompiler::new(self, pipeline_layout.clone(), io, None);

        let ray_generation = {
            let stage = validation::ShaderStageForValidation::RayGeneration;

            compiler.compile(&desc.ray_generation, stage)?
        };

        if desc.misses.is_empty() {
            return Err(ray_tracing_pipeline::RayTracingPipelineError::NoMissStage.into());
        }

        let misses = desc
            .misses
            .iter()
            .map(|miss| compiler.compile(miss, validation::ShaderStageForValidation::Miss))
            .collect::<Result<Vec<_>, _>>()?;

        if desc.intersections.len() > 1 << 24 {
            return Err(
                ray_tracing_pipeline::RayTracingPipelineError::TooManyIntersectionGroups(
                    desc.intersections.len(),
                )
                .into(),
            );
        }

        if desc.callables.len() > 1 << 24 {
            return Err(
                ray_tracing_pipeline::RayTracingPipelineError::TooManyCallables(
                    desc.callables.len(),
                )
                .into(),
            );
        }

        if desc.max_geometry_stride > 1 << 24 {
            return Err(
                ray_tracing_pipeline::RayTracingPipelineError::TooHighGeometryStride(
                    desc.max_geometry_stride,
                )
                .into(),
            );
        }

        if desc.max_recursion_depth > self.limits.max_ray_recursion_depth {
            return Err(
                ray_tracing_pipeline::RayTracingPipelineError::TooHighRayRecursionDepth(
                    desc.max_recursion_depth,
                    self.limits.max_ray_recursion_depth,
                )
                .into(),
            );
        }

        let mut intersections = Vec::with_capacity(desc.intersections.len());
        let mut compiled_intersections = Vec::with_capacity(desc.intersections.len());

        for (group, intersection) in desc.intersections.iter().enumerate() {
            let (closest_hit, any_hit, intersection, triangle) = match intersection {
                ray_tracing_pipeline::RayTracingIntersectionDescriptor::Triangle {
                    closest_hit,
                    any_hit,
                } => (Some(closest_hit), any_hit.as_ref(), None, true),
                ray_tracing_pipeline::RayTracingIntersectionDescriptor::Procedural {
                    intersection,
                    closest_hit,
                    any_hit,
                } => (
                    closest_hit.as_ref(),
                    any_hit.as_ref(),
                    Some(intersection),
                    false,
                ),
            };

            // The hit attribute a group's intersection stage writes is the
            // one its other stages read: barycentrics for triangles.
            let intersection = intersection
                .map(|desc| {
                    compile_hit_group_stage(
                        &mut compiler,
                        desc,
                        validation::ShaderStageForValidation::Intersection,
                    )
                })
                .transpose()?;
            let written = intersection
                .as_ref()
                .and_then(|(_, attribute, _)| attribute.clone());
            let mut hit_stages = [None, None];
            for (slot, desc, stage) in [
                (
                    0,
                    closest_hit,
                    validation::ShaderStageForValidation::ClosestHit { triangle },
                ),
                (
                    1,
                    any_hit,
                    validation::ShaderStageForValidation::AnyHit { triangle },
                ),
            ] {
                let Some(desc) = desc else {
                    continue;
                };
                let naga_stage = stage.to_naga();
                let (compiled, attribute, barycentrics) =
                    compile_hit_group_stage(&mut compiler, desc, stage)?;
                let error = match (triangle, attribute) {
                    (true, Some(_)) => Some(
                        ray_tracing_pipeline::RayTracingPipelineError::HitAttributeOfTriangles(
                            group, naga_stage,
                        ),
                    ),
                    (false, _) if barycentrics => Some(
                        ray_tracing_pipeline::RayTracingPipelineError::BarycentricsOfProcedural(
                            group, naga_stage,
                        ),
                    ),
                    (false, Some(read)) if Some(&read) != written.as_ref() => Some(
                        ray_tracing_pipeline::RayTracingPipelineError::HitAttributeMismatch(
                            group, naga_stage,
                        ),
                    ),
                    _ => None,
                };
                if let Some(error) = error {
                    return Err(error.into());
                }
                hit_stages[slot] = Some(compiled);
            }
            let [closest_hit, any_hit] = hit_stages;

            compiled_intersections.push((
                closest_hit,
                any_hit,
                intersection.map(|(compiled, _, _)| compiled),
            ));
        }

        let compiled_callables = desc
            .callables
            .iter()
            .map(|callable| {
                compiler.compile(callable, validation::ShaderStageForValidation::Callable)
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (closest_hit, any_hit, intersection) in compiled_intersections.iter() {
            intersections.push(hal::RayObjectIntersectionState {
                closest_hit: closest_hit.as_ref().map(|closest| closest.to_hal()),
                any_hit: any_hit.as_ref().map(|any| any.to_hal()),
                intersection: intersection.as_ref().map(|stage| stage.to_hal()),
            });
        }

        let StageCompiler { resources, .. } = compiler;

        let pipeline_layout = match resources.layout {
            validation::BindingLayoutSource::Provided(layout) => layout,
            validation::BindingLayoutSource::Derived(entries) => {
                self.create_derived_pipeline_layout(entries, resources.immediates.size())?
            }
        };

        let late_sized_buffer_groups = Device::make_late_sized_buffer_groups(
            &resources.minimum_binding_sizes,
            &pipeline_layout,
        );

        let cache = match desc.cache {
            Some(cache) => {
                cache.same_device(self)?;
                Some(cache)
            }
            None => None,
        };

        let raw = {
            let misses: Vec<_> = misses.iter().map(|miss| miss.to_hal()).collect();
            let callables: Vec<_> = compiled_callables
                .iter()
                .map(|callable| callable.to_hal())
                .collect();
            let pipeline_desc = hal::RayTracingPipelineDescriptor {
                label: desc.label.to_hal(self.instance_flags),
                layout: pipeline_layout.raw()?,
                ray_generation: ray_generation.to_hal(),
                misses: &misses,
                intersection: &intersections,
                callables: &callables,
                max_geometry_stride: desc.max_geometry_stride,
                max_recursion_depth: desc.max_recursion_depth,
                cache: cache.as_ref().map(|it| it.raw()).transpose()?,
            };
            unsafe { self.raw().create_ray_tracing_pipeline(&pipeline_desc) }.map_err(|err| {
                match err {
                    hal::PipelineError::Device(error) => {
                        CreatePipelineError::Device(self.handle_hal_error(error))
                    }
                    hal::PipelineError::Linkage(stage, message) => {
                        CreatePipelineError::Internal { stage, message }
                    }
                    hal::PipelineError::EntryPoint(stage) => CreatePipelineError::Internal {
                        stage: hal::auxil::map_naga_stage(stage),
                        message: ENTRYPOINT_FAILURE_ERROR.to_string(),
                    },
                    hal::PipelineError::PipelineConstants(stage, message) => {
                        CreatePipelineError::PipelineConstants { stage, message }
                    }
                }
            })?
        };

        let (shader_modules, intersection_types) = {
            let mut shader_modules = Vec::new();
            shader_modules.push(desc.ray_generation.module.clone());
            shader_modules.extend(desc.misses.iter().map(|miss| miss.module.clone()));
            shader_modules.reserve(desc.intersections.len());
            let mut intersection_types = Vec::with_capacity(desc.intersections.len());
            for intersection in &desc.intersections {
                match intersection {
                    ray_tracing_pipeline::RayTracingIntersectionDescriptor::Triangle {
                        closest_hit,
                        any_hit,
                    } => {
                        shader_modules.push(closest_hit.module.clone());
                        if let Some(any) = any_hit {
                            shader_modules.push(any.module.clone());
                        }
                        intersection_types
                            .push(ray_tracing_pipeline::RayTracingIntersectionType::Triangle);
                    }
                    ray_tracing_pipeline::RayTracingIntersectionDescriptor::Procedural {
                        intersection,
                        closest_hit,
                        any_hit,
                    } => {
                        shader_modules.push(intersection.module.clone());
                        shader_modules.extend(closest_hit.iter().map(|stage| stage.module.clone()));
                        shader_modules.extend(any_hit.iter().map(|stage| stage.module.clone()));
                        intersection_types
                            .push(ray_tracing_pipeline::RayTracingIntersectionType::AABB);
                    }
                }
            }
            shader_modules.extend(desc.callables.iter().map(|stage| stage.module.clone()));
            (
                shader_modules,
                Arc::from(intersection_types.into_boxed_slice()),
            )
        };

        // Won't panic because the length of `desc.intersections` is required to be below 2^24
        // (see `CreateRayTracingPipelineError::TooManyIntersectionGroups`)
        let shader_binding_data = match ray_tracing_pipeline::ShaderBindingData::from_raw_pipeline(
            self.clone(),
            raw.as_ref(),
            desc.misses.len(),
            desc.intersections.len(),
            desc.callables.len(),
        ) {
            Ok(sbd) => sbd,
            Err(e) => {
                // We need to destroy the raw ray tracing pipeline first.
                unsafe { self.raw().destroy_ray_tracing_pipeline(raw) };
                return Err(e);
            }
        };

        let naga::valid::ImmediateUsage::Valid {
            slots: immediate_slots_required,
            size: _,
        } = resources.immediates
        else {
            unreachable!("Immediates exceeding maxImmediateSize should have been rejected");
        };

        let pipeline = ray_tracing_pipeline::RayTracingPipeline {
            state: ResourceState::Valid(ray_tracing_pipeline::RayTracingPipelineState {
                raw: ManuallyDrop::new(raw),
                layout: pipeline_layout.clone(),
                _shader_modules: shader_modules,
                shader_binding_data,
                intersection_types,
                max_geometry_stride: desc.max_geometry_stride,
            }),
            device: self.clone(),
            late_sized_buffer_groups,
            immediate_slots_required,
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.ray_tracing_pipelines.clone()),
        };

        let pipeline = Arc::new(pipeline);

        if is_auto_layout {
            for bgl in pipeline_layout.bind_group_layouts.iter() {
                let Some(bgl) = bgl else {
                    continue;
                };

                // `bind_group_layouts` might contain duplicate entries, so we need to ignore the result.
                let _ = bgl.exclusive_pipeline.set((&pipeline).into());
            }
        }

        Ok(pipeline)
    }
}
