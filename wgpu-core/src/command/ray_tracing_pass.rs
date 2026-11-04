use core::{convert::Infallible, fmt};

use alloc::{borrow::Cow, boxed::Box, sync::Arc, vec::Vec};

use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    BufferAddress, DynamicOffset,
};

use crate::{
    api_log,
    binding_model::{BindError, BindGroup, ImmediateUploadError, LateMinBufferBindingSizeMismatch},
    command::{
        bind::{Binder, BinderError},
        memory_init::{fixup_discarded_surfaces, SurfacesInDiscardState},
        pass::{self, flush_bindings_helper, ImmediateState},
        pass_base, pass_try,
        query::record_pass_timestamp_writes,
        ray_tracing_pass_commands::ArcRayTracingCommand,
        ArcCommand, BasePass, BindGroupStateChange, CommandEncoder, CommandEncoderError,
        DebugGroupError, EncoderStateError, EncodingState, InnerCommandEncoder, MapPassErr,
        PassErrorScope, PassStateError, PassTimestampWrites, QueryUseError, StateChange,
        TimestampWritesError,
    },
    device::{Device, DeviceError, MissingDownlevelFlags, MissingFeatures},
    hal_label, id, impl_resource_type,
    init_tracker::MemoryInitKind,
    ray_tracing::AsAction,
    ray_tracing_pipeline::RayTracingPipeline,
    resource::{
        Buffer, DestroyedResourceError, InvalidOrDestroyedResourceError, InvalidResourceError,
        Labeled, MissingBufferUsageError, ParentDevice, QuerySet, RawResourceAccess, Trackable,
    },
    track::{ResourceUsageCompatibilityError, Tracker},
    Label,
};

pub type RayTracingBasePass = BasePass<ArcRayTracingCommand, RayTracingPassError>;

/// Very similar to [`super::compute::ComputePass`]
pub struct RayTracingPass {
    /// All pass data & records is stored here.
    base: RayTracingBasePass,

    /// Parent command encoder that this pass records commands into.
    ///
    /// Implications are the same as [`super::compute::ComputePass::parent`]
    parent: Option<Arc<CommandEncoder>>,

    device: Arc<Device>,

    timestamp_writes: Option<PassTimestampWrites>,

    current_bind_groups: BindGroupStateChange,
    current_pipeline: StateChange<Arc<RayTracingPipeline>>,
}

impl_resource_type!(RayTracingPass);

impl crate::storage::StorageItem for RayTracingPass {
    type Marker = id::markers::RayTracingPassEncoder;
}

impl fmt::Debug for RayTracingPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.parent {
            Some(ref cmd_enc) => {
                write!(f, "RayTracingPass {{ parent: {} }}", cmd_enc.error_ident())
            }
            None => write!(f, "RayTracingPass {{ parent: None }}"),
        }
    }
}

impl RayTracingPass {
    /// If the parent command encoder is invalid, the returned pass will be invalid.
    fn new(
        parent: Arc<CommandEncoder>,
        desc: RayTracingPassDescriptor,
    ) -> Result<Self, MissingFeatures> {
        parent
            .device
            .require_features(wgt::Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)?;

        let RayTracingPassDescriptor {
            label,
            timestamp_writes,
        } = desc;

        Ok(Self {
            base: BasePass::new(&label),
            device: parent.device.clone(),
            parent: Some(parent),
            timestamp_writes,

            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        })
    }

    fn new_invalid(parent: Arc<CommandEncoder>, label: &Label, err: RayTracingPassError) -> Self {
        Self {
            base: BasePass::new_invalid(label, err),
            device: parent.device.clone(),
            parent: Some(parent),
            timestamp_writes: None,
            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        }
    }

    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.base.label.as_deref()
    }

    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<Arc<BindGroup>>,
        offsets: &[DynamicOffset],
    ) {
        if let Err(err) = self.set_bind_group_inner(index, bind_group, offsets) {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::set_bind_group");
        }
    }

    pub fn set_bind_group_inner(
        &mut self,
        index: u32,
        bind_group: Option<Arc<BindGroup>>,
        offsets: &[DynamicOffset],
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetBindGroup;

        // This statement will return an error if the pass is ended. It's
        // important the error check comes before the early-out for
        // `set_and_check_redundant`.
        let base = pass_base!(self, scope);

        if self.current_bind_groups.set_and_check_redundant(
            &bind_group,
            index,
            &mut base.dynamic_offsets,
            offsets,
        ) {
            return Ok(());
        }

        let bind_group = if let Some(bind_group) = bind_group {
            pass_try!(base, scope, bind_group.check_is_valid());
            Some(bind_group)
        } else {
            None
        };

        base.commands.push(ArcRayTracingCommand::SetBindGroup {
            index,
            num_dynamic_offsets: offsets.len(),
            bind_group,
        });

        Ok(())
    }

    pub fn set_immediates_inner(&mut self, offset: u32, data: &[u8]) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetImmediate;
        let base = pass_base!(self, scope);

        pass_try!(
            base,
            scope,
            pass::validate_immediates_alignment(offset, data.len())
        );

        base.commands.push(ArcRayTracingCommand::SetImmediate {
            offset,
            data: data
                .chunks_exact(size_of::<u32>())
                .map(|ck| u32::from_le_bytes(ck.try_into().unwrap()))
                .collect(),
        });

        Ok(())
    }
    pub fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        if let Err(err) = self.set_immediates_inner(offset, data) {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::set_immediates");
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RayTracingPassDescriptor<'a, PTW = PassTimestampWrites> {
    pub label: Label<'a>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<PTW>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TraceRayError {
    #[error("Ray tracing pipeline must be set")]
    MissingPipeline(pass::MissingPipeline),
    #[error(transparent)]
    IncompatibleBindGroup(#[from] Box<BinderError>),
    #[error("Trace ray size ({current:?}) must be less or equal to {limit:?}")]
    InvalidDimensionSize { current: [u32; 3], limit: [u32; 3] },
    #[error("The total count of rays invocations ({current:?}) must be less or equal to {limit}")]
    TooManyTotal { current: u32, limit: u32 },
    #[error(transparent)]
    BindingSizeTooSmall(#[from] LateMinBufferBindingSizeMismatch),
    #[error("Not all immediate data required by the pipeline has been set via set_immediates (missing byte ranges: {missing})")]
    MissingImmediateData {
        missing: naga::valid::ImmediateSlots,
    },
}

impl WebGpuError for TraceRayError {
    fn webgpu_error_type(&self) -> ErrorType {
        ErrorType::Validation
    }
}

/// Error encountered when performing a ray tracing pass.
#[derive(Clone, Debug, Error)]
pub enum RayTracingPassErrorInner {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    EncoderState(#[from] EncoderStateError),
    #[error("Parent encoder is invalid")]
    InvalidParentEncoder,
    #[error(transparent)]
    DebugGroupError(#[from] DebugGroupError),
    #[error(transparent)]
    BindGroupIndexOutOfRange(#[from] pass::BindGroupIndexOutOfRange),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error("Indirect buffer offset {0:?} is not a multiple of 4")]
    UnalignedIndirectBufferOffset(BufferAddress),
    #[error("Indirect buffer uses bytes {offset}..{end_offset} which overruns indirect buffer of size {buffer_size}")]
    IndirectBufferOverrun {
        offset: u64,
        end_offset: u64,
        buffer_size: u64,
    },
    #[error(transparent)]
    ResourceUsageCompatibility(#[from] ResourceUsageCompatibilityError),
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error(transparent)]
    TraceRay(#[from] TraceRayError),
    #[error(transparent)]
    Bind(#[from] BindError),
    #[error(transparent)]
    ImmediateData(#[from] ImmediateUploadError),
    #[error("Immediate data offset must be aligned to 4 bytes")]
    ImmediateOffsetAlignment,
    #[error("Immediate data size must be aligned to 4 bytes")]
    ImmediateDataizeAlignment,
    #[error("Ran out of immediate data space. Don't set 4gb of immediates per RayTracingPass.")]
    ImmediateOutOfMemory,
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error("The ray tracing pass has already been ended and no further commands can be recorded")]
    PassEnded,
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error(transparent)]
    QueryUse(#[from] QueryUseError),
    #[error(transparent)]
    TimestampWrites(#[from] TimestampWritesError),
}

impl From<InvalidOrDestroyedResourceError> for RayTracingPassErrorInner {
    fn from(error: InvalidOrDestroyedResourceError) -> Self {
        match error {
            InvalidOrDestroyedResourceError::InvalidResource(e) => Self::InvalidResource(e),
            InvalidOrDestroyedResourceError::DestroyedResource(e) => Self::DestroyedResource(e),
        }
    }
}

/// Error encountered when performing a ray tracing pass, stored for later reporting
/// when encoding ends.
#[derive(Clone, Debug, Error)]
#[error("{scope}")]
pub struct RayTracingPassError {
    pub scope: PassErrorScope,
    #[source]
    pub(super) inner: Box<RayTracingPassErrorInner>,
}

impl From<pass::MissingPipeline> for RayTracingPassErrorInner {
    fn from(value: pass::MissingPipeline) -> Self {
        Self::TraceRay(TraceRayError::MissingPipeline(value))
    }
}

impl<E> MapPassErr<RayTracingPassError> for E
where
    E: Into<RayTracingPassErrorInner>,
{
    fn map_pass_err(self, scope: PassErrorScope) -> RayTracingPassError {
        RayTracingPassError {
            scope,
            inner: Box::new(self.into()),
        }
    }
}

impl WebGpuError for RayTracingPassError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self.inner.as_ref() {
            RayTracingPassErrorInner::Device(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::EncoderState(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::DebugGroupError(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::DestroyedResource(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::ResourceUsageCompatibility(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::MissingBufferUsage(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::TraceRay(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::Bind(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::ImmediateData(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::MissingFeatures(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::MissingDownlevelFlags(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::InvalidResource(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::QueryUse(e) => e.webgpu_error_type(),
            RayTracingPassErrorInner::TimestampWrites(e) => e.webgpu_error_type(),

            RayTracingPassErrorInner::InvalidParentEncoder
            | RayTracingPassErrorInner::BindGroupIndexOutOfRange { .. }
            | RayTracingPassErrorInner::UnalignedIndirectBufferOffset(_)
            | RayTracingPassErrorInner::IndirectBufferOverrun { .. }
            | RayTracingPassErrorInner::ImmediateOffsetAlignment
            | RayTracingPassErrorInner::ImmediateDataizeAlignment
            | RayTracingPassErrorInner::ImmediateOutOfMemory
            | RayTracingPassErrorInner::PassEnded => ErrorType::Validation,
        }
    }
}

struct State<'scope, 'snatch_guard, 'cmd_enc> {
    pipeline: Option<Arc<RayTracingPipeline>>,

    pass: pass::PassState<'scope, 'snatch_guard, 'cmd_enc>,

    intermediate_trackers: Tracker,
}

impl<'scope, 'snatch_guard, 'cmd_enc> State<'scope, 'snatch_guard, 'cmd_enc> {
    fn is_ready(&self) -> Result<(), TraceRayError> {
        if let Some(pipeline) = self.pipeline.as_ref() {
            self.pass.binder.check_compatibility(pipeline.as_ref())?;
            self.pass.binder.check_late_buffer_bindings()?;
            if !self
                .pass
                .immediate_state
                .immediate_slots_set
                .contains(pipeline.immediate_slots_required)
            {
                return Err(TraceRayError::MissingImmediateData {
                    missing: pipeline
                        .immediate_slots_required
                        .difference(self.pass.immediate_state.immediate_slots_set),
                });
            }
            Ok(())
        } else {
            Err(TraceRayError::MissingPipeline(pass::MissingPipeline))
        }
    }

    fn flush_immediates(&mut self) {
        let pipeline = self.pipeline.as_ref().unwrap();
        let layout = pipeline.layout().unwrap();
        self.pass
            .immediate_state
            .flush_immediates(layout, self.pass.base.raw_encoder);
    }

    /// Flush binding state in preparation for a trace rays call.
    ///
    /// This behaves almost the same as a compute pass' equivalent
    /// function, but also adds a `AsAction::TraceTlas` as trace ray
    /// calls need a bit more validation.
    ///
    /// This differs from a render pass, which the equivalent method
    /// on compute pass' state discuss more.
    ///
    /// An indirect trace's buffer is checked for conflicting usages and
    /// moved into its indirect usage, as a compute pass' unvalidated
    /// indirect dispatch's is.
    fn flush_bindings(
        &mut self,
        indirect_buffer: Option<&Arc<Buffer>>,
    ) -> Result<(), RayTracingPassErrorInner> {
        for bind_group in self.pass.binder.list_active() {
            unsafe { self.pass.scope.merge_bind_group(&bind_group.used)? };
        }

        if let Some(buffer) = indirect_buffer {
            self.pass
                .scope
                .buffers
                .merge_single(buffer, wgt::BufferUses::INDIRECT)?;
        }
        // For ray tracing passes, usage scopes are associated with each trace
        // ray (in the same way as compute passes) and not with the pass as a
        // whole. However, because the cost of creating and dropping
        // `UsageScope`s is significant (even with the pool), we add and then
        // remove usage from a single usage scope.

        let intersection_types = self.pipeline.as_ref().unwrap().intersection_types()?;

        for bind_group in self.pass.binder.list_active() {
            self.intermediate_trackers
                .set_and_remove_from_usage_scope_sparse(&mut self.pass.scope, &bind_group.used);

            for tlas in bind_group.used.acceleration_structures.into_iter() {
                self.pass.base.as_actions.push(AsAction::TraceTlas(
                    tlas.clone(),
                    intersection_types.clone(),
                ));
            }
        }

        if let Some(buffer) = indirect_buffer {
            self.intermediate_trackers
                .buffers
                .set_and_remove_from_usage_scope_sparse(
                    &mut self.pass.scope.buffers,
                    Some(buffer.tracker_index()),
                );
        }

        flush_bindings_helper(&mut self.pass)?;

        CommandEncoder::drain_barriers(
            self.pass.base.raw_encoder,
            &mut self.intermediate_trackers,
            self.pass.base.snatch_guard,
        );
        Ok(())
    }
}

// Ray tracing pass commands

impl RayTracingPass {
    pub fn set_pipeline(&mut self, pipeline: Arc<RayTracingPipeline>) {
        if let Err(err) = self.set_pipeline_inner(pipeline) {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::set_pipeline");
        }
    }
    pub fn set_pipeline_inner(
        &mut self,
        pipeline: Arc<RayTracingPipeline>,
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::SetPipelineRayTracing;

        let redundant = self.current_pipeline.set_and_check_redundant(&pipeline);

        // This statement will return an error if the pass is ended.
        // Its important the error check comes before the early-out for `redundant`.
        let base = pass_base!(self, scope);

        if redundant {
            return Ok(());
        }

        pass_try!(base, scope, pipeline.check_valid());

        base.commands
            .push(ArcRayTracingCommand::SetPipeline(pipeline));

        Ok(())
    }
    pub fn end_inner(&mut self) -> Result<(), EncoderStateError> {
        profiling::scope!(
            "CommandEncoder::run_ray_tracing_pass {}",
            self.base.label.as_deref().unwrap_or("")
        );

        let cmd_enc = self.parent.take().ok_or(EncoderStateError::Ended)?;
        let mut cmd_buf_data = cmd_enc.data.lock();

        cmd_buf_data.unlock_encoder()?;

        let base = self.base.take();

        if let Err(RayTracingPassError { inner, scope: _ }) = &base {
            if let RayTracingPassErrorInner::EncoderState(
                err @ (EncoderStateError::Locked | EncoderStateError::Ended),
            ) = inner.as_ref()
            {
                // Most encoding errors are detected and raised within `finish()`.
                //
                // However, we raise a validation error here if the pass was opened
                // within another pass, or on a finished encoder. The latter is
                // particularly important, because in that case reporting errors via
                // `CommandEncoder::finish` is not possible.
                return Err(err.clone());
            }
        }

        cmd_buf_data.push_with(|| -> Result<_, RayTracingPassError> {
            Ok(ArcCommand::RunRayTracingPass {
                pass: base?,
                timestamp_writes: self.timestamp_writes.take(),
            })
        })
    }

    pub fn end(&mut self) {
        if let Err(err) = self.end_inner() {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::end");
        }
    }

    pub fn push_debug_group_inner(
        &mut self,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let base = pass_base!(self, PassErrorScope::PushDebugGroup);

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcRayTracingCommand::PushDebugGroup {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn push_debug_group(&mut self, label: &str, color: u32) {
        if let Err(err) = self.push_debug_group_inner(label, color) {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::push_debug_group");
        }
    }

    pub fn pop_debug_group_inner(&mut self) -> Result<(), PassStateError> {
        let base = pass_base!(self, PassErrorScope::PopDebugGroup);

        base.commands.push(ArcRayTracingCommand::PopDebugGroup);

        Ok(())
    }

    pub fn pop_debug_group(&mut self) {
        if let Err(err) = self.pop_debug_group_inner() {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::pop_debug_group");
        }
    }

    pub fn insert_debug_marker_inner(
        &mut self,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let base = pass_base!(self, PassErrorScope::InsertDebugMarker);

        let bytes = label.as_bytes();
        base.string_data.extend_from_slice(bytes);

        base.commands.push(ArcRayTracingCommand::InsertDebugMarker {
            color,
            len: bytes.len(),
        });

        Ok(())
    }

    pub fn insert_debug_marker(&mut self, label: &str, color: u32) {
        if let Err(err) = self.insert_debug_marker_inner(label, color) {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::insert_debug_marker");
        }
    }

    pub fn trace_rays_inner(
        &mut self,
        count_x: u32,
        count_y: u32,
        count_z: u32,
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::TraceRays;

        pass_base!(self, scope)
            .commands
            .push(ArcRayTracingCommand::TraceRays([count_x, count_y, count_z]));

        Ok(())
    }

    pub fn trace_rays(&mut self, count_x: u32, count_y: u32, count_z: u32) {
        if let Err(err) = self.trace_rays_inner(count_x, count_y, count_z) {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::trace_rays");
        }
    }

    pub fn trace_rays_indirect_inner(
        &mut self,
        buffer: Arc<Buffer>,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        let scope = PassErrorScope::TraceRaysIndirect;

        pass_base!(self, scope)
            .commands
            .push(ArcRayTracingCommand::TraceRaysIndirect { buffer, offset });

        Ok(())
    }

    pub fn trace_rays_indirect(&mut self, buffer: Arc<Buffer>, offset: BufferAddress) {
        if let Err(err) = self.trace_rays_indirect_inner(buffer, offset) {
            self.device
                .handle_error(err, self.label(), "RayTracingPass::trace_rays_indirect");
        }
    }
}

impl CommandEncoder {
    pub fn begin_ray_tracing_pass_inner(
        self: &Arc<Self>,
        desc: &RayTracingPassDescriptor<'_, PassTimestampWrites<Arc<QuerySet>>>,
    ) -> (RayTracingPass, Option<CommandEncoderError>) {
        use EncoderStateError as SErr;

        let label = desc.label.as_deref().map(Cow::Borrowed);

        let scope = PassErrorScope::Pass;

        let mut cmd_buf_data = self.data.lock();

        match cmd_buf_data.lock_encoder() {
            Ok(()) => {
                drop(cmd_buf_data);
                if let Err(err) = self.device.check_is_valid() {
                    return (
                        RayTracingPass::new_invalid(self.clone(), &label, err.map_pass_err(scope)),
                        None,
                    );
                }

                let timestamp_writes = match desc
                    .timestamp_writes
                    .as_ref()
                    .map(|tw| {
                        Self::validate_pass_timestamp_writes::<RayTracingPassErrorInner>(
                            &self.device,
                            tw,
                        )
                    })
                    .transpose()
                {
                    Ok(timestamp_writes) => timestamp_writes,
                    Err(err) => {
                        return (
                            RayTracingPass::new_invalid(
                                self.clone(),
                                &label,
                                err.map_pass_err(scope),
                            ),
                            None,
                        );
                    }
                };

                let pass = match RayTracingPass::new(
                    self.clone(),
                    RayTracingPassDescriptor {
                        label: label.clone(),
                        timestamp_writes,
                    },
                ) {
                    Ok(pass) => pass,
                    Err(err) => {
                        return (
                            RayTracingPass::new_invalid(
                                self.clone(),
                                &label,
                                err.map_pass_err(scope),
                            ),
                            None,
                        );
                    }
                };

                (pass, None)
            }
            Err(err @ SErr::Locked) => {
                // Attempting to open a new pass while the encoder is locked
                // invalidates the encoder, but does not generate a validation
                // error.
                cmd_buf_data.invalidate(err.clone());
                drop(cmd_buf_data);
                (
                    RayTracingPass::new_invalid(self.clone(), &label, err.map_pass_err(scope)),
                    None,
                )
            }
            Err(err @ (SErr::Ended | SErr::Submitted)) => {
                // Attempting to open a new pass after the encode has ended
                // generates an immediate validation error.
                drop(cmd_buf_data);
                (
                    RayTracingPass::new_invalid(
                        self.clone(),
                        &label,
                        err.clone().map_pass_err(scope),
                    ),
                    Some(err.into()),
                )
            }
            Err(err @ SErr::Invalid) => {
                // Passes can be opened even on an invalid encoder. Such passes
                // are even valid, but since there's no visible side-effect of
                // the pass being valid and there's no point in storing recorded
                // commands that will ultimately be discarded, we open an
                // invalid pass to save that work.
                drop(cmd_buf_data);
                (
                    RayTracingPass::new_invalid(self.clone(), &label, err.map_pass_err(scope)),
                    None,
                )
            }
            Err(SErr::Unlocked) => {
                unreachable!("lock_encoder cannot fail due to the encoder being unlocked")
            }
        }
    }

    pub fn begin_ray_tracing_pass(
        self: &Arc<Self>,
        desc: &RayTracingPassDescriptor<'_, PassTimestampWrites<Arc<QuerySet>>>,
    ) -> RayTracingPass {
        let (pass, err) = self.begin_ray_tracing_pass_inner(desc);
        if let Some(err) = err {
            self.device
                .handle_error(err, pass.label(), "CommandEncoder::begin_ray_tracing_pass");
        }
        pass
    }
}

pub(super) fn encode_ray_tracing_pass(
    parent_state: &mut EncodingState<InnerCommandEncoder>,
    mut base: BasePass<ArcRayTracingCommand, Infallible>,
    mut timestamp_writes: Option<PassTimestampWrites>,
) -> Result<(), RayTracingPassError> {
    let pass_scope = PassErrorScope::Pass;

    let device = parent_state.device;

    // We automatically keep extending command buffers over time, and because
    // we want to insert a command buffer _before_ what we're about to record,
    // we need to make sure to close the previous one.
    parent_state
        .raw_encoder
        .close_if_open()
        .map_pass_err(pass_scope)?;
    let raw_encoder = parent_state
        .raw_encoder
        .open_pass(base.label.as_deref())
        .map_pass_err(pass_scope)?;

    let mut debug_scope_depth = 0;

    let mut state = State {
        pipeline: None,

        pass: pass::PassState {
            base: EncodingState {
                device,
                raw_encoder,
                tracker: parent_state.tracker,
                buffer_memory_init_actions: parent_state.buffer_memory_init_actions,
                texture_memory_actions: parent_state.texture_memory_actions,
                as_actions: parent_state.as_actions,
                temp_resources: parent_state.temp_resources,
                indirect_draw_validation_resources: parent_state.indirect_draw_validation_resources,
                snatch_guard: parent_state.snatch_guard,
                debug_scope_depth: &mut debug_scope_depth,
                query_set_writes: parent_state.query_set_writes,
                deferred_query_set_resolves: parent_state.deferred_query_set_resolves,
            },
            binder: Binder::new(),
            temp_offsets: Vec::new(),
            dynamic_offset_count: 0,
            pending_discard_init_fixups: SurfacesInDiscardState::new(),
            scope: device.new_usage_scope(),
            string_offset: 0,

            immediate_state: ImmediateState::default(),
        },

        intermediate_trackers: Tracker::new(
            device.ordered_buffer_usages,
            device.ordered_texture_usages,
        ),
    };

    let indices = &device.tracker_indices;
    state
        .pass
        .base
        .tracker
        .buffers
        .set_size(indices.buffers.size());
    state
        .pass
        .base
        .tracker
        .textures
        .set_size(indices.textures.size());

    // As a compute pass' timestamps: reset here, as there is no auxiliary
    // pass to reset them in.
    let timestamp_writes: Option<hal::PassTimestampWrites<'_, dyn hal::DynQuerySet>> =
        if let Some(tw) = timestamp_writes.take() {
            tw.query_set.same_device(device).map_pass_err(pass_scope)?;

            record_pass_timestamp_writes(&tw, state.pass.base.query_set_writes);

            let query_set = state
                .pass
                .base
                .tracker
                .query_sets
                .insert_single(tw.query_set);

            let range = if let (Some(index_a), Some(index_b)) =
                (tw.beginning_of_pass_write_index, tw.end_of_pass_write_index)
            {
                Some(index_a.min(index_b)..index_a.max(index_b) + 1)
            } else {
                tw.beginning_of_pass_write_index
                    .or(tw.end_of_pass_write_index)
                    .map(|i| i..i + 1)
            };
            let raw_query_set = query_set
                .try_raw(parent_state.snatch_guard)
                .map_pass_err(pass_scope)?;
            if let Some(range) = range {
                unsafe {
                    state
                        .pass
                        .base
                        .raw_encoder
                        .reset_queries(raw_query_set, range);
                }
            }

            Some(hal::PassTimestampWrites {
                query_set: raw_query_set,
                beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                end_of_pass_write_index: tw.end_of_pass_write_index,
            })
        } else {
            None
        };

    let hal_desc = hal::RayTracingPassDescriptor {
        label: hal_label(base.label.as_deref(), device.instance_flags),
        timestamp_writes,
    };

    unsafe {
        state
            .pass
            .base
            .raw_encoder
            .begin_ray_tracing_pass(&hal_desc);
    }

    for command in base.commands.drain(..) {
        match command {
            ArcRayTracingCommand::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => {
                let scope = PassErrorScope::SetBindGroup;
                pass::set_bind_group::<RayTracingPassErrorInner>(
                    &mut state.pass,
                    device,
                    &base.dynamic_offsets,
                    index,
                    num_dynamic_offsets,
                    bind_group,
                    false,
                )
                .map_pass_err(scope)?;
            }
            ArcRayTracingCommand::SetPipeline(pipeline) => {
                let scope = PassErrorScope::SetPipelineRayTracing;
                set_pipeline(&mut state, device, pipeline).map_pass_err(scope)?;
            }
            ArcRayTracingCommand::SetImmediate { offset, data } => {
                let scope = PassErrorScope::SetImmediate;
                state
                    .pass
                    .immediate_state
                    .set_immediates::<RayTracingPassErrorInner>(
                        &state.pass.base.device.limits,
                        offset,
                        &data,
                    )
                    .map_pass_err(scope)?;
            }
            ArcRayTracingCommand::PushDebugGroup { color: _, len } => {
                pass::push_debug_group(&mut state.pass, &base.string_data, len);
            }
            ArcRayTracingCommand::PopDebugGroup => {
                let scope = PassErrorScope::PopDebugGroup;
                pass::pop_debug_group::<RayTracingPassErrorInner>(&mut state.pass)
                    .map_pass_err(scope)?;
            }
            ArcRayTracingCommand::InsertDebugMarker { color: _, len } => {
                pass::insert_debug_marker(&mut state.pass, &base.string_data, len);
            }
            ArcRayTracingCommand::TraceRays(groups) => {
                let scope = PassErrorScope::TraceRays;
                trace_rays(&mut state, groups, device).map_pass_err(scope)?;
            }
            ArcRayTracingCommand::TraceRaysIndirect { buffer, offset } => {
                let scope = PassErrorScope::TraceRaysIndirect;
                trace_rays_indirect(&mut state, device, buffer, offset).map_pass_err(scope)?;
            }
        }
    }

    if *state.pass.base.debug_scope_depth > 0 {
        Err(
            RayTracingPassErrorInner::DebugGroupError(DebugGroupError::MissingPop)
                .map_pass_err(pass_scope),
        )?;
    }

    unsafe {
        state.pass.base.raw_encoder.end_ray_tracing_pass();
    }

    let State {
        pass: pass::PassState {
            pending_discard_init_fixups,
            ..
        },
        intermediate_trackers,
        ..
    } = state;

    // Stop the current command encoder.
    parent_state.raw_encoder.close().map_pass_err(pass_scope)?;

    // Create a new command encoder, which we will insert _before_ the body of the ray tracing pass.
    //
    // Use that buffer to insert barriers and clear discarded images.
    let transit = parent_state
        .raw_encoder
        .open_pass(hal_label(
            Some("(wgpu internal) Pre Pass"),
            device.instance_flags,
        ))
        .map_pass_err(pass_scope)?;
    // If the ray tracing pass reads any surfaces that were discarded by a previous
    // render pass in the same command buffer, initialize them.
    fixup_discarded_surfaces(
        pending_discard_init_fixups.into_iter(),
        transit,
        &mut parent_state.tracker.textures,
        device,
        parent_state.snatch_guard,
    );
    CommandEncoder::insert_barriers_from_tracker(
        transit,
        parent_state.tracker,
        &intermediate_trackers,
        parent_state.snatch_guard,
    );
    // Close the command encoder, and swap it with the previous.
    parent_state
        .raw_encoder
        .close_and_swap()
        .map_pass_err(pass_scope)?;

    Ok(())
}

fn set_pipeline(
    state: &mut State,
    device: &Arc<Device>,
    pipeline: Arc<RayTracingPipeline>,
) -> Result<(), RayTracingPassErrorInner> {
    pipeline.same_device(device)?;

    state.pipeline = Some(pipeline.clone());

    let pipeline = state
        .pass
        .base
        .tracker
        .ray_tracing_pipelines
        .insert_single(pipeline)
        .clone();

    unsafe {
        state
            .pass
            .base
            .raw_encoder
            .set_ray_tracing_pipeline(pipeline.raw()?);
    }

    // Rebind resources
    let pipeline_layout = pipeline.layout()?;
    pass::change_pipeline_layout::<RayTracingPassErrorInner>(
        &mut state.pass,
        pipeline_layout,
        &pipeline.late_sized_buffer_groups,
    )
}

fn trace_rays(
    state: &mut State,
    dims: [u32; 3],
    device: &Device,
) -> Result<(), RayTracingPassErrorInner> {
    api_log!("RayTracingPass::trace_rays {dims:?}");

    state.is_ready()?;

    state.flush_bindings(None)?;
    state.flush_immediates();

    let limits = &state.pass.base.device.limits;

    // `saturating_mul` is fine here as it is the limit, so a lower limit if it would overflow is just a
    // slightly greater restriction.
    let dim_size_limit = [
        limits
            .max_compute_workgroup_size_x
            .saturating_mul(limits.max_compute_workgroups_per_dimension),
        limits
            .max_compute_workgroup_size_y
            .saturating_mul(limits.max_compute_workgroups_per_dimension),
        limits
            .max_compute_workgroup_size_z
            .saturating_mul(limits.max_compute_workgroups_per_dimension),
    ];

    if dims[0] > dim_size_limit[0] {
        return Err(RayTracingPassErrorInner::TraceRay(
            TraceRayError::InvalidDimensionSize {
                current: dims,
                limit: dim_size_limit,
            },
        ));
    }

    if dims[1] > dim_size_limit[1] {
        return Err(RayTracingPassErrorInner::TraceRay(
            TraceRayError::InvalidDimensionSize {
                current: dims,
                limit: dim_size_limit,
            },
        ));
    }

    if dims[2] > dim_size_limit[2] {
        return Err(RayTracingPassErrorInner::TraceRay(
            TraceRayError::InvalidDimensionSize {
                current: dims,
                limit: dim_size_limit,
            },
        ));
    }

    let tot_rays = dims[0]
        .checked_mul(dims[1])
        .and_then(|tmp| tmp.checked_mul(dims[2]));

    if tot_rays.is_none_or(|tot_rays| tot_rays > limits.max_ray_dispatch_count) {
        return Err(RayTracingPassErrorInner::TraceRay(
            TraceRayError::TooManyTotal {
                current: tot_rays.unwrap_or(u32::MAX),
                limit: limits.max_ray_dispatch_count,
            },
        ));
    }

    let current_pipeline = state.pipeline.as_ref().unwrap();
    let shader_binding_data = current_pipeline.shader_binding_data()?;
    let [ray_generation, miss, intersection] = group_data(shader_binding_data, device);

    unsafe {
        state
            .pass
            .base
            .raw_encoder
            .trace_rays(dims, ray_generation, miss, intersection);
    }
    Ok(())
}

/// The ray generation, miss and intersection regions of a pipeline's
/// shader binding table.
fn group_data<'a>(
    shader_binding_data: &'a crate::ray_tracing_pipeline::ShaderBindingData,
    device: &Device,
) -> [hal::PipelineGroupData<'a, dyn hal::DynBuffer>; 3] {
    let stride = device.alignments.ray_tracing_pipeline_group_data_size as BufferAddress;
    let region = |offset, count: u32| hal::PipelineGroupData {
        buffer: shader_binding_data.raw.as_ref(),
        offset,
        stride,
        count: count as _,
    };
    [
        region(0, 1),
        region(shader_binding_data.miss_offset, shader_binding_data.num_misses),
        region(
            shader_binding_data.intersection_offset,
            shader_binding_data.intersection_records,
        ),
    ]
}

/// A trace of the size `buffer` holds at `offset`.
///
/// Unlike a compute pass' indirect dispatch, the size is not checked
/// against the device's limits: a trace past `max_ray_dispatch_count` rays
/// is the driver's to handle.
fn trace_rays_indirect(
    state: &mut State,
    device: &Arc<Device>,
    buffer: Arc<Buffer>,
    offset: BufferAddress,
) -> Result<(), RayTracingPassErrorInner> {
    api_log!("RayTracingPass::trace_rays_indirect");

    buffer.same_device(device)?;

    state.is_ready()?;

    state
        .pass
        .base
        .device
        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)?;

    buffer.check_usage(wgt::BufferUsages::INDIRECT)?;

    if !offset.is_multiple_of(4) {
        return Err(RayTracingPassErrorInner::UnalignedIndirectBufferOffset(
            offset,
        ));
    }

    // Three `u32`s: width, height and depth.
    let args_size = 3 * 4;
    if buffer.size < args_size || buffer.size - args_size < offset {
        return Err(RayTracingPassErrorInner::IndirectBufferOverrun {
            offset,
            end_offset: offset + args_size,
            buffer_size: buffer.size,
        });
    }

    buffer.check_destroyed(state.pass.base.snatch_guard)?;

    state.pass.base.buffer_memory_init_actions.extend(
        buffer.initialization_status.read().create_action(
            &buffer,
            offset..(offset + args_size),
            MemoryInitKind::NeedsInitializedMemory,
        ),
    );

    state.flush_bindings(Some(&buffer))?;
    state.flush_immediates();

    let buf_raw = buffer.try_raw(state.pass.base.snatch_guard)?;
    let current_pipeline = state.pipeline.as_ref().unwrap();
    let shader_binding_data = current_pipeline.shader_binding_data()?;
    let [ray_generation, miss, intersection] = group_data(shader_binding_data, device);

    unsafe {
        state.pass.base.raw_encoder.trace_rays_indirect(
            ray_generation,
            miss,
            intersection,
            buf_raw,
            offset,
        );
    }
    Ok(())
}
