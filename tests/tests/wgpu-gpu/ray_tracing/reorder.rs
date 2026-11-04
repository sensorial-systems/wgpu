use std::{mem, sync::mpsc};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BindGroupDescriptor, BindGroupEntry, BlasBuildEntry,
    BlasGeometries, BlasGeometrySizeDescriptors, BlasTriangleGeometry,
    BlasTriangleGeometrySizeDescriptor, BufferAddress, BufferUsages, CreateBlasDescriptor,
    CreateTlasDescriptor, Features, Limits, RayTracingIntersectionDescriptor,
    RayTracingPassDescriptor, RayTracingPipelineDescriptor, RayTracingStage,
    ShaderModuleDescriptor, TlasInstance, VertexFormat,
};
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(HIT_OBJECTS_EXECUTE_WHAT_THEY_HOLD);
}

#[apply(gpu_test!)]
static HIT_OBJECTS_EXECUTE_WHAT_THEY_HOLD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::EXPERIMENTAL_RAY_TRACING_PIPELINES
                    | Features::EXPERIMENTAL_RAY_TRACING_INVOCATION_REORDER,
            )
            .limits(
                Limits::defaults()
                    .using_minimum_supported_acceleration_structure_values()
                    .using_minimum_supported_ray_tracing_pipeline_values(),
            ),
    )
    .run_sync(hit_objects_execute_what_they_hold);

/// Rays traced into hit objects, which hit a triangle or miss it, and hit
/// objects made misses and emptied; all reordered, then executed: each runs
/// the closest-hit stage of its hit, the miss stage of its miss, or nothing,
/// and reads back what it holds.
fn hit_objects_execute_what_they_hold(ctx: TestingContext) {
    // A triangle in the z = 5 plane around the origin's ray.
    let corners: [[f32; 3]; 3] = [[-3.0, -2.0, 5.0], [3.0, -2.0, 5.0], [0.0, 4.0, 5.0]];
    let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&corners),
        usage: BufferUsages::BLAS_INPUT,
    });
    let size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
        opacity_micromap: Vec::new(),
    };
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: None,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size.clone()],
        },
    );
    let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: None,
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    tlas[0] = Some(TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        0,
        0xFF,
        0,
    ));
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: mem::size_of::<[f32; 3]>() as BufferAddress,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
                end_vertex_buffer: None,
                opacity_micromap: None,
            }]),
        }],
        [&tlas],
    );
    ctx.queue.submit([encoder.finish()]);

    let gen = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray generation"),
        source: wgpu::ShaderSource::Wgsl(
            "
            enable wgpu_ray_tracing_pipeline;
            enable wgpu_ray_tracing_invocation_reorder;

            var<ray_payload> payload: f32;
            @group(0) @binding(0) var acc_struct: acceleration_structure;
            @group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>, 16>;

            // Ray i % 4: traced at the triangle, traced past it, made a miss,
            // emptied.
            @ray_generation
            fn gen(@builtin(ray_invocation_id) id: vec3<u32>) {
                let x = select(0.0, 100.0, id.x % 4u == 1u);
                let ray = RayDesc(0u, 0xFFu, 0.001, 100.0, vec3f(x, 0.0, 0.0), vec3f(0.0, 0.0, 1.0));
                var hit: hit_object;
                switch id.x % 4u {
                    case 0u, 1u: { hitObjectTraceRay(&hit, acc_struct, ray, &payload); }
                    case 2u: { hitObjectRecordMiss(&hit, 0u, ray); }
                    default: { hitObjectRecordEmpty(&hit); }
                }
                reorderThread(&hit, id.x, 3u);
                reorderThread(&hit);
                reorderThreadHint(id.x, 2u);
                payload = 0.0;
                hitObjectExecute(&hit, &payload);
                let held = vec3f(
                    select(0.0, 1.0, hitObjectIsHit(&hit)),
                    select(0.0, 1.0, hitObjectIsMiss(&hit)),
                    select(0.0, 1.0, hitObjectIsEmpty(&hit)),
                );
                out[2u * id.x] = vec4f(held, payload);
                if hitObjectIsHit(&hit) {
                    out[2u * id.x + 1u] = vec4f(
                        hitObjectRayTMax(&hit),
                        f32(hitObjectInstanceIndex(&hit)),
                        f32(hitObjectPrimitiveIndex(&hit)),
                        f32(hitObjectHitKind(&hit)),
                    );
                }
            }
            "
            .into(),
        ),
    });
    let stages = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("hit and miss"),
        source: wgpu::ShaderSource::Wgsl(
            "
            enable wgpu_ray_tracing_pipeline;

            var<incoming_ray_payload> payload: f32;

            @closest_hit
            @incoming_payload(payload)
            fn hit() {
                payload = 2.0;
            }

            @miss
            @incoming_payload(payload)
            fn miss() {
                payload = 3.0;
            }
            "
            .into(),
        ),
    });
    let stage = |module, entry_point| RayTracingStage {
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
    };
    let pipeline = ctx
        .device
        .create_ray_tracing_pipeline(&RayTracingPipelineDescriptor {
            label: None,
            layout: None,
            ray_generation: stage(&gen, "gen"),
            misses: &[stage(&stages, "miss")],
            intersections: &[RayTracingIntersectionDescriptor::Triangle {
                closest_hit: stage(&stages, "hit"),
                any_hit: None,
            }],
            callables: &[],
            max_geometry_stride: 0,
            max_recursion_depth: 1,
            cache: None,
        });

    let out = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &[0; 16 * 16],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16 * 16,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: tlas.as_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.trace_rays(8, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, out.size());
    ctx.queue.submit([encoder.finish()]);

    let (send, recv) = mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |res| {
        res.unwrap();
        send.send(()).unwrap()
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    recv.recv().unwrap();
    let range = readback.get_mapped_range(..).unwrap();
    let results: &[[f32; 4]] = bytemuck::cast_slice(&range);

    for ray in 0..8 {
        let (held, extra) = (results[2 * ray], results[2 * ray + 1]);
        let expected = match ray % 4 {
            // A hit at t = 5 on primitive 0 of instance 0, its back face
            // (0xFF), its normal along the ray: the closest-hit stage ran.
            0 => ([1.0, 0.0, 0.0, 2.0], [5.0, 0.0, 0.0, 255.0]),
            // Misses: the miss stage ran.
            1 | 2 => ([0.0, 1.0, 0.0, 3.0], [0.0; 4]),
            // Empty: nothing ran.
            _ => ([0.0, 0.0, 1.0, 0.0], [0.0; 4]),
        };
        assert_eq!(held, expected.0, "ray {ray}: what it held and what ran");
        assert!(
            extra
                .iter()
                .zip(expected.1)
                .all(|(got, wanted)| (got - wanted).abs() < 1e-4),
            "ray {ray}: the hit {extra:?}, not {:?}",
            expected.1
        );
    }
}
