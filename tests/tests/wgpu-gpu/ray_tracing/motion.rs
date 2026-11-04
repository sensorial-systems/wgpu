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
    apply, fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(MOTION_TRACES_MEET_GEOMETRY_WHERE_IT_IS);
    tests.push(MOTION_DATA_ONLY_IN_STRUCTURES_THAT_MOVE);
}

fn parameters() -> TestParameters {
    TestParameters::default()
        .features(
            Features::EXPERIMENTAL_RAY_TRACING_PIPELINES
                | Features::EXPERIMENTAL_RAY_TRACING_MOTION_BLUR,
        )
        .limits(
            Limits::defaults()
                .using_minimum_supported_acceleration_structure_values()
                .using_minimum_supported_ray_tracing_pipeline_values(),
        )
}

const IDENTITY: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

const fn translation(x: f32) -> [f32; 12] {
    [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

/// A triangle in the `z = 5` plane, 6 wide at `y - 2`, its apex at `y + 4`, centered on `x`:
/// 4 wide at `y`.
fn triangle(x: f32, y: f32) -> [[f32; 3]; 3] {
    [
        [x - 3.0, y - 2.0, 5.0],
        [x + 3.0, y - 2.0, 5.0],
        [x, y + 4.0, 5.0],
    ]
}

fn vertex_buffer(ctx: &TestingContext, vertices: [[f32; 3]; 3]) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vertices),
        usage: BufferUsages::BLAS_INPUT,
    })
}

fn triangle_size() -> BlasTriangleGeometrySizeDescriptor {
    BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
        opacity_micromap: Vec::new(),
    }
}

fn triangles<'a>(
    size: &'a BlasTriangleGeometrySizeDescriptor,
    vertices: &'a wgpu::Buffer,
    end_vertices: Option<&'a wgpu::Buffer>,
) -> BlasGeometries<'a> {
    BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
        size,
        vertex_buffer: vertices,
        first_vertex: 0,
        vertex_stride: mem::size_of::<[f32; 3]>() as BufferAddress,
        index_buffer: None,
        first_index: None,
        transform_buffer: None,
        transform_buffer_offset: None,
        end_vertex_buffer: end_vertices,
        opacity_micromap: None,
    }])
}

fn create_blas(ctx: &TestingContext, flags: AccelerationStructureFlags) -> wgpu::Blas {
    ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: None,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE | flags,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![triangle_size()],
        },
    )
}

fn create_tlas(ctx: &TestingContext, flags: AccelerationStructureFlags) -> wgpu::Tlas {
    ctx.device.create_tlas(&CreateTlasDescriptor {
        label: None,
        max_instances: 2,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE | flags,
        update_mode: AccelerationStructureUpdateMode::Build,
    })
}

#[apply(gpu_test!)]
static MOTION_TRACES_MEET_GEOMETRY_WHERE_IT_IS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_sync(motion_traces_meet_geometry_where_it_is);

/// A triangle whose vertices move from `x = -5` to `x = 5`, and an instance
/// of a still triangle whose transform does, are hit by rays at `x = -5`
/// early and by rays at `x = 5` late: each is 4 wide where the rays cross
/// it, so a ray meets it while its center is within 2 of the ray.
fn motion_traces_meet_geometry_where_it_is(ctx: TestingContext) {
    let size = triangle_size();
    let moving_start = vertex_buffer(&ctx, triangle(-5.0, -1.0));
    let moving_end = vertex_buffer(&ctx, triangle(5.0, -1.0));
    let still = vertex_buffer(&ctx, triangle(0.0, 9.0));

    let moving_blas = create_blas(&ctx, AccelerationStructureFlags::ALLOW_MOTION);
    let still_blas = create_blas(&ctx, AccelerationStructureFlags::empty());
    let mut tlas = create_tlas(&ctx, AccelerationStructureFlags::ALLOW_MOTION);
    tlas[0] = Some(TlasInstance::new(&moving_blas, IDENTITY, 0, 0xFF, 0));
    let mut sliding = TlasInstance::new(&still_blas, translation(-5.0), 0, 0xFF, 0);
    sliding.end_transform = Some(translation(5.0));
    tlas[1] = Some(sliding);

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(
        [
            &BlasBuildEntry {
                blas: &moving_blas,
                geometry: triangles(&size, &moving_start, Some(&moving_end)),
            },
            &BlasBuildEntry {
                blas: &still_blas,
                geometry: triangles(&size, &still, None),
            },
        ],
        [&tlas],
    );
    ctx.queue.submit([encoder.finish()]);

    let gen = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray generation"),
        source: wgpu::ShaderSource::Wgsl(
            "
            enable wgpu_ray_tracing_pipeline;
            enable wgpu_ray_tracing_motion_blur;

            var<ray_payload> payload: u32;
            @group(0) @binding(0) var acc_struct: acceleration_structure;
            @group(0) @binding(1) var<storage, read_write> out: array<u32, 32>;

            // Rows of eight rays at times 0, 1/7, ..., 1: from x = -5 and from x = 5,
            // at the moving triangle's height and at the instance's.
            @ray_generation
            fn gen(@builtin(ray_invocation_id) id: vec3<u32>) {
                let x = select(-5.0, 5.0, id.y % 2u == 1u);
                let y = select(0.0, 10.0, id.y >= 2u);
                let ray = RayDesc(0u, 0xFFu, 0.001, 100.0, vec3f(x, y, 0.0), vec3f(0.0, 0.0, 1.0));
                payload = 0u;
                traceRayMotion(acc_struct, ray, f32(id.x) / 7.0, &payload);
                out[id.y * 8u + id.x] = payload;
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
            enable wgpu_ray_tracing_motion_blur;

            var<incoming_ray_payload> payload: u32;

            // The instance hit, and the time the hit was at in sevenths.
            @closest_hit
            @incoming_payload(payload)
            fn hit(
                @builtin(instance_index) instance: u32,
                @builtin(current_ray_time) time: f32,
            ) {
                payload = (instance + 1u) | (u32(round(time * 7.0)) << 8u);
            }

            @miss
            @incoming_payload(payload)
            fn miss() {
                payload = 0u;
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

    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32 * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32 * 4,
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
        pass.trace_rays(8, 4, 1);
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
    let results: &[u32] = bytemuck::cast_slice(&range);

    // A ray from x meets geometry centered at -5 + 10 t when the center is
    // within 2 of it: at x = -5 until t = 0.2, at x = 5 from t = 0.8.
    let expected: Vec<u32> = (0..4u32)
        .flat_map(|row| {
            (0..8u32).map(move |time| {
                let instance = 1 + row / 2;
                let hit = match row % 2 {
                    0 => time <= 1,
                    _ => time >= 6,
                };
                if hit {
                    instance | (time << 8)
                } else {
                    0
                }
            })
        })
        .collect();
    assert_eq!(results, &expected[..]);
}

#[apply(gpu_test!)]
static MOTION_DATA_ONLY_IN_STRUCTURES_THAT_MOVE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_sync(motion_data_only_in_structures_that_move);

fn motion_data_only_in_structures_that_move(ctx: TestingContext) {
    let size = triangle_size();
    let vertices = vertex_buffer(&ctx, triangle(0.0, 0.0));

    // A structure that moves needs where its triangles end.
    let moving_blas = create_blas(&ctx, AccelerationStructureFlags::ALLOW_MOTION);
    fail(
        &ctx.device,
        || {
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            encoder.build_acceleration_structures(
                [&BlasBuildEntry {
                    blas: &moving_blas,
                    geometry: triangles(&size, &vertices, None),
                }],
                [],
            );
            encoder.finish()
        },
        Some("needs its end vertices"),
    );

    // One that does not move has none.
    let still_blas = create_blas(&ctx, AccelerationStructureFlags::empty());
    fail(
        &ctx.device,
        || {
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            encoder.build_acceleration_structures(
                [&BlasBuildEntry {
                    blas: &still_blas,
                    geometry: triangles(&size, &vertices, Some(&vertices)),
                }],
                [],
            );
            encoder.finish()
        },
        Some("have no end vertices"),
    );

    // Nor do its instances have end transforms.
    let mut tlas = create_tlas(&ctx, AccelerationStructureFlags::empty());
    let mut ending = TlasInstance::new(&still_blas, IDENTITY, 0, 0xFF, 0);
    ending.end_transform = Some(IDENTITY);
    tlas[0] = Some(ending);
    fail(
        &ctx.device,
        || {
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            encoder.build_acceleration_structures(
                [&BlasBuildEntry {
                    blas: &still_blas,
                    geometry: triangles(&size, &vertices, None),
                }],
                [&tlas],
            );
            encoder.finish()
        },
        Some("has no end transform"),
    );

    // Nor does it hold structures that move.
    tlas[0] = Some(TlasInstance::new(&moving_blas, IDENTITY, 0, 0xFF, 0));
    fail(
        &ctx.device,
        || {
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            encoder.build_acceleration_structures(
                [&BlasBuildEntry {
                    blas: &moving_blas,
                    geometry: triangles(&size, &vertices, Some(&vertices)),
                }],
                [&tlas],
            );
            encoder.finish()
        },
        Some("which does not"),
    );
}
