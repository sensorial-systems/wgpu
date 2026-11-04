use std::{mem, sync::mpsc};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BindGroupDescriptor, BindGroupEntry, BlasBuildEntry,
    BlasGeometries, BlasGeometrySizeDescriptors, BlasOpacityMicromap, BlasTriangleGeometry,
    BlasTriangleGeometrySizeDescriptor, BufferAddress, BufferUsages, CreateBlasDescriptor,
    CreateTlasDescriptor, Features, Limits, OpacityMicromapFormat, OpacityMicromapUsage,
    RayTracingIntersectionDescriptor, RayTracingPassDescriptor, RayTracingPipelineDescriptor,
    RayTracingStage, ShaderModuleDescriptor, TlasInstance, VertexFormat,
};
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(MICROMAPS_DECIDE_WHAT_RAYS_HIT);
}

/// The index of the micro-triangle at barycentrics `u` (the second corner's
/// weight) and `v` (the third's) of a triangle split `level` times: the
/// Vulkan specification's reference code.
fn micro_triangle(u: f32, v: f32, level: u32) -> u32 {
    let (u, v) = (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
    let side = 1u32 << level;
    let (fu, fv) = (u * side as f32, v * side as f32);
    let (mut iu, mut iv) = (fu as u32, fv as u32);
    let (uf, vf) = (fu - iu as f32, fv - iv as f32);
    iu = iu.min(side - 1);
    iv = iv.min(side - 1);
    let iuv = iu + iv;
    if iuv >= side {
        iu -= iuv - side + 1;
    }
    let mut iw = !(iu + iv);
    if uf + vf >= 1.0 && iuv < side - 1 {
        iw = iw.wrapping_sub(1);
    }
    let mut b0 = !(iu ^ iw) & (side - 1);
    let t = (iu ^ iv) & b0;
    let mut f = t;
    f ^= f >> 1;
    f ^= f >> 2;
    f ^= f >> 4;
    f ^= f >> 8;
    let mut b1 = ((f ^ iu) & !b0) | t;
    for (shift, mask) in [
        (8, 0x00ff00ffu32),
        (4, 0x0f0f0f0f),
        (2, 0x33333333),
        (1, 0x55555555),
    ] {
        b0 = (b0 | (b0 << shift)) & mask;
        b1 = (b1 | (b1 << shift)) & mask;
    }
    b0 | (b1 << 1)
}

#[apply(gpu_test!)]
static MICROMAPS_DECIDE_WHAT_RAYS_HIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::EXPERIMENTAL_RAY_TRACING_PIPELINES
                    | Features::EXPERIMENTAL_RAY_TRACING_OPACITY_MICROMAP,
            )
            .limits(
                Limits::defaults()
                    .using_minimum_supported_acceleration_structure_values()
                    .using_minimum_supported_ray_tracing_pipeline_values(),
            ),
    )
    .run_sync(micromaps_decide_what_rays_hit);

/// A triangle split twice into 16 micro-triangles in the 4-state format,
/// each transparent, opaque, unknown and transparent, or unknown and
/// opaque, in turn. Rays at points across it hit the opaque ones; an
/// any-hit stage that ignores every candidate passes the unknown ones over;
/// forced to two states, rays hit the unknown opaque ones too.
fn micromaps_decide_what_rays_hit(ctx: TestingContext) {
    const LEVEL: u32 = 2;
    // The triangle at z = 5, its barycentrics x / 4 and y / 4.
    let corners: [[f32; 3]; 3] = [[0.0, 0.0, 5.0], [4.0, 0.0, 5.0], [0.0, 4.0, 5.0]];
    let state = |micro: u32| micro % 4;
    let mut packed = 0u32;
    for micro in 0..16 {
        packed |= state(micro) << (2 * micro);
    }
    let buffer = |contents: &[u8]| {
        ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents,
            usage: BufferUsages::BLAS_INPUT,
        })
    };
    let vertices = buffer(bytemuck::cast_slice(&corners));
    let data = buffer(bytemuck::bytes_of(&packed));
    // One micromap triangle: its states from byte 0, level 2, 4-state.
    let record: [u32; 2] = [0, LEVEL | (2 << 16)];
    let triangles = buffer(bytemuck::cast_slice(&record));

    let size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::empty(),
        opacity_micromap: vec![OpacityMicromapUsage {
            count: 1,
            subdivision_level: LEVEL,
            format: OpacityMicromapFormat::FourState,
        }],
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
                opacity_micromap: Some(BlasOpacityMicromap {
                    data: &data,
                    data_offset: 0,
                    triangles: &triangles,
                    triangles_offset: 0,
                    indices: None,
                }),
            }]),
        }],
        [&tlas],
    );
    ctx.queue.submit([encoder.finish()]);

    // Points across the triangle, kept off its micro-triangles' edges.
    const GRID: u32 = 32;
    let mut points = Vec::new();
    for j in 0..GRID {
        for i in 0..GRID {
            let (u, v) = (
                (i as f32 + 0.5) / GRID as f32,
                (j as f32 + 0.5) / GRID as f32,
            );
            let near = |x: f32| {
                let scaled = x * (1 << LEVEL) as f32;
                (scaled - scaled.round()).abs() < 0.1
            };
            if u + v < 0.97 && !near(u) && !near(v) && !near(u + v) {
                points.push([u * 4.0, v * 4.0]);
            }
        }
    }
    let rays = points.len() as u32;
    let points_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&points),
        usage: BufferUsages::STORAGE,
    });

    let gen = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray generation"),
        source: wgpu::ShaderSource::Wgsl(
            "
            enable wgpu_ray_tracing_pipeline;
            enable wgpu_ray_tracing_opacity_micromap;

            var<ray_payload> payload: u32;
            @group(0) @binding(0) var acc_struct: acceleration_structure;
            @group(0) @binding(1) var<storage, read> points: array<vec2<f32>>;
            @group(0) @binding(2) var<storage, read_write> out: array<vec2<u32>>;

            @ray_generation
            fn gen(@builtin(ray_invocation_id) id: vec3<u32>) {
                let point = points[id.x];
                let origin = vec3f(point, 0.0);
                payload = 0u;
                traceRay(acc_struct, RayDesc(0u, 0xFFu, 0.001, 100.0, origin, vec3f(0.0, 0.0, 1.0)), &payload);
                let plain = payload;
                payload = 0u;
                traceRay(acc_struct, RayDesc(RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, 0xFFu, 0.001, 100.0, origin, vec3f(0.0, 0.0, 1.0)), &payload);
                out[id.x] = vec2<u32>(plain, payload);
            }
            "
            .into(),
        ),
    });
    let stages = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("hit stages"),
        source: wgpu::ShaderSource::Wgsl(
            "
            enable wgpu_ray_tracing_pipeline;

            var<incoming_ray_payload> payload: u32;

            @closest_hit
            @incoming_payload(payload)
            fn hit() {
                payload = 1u;
            }

            // Every candidate that is not opaque is passed over.
            @any_hit
            @incoming_payload(payload)
            fn any() {
                ignoreIntersection();
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
                any_hit: Some(stage(&stages, "any")),
            }],
            callables: &[],
            max_geometry_stride: 0,
            max_recursion_depth: 1,
            cache: None,
        });

    let size_bytes = rays as u64 * 8;
    let out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_bytes,
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
                resource: points_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: out.as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_ray_tracing_pass(&RayTracingPassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.trace_rays(rays, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, size_bytes);
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
    let results: &[[u32; 2]] = bytemuck::cast_slice(&range);

    let mut states = [0u32; 4];
    for (point, result) in points.iter().zip(results) {
        let micro = micro_triangle(point[0] / 4.0, point[1] / 4.0, LEVEL);
        let state = state(micro);
        states[state as usize] += 1;
        // Hit where opaque; forced, where unknown and opaque too.
        let expected = [u32::from(state == 1), u32::from(state == 1 || state == 3)];
        assert_eq!(
            *result, expected,
            "the ray at {point:?}, micro-triangle {micro} of state {state}"
        );
    }
    assert!(
        states.iter().all(|&count| count > 0),
        "every state is tried: {states:?}"
    );
}
