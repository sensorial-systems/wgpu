use std::{mem, sync::mpsc};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BindGroupDescriptor, BindGroupEntry, Blas, BlasBuildEntry,
    BlasGeometries, BlasGeometrySizeDescriptors, BlasSphereGeometry,
    BlasSphereGeometrySizeDescriptor, BufferAddress, BufferUsages, CreateBlasDescriptor,
    CreateTlasDescriptor, Features, IndexFormat, Limits, LinearSweptSphereEndCaps,
    LinearSweptSphereIndexing, LinearSweptSpheres, RayTracingIntersectionDescriptor,
    RayTracingPassDescriptor, RayTracingPipelineDescriptor, RayTracingStage,
    ShaderModuleDescriptor, Tlas, TlasInstance, VertexFormat,
};
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(RAYS_HIT_SPHERES_AND_WHAT_THEY_SWEEP);
    tests.push(HIT_OBJECTS_HOLD_SPHERES);
}

fn parameters(features: Features) -> TestParameters {
    TestParameters::default()
        .features(
            Features::EXPERIMENTAL_RAY_TRACING_PIPELINES
                | Features::EXPERIMENTAL_RAY_TRACING_LINEAR_SWEPT_SPHERES
                | features,
        )
        .limits(
            Limits::defaults()
                .using_minimum_supported_acceleration_structure_values()
                .using_minimum_supported_ray_tracing_pipeline_values(),
        )
}

#[apply(gpu_test!)]
static RAYS_HIT_SPHERES_AND_WHAT_THEY_SWEEP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters(Features::empty()))
    .run_sync(|ctx| rays_hit_spheres(ctx, false));

#[apply(gpu_test!)]
static HIT_OBJECTS_HOLD_SPHERES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters(
        Features::EXPERIMENTAL_RAY_TRACING_INVOCATION_REORDER,
    ))
    .run_sync(|ctx| rays_hit_spheres(ctx, true));

/// Two spheres, at the origin (radius 1) and at x = 3 (radius 0.5), and a
/// linear swept sphere from x = -1 (radius 0.5) to x = 3 (radius 1) along
/// y = 5, both ends capped: in instances moved to z = 10.
const SPHERES: [([f32; 3], f32); 2] = [([0.0, 0.0, 0.0], 1.0), ([3.0, 0.0, 0.0], 0.5)];
const SWEPT: [([f32; 3], f32); 2] = [([-1.0, 5.0, 0.0], 0.5), ([3.0, 5.0, 0.0], 1.0)];
const Z: f32 = 10.0;

/// What a ray from `(x, y, z0)` along +z meets: 0 for nothing, 1 for a
/// sphere, 2 for the swept sphere, the distance, and, for the swept sphere,
/// where along it from 0 to 1. Rays from inside meet nothing.
fn expected_hit(x: f32, y: f32, z0: f32) -> (u32, f32, f32) {
    let mut best = (0, f32::INFINITY, 0.0);
    let mut consider = |kind, t: f32, u| {
        if t > 0.0 && t < best.1 {
            best = (kind, t, u);
        }
    };
    // The distance along the ray to a sphere's near side, if it is there.
    let sphere = |center: [f32; 3], radius: f32| {
        let d2 = (x - center[0]).powi(2) + (y - center[1]).powi(2);
        (d2 < radius * radius).then(|| Z + center[2] - (radius * radius - d2).sqrt() - z0)
    };
    for (center, radius) in SPHERES {
        if let Some(t) = sphere(center, radius) {
            consider(1, t, 0.0);
        }
    }
    let ([a, ra], [b, rb]) = ([SWEPT[0].0[0], SWEPT[0].1], [SWEPT[1].0[0], SWEPT[1].1]);
    let length = b - a;
    for (end, (center, radius)) in SWEPT.into_iter().enumerate() {
        if let Some(t) = sphere(center, radius) {
            consider(2, t, end as f32);
        }
    }
    // The cone tangent to both ends, between where it touches them.
    let sin = (rb - ra) / length;
    let cos = (1.0 - sin * sin).sqrt();
    let (from, to) = (a - ra * sin, b - rb * sin);
    if (from..=to).contains(&x) {
        let rho = ra * cos + (x - from) * sin / cos;
        let dy = y - SWEPT[0].0[1];
        if dy.abs() < rho {
            let t = Z - (rho * rho - dy * dy).sqrt() - z0;
            // The swept sphere that touches the cone there.
            let center = x + rho * sin / cos;
            consider(2, t, (center - a) / length);
        }
    }
    best
}

fn blas(ctx: &TestingContext, spheres: &[([f32; 3], f32)], swept: bool) -> Blas {
    let centers: Vec<[f32; 3]> = spheres.iter().map(|&(center, _)| center).collect();
    let radii: Vec<f32> = spheres.iter().map(|&(_, radius)| radius).collect();
    let buffer = |contents: &[u8]| {
        ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents,
            usage: BufferUsages::BLAS_INPUT,
        })
    };
    let vertices = buffer(bytemuck::cast_slice(&centers));
    let radius_buffer = buffer(bytemuck::cast_slice(&radii));
    // The swept sphere is indexed, the spheres are not.
    let indices = buffer(bytemuck::cast_slice(&[0u32, 1]));
    let size = BlasSphereGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: spheres.len() as u32,
        index_format: swept.then_some(IndexFormat::Uint32),
        index_count: swept.then_some(2),
        linear_swept: swept.then_some(LinearSweptSpheres {
            indexing: LinearSweptSphereIndexing::List,
            end_caps: LinearSweptSphereEndCaps::Chained,
        }),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: None,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Spheres {
            descriptors: vec![size.clone()],
        },
    );
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::SphereGeometries(vec![BlasSphereGeometry {
                size: &size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: mem::size_of::<[f32; 3]>() as BufferAddress,
                radius_buffer: &radius_buffer,
                radius_stride: mem::size_of::<f32>() as BufferAddress,
                index_buffer: swept.then_some(&indices),
                first_index: swept.then_some(0),
            }]),
        }],
        [],
    );
    ctx.queue.submit([encoder.finish()]);
    blas
}

fn tlas(ctx: &TestingContext, blases: &[&Blas]) -> Tlas {
    let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: None,
        max_instances: blases.len() as u32,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    for (index, blas) in blases.iter().enumerate() {
        tlas[index] = Some(TlasInstance::new(
            blas,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, Z],
            0,
            0xFF,
            0,
        ));
    }
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures([], [&tlas]);
    ctx.queue.submit([encoder.finish()]);
    tlas
}

/// Rays at the spheres, the swept sphere's cone and caps, past them all, and
/// from inside a sphere: they hit what [`expected_hit`] says, read in their
/// closest hit stage's built-ins, or, with `hit_objects`, in hit objects.
fn rays_hit_spheres(ctx: TestingContext, hit_objects: bool) {
    let spheres = blas(&ctx, &SPHERES, false);
    let swept = blas(&ctx, &SWEPT, true);
    let tlas = tlas(&ctx, &[&spheres, &swept]);

    let origins: [[f32; 4]; 8] = [
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 0.3, 0.0, 0.0],
        [1.0, 5.0, 0.0, 0.0],
        [2.0, 5.3, 0.0, 0.0],
        [-1.4, 5.0, 0.0, 0.0],
        [3.5, 5.2, 0.0, 0.0],
        [3.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, Z, 0.0],
    ];
    let rays = origins.len() as u32;
    let origins_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&origins),
        usage: BufferUsages::STORAGE,
    });

    // A hit, as 12 floats: its kind (0 none, 1 a sphere, 2 a swept sphere),
    // its distance, where along a swept sphere, its two radii, its two
    // centers -- a sphere's in the first of each -- and its `hit_kind`.
    let reorder = match hit_objects {
        true => "enable wgpu_ray_tracing_invocation_reorder;",
        false => "",
    };
    let trace = if hit_objects {
        "
        var hit: hit_object;
        hitObjectTraceRay(&hit, acc_struct, ray, &payload);
        var result = array<f32, 12>();
        if hitObjectIsSphereHit(&hit) {
            result[0] = 1.0;
            result[3] = hitObjectSphereRadius(&hit);
            let center = hitObjectSpherePosition(&hit);
            result[5] = center.x;
            result[6] = center.y;
            result[7] = center.z;
        } else if hitObjectIsLssHit(&hit) {
            result[0] = 2.0;
            let radii = hitObjectLssRadii(&hit);
            let ends = hitObjectLssPositions(&hit);
            result[3] = radii.x;
            result[4] = radii.y;
            result[5] = ends[0].x;
            result[6] = ends[0].y;
            result[7] = ends[0].z;
            result[8] = ends[1].x;
            result[9] = ends[1].y;
            result[10] = ends[1].z;
        }
        if hitObjectIsHit(&hit) {
            result[1] = hitObjectRayTMax(&hit);
        }
        out[id.x] = result;
        "
    } else {
        "
        traceRay(acc_struct, ray, &payload);
        out[id.x] = payload;
        "
    };
    let gen_source = format!(
        "
        enable wgpu_ray_tracing_pipeline;
        enable wgpu_ray_tracing_linear_swept_spheres;
        {reorder}

        var<ray_payload> payload: array<f32, 12>;
        @group(0) @binding(0) var acc_struct: acceleration_structure;
        @group(0) @binding(1) var<storage, read> origins: array<vec4<f32>>;
        @group(0) @binding(2) var<storage, read_write> out: array<array<f32, 12>>;

        @ray_generation
        fn gen(@builtin(ray_invocation_id) id: vec3<u32>) {{
            let ray = RayDesc(0u, 0xFFu, 0.0, 100.0, origins[id.x].xyz, vec3f(0.0, 0.0, 1.0));
            payload = array<f32, 12>();
            {trace}
        }}
        "
    );
    let gen = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray generation"),
        source: wgpu::ShaderSource::Wgsl(gen_source.into()),
    });
    let stages = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("hit stages"),
        source: wgpu::ShaderSource::Wgsl(
            "
            enable wgpu_ray_tracing_pipeline;
            enable wgpu_ray_tracing_linear_swept_spheres;

            var<incoming_ray_payload> payload: array<f32, 12>;

            @closest_hit
            @incoming_payload(payload)
            fn hit(
                @builtin(hit_is_sphere) is_sphere: bool,
                @builtin(hit_is_lss) is_lss: bool,
                @builtin(hit_sphere_position) center: vec3<f32>,
                @builtin(hit_sphere_radius) radius: f32,
                @builtin(hit_lss_positions) ends: array<vec3<f32>, 2>,
                @builtin(hit_lss_radii) radii: array<f32, 2>,
                @builtin(hit_lss_value) along: f32,
                @builtin(ray_t_current_max) t: f32,
                @builtin(hit_kind) kind: u32,
            ) {
                payload[1] = t;
                payload[11] = f32(kind);
                payload[2] = along;
                if is_sphere {
                    payload[0] = 1.0;
                    payload[3] = radius;
                    payload[5] = center.x;
                    payload[6] = center.y;
                    payload[7] = center.z;
                } else if is_lss {
                    payload[0] = 2.0;
                    payload[3] = radii[0];
                    payload[4] = radii[1];
                    payload[5] = ends[0].x;
                    payload[6] = ends[0].y;
                    payload[7] = ends[0].z;
                    payload[8] = ends[1].x;
                    payload[9] = ends[1].y;
                    payload[10] = ends[1].z;
                } else {
                    payload[0] = 3.0;
                }
            }

            @miss
            @incoming_payload(payload)
            fn miss() {
                payload = array<f32, 12>();
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

    let size_bytes = u64::from(rays) * 48;
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
                resource: origins_buffer.as_entire_binding(),
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
    let results: &[[f32; 12]] = bytemuck::cast_slice(&range);

    let close = |a: f32, b: f32| (a - b).abs() < 2e-3;
    for (origin, result) in origins.iter().zip(results) {
        let (kind, t, along) = expected_hit(origin[0], origin[1], origin[2]);
        let message =
            format!("the ray from {origin:?}, expecting {kind} at {t} ({along}): {result:?}");
        assert_eq!(result[0], kind as f32, "{message}");
        if kind == 0 {
            continue;
        }
        assert!(close(result[1], t), "{message}");
        if !hit_objects {
            assert!(close(result[2], along), "{message}");
            // Hit kinds of their own: 0xCE a sphere's, 0xCC a swept sphere's.
            assert_eq!(result[11], [0.0, 206.0, 204.0][kind as usize], "{message}");
        }
        // Centers in object space, as the structure has them.
        if kind == 1 {
            let (center, radius) = SPHERES
                .into_iter()
                .find(|&(center, _)| close(center[0], result[5]))
                .unwrap_or_else(|| panic!("{message}"));
            assert_eq!(result[3], radius, "{message}");
            assert_eq!(&result[5..8], &center, "{message}");
        } else {
            assert_eq!(&result[3..5], &[SWEPT[0].1, SWEPT[1].1], "{message}");
            assert_eq!(&result[5..8], &SWEPT[0].0, "{message}");
            assert_eq!(&result[8..11], &SWEPT[1].0, "{message}");
        }
    }
}
