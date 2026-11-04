enable primitive_index;
enable wgpu_ray_tracing_pipeline;

struct RayDesc {
    flags: u32,
    cull_mask: u32,
    tmin: f32,
    tmax: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
}

var<ray_payload> payload: u32;
@group(0) @binding(0)
var acc_struct: acceleration_structure;
var<incoming_ray_payload> incoming: u32;

@ray_generation
fn ray_gen_main(@builtin(ray_invocation_id) id: vec3<u32>) {
    let ray = RayDesc(0u, 255u, 0.01f, 100f, vec3(0f), vec3<f32>(0f, 1f, 0f));
    traceRay(acc_struct, ray, (&payload));
    traceRay(acc_struct, ray, (&payload), (id.x % 3u), (id.y % 2u));
    return;
}

@miss @incoming_payload(incoming)
fn miss_main() {
    incoming = 0u;
    return;
}

@any_hit @incoming_payload(incoming)
fn any_hit_main(@builtin(primitive_index) primitive: u32) {
    if (primitive == 1u) {
        ignoreIntersection();
    }
    if (primitive == 2u) {
        incoming = 2u;
        terminateRay();
    }
    incoming = primitive;
    return;
}

@closest_hit @incoming_payload(incoming)
fn closest_hit_main(@builtin(primitive_index) primitive_1: u32) {
    incoming = (primitive_1 + 1u);
    return;
}
