enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_motion_blur;

struct RayDesc {
    flags: u32,
    cull_mask: u32,
    tmin: f32,
    tmax: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
}

var<ray_payload> payload: vec4<f32>;
@group(0) @binding(0)
var acc_struct: acceleration_structure;
var<incoming_ray_payload> incoming: vec4<f32>;

@ray_generation
fn ray_gen_main(@builtin(ray_invocation_id) id: vec3<u32>) {
    let ray = RayDesc(0u, 255u, 0.01f, 100f, vec3(0f), vec3<f32>(0f, 1f, 0f));
    let time_3 = (f32((id.x % 8u)) / 7f);
    traceRayMotion(acc_struct, ray, time_3, (&payload));
    traceRayMotion(acc_struct, ray, time_3, (&payload), 1u, 0u);
    traceRayMotion(acc_struct, ray, time_3, (&payload), 1u, 1u, 0u);
    traceRay(acc_struct, ray, (&payload));
    return;
}

@closest_hit @incoming_payload(incoming)
fn closest_hit_main(@builtin(current_ray_time) time: f32) {
    incoming = vec4<f32>(time, 1f, 0f, 1f);
    return;
}

@any_hit @incoming_payload(incoming)
fn any_hit_main(@builtin(current_ray_time) time_1: f32) {
    if (time_1 > 0.5f) {
        ignoreIntersection();
    } else {
        return;
    }
}

@miss @incoming_payload(incoming)
fn miss_main(@builtin(current_ray_time) time_2: f32) {
    incoming = vec4<f32>(time_2, 0f, 0f, 1f);
    return;
}
