// Rays traced at a time through structures that move, and the stages that
// read the time they were traced at.
enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_motion_blur;

var<ray_payload> payload: vec4<f32>;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@ray_generation
fn ray_gen_main(@builtin(ray_invocation_id) id: vec3<u32>) {
    let ray = RayDesc(RAY_FLAG_NONE, 0xff, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0));
    let time = f32(id.x % 8u) / 7.0;
    traceRayMotion(acc_struct, ray, time, &payload);
    traceRayMotion(acc_struct, ray, time, &payload, 1u, 0u);
    traceRayMotion(acc_struct, ray, time, &payload, 1u, 1u, 0u);
    traceRay(acc_struct, ray, &payload);
}

var<incoming_ray_payload> incoming: vec4<f32>;

@closest_hit
@incoming_payload(incoming)
fn closest_hit_main(@builtin(current_ray_time) time: f32) {
    incoming = vec4(time, 1.0, 0.0, 1.0);
}

@any_hit
@incoming_payload(incoming)
fn any_hit_main(@builtin(current_ray_time) time: f32) {
    if time > 0.5 {
        ignoreIntersection();
    }
}

@miss
@incoming_payload(incoming)
fn miss_main(@builtin(current_ray_time) time: f32) {
    incoming = vec4(time, 0.0, 0.0, 1.0);
}
