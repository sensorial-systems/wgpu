// Hit groups and misses chosen per trace, and an any-hit stage's verdicts.
enable wgpu_ray_tracing_pipeline;
enable primitive_index;

var<ray_payload> payload: u32;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@ray_generation
fn ray_gen_main(@builtin(ray_invocation_id) id: vec3<u32>) {
    let ray = RayDesc(RAY_FLAG_NONE, 0xff, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0));
    traceRay(acc_struct, ray, &payload);
    traceRay(acc_struct, ray, &payload, id.x % 3u, id.y % 2u);
}

var<incoming_ray_payload> incoming: u32;

@miss
@incoming_payload(incoming)
fn miss_main() {
    incoming = 0u;
}

@any_hit
@incoming_payload(incoming)
fn any_hit_main(@builtin(primitive_index) primitive: u32) {
    if primitive == 1u {
        ignoreIntersection();
    }
    if primitive == 2u {
        incoming = 2u;
        terminateRay();
    }
    incoming = primitive;
}

@closest_hit
@incoming_payload(incoming)
fn closest_hit_main(@builtin(primitive_index) primitive: u32) {
    incoming = primitive + 1u;
}
