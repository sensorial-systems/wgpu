// Hit objects: traced into, made misses and empty, read, reordered by and
// executed.
enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_motion_blur;
enable wgpu_ray_tracing_invocation_reorder;

var<ray_payload> payload: vec4<f32>;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@ray_generation
fn ray_gen_main(@builtin(ray_invocation_id) id: vec3<u32>) {
    let ray = RayDesc(RAY_FLAG_NONE, 0xff, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0));
    var hit: hit_object;
    hitObjectTraceRay(&hit, acc_struct, ray, &payload);
    hitObjectTraceRay(&hit, acc_struct, ray, &payload, 1u, 1u);
    hitObjectTraceRayMotion(&hit, acc_struct, ray, 0.5, &payload, 1u, 1u, 1u);
    if hitObjectIsHit(&hit) {
        let t = hitObjectRayTMax(&hit) - hitObjectRayTMin(&hit);
        let instance = hitObjectInstanceIndex(&hit) + hitObjectInstanceCustomData(&hit);
        let primitive = hitObjectGeometryIndex(&hit) + hitObjectPrimitiveIndex(&hit) + hitObjectHitKind(&hit);
        let origin = hitObjectWorldRayOrigin(&hit) + hitObjectWorldRayDirection(&hit);
        let moved = hitObjectObjectToWorld(&hit) * vec4(origin, 1.0) + hitObjectWorldToObject(&hit) * vec4(origin, 0.0);
        payload = vec4(moved, t + f32(instance + primitive));
    } else if hitObjectIsMiss(&hit) {
        hitObjectRecordMiss(&hit, 1u, ray);
    } else if hitObjectIsEmpty(&hit) {
        hitObjectRecordEmpty(&hit);
    }
    reorderThread(&hit);
    reorderThread(&hit, id.x, 4u);
    reorderThreadHint(id.y, 2u);
    hitObjectExecute(&hit, &payload);
}

var<incoming_ray_payload> incoming: vec4<f32>;

// A hit stage traces into hit objects and executes them too, with a
// payload of its own: an incoming payload passes on to traces alone.
@closest_hit
@incoming_payload(incoming)
fn closest_hit_main() {
    var hit: hit_object;
    hitObjectTraceRay(&hit, acc_struct, RayDesc(RAY_FLAG_NONE, 0xff, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0)), &incoming);
    hitObjectRecordEmpty(&hit);
    hitObjectExecute(&hit, &payload);
    incoming = payload;
}

@miss
@incoming_payload(incoming)
fn miss_main() {
    incoming = vec4(0.0);
}
