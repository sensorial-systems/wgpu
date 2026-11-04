enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_motion_blur;
enable wgpu_ray_tracing_invocation_reorder;

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
    var hit: hit_object;

    let ray = RayDesc(0u, 255u, 0.01f, 100f, vec3(0f), vec3<f32>(0f, 1f, 0f));
    hitObjectTraceRay((&hit), acc_struct, ray, (&payload));
    hitObjectTraceRay((&hit), acc_struct, ray, (&payload), 1u, 1u);
    hitObjectTraceRayMotion((&hit), acc_struct, ray, 0.5f, (&payload), 1u, 1u, 1u);
    if hitObjectIsHit((&hit)) {
        let t = (hitObjectRayTMax((&hit)) - hitObjectRayTMin((&hit)));
        let instance = (hitObjectInstanceIndex((&hit)) + hitObjectInstanceCustomData((&hit)));
        let primitive = ((hitObjectGeometryIndex((&hit)) + hitObjectPrimitiveIndex((&hit))) + hitObjectHitKind((&hit)));
        let origin = (hitObjectWorldRayOrigin((&hit)) + hitObjectWorldRayDirection((&hit)));
        let moved = ((hitObjectObjectToWorld((&hit)) * vec4<f32>(origin, 1f)) + (hitObjectWorldToObject((&hit)) * vec4<f32>(origin, 0f)));
        payload = vec4<f32>(moved, (t + f32((instance + primitive))));
    } else {
        if hitObjectIsMiss((&hit)) {
            hitObjectRecordMiss((&hit), 1u, ray);
        } else {
            if hitObjectIsEmpty((&hit)) {
                hitObjectRecordEmpty((&hit));
            }
        }
    }
    reorderThread((&hit));
    reorderThread((&hit), id.x, 4u);
    reorderThreadHint(id.y, 2u);
    hitObjectExecute((&hit), (&payload));
    return;
}

@closest_hit @incoming_payload(incoming)
fn closest_hit_main() {
    var hit_1: hit_object;

    hitObjectTraceRay((&hit_1), acc_struct, RayDesc(0u, 255u, 0.01f, 100f, vec3(0f), vec3<f32>(0f, 1f, 0f)), (&incoming));
    hitObjectRecordEmpty((&hit_1));
    hitObjectExecute((&hit_1), (&payload));
    let _e17 = payload;
    incoming = _e17;
    return;
}

@miss @incoming_payload(incoming)
fn miss_main() {
    incoming = vec4(0f);
    return;
}
