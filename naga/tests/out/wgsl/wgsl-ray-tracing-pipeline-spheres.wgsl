enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_invocation_reorder;
enable wgpu_ray_tracing_linear_swept_spheres;

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
    if hitObjectIsSphereHit((&hit)) {
        payload = vec4<f32>(hitObjectSpherePosition((&hit)), hitObjectSphereRadius((&hit)));
        return;
    } else {
        if hitObjectIsLssHit((&hit)) {
            let ends_1 = hitObjectLssPositions((&hit));
            let radii_1 = hitObjectLssRadii((&hit));
            payload = vec4<f32>((ends_1[0] + ends_1[1]), (radii_1.x + radii_1.y));
            return;
        } else {
            return;
        }
    }
}

@closest_hit @incoming_payload(incoming)
fn closest_hit_main(@builtin(hit_is_sphere) is_sphere: bool, @builtin(hit_is_lss) is_lss: bool, @builtin(hit_sphere_position) center: vec3<f32>, @builtin(hit_sphere_radius) radius: f32, @builtin(hit_lss_positions) ends: array<vec3<f32>, 2>, @builtin(hit_lss_radii) radii: array<f32, 2>, @builtin(hit_lss_value) along: f32) {
    if is_sphere {
        incoming = vec4<f32>(center, radius);
        return;
    } else {
        if is_lss {
            incoming = vec4<f32>(mix(ends[0], ends[1], along), mix(radii[0], radii[1], along));
            return;
        } else {
            return;
        }
    }
}

@miss @incoming_payload(incoming)
fn miss_main() {
    incoming = vec4(0f);
    return;
}
