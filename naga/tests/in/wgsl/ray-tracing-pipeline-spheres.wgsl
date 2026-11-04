// Spheres and linear swept spheres: the built-ins of their hits, and their
// hit objects' queries.
enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_invocation_reorder;
enable wgpu_ray_tracing_linear_swept_spheres;

var<ray_payload> payload: vec4<f32>;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@ray_generation
fn ray_gen_main(@builtin(ray_invocation_id) id: vec3<u32>) {
    let ray = RayDesc(RAY_FLAG_NONE, 0xff, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0));
    var hit: hit_object;
    hitObjectTraceRay(&hit, acc_struct, ray, &payload);
    if hitObjectIsSphereHit(&hit) {
        payload = vec4(hitObjectSpherePosition(&hit), hitObjectSphereRadius(&hit));
    } else if hitObjectIsLssHit(&hit) {
        let ends = hitObjectLssPositions(&hit);
        let radii = hitObjectLssRadii(&hit);
        payload = vec4(ends[0] + ends[1], radii.x + radii.y);
    }
}

var<incoming_ray_payload> incoming: vec4<f32>;

@closest_hit
@incoming_payload(incoming)
fn closest_hit_main(
    @builtin(hit_is_sphere) is_sphere: bool,
    @builtin(hit_is_lss) is_lss: bool,
    @builtin(hit_sphere_position) center: vec3<f32>,
    @builtin(hit_sphere_radius) radius: f32,
    @builtin(hit_lss_positions) ends: array<vec3<f32>, 2>,
    @builtin(hit_lss_radii) radii: array<f32, 2>,
    @builtin(hit_lss_value) along: f32,
) {
    if is_sphere {
        incoming = vec4(center, radius);
    } else if is_lss {
        incoming = vec4(mix(ends[0], ends[1], along), mix(radii[0], radii[1], along));
    }
}

@miss
@incoming_payload(incoming)
fn miss_main() {
    incoming = vec4(0.0);
}
