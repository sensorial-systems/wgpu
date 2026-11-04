// Procedural hits reported by an intersection stage, with attributes of
// their own; callable stages; traces whose hit groups step with the
// geometry; and the flags a trace was given.
enable wgpu_ray_tracing_pipeline;
enable primitive_index;

struct Sphere {
    normal: vec3<f32>,
    inside: u32,
}

var<ray_payload> payload: vec4<f32>;
var<callable_data> shade: vec4<f32>;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@ray_generation
fn ray_gen_main(@builtin(ray_invocation_id) id: vec3<u32>) {
    let ray = RayDesc(RAY_FLAG_NONE, 0xff, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0));
    traceRay(acc_struct, ray, &payload, 1u, 2u, id.y % 2u);
    shade = payload;
    executeCallable(id.x % 2u, &shade);
}

var<ray_hit_attribute> sphere: Sphere;

@intersection
fn intersection_main(
    @builtin(object_ray_origin) origin: vec3<f32>,
    @builtin(object_ray_direction) direction: vec3<f32>,
    @builtin(ray_t_min) t_min: f32,
    @builtin(primitive_index) primitive: u32,
) {
    // A unit sphere at the origin of the primitive's space.
    let b = dot(origin, direction);
    let c = dot(origin, origin) - 1.0;
    let a = dot(direction, direction);
    let discriminant = b * b - a * c;
    if discriminant < 0.0 {
        return;
    }
    let root = sqrt(discriminant);
    let near = (-b - root) / a;
    let far = (-b + root) / a;
    sphere.normal = normalize(origin + near * direction);
    sphere.inside = 0u;
    if !reportIntersection(near, primitive % 128u) {
        sphere.normal = -normalize(origin + far * direction);
        sphere.inside = 1u;
        _ = reportIntersection(far, primitive % 128u);
    }
}

var<incoming_ray_payload> incoming: vec4<f32>;

@any_hit
@incoming_payload(incoming)
fn any_hit_main(@builtin(incoming_ray_flags) flags: u32) {
    if sphere.inside == 1u && (flags & RAY_FLAG_CULL_BACK_FACING) != 0u {
        ignoreIntersection();
    }
}

@closest_hit
@incoming_payload(incoming)
fn closest_hit_main(@builtin(hit_kind) kind: u32) {
    incoming = vec4(sphere.normal, f32(kind));
}

@miss
@incoming_payload(incoming)
fn miss_main(@builtin(incoming_ray_flags) flags: u32) {
    incoming = vec4(f32(flags));
}

var<incoming_callable_data> callable_in: vec4<f32>;

@callable
@incoming_payload(callable_in)
fn callable_main() {
    callable_in = callable_in * 0.5;
}
