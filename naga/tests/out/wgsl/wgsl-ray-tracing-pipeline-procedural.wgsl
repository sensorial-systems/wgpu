enable primitive_index;
enable wgpu_ray_tracing_pipeline;

struct Sphere {
    normal: vec3<f32>,
    inside: u32,
}

struct RayDesc {
    flags: u32,
    cull_mask: u32,
    tmin: f32,
    tmax: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
}

var<ray_payload> payload: vec4<f32>;
var<callable_data> shade: vec4<f32>;
@group(0) @binding(0)
var acc_struct: acceleration_structure;
var<ray_hit_attribute> sphere: Sphere;
var<incoming_ray_payload> incoming: vec4<f32>;
var<incoming_callable_data> callable_in: vec4<f32>;

@ray_generation
fn ray_gen_main(@builtin(ray_invocation_id) id: vec3<u32>) {
    let ray = RayDesc(0u, 255u, 0.01f, 100f, vec3(0f), vec3<f32>(0f, 1f, 0f));
    traceRay(acc_struct, ray, (&payload), 1u, 2u, (id.y % 2u));
    let _e21 = payload;
    shade = _e21;
    executeCallable((id.x % 2u), (&shade));
    return;
}

@intersection
fn intersection_main(@builtin(object_ray_origin) origin: vec3<f32>, @builtin(object_ray_direction) direction: vec3<f32>, @builtin(ray_t_min) t_min: f32, @builtin(primitive_index) primitive: u32) {
    let b = dot(origin, direction);
    let c = (dot(origin, origin) - 1f);
    let a = dot(direction, direction);
    let discriminant = ((b * b) - (a * c));
    if (discriminant < 0f) {
        return;
    }
    let root = sqrt(discriminant);
    let near = ((-(b) - root) / a);
    let far = ((-(b) + root) / a);
    sphere.normal = normalize((origin + (near * direction)));
    sphere.inside = 0u;
    let _e31 = reportIntersection(near, (primitive % 128u));
    if !(_e31) {
        sphere.normal = -(normalize((origin + (far * direction))));
        sphere.inside = 1u;
        let _e44 = reportIntersection(far, (primitive % 128u));
        return;
    } else {
        return;
    }
}

@any_hit @incoming_payload(incoming)
fn any_hit_main(@builtin(incoming_ray_flags) flags: u32) {
    var local: bool;

    let _e3 = sphere.inside;
    if (_e3 == 1u) {
        local = ((flags & 16u) != 0u);
    } else {
        local = false;
    }
    let _e13 = local;
    if _e13 {
        ignoreIntersection();
    } else {
        return;
    }
}

@closest_hit @incoming_payload(incoming)
fn closest_hit_main(@builtin(hit_kind) kind: u32) {
    let _e4 = sphere.normal;
    incoming = vec4<f32>(_e4, f32(kind));
    return;
}

@miss @incoming_payload(incoming)
fn miss_main(@builtin(incoming_ray_flags) flags_1: u32) {
    incoming = vec4(f32(flags_1));
    return;
}

@callable @incoming_payload(callable_in)
fn callable_main() {
    let _e2 = callable_in;
    callable_in = (_e2 * 0.5f);
    return;
}
