enable wgpu_binding_array;

struct Tint {
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> tints: binding_array<Tint, 4>;

@fragment
fn main() -> @location(0) vec4<f32> {
    let _e3 = tints[0].color;
    let _e7 = tints[3].color;
    return (_e3 + _e7);
}
