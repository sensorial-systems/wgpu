// A binding array of uniform buffers: each element's struct is a `Block`.
enable wgpu_binding_array;

struct Tint {
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> tints: binding_array<Tint, 4>;

@fragment
fn main() -> @location(0) vec4<f32> {
    return tints[0].color + tints[3].color;
}
