# sensorial's wgpu

The `sensorial` branch is wgpu as sensorial's workspaces build it: gfx-rs/wgpu's
`trunk` with our changes on top. Every workspace that draws with prism
patches crates.io's wgpu crates to this checkout, all of them, since they
depend on one another by path:

```toml
[patch.crates-io]
naga = { path = "<path to systems>/wgpu/naga" }
naga-types = { path = "<path to systems>/wgpu/naga-types" }
wgpu = { path = "<path to systems>/wgpu/wgpu" }
wgpu-core = { path = "<path to systems>/wgpu/wgpu-core" }
wgpu-core-deps-apple = { path = "<path to systems>/wgpu/wgpu-core/platform-deps/apple" }
wgpu-core-deps-emscripten = { path = "<path to systems>/wgpu/wgpu-core/platform-deps/emscripten" }
wgpu-core-deps-wasm = { path = "<path to systems>/wgpu/wgpu-core/platform-deps/wasm" }
wgpu-core-deps-linux-android-bsd = { path = "<path to systems>/wgpu/wgpu-core/platform-deps/linux-android-bsd" }
wgpu-core-deps-windows = { path = "<path to systems>/wgpu/wgpu-core/platform-deps/windows" }
wgpu-hal = { path = "<path to systems>/wgpu/wgpu-hal" }
wgpu-naga-bridge = { path = "<path to systems>/wgpu/wgpu-naga-bridge" }
wgpu-types = { path = "<path to systems>/wgpu/wgpu-types" }
```

A `[patch]` only applies from the workspace being built, never through a
dependency, so each workspace needs those lines itself. Trunk's crates are
versioned 30.0.0, which every workspace's wgpu 30 requirement takes.

`upstream` is gfx-rs/wgpu. Changes worth having there go to it as pull
requests against `trunk`; keep them as separate commits here so they can.
`sensorial-v30` is the branch this one replaced: v30.0.1 with the same
changes, kept for reference.

## Changes

- **Ray tracing pipelines in wgpu-core and wgpu**: gfx-rs/wgpu#10144, merged
  in while it is in review. Ray tracing pipelines and passes, shader
  binding tables, and an intersection index on `TlasInstance`.
- **What ray tracing pipelines lacked**, on top of #10144:
  - naga: `ignoreIntersection()` and `terminateRay()` in any-hit stages
    (`RayPipelineFunction::IgnoreIntersection`, `TerminateRay`), and
    `traceRay(structure, descriptor, &payload, hit_group_offset,
    miss_index)` (`RayGroups`, gfx-rs/wgpu#8894). Out-of-range choices are
    safe where the SPIR-V writer is told the pipeline's counts
    (`Options::ray_tracing_pipeline_groups`, which wgpu-hal sets): both
    are clamped, and the shader binding table holds as many empty hit
    groups past the last as an offset can reach.
  - wgpu: several miss stages (`RayTracingPipelineDescriptor::misses`),
    `RayTracingPass::trace_rays_indirect`, and timestamp writes on a ray
    tracing pass. An indirect trace's size is not checked against the
    device's limits, as an indirect dispatch's is.
- **Incoming ray payloads through compaction**: naga's compaction, which
  wgpu-core runs on every pipeline stage, neither kept a ray tracing
  stage's incoming payload nor remapped its handle, so a miss stage that
  never touches its payload failed pipeline creation with a panic.
- **Binding arrays of uniform buffers in SPIR-V**: naga decorated the
  structs of storage buffer binding arrays as `Block`s, but not those of
  uniform buffers, which Vulkan requires of both.
- **No timeline semaphores on PowerVR** (`wgpu-hal/src/vulkan/adapter.rs`,
  beside trunk's VeriSilicon exclusion): MediaTek's PowerVR drivers
  dereference null inside `vkQueueSubmit` with them. Revving crashed on its
  first frames on a moto g54 5G (`vulkan.mtk.so`) and a moto g play 2023
  (`vulkan.mt6765.so`) in Firebase Test Lab, and runs on both with fences.

## What trunk changed for its users

Moving the workspaces from 30.0.1 to trunk took: `..Default::default()` in
`DeviceDescriptor` and `TextureViewDescriptor` literals (`default_queue`,
`swizzle`), `Queue::submit` returning `Option<SubmissionIndex>` (`None` on
WebGPU), `TextureFormat::is_srgb` renamed `has_srgb_suffix`, wgpu-hal's
`create_buffer` returning the allocated size too, `TlasInstance::new` taking
an intersection index, and naga's `RayQueryFunction::Begin`. Its WebGPU
bindings need js-sys/web-sys 0.3.104 or later.

naga's snapshot tests (`cargo test -p naga --test naga snapshots`) need
`spirv-cross` and `spirv-as` from the Vulkan SDK on the path.
