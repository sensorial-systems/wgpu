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
