# sensorial's wgpu

The `sensorial` branch is wgpu as sensorial's workspaces build it: a release
of gfx-rs/wgpu with our changes on top. Every workspace that draws with prism
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
wgpu-core-deps-windows-linux-android = { path = "<path to systems>/wgpu/wgpu-core/platform-deps/windows-linux-android" }
wgpu-hal = { path = "<path to systems>/wgpu/wgpu-hal" }
wgpu-naga-bridge = { path = "<path to systems>/wgpu/wgpu-naga-bridge" }
wgpu-types = { path = "<path to systems>/wgpu/wgpu-types" }
```

A `[patch]` only applies from the workspace being built, never through a
dependency, so each workspace needs those lines itself. `upstream` is
gfx-rs/wgpu; changes worth having there go to it as pull requests against
`trunk`.

## Base

v30.0.1, which every workspace's wgpu 30 requirement takes. Trunk is past it
by breaking changes (new required fields in `DeviceDescriptor` and
`TextureViewDescriptor`) that cubecl, ui-kit, burn-models and glyphon would
all need; move the branch when wgpu 31 ships or a pull request needs it.

## Changes

- **30.0.0's web dependency minimums**, and **30.0.0's WebGPU bindings** (the
  backport of #10034 reverted): 30.0.1's need js-sys 0.3.104 (`JsNullable`),
  and the cross workspace pins `web-sys = "=0.3.103"`. Drop both once that
  pin moves.
- **No timeline semaphores on PowerVR** (`wgpu-hal/src/vulkan/adapter.rs`):
  MediaTek's PowerVR drivers dereference null inside `vkQueueSubmit` with
  them. Revving crashed on its first frames on a moto g54 5G
  (`vulkan.mtk.so`) and a moto g play 2023 (`vulkan.mt6765.so`) in Firebase
  Test Lab, and runs on both with fences.
- **`AccelerationStructure::raw_handle`** (`wgpu-hal/src/vulkan/mod.rs`), as
  `Buffer` and `TextureView` have theirs: prism-ray-tracing's native path
  writes its descriptor sets itself, since wgpu-core lets a thread hold one
  `as_hal` guard at a time.

30.0.1's own fixes over 30.0.0 come with the base: on Android no fence per
`vkAcquireNextImageKHR` (the loader attached a `sync_file` to each, about 60
a second, until the app ran out of file descriptors), and Display P3 and
BT.2100 looked up at run time on Apple platforms.
