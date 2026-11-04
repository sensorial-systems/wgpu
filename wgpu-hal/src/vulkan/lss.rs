//! `VK_NV_ray_tracing_linear_swept_spheres`, which ash does not have yet.

use alloc::boxed::Box;
use core::{ffi::c_void, ffi::CStr, ptr};

use ash::vk;

use super::conv;

pub const NAME: &CStr = c"VK_NV_ray_tracing_linear_swept_spheres";

const STRUCTURE_TYPE_FEATURES: vk::StructureType = vk::StructureType::from_raw(1000429008);
const STRUCTURE_TYPE_LINEAR_SWEPT_SPHERES_DATA: vk::StructureType =
    vk::StructureType::from_raw(1000429009);
const STRUCTURE_TYPE_SPHERES_DATA: vk::StructureType = vk::StructureType::from_raw(1000429010);
const GEOMETRY_TYPE_SPHERES: vk::GeometryTypeKHR = vk::GeometryTypeKHR::from_raw(1000429004);
const GEOMETRY_TYPE_LINEAR_SWEPT_SPHERES: vk::GeometryTypeKHR =
    vk::GeometryTypeKHR::from_raw(1000429005);

/// `VK_PIPELINE_CREATE_2_RAY_TRACING_ALLOW_SPHERES_AND_LINEAR_SWEPT_SPHERES_BIT_NV`, which only
/// `VkPipelineCreateFlags2CreateInfo` (`VK_KHR_maintenance5`) can give.
pub const PIPELINE_CREATE_2_ALLOW_SPHERES: vk::PipelineCreateFlags2KHR =
    vk::PipelineCreateFlags2KHR::from_raw(0x2_0000_0000);

/// `VkPhysicalDeviceRayTracingLinearSweptSpheresFeaturesNV`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PhysicalDeviceFeatures {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub spheres: vk::Bool32,
    pub linear_swept_spheres: vk::Bool32,
}

// SAFETY: as ash's own structures, it holds no pointer it owns.
unsafe impl Send for PhysicalDeviceFeatures {}
// SAFETY: as above.
unsafe impl Sync for PhysicalDeviceFeatures {}
// SAFETY: it starts as every Vulkan structure in a chain does.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for PhysicalDeviceFeatures {}
// SAFETY: as above.
unsafe impl vk::ExtendsDeviceCreateInfo for PhysicalDeviceFeatures {}

impl Default for PhysicalDeviceFeatures {
    fn default() -> Self {
        Self {
            s_type: STRUCTURE_TYPE_FEATURES,
            p_next: ptr::null_mut(),
            spheres: vk::FALSE,
            linear_swept_spheres: vk::FALSE,
        }
    }
}

impl PhysicalDeviceFeatures {
    /// Both kinds of geometry, which wgpu's one feature has.
    pub fn enabled() -> Self {
        Self {
            spheres: vk::TRUE,
            linear_swept_spheres: vk::TRUE,
            ..Self::default()
        }
    }

    pub const fn supported(&self) -> bool {
        self.spheres != 0 && self.linear_swept_spheres != 0
    }
}

/// `VkAccelerationStructureGeometrySpheresDataNV`.
#[repr(C)]
pub struct SpheresData {
    s_type: vk::StructureType,
    p_next: *const c_void,
    vertex_format: vk::Format,
    vertex_data: vk::DeviceOrHostAddressConstKHR,
    vertex_stride: vk::DeviceSize,
    radius_format: vk::Format,
    radius_data: vk::DeviceOrHostAddressConstKHR,
    radius_stride: vk::DeviceSize,
    index_type: vk::IndexType,
    index_data: vk::DeviceOrHostAddressConstKHR,
    index_stride: vk::DeviceSize,
}

/// `VkAccelerationStructureGeometryLinearSweptSpheresDataNV`.
#[repr(C)]
pub struct LinearSweptSpheresData {
    spheres: SpheresData,
    /// `VkRayTracingLssIndexingModeNV`: list 0, successive 1.
    indexing_mode: i32,
    /// `VkRayTracingLssPrimitiveEndCapsModeNV`: none 0, chained 1.
    end_caps_mode: i32,
}

/// The data a sphere geometry's `p_next` points to, kept where it is until
/// the geometry is used.
pub enum GeometryData {
    Spheres(#[expect(dead_code, reason = "only pointed to")] Box<SpheresData>),
    LinearSweptSpheres(
        #[expect(dead_code, reason = "only pointed to")] Box<LinearSweptSpheresData>,
    ),
}

/// The device addresses of a sphere geometry's data, all zero where only
/// sizes are asked for.
pub struct Addresses {
    pub vertices: vk::DeviceAddress,
    pub radii: vk::DeviceAddress,
    pub indices: vk::DeviceAddress,
}

/// A geometry of spheres or linear swept spheres, and the data it points to.
pub fn geometry<B: crate::DynBuffer + ?Sized>(
    spheres: &crate::AccelerationStructureSpheres<'_, B>,
    addresses: Addresses,
) -> (vk::AccelerationStructureGeometryKHR<'static>, GeometryData) {
    let (index_type, index_stride) = match spheres.indices {
        Some(ref indices) => (
            conv::map_index_format(indices.format),
            u64::from(indices.format.byte_size()),
        ),
        None => (vk::IndexType::NONE_KHR, 0),
    };
    let address = |device_address| vk::DeviceOrHostAddressConstKHR { device_address };
    let data = SpheresData {
        s_type: STRUCTURE_TYPE_SPHERES_DATA,
        p_next: ptr::null(),
        vertex_format: conv::map_vertex_format(spheres.vertex_format),
        vertex_data: address(addresses.vertices),
        vertex_stride: spheres.vertex_stride,
        radius_format: vk::Format::R32_SFLOAT,
        radius_data: address(addresses.radii),
        radius_stride: spheres.radius_stride,
        index_type,
        index_data: address(if spheres.indices.is_some() {
            addresses.indices
        } else {
            0
        }),
        index_stride,
    };
    let (geometry_type, data, p_next) = match spheres.linear_swept {
        None => {
            let data = Box::new(data);
            let p_next = <*const SpheresData>::cast::<c_void>(&*data);
            (GEOMETRY_TYPE_SPHERES, GeometryData::Spheres(data), p_next)
        }
        Some(linear_swept) => {
            let data = Box::new(LinearSweptSpheresData {
                spheres: SpheresData {
                    s_type: STRUCTURE_TYPE_LINEAR_SWEPT_SPHERES_DATA,
                    ..data
                },
                indexing_mode: match linear_swept.indexing {
                    wgt::LinearSweptSphereIndexing::List => 0,
                    wgt::LinearSweptSphereIndexing::Successive => 1,
                },
                end_caps_mode: match linear_swept.end_caps {
                    wgt::LinearSweptSphereEndCaps::None => 0,
                    wgt::LinearSweptSphereEndCaps::Chained => 1,
                },
            });
            let p_next = <*const LinearSweptSpheresData>::cast::<c_void>(&*data);
            (
                GEOMETRY_TYPE_LINEAR_SWEPT_SPHERES,
                GeometryData::LinearSweptSpheres(data),
                p_next,
            )
        }
    };
    let mut geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(geometry_type)
        .geometry(vk::AccelerationStructureGeometryDataKHR::default())
        .flags(conv::map_acceleration_structure_geometry_flags(
            spheres.flags,
        ));
    geometry.p_next = p_next;
    (geometry, data)
}
