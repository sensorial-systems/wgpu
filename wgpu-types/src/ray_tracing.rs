use alloc::vec::Vec;

#[cfg(any(feature = "serde", test))]
use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::{Features, VertexFormat};

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for all size defining attributes of a single triangle geometry inside a bottom level acceleration structure.
pub struct BlasTriangleGeometrySizeDescriptor {
    /// Format of a vertex position, must be [`VertexFormat::Float32x3`]
    /// with just [`Features::EXPERIMENTAL_RAY_QUERY`]
    /// but [`Features::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS`] adds more.
    pub vertex_format: crate::VertexFormat,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Format of an index. Only needed if an index buffer is used.
    /// If `index_format` is provided `index_count` is required.
    pub index_format: Option<crate::IndexFormat>,
    /// Number of indices. Only needed if an index buffer is used.
    /// If `index_count` is provided `index_format` is required.
    pub index_count: Option<u32>,
    /// Flags for the geometry.
    pub flags: AccelerationStructureGeometryFlags,
    /// The micro-triangles of the geometry's opacity micromap, by subdivision level and
    /// format: none for a geometry without one. These bound both the micromap's own triangles
    /// and the geometry's triangles that use them, at every build.
    ///
    /// Requires [`Features::EXPERIMENTAL_RAY_TRACING_OPACITY_MICROMAP`].
    ///
    /// [`Features::EXPERIMENTAL_RAY_TRACING_OPACITY_MICROMAP`]: crate::Features::EXPERIMENTAL_RAY_TRACING_OPACITY_MICROMAP
    #[cfg_attr(feature = "serde", serde(default))]
    pub opacity_micromap: Vec<OpacityMicromapUsage>,
}

/// How many triangles of an opacity micromap are at one subdivision level, in one format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OpacityMicromapUsage {
    /// How many triangles.
    pub count: u32,
    /// Their subdivision level: each is split into `4^level` micro-triangles, at most 12.
    pub subdivision_level: u32,
    /// How each micro-triangle's opacity is kept.
    pub format: OpacityMicromapFormat,
}

/// How an opacity micromap keeps a micro-triangle's opacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OpacityMicromapFormat {
    /// One bit: 0 transparent, 1 opaque.
    TwoState = 1,
    /// Two bits: 0 transparent, 1 opaque, 2 unknown and transparent, 3 unknown and opaque.
    /// Unknown micro-triangles are hit as non-opaque geometry is, unless a 2-state force says
    /// otherwise.
    FourState = 2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for size-defining attributes of one AABB geometry group in a bottom level acceleration structure.
///
/// Each primitive is one axis-aligned bounding box, typically stored as two `vec3<f32>` min/max corners.
pub struct BlasAABBGeometrySizeDescriptor {
    /// Number of AABB primitives in this geometry.
    pub primitive_count: u32,
    /// Flags for the geometry.
    pub flags: AccelerationStructureGeometryFlags,
}

/// Minimum stride for AABB geometry (24 bytes: two `vec3<f32>`).
pub const AABB_GEOMETRY_MIN_STRIDE: crate::BufferAddress = 24;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for all size defining attributes of all geometries inside a bottom level acceleration structure.
pub enum BlasGeometrySizeDescriptors {
    /// Triangle geometry version.
    Triangles {
        /// Descriptor for each triangle geometry.
        descriptors: Vec<BlasTriangleGeometrySizeDescriptor>,
    },
    /// AABB geometry version.
    AABBs {
        /// Descriptor for each AABB geometry.
        descriptors: Vec<BlasAABBGeometrySizeDescriptor>,
    },
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Update mode for acceleration structure builds.
pub enum AccelerationStructureUpdateMode {
    /// Always perform a full build.
    Build,
    /// If possible, perform an incremental update.
    ///
    /// Not advised for major topology changes.
    /// (Useful for e.g. skinning)
    PreferUpdate,
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for creating a bottom level acceleration structure.
pub struct CreateBlasDescriptor<L> {
    /// Label for the bottom level acceleration structure.
    pub label: L,
    /// Flags for the bottom level acceleration structure.
    pub flags: AccelerationStructureFlags,
    /// Update mode for the bottom level acceleration structure.
    pub update_mode: AccelerationStructureUpdateMode,
}

impl<L> CreateBlasDescriptor<L> {
    /// Takes a closure and maps the label of the blas descriptor into another.
    pub fn map_label<'a, K>(&'a self, fun: impl FnOnce(&'a L) -> K) -> CreateBlasDescriptor<K> {
        CreateBlasDescriptor {
            label: fun(&self.label),
            flags: self.flags,
            update_mode: self.update_mode,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// Descriptor for creating a top level acceleration structure.
pub struct CreateTlasDescriptor<L> {
    /// Label for the top level acceleration structure.
    pub label: L,
    /// Number of instances that can be stored in the acceleration structure.
    pub max_instances: u32,
    /// Flags for the bottom level acceleration structure.
    pub flags: AccelerationStructureFlags,
    /// Update mode for the bottom level acceleration structure.
    pub update_mode: AccelerationStructureUpdateMode,
}

impl<L> CreateTlasDescriptor<L> {
    /// Takes a closure and maps the label of the blas descriptor into another.
    pub fn map_label<'a, K>(&'a self, fun: impl FnOnce(&'a L) -> K) -> CreateTlasDescriptor<K> {
        CreateTlasDescriptor {
            label: fun(&self.label),
            flags: self.flags,
            update_mode: self.update_mode,
            max_instances: self.max_instances,
        }
    }
}

bitflags::bitflags!(
    /// Flags for acceleration structures
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AccelerationStructureFlags: u8 {
        /// Allow for incremental updates (no change in size), currently this is unimplemented
        /// and will build as normal (this is fine, update vs build should be unnoticeable)
        const ALLOW_UPDATE = 1 << 0;
        /// Allow the acceleration structure to be compacted in a copy operation
        /// (`Blas::prepare_for_compaction`, `CommandEncoder::compact_blas`).
        ///
        /// Some backends allocate structures with this flag from memory meant for
        /// short-lived resources. Only set it on structures you compact soon after building.
        const ALLOW_COMPACTION = 1 << 1;
        /// Optimize for fast ray tracing performance, recommended if the geometry is unlikely
        /// to change (e.g. in a game: non-interactive scene geometry)
        const PREFER_FAST_TRACE = 1 << 2;
        /// Optimize for fast build time, recommended if geometry is likely to change frequently
        /// (e.g. in a game: player model).
        const PREFER_FAST_BUILD = 1 << 3;
        /// Optimize for low memory footprint (both while building and in the output BLAS).
        const LOW_MEMORY = 1 << 4;
        /// Use `BlasTriangleGeometry::transform_buffer` when building a BLAS (only allowed in
        /// BLAS creation)
        const USE_TRANSFORM = 1 << 5;
        /// Allow retrieval of the vertices of the triangle hit by a ray.
        const ALLOW_RAY_HIT_VERTEX_RETURN = 1 << 6;
        /// The structure moves over a ray's time: a bottom level one's triangles go from
        /// their vertices to their end vertices, a top level one's instances from their
        /// transforms to their end transforms. Rays traced at a time between 0 and 1 meet
        /// them where they are at that time.
        ///
        /// Requires [`Features::EXPERIMENTAL_RAY_TRACING_MOTION_BLUR`].
        ///
        /// [`Features::EXPERIMENTAL_RAY_TRACING_MOTION_BLUR`]: crate::Features::EXPERIMENTAL_RAY_TRACING_MOTION_BLUR
        const ALLOW_MOTION = 1 << 7;
    }
);

bitflags::bitflags!(
    /// Flags for acceleration structure geometries
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AccelerationStructureGeometryFlags: u8 {
        /// Is OPAQUE (is there no alpha test) recommended as currently in naga there is no
        /// candidate intersections yet so currently BLASes without this flag will not have hits.
        /// Not enabling this makes the BLAS unable to be interacted with in WGSL.
        const OPAQUE = 1 << 0;
        /// NO_DUPLICATE_ANY_HIT_INVOCATION, not useful unless using hal with wgpu, ray-tracing
        /// pipelines are not supported in wgpu so any-hit shaders do not exist. For when any-hit
        /// shaders are implemented (or experienced users who combine this with an underlying library:
        /// for any primitive (triangle or AABB) multiple any-hit shaders sometimes may be invoked
        /// (especially in AABBs like a sphere), if this flag in present only one hit on a primitive may
        /// invoke an any-hit shader.
        const NO_DUPLICATE_ANY_HIT_INVOCATION = 1 << 1;
    }
);

bitflags::bitflags!(
    /// How rays treat an instance of a top level acceleration structure.
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct TlasInstanceFlags: u8 {
        /// No face of the instance's triangles is culled, whatever the ray's
        /// flags ask.
        const TRIANGLE_CULL_DISABLE = 1 << 0;
        /// The instance's triangles face the other way: front where they
        /// would be back facing.
        const TRIANGLE_FLIP_FACING = 1 << 1;
        /// Every geometry of the instance is opaque, whatever its own flags.
        /// Rays' flags still override this.
        const FORCE_OPAQUE = 1 << 2;
        /// No geometry of the instance is opaque, whatever its own flags.
        /// Rays' flags still override this.
        const FORCE_NO_OPAQUE = 1 << 3;
        /// The instance's opacity micromaps' unknown micro-triangles are what they lean to:
        /// transparent or opaque, never non-opaque.
        const FORCE_OPACITY_MICROMAP_2_STATE = 1 << 4;
        /// The instance's opacity micromaps are not looked at.
        const DISABLE_OPACITY_MICROMAPS = 1 << 5;
    }
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// What a copy between acceleration structures should do
pub enum AccelerationStructureCopy {
    /// Directly duplicate an acceleration structure to another
    Clone,
    /// Duplicate and compact an acceleration structure
    Compact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// What type the data of an acceleration structure is
pub enum AccelerationStructureType {
    /// The types of the acceleration structure are triangles
    Triangles,
    /// The types of the acceleration structure are axis aligned bounding boxes
    AABBs,
    /// The types of the acceleration structure are instances
    Instances,
}

/// Alignment requirement for transform buffers used in acceleration structure builds
pub const TRANSFORM_BUFFER_ALIGNMENT: crate::BufferAddress = 16;

/// Alignment requirement for instance buffers used in acceleration structure builds (`build_acceleration_structures_unsafe_tlas`)
pub const INSTANCE_BUFFER_ALIGNMENT: crate::BufferAddress = 16;
