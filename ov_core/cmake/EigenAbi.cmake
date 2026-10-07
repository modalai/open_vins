# Public Eigen objects cross library boundaries. Keep their layout and the
# aligned allocator/free convention independent of native SIMD code generation.
# Sixteen-byte object alignment matches the ARM NEON target; native kernels
# remain enabled. The allocator protocol is explicit for mixed SIMD consumers.
set(OV_EIGEN_ABI_DEFINITIONS
    EIGEN_MAX_ALIGN_BYTES=16
    EIGEN_MAX_STATIC_ALIGN_BYTES=16
    EIGEN_MALLOC_ALREADY_ALIGNED=0)
foreach(definition ${OV_EIGEN_ABI_DEFINITIONS})
    add_definitions(-D${definition})
endforeach()
