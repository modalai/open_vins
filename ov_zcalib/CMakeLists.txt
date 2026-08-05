# ov_zcalib: the targetless in-flight visual-inertial calibrator (upstream-OpenVINS
# layout, minimal instrumentation). Built behind OV_BUILD_CALIB; the server's
# --calibrate mode links ov_zcalib_lib, and the ov_zcalibrate CLI is the host
# replay/forensics tool.
#
# Layout follows upstream module nouns: types/ (models), cpi/ (preintegration),
# init/ (bootstrap), sim/ (synthetic worlds), utils/ (record/writeback/feeders),
# core/ (session runner), solve/ + window/ (calib-specific, no upstream analog),
# run_calibrate.cpp entrypoint, test_*.cpp at src root.
cmake_minimum_required(VERSION 3.3)
project(ov_zcalib)

find_package(Eigen3 REQUIRED)

include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../ov_core/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../ov_init/src
)

# Flags MUST match ov_core/ov_init exactly: Eigen objects cross the .so boundary, and a
# vectorization/alignment mismatch (e.g. AVX in ov_init, SSE2 here) makes Eigen allocate
# with one allocator and free with another -- glibc's "double free or corruption (out)".
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_CROSSCOMPILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a -O3 -ffast-math -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -ftree-vectorize")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3 -ffast-math -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -ftree-vectorize")
endif()

# Port ladder (grows with task 4): core/ (runner) + run_calibrate.cpp remain; each stage
# gated on its oracle tests before the next lands. EurocFeeder is NOT ported (operator
# standing order: Stinger/VOXL data only — the cruft dies by omission).
set(OV_ZCALIB_LIB_SOURCES
        src/cpi/AciCalibPreint.cpp
        src/solve/Factor_ImuAci3.cpp
        src/solve/WindowBA.cpp
        src/solve/JointCalib.cpp
        src/window/LinearSeed.cpp
        src/window/WindowHarvester.cpp
        src/window/WindowScorer.cpp
        src/init/HandEyeWahba.cpp
        src/utils/SessionRecord.cpp
        src/core/CalibSessionRunner.cpp
)

# Host-only ingest front-end (TrackKLT-based; the device superbuild strips KLT --
# TrackOCL is the only tracker on target, fed in-process by the server).
if (NOT DISABLE_TRACK_KLT)
    list(APPEND OV_ZCALIB_LIB_SOURCES
            src/utils/VoxlLogFeeder.cpp
    )
endif ()

add_library(ov_zcalib_lib SHARED ${OV_ZCALIB_LIB_SOURCES})
target_link_libraries(ov_zcalib_lib ov_core_lib ov_init_lib)
target_include_directories(ov_zcalib_lib PUBLIC src/)

install(TARGETS ov_zcalib_lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Test oracles. Target names carry the zcalib_ prefix; ctest names match the targets.
option(OV_ZCALIB_BUILD_TESTS "Build ov_zcalib test/oracle executables (dev/CI only)" ON)
if (OV_ZCALIB_BUILD_TESTS AND NOT DISABLE_TRACK_KLT)
    enable_testing()
    add_executable(zcalib_test_aci3_fd src/test_aci3_fd.cpp)
    target_link_libraries(zcalib_test_aci3_fd ov_zcalib_lib)
    add_test(NAME zcalib_test_aci3_fd COMMAND zcalib_test_aci3_fd)
    add_executable(zcalib_test_preint_chain src/test_preint_chain.cpp)
    target_link_libraries(zcalib_test_preint_chain ov_zcalib_lib)
    add_test(NAME zcalib_test_preint_chain COMMAND zcalib_test_preint_chain)
    add_executable(zcalib_test_convention_ports src/test_convention_ports.cpp)
    target_link_libraries(zcalib_test_convention_ports ov_zcalib_lib)
    add_test(NAME zcalib_test_convention_ports COMMAND zcalib_test_convention_ports)
    add_executable(zcalib_test_frontend src/test_frontend.cpp)
    target_link_libraries(zcalib_test_frontend ov_zcalib_lib)
    add_test(NAME zcalib_test_frontend COMMAND zcalib_test_frontend)
    add_executable(zcalib_test_calib_e2e src/test_calib_e2e.cpp)
    target_link_libraries(zcalib_test_calib_e2e ov_zcalib_lib)
    add_test(NAME zcalib_test_calib_e2e COMMAND zcalib_test_calib_e2e)
    add_executable(zcalib_test_session_e2e src/test_session_e2e.cpp)
    target_link_libraries(zcalib_test_session_e2e ov_zcalib_lib)
    add_test(NAME zcalib_test_session_e2e COMMAND zcalib_test_session_e2e)
    add_executable(zcalib_test_tg_e2e src/test_tg_e2e.cpp)
    target_link_libraries(zcalib_test_tg_e2e ov_zcalib_lib)
    add_test(NAME zcalib_test_tg_e2e COMMAND zcalib_test_tg_e2e)
    add_executable(zcalib_test_wald_mc src/test_wald_mc.cpp)
    target_link_libraries(zcalib_test_wald_mc ov_zcalib_lib)
    add_test(NAME zcalib_test_wald_mc COMMAND zcalib_test_wald_mc)
    add_executable(zcalib_test_export_parity src/test_export_parity.cpp)
    target_link_libraries(zcalib_test_export_parity ov_zcalib_lib)
    add_test(NAME zcalib_test_export_parity COMMAND zcalib_test_export_parity)
endif ()

# The host CLI (upstream run_*.cpp entrypoint pattern).
if (NOT DISABLE_TRACK_KLT)
    add_executable(ov_zcalibrate src/run_calibrate.cpp)
    target_link_libraries(ov_zcalibrate ov_zcalib_lib)
endif ()
