cmake_minimum_required(VERSION 3.3)

# Find ROS build system
find_package(catkin QUIET COMPONENTS roscpp ov_core)

# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (catkin_FOUND AND ENABLE_ROS)
    add_definitions(-DROS_AVAILABLE=1)
    catkin_package(
            CATKIN_DEPENDS roscpp cv_bridge ov_core
            INCLUDE_DIRS src/
            LIBRARIES ov_init_lib
    )
else ()
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
    include(GNUInstallDirs)
    set(CATKIN_PACKAGE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    set(CATKIN_PACKAGE_BIN_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    set(CATKIN_GLOBAL_INCLUDE_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif ()

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
# Threads::Threads provides pthread for the lock-free worker pool in ceres_free.
find_package(Threads REQUIRED)
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CERES_LIBRARIES}
        ${catkin_LIBRARIES}
        Threads::Threads
)

# For native builds without ROS, add ov_core library path early
if (NOT catkin_FOUND OR NOT ENABLE_ROS)
    if (EXISTS "${CMAKE_SOURCE_DIR}/../ov_core/build_native/libov_core_lib.so")
        link_directories(${CMAKE_SOURCE_DIR}/../ov_core/build_native)
    endif()
endif()

##################################################
# Make the shared library
##################################################

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/init/InertialInitializer.cpp
        src/dynamic/DynamicInitializer.cpp
        src/static/StaticInitializer.cpp
        src/sim/SimulatorInit.cpp
)

# Ceres-free initialization backend (ov_init::zbft_sfm). Depends only on
# Eigen + ov_core + pthread.
if (OV_INIT_CERES_FREE)
    list(APPEND LIBRARY_SOURCES
        src/ceres_free/Parallel.cpp
        src/ceres_free/Problem.cpp
        src/ceres_free/State_JPLQuatLocal.cpp
        src/ceres_free/Factor_GenericPrior.cpp
        src/ceres_free/Factor_ImageReprojCalib.cpp
        src/ceres_free/Factor_ImuCPIv1.cpp
    )
else()
    # Original Ceres-based factors (require Ceres with LocalParameterization API)
    list(APPEND LIBRARY_SOURCES
        src/ceres/Factor_GenericPrior.cpp
        src/ceres/Factor_ImageReprojCalib.cpp
        src/ceres/Factor_ImuCPIv1.cpp
        src/ceres/State_JPLQuatLocal.cpp
    )
endif()
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_init_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})

# If we are not building with ROS then we need to manually link to its headers
# This isn't that elegant of a way, but this at least allows for building without ROS
# See this stackoverflow answer: https://stackoverflow.com/a/11217008/7718197
if (NOT catkin_FOUND OR NOT ENABLE_ROS)
    message(STATUS "MANUALLY LINKING TO OV_CORE LIBRARY....")
    include_directories(${CMAKE_SOURCE_DIR}/../ov_core/src/)
    target_link_libraries(ov_init_lib ov_core_lib)
endif ()

target_link_libraries(ov_init_lib ${thirdparty_libraries})
target_include_directories(ov_init_lib PUBLIC src/)
install(TARGETS ov_init_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY src/
        DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)


##################################################
# Make binary files!
##################################################

# add_executable(test_simulation src/test_simulation.cpp)
# target_link_libraries(test_simulation ov_init_lib ${thirdparty_libraries})
# install(TARGETS test_simulation
#         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# ---------------------------------------------------------------------------------------------------
# Test & benchmark executables.
# OFF by default so PRODUCTION / Debian package builds do NOT compile any of them. bench_init in
# particular links Ceres, so gating it is also a prerequisite for eventually dropping the
# voxl-ceres-solver dependency. Enable for dev/CI with -DOV_INIT_BUILD_TESTS=ON; the Eigen-only
# self-tests are registered with CTest (run: ctest --test-dir <build> --output-on-failure).
# OV_INIT_BUILD_MINI_TESTS is kept as a narrower switch that builds ONLY the Eigen-only self-tests.
# (test_init_ab_compare.cpp and bench_zbft_s2.cpp remain manual-build; see their file-header recipes.)
# ---------------------------------------------------------------------------------------------------
option(OV_INIT_BUILD_TESTS "Build ov_init test & benchmark executables (dev/CI only)" OFF)
option(OV_INIT_BUILD_MINI_TESTS "Build only the ceres-free Eigen-only solver self-tests" OFF)

find_package(Threads REQUIRED)

# Eigen-only ceres-free solver self-tests -- no ov_core / Ceres / OpenCV. Fast; registered with CTest.
if (OV_INIT_BUILD_TESTS OR OV_INIT_BUILD_MINI_TESTS)
    enable_testing()
    add_executable(test_mini_solver src/ceres_free/test_mini_solver.cpp src/ceres_free/Problem.cpp src/ceres_free/Parallel.cpp)
    target_include_directories(test_mini_solver PRIVATE src/ceres_free ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(test_mini_solver Threads::Threads)
    add_test(NAME test_mini_solver COMMAND test_mini_solver)

    add_executable(test_warmstart_cov src/ceres_free/test_warmstart_cov.cpp src/ceres_free/Problem.cpp src/ceres_free/Parallel.cpp)
    target_include_directories(test_warmstart_cov PRIVATE src/ceres_free ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(test_warmstart_cov Threads::Threads)
    add_test(NAME test_warmstart_cov COMMAND test_warmstart_cov)
endif ()

# Tests/benches that need ov_core (and, for bench_init, Ceres). Dev/CI only.
if (OV_INIT_BUILD_TESTS)
    # test_dynamic_init using TrackSIM (no modal_flow needed for simulation)
    add_executable(test_dynamic_init src/test_dynamic_init.cpp)
    target_link_libraries(test_dynamic_init ov_init_lib ${thirdparty_libraries})

    # NEES consistency gold standard for the ceres-free S2 dynamic init (no OpenCV/Ceres).
    add_executable(test_init_consistency src/test_init_consistency.cpp
            src/ceres_free/Problem.cpp src/ceres_free/Parallel.cpp src/ceres_free/State_JPLQuatLocal.cpp
            src/ceres_free/Factor_ImuCPIv1.cpp src/ceres_free/Factor_GenericPrior.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../ov_core/src/cpi/CpiV1.cpp)
    target_include_directories(test_init_consistency PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/../ov_core/src ${EIGEN3_INCLUDE_DIR})
    target_link_libraries(test_init_consistency Threads::Threads)
    add_test(NAME test_init_consistency COMMAND test_init_consistency)

    # Ceres vs ov_init::zbft_sfm parity/performance benchmark. Compiles the src/ceres/* factors, so it
    # needs Ceres regardless of OV_INIT_CERES_FREE. Look it up locally (QUIET) and skip if unavailable --
    # so enabling tests on a ceres-free image (no voxl-ceres-solver) does not break the build.
    find_package(Ceres QUIET)
    if (Ceres_FOUND)
        add_executable(bench_init
                src/bench_init.cpp
                src/ceres/Factor_ImuCPIv1.cpp
                src/ceres/Factor_GenericPrior.cpp
                src/ceres/State_JPLQuatLocal.cpp
                src/ceres_free/Factor_ImuCPIv1.cpp
                src/ceres_free/Factor_GenericPrior.cpp
                src/ceres_free/State_JPLQuatLocal.cpp
                src/ceres_free/Problem.cpp
                src/ceres_free/Parallel.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../ov_core/src/cpi/CpiV1.cpp
        )
        target_include_directories(bench_init PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ov_core/src ${EIGEN3_INCLUDE_DIR} ${CERES_INCLUDE_DIRS})
        target_link_libraries(bench_init ${thirdparty_libraries} ${CERES_LIBRARIES} Threads::Threads)
    else ()
        message(STATUS "ov_init: bench_init skipped (Ceres not found; it is the Ceres-vs-free benchmark)")
    endif ()
endif ()


