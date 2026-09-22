cmake_minimum_required(VERSION 3.5)

# Find ros dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)

# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (NOT ENABLE_ROS)
    message(FATAL_ERROR "Build with ROS1.cmake if you don't have ROS.")
endif ()
add_definitions(-DROS_AVAILABLE=2)

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
)

##################################################
# Make the core library
##################################################

option(DISABLE_TRACK_KLT "Disable TrackKLT to avoid opencv video module dependency" OFF)
if (DISABLE_TRACK_KLT)
    add_definitions(-DDISABLE_TRACK_KLT=1)
endif ()

list(APPEND LIBRARY_SOURCES
        src/dummy.cpp
        src/cpi/CpiV1.cpp
        src/cpi/CpiV2.cpp
        src/sim/BsplineSE3.cpp
        src/track/TrackBase.cpp
        src/track/TrackAruco.cpp
        src/track/TrackDescriptor.cpp
        src/track/TrackSIM.cpp
        src/types/Landmark.cpp
        src/feat/Feature.cpp
        src/feat/FeatureDatabase.cpp
        src/feat/FeatureInitializer.cpp
        src/utils/print.cpp
)
# Match the optional tracker selection and direct dependency used by ROS1.
# A CPU-only ROS installation does not have the VOXL modal-flow dependency.
if (OV_USE_MODAL_FLOW)
    list(APPEND LIBRARY_SOURCES src/track/TrackOCL/TrackOCL.cpp)
endif()
if (NOT DISABLE_TRACK_KLT)
    list(APPEND LIBRARY_SOURCES src/track/TrackKLT.cpp)
endif ()
file(GLOB_RECURSE LIBRARY_HEADERS "src/*.h")
add_library(ov_core_lib SHARED ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_compile_definitions(ov_core_lib PUBLIC ${OV_EIGEN_ABI_DEFINITIONS}
    HAVE_OPENCL=${OV_USE_MODAL_FLOW} OV_HAVE_MODAL_FLOW=${OV_USE_MODAL_FLOW})
if (OV_USE_MODAL_FLOW)
    target_include_directories(ov_core_lib PUBLIC ${OpenCL_INCLUDE_DIRS} ${MODAL_FLOW_INCLUDE_DIR})
endif()
ament_target_dependencies(ov_core_lib rclcpp cv_bridge)
target_link_libraries(ov_core_lib ${thirdparty_libraries})
target_include_directories(ov_core_lib PUBLIC src/)
install(TARGETS ov_core_lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
)
install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
ament_export_include_directories(include)
ament_export_libraries(ov_core_lib)
ament_export_definitions(${OV_EIGEN_ABI_DEFINITIONS}
    HAVE_OPENCL=${OV_USE_MODAL_FLOW} OV_HAVE_MODAL_FLOW=${OV_USE_MODAL_FLOW})
ament_export_dependencies(rclcpp cv_bridge Eigen3)
if (OV_USE_MODAL_FLOW)
    ament_export_include_directories(${OpenCL_INCLUDE_DIRS} ${MODAL_FLOW_INCLUDE_DIR})
endif()

##################################################
# Make binary files!
##################################################

# TODO: UPGRADE THIS TO ROS2 AS ANOTHER FILE!!
#if (catkin_FOUND AND ENABLE_ROS)
#    add_executable(test_tracking src/test_tracking.cpp)
#    target_link_libraries(test_tracking ov_core_lib ${thirdparty_libraries})
#endif ()

if (NOT DISABLE_TRACK_KLT)
    add_executable(test_webcam src/test_webcam.cpp)
    ament_target_dependencies(test_webcam rclcpp cv_bridge)
    target_link_libraries(test_webcam ov_core_lib ${thirdparty_libraries})
    install(TARGETS test_webcam DESTINATION lib/${PROJECT_NAME})
endif ()

# finally define this as the package
ament_package()
