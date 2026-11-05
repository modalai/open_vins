cmake_minimum_required(VERSION 3.10)
project(zstudio)

option(ZSTUDIO_STANDALONE "Install the studio without a ROS build system" OFF)
# requirements.txt is resolved at install time: requirements the Python
# interpreter already provides (distribution / rosdep packages) are kept, the
# missing ones are pip-installed into <libdir>/zstudio/python-deps (never
# into the system Python), which zstudio appends to sys.path.
option(ZSTUDIO_INSTALL_PYTHON_DEPS "Install missing Python requirements (requirements.txt) during install" ON)
option(ZSTUDIO_PYTHON_DEPS_STRICT "Fail the install when a Python requirement cannot be installed" OFF)
find_package(Python3 COMPONENTS Interpreter)

# Install-time hook for requirements.txt; <dest> is relative to the install prefix.
function(zstudio_install_requirements dest)
    install(FILES requirements.txt DESTINATION ${dest})
    if (NOT ZSTUDIO_INSTALL_PYTHON_DEPS)
        return()
    endif ()
    if (NOT Python3_EXECUTABLE)
        message(WARNING "zstudio: no Python 3 interpreter found; Python requirements are not installed")
        return()
    endif ()
    set(strict "")
    if (ZSTUDIO_PYTHON_DEPS_STRICT)
        set(strict "--strict")
    endif ()
    install(CODE "
        set(_ov_target \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${dest}/python-deps\")
        execute_process(
            COMMAND \"${Python3_EXECUTABLE}\" \"${CMAKE_CURRENT_SOURCE_DIR}/scripts/install_requirements.py\"
                    --requirements \"${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt\"
                    --target \"\${_ov_target}\" --python \"${Python3_EXECUTABLE}\" ${strict}
            RESULT_VARIABLE _ov_result)
        if (NOT _ov_result EQUAL 0)
            message(FATAL_ERROR \"zstudio: Python requirements could not be installed (see above)\")
        endif ()
    ")
endfunction()
# OpenVINS estimator presets (repository config/) double as VIO config templates.
set(OV_CONFIG_PRESETS "${CMAKE_CURRENT_SOURCE_DIR}/../config" CACHE PATH "Directory of OpenVINS config presets to install for the studio's VIO page")
if (NOT ZSTUDIO_STANDALONE AND (CATKIN_DEVEL_PREFIX OR "$ENV{ROS_VERSION}" STREQUAL "1"))
    find_package(catkin REQUIRED)
    catkin_python_setup()
    catkin_package()
    catkin_install_python(PROGRAMS scripts/zstudio scripts/ov_zcalib_realsense_bridge scripts/ov_log_inspector DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
    install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
    # OpenVINS presets used as VIO config templates (repository config/).
    install(DIRECTORY ${OV_CONFIG_PRESETS}/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/openvins_config)
    zstudio_install_requirements(${CATKIN_GLOBAL_LIB_DESTINATION}/${PROJECT_NAME})
elseif (NOT ZSTUDIO_STANDALONE AND "$ENV{ROS_VERSION}" STREQUAL "2")
    find_package(ament_cmake REQUIRED)
    find_package(ament_cmake_python REQUIRED)
    ament_python_install_package(zstudio PACKAGE_DIR src/zstudio)
    install(PROGRAMS scripts/zstudio scripts/ov_zcalib_realsense_bridge scripts/ov_log_inspector DESTINATION lib/${PROJECT_NAME})
    install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
    install(DIRECTORY ${OV_CONFIG_PRESETS}/ DESTINATION share/${PROJECT_NAME}/openvins_config)
    zstudio_install_requirements(lib/${PROJECT_NAME})
    ament_package()
else ()
    # Keep a private module directory beside the executable: no pip, ROS, or
    # system Python installation is needed for a relocatable desktop install.
    include(GNUInstallDirs)
    install(PROGRAMS scripts/zstudio scripts/ov_zcalib_realsense_bridge scripts/ov_log_inspector DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY config DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME})
    install(DIRECTORY ${OV_CONFIG_PRESETS}/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/openvins_config)
    install(DIRECTORY src/zstudio DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
        PATTERN "__pycache__" EXCLUDE)
    zstudio_install_requirements(${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
endif ()

include(CTest)
if (BUILD_TESTING)
    if (NOT Python3_EXECUTABLE)
        message(FATAL_ERROR "zstudio tests need a Python 3 interpreter")
    endif ()
    add_test(NAME zstudio_contracts COMMAND ${Python3_EXECUTABLE} -m unittest discover
        -s ${CMAKE_CURRENT_SOURCE_DIR}/tests -p "test_*.py")
    set_tests_properties(zstudio_contracts PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src;QT_QPA_PLATFORM=offscreen")
endif ()
