cmake_minimum_required(VERSION 3.16)

project(robot_occlusion_recovery_middleware LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(realsense2 REQUIRED IMPORTED_TARGET realsense2)
pkg_check_modules(libcurl REQUIRED IMPORTED_TARGET libcurl)

set(FAIRINO_SDK_DIR "" CACHE PATH "Path to the FAIRINO C++ SDK libfairino directory")
if(NOT EXISTS "${FAIRINO_SDK_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Set FAIRINO_SDK_DIR to the SDK libfairino directory")
endif()

add_subdirectory("${FAIRINO_SDK_DIR}" "${CMAKE_BINARY_DIR}/fairino-sdk"
                 EXCLUDE_FROM_ALL)
set_target_properties(
  fairino
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/fairino-sdk"
             ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/fairino-sdk")

add_executable(run_fr5_openvla integration/run_fr5_openvla.cpp)
target_include_directories(
  run_fr5_openvla
  PRIVATE "${FAIRINO_SDK_DIR}/src/include/Robot-EN" ${OpenCV_INCLUDE_DIRS})
target_compile_options(
  run_fr5_openvla
  PRIVATE "$<$<CONFIG:Release>:-O2;-DNDEBUG>" -Wall -Wextra -Wpedantic)
target_link_libraries(
  run_fr5_openvla
  PRIVATE fairino PkgConfig::realsense2 PkgConfig::libcurl ${OpenCV_LIBS}
          nlohmann_json::nlohmann_json Threads::Threads)
