cmake_minimum_required(VERSION 3.20)
project(rgbd_slam LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Keep the imported project and both teammates' sources byte-for-byte unchanged.
get_filename_component(SLAM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)
option(SLAM_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(SLAM_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

add_library(slam_frontend
    ${SLAM_ROOT}/src/our/feature_extractor.cpp
    ${SLAM_ROOT}/src/our/processor.cpp
    ${SLAM_ROOT}/src/our/pose_estimation.cpp)
target_include_directories(slam_frontend PUBLIC ${SLAM_ROOT}/src/our ${OpenCV_INCLUDE_DIRS})
target_link_libraries(slam_frontend PUBLIC ${OpenCV_LIBS})

set(MAPPING_DIR ${SLAM_ROOT}/src/our/mapping)
add_library(slam_mapping
    ${MAPPING_DIR}/linalg.cpp
    ${MAPPING_DIR}/se3.cpp
    ${MAPPING_DIR}/integration_adapters.cpp
    ${MAPPING_DIR}/teammate_interfaces.cpp
    ${MAPPING_DIR}/map.cpp
    ${MAPPING_DIR}/bundle_adjustment.cpp
    ${MAPPING_DIR}/pose_graph.cpp
    ${MAPPING_DIR}/loop_closing.cpp
    ${MAPPING_DIR}/map_pipeline.cpp)
target_include_directories(slam_mapping PUBLIC ${MAPPING_DIR})
target_link_libraries(slam_mapping PUBLIC slam_frontend Threads::Threads)

add_library(slam_pipeline ${SLAM_ROOT}/src/our/rgbd_pipeline.cpp)
target_link_libraries(slam_pipeline PUBLIC slam_mapping)
add_executable(slam_rgbd ${SLAM_ROOT}/apps/slam_rgbd.cpp)
target_link_libraries(slam_rgbd PRIVATE slam_pipeline)
add_executable(slam_mapping_sequence ${SLAM_ROOT}/examples/mapping_sequence.cpp)
target_link_libraries(slam_mapping_sequence PRIVATE slam_mapping)
add_executable(slam_mapping_bench ${SLAM_ROOT}/bench/mapping_bench.cpp)
target_link_libraries(slam_mapping_bench PRIVATE slam_mapping)

include(CTest)
if(BUILD_TESTING)
    foreach(test IN ITEMS pose_estimation_tests mapping_adapter_tests mapping_modes_tests slam_pipeline_tests)
        add_executable(slam_${test} ${SLAM_ROOT}/tests/${test}.cpp)
        target_link_libraries(slam_${test} PRIVATE slam_pipeline)
        add_test(NAME ${test} COMMAND slam_${test})
    endforeach()
endif()
