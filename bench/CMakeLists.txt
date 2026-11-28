# Pipeline benchmarks in the lab3 style: every operation is built once per variant and
# written to bench/compiled/<operation>_<variant>; bench/run_benchmarks.py drives them.
#   <op>_serial, <op>_simd: our code in that execution mode
#   <op>_opencv:            the OpenCV counterpart (see BENCHMARK.md)
# The binaries and all the code they measure are compiled with -O2. A target option comes
# after the build-type flags on the command line, so it overrides Release's -O3.

find_package(Boost CONFIG QUIET COMPONENTS program_options)
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs features2d calib3d)
if (NOT Boost_FOUND OR NOT OpenCV_FOUND)
    message(STATUS "Pipeline benchmarks skipped: they need Boost.Program_options and OpenCV with features2d and calib3d")
    return()
endif ()

set(BENCH_OPTIMIZATION -O2)

# persons 1, 2 and 3 compiled again at -O2; the sources are used as they are
set(BENCH_MAPPING_DIR ${SLAM_ROOT}/src/our/mapping)
add_library(bench_slam STATIC
        ${SLAM_ROOT}/src/our/feature_extractor.cpp
        ${SLAM_ROOT}/src/our/processor.cpp
        ${SLAM_ROOT}/src/our/pose_estimation.cpp
        ${BENCH_MAPPING_DIR}/linalg.cpp
        ${BENCH_MAPPING_DIR}/se3.cpp
        ${BENCH_MAPPING_DIR}/integration_adapters.cpp
        ${BENCH_MAPPING_DIR}/teammate_interfaces.cpp
        ${BENCH_MAPPING_DIR}/map.cpp
        ${BENCH_MAPPING_DIR}/bundle_adjustment.cpp
        ${BENCH_MAPPING_DIR}/pose_graph.cpp
        ${BENCH_MAPPING_DIR}/loop_closing.cpp
        ${BENCH_MAPPING_DIR}/map_pipeline.cpp)
target_include_directories(bench_slam PUBLIC ${SLAM_ROOT}/src/our ${BENCH_MAPPING_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bench_slam PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_compile_options(bench_slam PRIVATE ${BENCH_OPTIMIZATION})

# operation -> variants
set(BENCH_OPERATIONS features pose mapping pipeline)
set(BENCH_VARIANTS_features serial opencv)
set(BENCH_VARIANTS_pose serial opencv)
set(BENCH_VARIANTS_mapping serial simd opencv)
set(BENCH_VARIANTS_pipeline serial opencv)

foreach (operation IN LISTS BENCH_OPERATIONS)
    foreach (variant IN LISTS BENCH_VARIANTS_${operation})
        set(target bench_${operation}_${variant})
        add_executable(${target} ${operation}.cpp ${CMAKE_CURRENT_SOURCE_DIR}/options_parser/options_parser.cpp)
        target_include_directories(${target} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/options_parser ${SLAM_ROOT}/src/utils)
        target_link_libraries(${target} PRIVATE bench_slam Boost::program_options)
        target_compile_options(${target} PRIVATE ${BENCH_OPTIMIZATION})
        string(TOUPPER ${variant} variant_macro)
        target_compile_definitions(${target} PRIVATE BENCH_${variant_macro})
        set_target_properties(${target} PROPERTIES
                OUTPUT_NAME ${operation}_${variant} RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/compiled)
    endforeach ()
endforeach ()
