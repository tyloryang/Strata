# Strata on Ascend - the CANN backend (Phase 0).
#
# This is a separate project on purpose, mirroring sycl/. The top-level
# build describes the CUDA engine; with -DSTRATA_ENABLE_CANN=ON it
# returns after add_subdirectory(cann), so nothing below is configured
# and the default build is byte-identical to the build without the flag.
# Upstream files are never edited by this port; the port is the content
# of this directory (plus docs/CANN.md and docs/CANN_PORT_PLAN.md).
#
#   cmake -S cann -B build-cann -G Ninja   # or: cmake -S . -B build -DSTRATA_ENABLE_CANN=ON
#   cmake --build build-cann
#
# Phase 0 builds exactly one thing: strata_cann_probe, a no-model
# self-test that calls only ACL runtime APIs whose signatures were
# checked against /usr/local/Ascend/cann-9.0.0/include/acl/acl_rt.h
# on the first verified target (910B2C, CANN 9.0.0, 2026-11-04).
# It does not load a model and it contains no kernel yet. If CANN is
# not installed, configure fails with a plain message instead of
# silently building a stub.
cmake_minimum_required(VERSION 3.24)
project(strata_cann VERSION 0.1.56 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# CANN location: $ASCEND_HOME_PATH, $ASCEND_TOOLKIT_HOME, or the
# standard install path. All three were present on the first target.
set(_cann_roots
  $ENV{ASCEND_HOME_PATH}
  $ENV{ASCEND_TOOLKIT_HOME}
  /usr/local/Ascend/cann-9.0.0
  /usr/local/Ascend/ascend-toolkit/latest
  /usr/local/Ascend/cann
)
set(STRATA_CANN_ROOT "" CACHE PATH "CANN install root (default: search ASCEND_HOME_PATH and /usr/local/Ascend)")
if(STRATA_CANN_ROOT)
  list(PREPEND _cann_roots ${STRATA_CANN_ROOT})
endif()
find_path(STRATA_CANN_INCLUDE_DIR acl/acl_rt.h
  HINTS ${_cann_roots}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)
find_library(STRATA_CANN_ASCENDCL_LIB NAMES ascendcl
  HINTS ${_cann_roots}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
)
if(NOT STRATA_CANN_INCLUDE_DIR OR NOT STRATA_CANN_ASCENDCL_LIB)
  message(FATAL_ERROR
    "Strata CANN: CANN not found (looked for acl/acl_rt.h and libascendcl.so under ASCEND_HOME_PATH / /usr/local/Ascend). "
    "Install CANN 9.0.0 (the version on the first verified target, 910B2C) or pass -DSTRATA_CANN_ROOT=<path>. "
    "This is not a default-build failure: the default build never enters cann/.")
endif()
message(STATUS "Strata CANN: CANN found at ${STRATA_CANN_INCLUDE_DIR} (${STRATA_CANN_ASCENDCL_LIB})")

add_executable(strata_cann_probe src/cann_probe.cpp)
target_include_directories(strata_cann_probe PRIVATE ${STRATA_CANN_INCLUDE_DIR})
target_link_libraries(strata_cann_probe PRIVATE ${STRATA_CANN_ASCENDCL_LIB})
target_compile_definitions(strata_cann_probe PRIVATE STRATA_VERSION="0.1.56")

# --- Phase 1: CANN aclnn ops (GEMM + GDN attention) ---
# Both branches were written independently and each defined
# STRATA_CANN_OPAPI_LIB for a different library; the merge renames
# them apart (find_library caches by variable name, so sharing one
# variable would link one target against the wrong .so).
#   libopapi_nn.so: aclnnMatmul (GEMM)
#   libopapi.so:    the elementwise aclnn ops (attention)
#   libnnopbase.so: aclCreateTensor etc. (shared, found once)
find_library(STRATA_CANN_OPAPI_NN_LIB NAMES opapi_nn
  HINTS ${_cann_roots}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
)
find_library(STRATA_CANN_OPAPI_LIB NAMES opapi
  HINTS ${_cann_roots}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
)
find_library(STRATA_CANN_NNOPBASE_LIB NAMES nnopbase
  HINTS ${_cann_roots}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
)
if(NOT STRATA_CANN_OPAPI_NN_LIB OR NOT STRATA_CANN_OPAPI_LIB OR NOT STRATA_CANN_NNOPBASE_LIB)
  message(FATAL_ERROR
    "Strata CANN: libopapi_nn.so / libopapi.so / libnnopbase.so not found under ASCEND_HOME_PATH / /usr/local/Ascend "
    "(aclnnMatmul lives in opapi_nn, the elementwise aclnn ops in opapi, aclCreateTensor in nnopbase; none is in libascendcl).")
endif()
message(STATUS "Strata CANN: libopapi_nn found at ${STRATA_CANN_OPAPI_NN_LIB}")
message(STATUS "Strata CANN: libopapi found at ${STRATA_CANN_OPAPI_LIB}")
message(STATUS "Strata CANN: libnnopbase found at ${STRATA_CANN_NNOPBASE_LIB}")

# GEMM (src/cann_gemm.{h,cpp}, tests/test_cann_gemm.cpp)
add_library(strata_cann_gemm src/cann_gemm.cpp)
target_include_directories(strata_cann_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src PRIVATE ${STRATA_CANN_INCLUDE_DIR})
target_link_libraries(strata_cann_gemm PUBLIC ${STRATA_CANN_ASCENDCL_LIB} ${STRATA_CANN_OPAPI_NN_LIB} ${STRATA_CANN_NNOPBASE_LIB})

add_executable(test_cann_gemm tests/test_cann_gemm.cpp)
target_include_directories(test_cann_gemm PRIVATE ${STRATA_CANN_INCLUDE_DIR})
# The CPU reference is a plain triple loop; -march=native (test binary
# only, never the library) is what makes it finish in reasonable time.
target_compile_options(test_cann_gemm PRIVATE -O3 -march=native)
target_link_libraries(test_cann_gemm PRIVATE strata_cann_gemm)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(test_cann_gemm PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(test_cann_gemm PRIVATE STRATA_CANN_GEMM_TEST_OPENMP)
endif()

# GDN attention + RMSNorm / SiLU / RoPE
# (src/cann_attention.{h,cpp}, src/cann_elementwise.{h,cpp}, tests/test_cann_attention.cpp)
add_library(strata_cann_attention src/cann_attention.cpp src/cann_elementwise.cpp)
target_include_directories(strata_cann_attention PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
                                                         ${STRATA_CANN_INCLUDE_DIR})
target_link_libraries(strata_cann_attention PUBLIC ${STRATA_CANN_ASCENDCL_LIB} ${STRATA_CANN_OPAPI_LIB} ${STRATA_CANN_NNOPBASE_LIB})

add_executable(test_cann_attention tests/test_cann_attention.cpp)
target_link_libraries(test_cann_attention PRIVATE strata_cann_attention)
# --- end Phase 1 ---
