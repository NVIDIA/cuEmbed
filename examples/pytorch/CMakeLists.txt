find_package(Python REQUIRED COMPONENTS Development)

set(TORCH_CUDA_ARCH_LIST "8.0;9.0;10.0" CACHE STRING "List of target GPU architectures")

message(STATUS "add `python -c 'import torch;print(torch.utils.cmake_prefix_path)'` to CMAKE_PREFIX_PATH")
find_package(Torch REQUIRED)

add_library(cuembed_pyt SHARED cuembed_embedding.cu)
target_link_libraries(cuembed_pyt PRIVATE ${TORCH_LIBRARIES} Python::Python cuembed_hdrs)
