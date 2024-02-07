find_package( CUDA REQUIRED )

set_source_files_properties(operations.cu PROPERTIES LANGUAGE CUDA)

enable_language(CUDA)


add_library(Kernels SHARED operations.cu)
set_target_properties(Kernels PROPERTIES CUDA_ARCHITECTURES "89")

