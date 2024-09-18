add_executable(08_Reductions
	../../shared/include/utility.h
	src/main.cu
	src/private.h
	src/broadcast_value_cross_a_warp.cu
)

target_include_directories(08_Reductions PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})