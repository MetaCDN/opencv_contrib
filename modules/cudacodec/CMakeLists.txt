if(IOS OR APPLE OR WINRT OR (NOT HAVE_CUDA AND NOT BUILD_CUDA_STUBS))
  ocv_module_disable(cudacodec)
endif()

set(the_description "CUDA-accelerated Video Encoding/Decoding")

if(WIN32)
  ocv_warnings_disable(CMAKE_CXX_FLAGS /wd4127 /wd4324 /wd4512)
else()
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wundef -Wshadow -Wsign-compare -Wenum-compare)
endif()

ocv_add_module(cudacodec opencv_core opencv_videoio OPTIONAL opencv_cudev WRAP python)

ocv_module_include_directories()
ocv_glob_module_sources()

set(extra_libs "")

<<<<<<< HEAD
if(HAVE_NVCUVID)
  list(APPEND extra_libs ${CUDA_CUDA_LIBRARY} ${CUDA_nvcuvid_LIBRARY})
endif()

if(HAVE_NVCUVENC)
  if(WIN32)
    list(APPEND extra_libs ${CUDA_nvcuvenc_LIBRARY})
=======
if(WITH_NVCUVID AND NOT HAVE_NVCUVID)
  message(WARNING "cudacodec::VideoReader requires Nvidia Video Codec SDK. Please resolve dependency or disable WITH_NVCUVID=OFF")
endif()

if(WITH_NVCUVENC AND NOT HAVE_NVCUVENC)
  message(WARNING "cudacodec::VideoWriter requires Nvidia Video Codec SDK. Please resolve dependency or disable WITH_NVCUVENC=OFF")
endif()

if(HAVE_NVCUVID OR HAVE_NVCUVENC)
  if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
      list(APPEND extra_libs CUDA::cuda_driver)
  else()
    list(APPEND extra_libs ${CUDA_CUDA_LIBRARY})
  endif()
  if(HAVE_NVCUVID)
    list(APPEND extra_libs ${CUDA_nvcuvid_LIBRARY})
    if(ENABLE_CUDA_FIRST_CLASS_LANGUAGE)
      list(APPEND extra_libs CUDA::nppicc${CUDA_LIB_EXT})
    endif()
  endif()
  if(HAVE_NVCUVENC)
      if(WIN32)
        list(APPEND extra_libs ${CUDA_nvencodeapi_LIBRARY})
      else()
        list(APPEND extra_libs ${CUDA_nvidia-encode_LIBRARY})
      endif()
>>>>>>> 80f1ca2442982ed518076cd88cf08c71155b30f6
  endif()
endif()

ocv_create_module(${extra_libs})

ocv_add_accuracy_tests()
ocv_add_perf_tests()
