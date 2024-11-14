cmake_minimum_required(VERSION 3.10)
project(onnx_squeezenet_inference)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

# Path to ONNX Runtime includes and libraries
include_directories(/usr/local/onnxruntime/include)
link_directories(/usr/local/onnxruntime/lib)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Add your source files
add_executable(onnx_squeezenet_inference main.cpp src/helpers.cpp)

# Link the OpenCV and NXX Runtime libraries
target_link_libraries(onnx_squeezenet_inference
                      onnxruntime                     
                      ${OpenCV_LIBS})