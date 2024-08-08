cmake_minimum_required(VERSION 3.10)
project(TensorRTCalibrator)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find TensorRT
find_package(TensorRT REQUIRED)
message(STATUS "TensorRT Version: ${TensorRT_VERSION_STRING}")
message(STATUS "Find TensorRT include at ${TensorRT_INCLUDE_DIRS}")
message(STATUS "Find TensorRT libraries: ${TensorRT_LIBRARIES}")

# Find Threads
find_package(Threads)

# Find OpenCV
find_package(OpenCV REQUIRED)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include 
    ${OpenCV_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS} 
    ${TensorRT_INCLUDE_DIRS} 
    ${TensorRT_SAMPLE_DIR}/common
)

link_directories(
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES} 
    ${TensorRT_LIBRARIES}
)

# Define source files
file(GLOB My_Source_Files ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB My_Include_Files ${PROJECT_SOURCE_DIR}/include/*.h)

# Add executable
add_executable(TensorRTCalibrator ${My_Source_Files} ${My_Include_Files})

# Include directories
target_include_directories(TensorRTCalibrator PUBLIC 
    ${PROJECT_SOURCE_DIR}/include 
    ${OpenCV_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS} 
    ${TensorRT_SAMPLE_DIR}/common
)

# Link libraries
target_link_libraries(TensorRTCalibrator PUBLIC 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT}
    ${TensorRT_LIBRARIES}
)

# Set C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors")