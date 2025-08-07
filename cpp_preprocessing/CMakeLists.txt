cmake_minimum_required(VERSION 3.12)
project(TrackNetV3_Preprocessing)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(preprocess preprocess.cpp)

# Link libraries
target_link_libraries(preprocess ${OpenCV_LIBS})

# Enable optimizations for release build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(preprocess PRIVATE -O3 -march=native)
endif()

# Enable threading
find_package(Threads REQUIRED)
target_link_libraries(preprocess Threads::Threads)

# Platform-specific settings
if(APPLE)
    # macOS specific settings
    target_compile_definitions(preprocess PRIVATE -D_DARWIN_C_SOURCE)
elseif(UNIX)
    # Linux specific settings
    target_compile_definitions(preprocess PRIVATE -D_GNU_SOURCE)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(preprocess PRIVATE /W4)
else()
    target_compile_options(preprocess PRIVATE -Wall -Wextra -Wpedantic)
endif()
