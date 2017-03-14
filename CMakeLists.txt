cmake_minimum_required(VERSION 2.8)
project (addon)

# Essential node addon include files
include_directories(${CMAKE_JS_INC})

# Include Cortex Library
add_subdirectory (lib/Cortex)
include_directories (lib/Cortex)

# Location of addon.cpp
file(GLOB SOURCE_FILES "src/addon.cpp" "src/addon.h")

# This line will tell CMake that we're building a shared library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

# This line will give our library file a .node extension without any "lib" prefix
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

target_link_libraries (${PROJECT_NAME} Cortex)
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})
