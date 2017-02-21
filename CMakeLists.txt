cmake_minimum_required (VERSION 2.8.0)
project (Arcturus)

set (Arcturus_MAJOR_VERSION 1)
set (Arcturus_MINOR_VERSION 0)

add_subdirectory (lib/Cortex)
include_directories (lib/Cortex)

add_executable (Arcturus src/main.cpp)

target_link_libraries (Arcturus Cortex)

install (TARGETS Arcturus DESTINATION bin)
