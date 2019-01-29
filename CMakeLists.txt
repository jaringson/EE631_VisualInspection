cmake_minimum_required(VERSION 3.4)
project(VisualInspection)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(project project.cpp)
target_link_libraries(project ${OpenCV_LIBS})

add_executable(inspect inspection.cpp)
target_link_libraries(inspect inspection.cpp)
