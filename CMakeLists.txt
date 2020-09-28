cmake_minimum_required(VERSION 3.9)
project(hpomdp)

# ----------- AIToolbox related --------------
##############################
## Compiler/Linker Settings ##
##############################

# Set default cmake build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are:
          Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(
#    -Wall
#    -Wextra
)

##############################
##     Project Settings     ##
##############################

# For additional Find library scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

##############################
##       Dependencies       ##
##############################

set(BOOST_VERSION_REQUIRED 1.54)
set(EIGEN_VERSION_REQUIRED 3.2.92)

find_package(Boost ${BOOST_VERSION_REQUIRED} REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(Eigen3 ${EIGEN_VERSION_REQUIRED} REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(LpSolve REQUIRED)
include_directories(SYSTEM ${LPSOLVE_INCLUDE_DIR})
#---------------------------------------------

##############################
##      Project Start       ##
##############################

#OpenCV dependencies
set(opencv_LOCAL_INCLUDE_DIRS "path-to-your-opencv-installation/include")
set(opencv_LOCAL_LIBS_DIR "path-to-your-opencv-installation/lib")
set(opencv_LOCAL_LIBS 
  -lopencv_core 
  -lopencv_dnn 
  -lopencv_imgproc 
  -lopencv_highgui 
  -lopencv_video 
  -lopencv_videoio 
  -lopencv_imgcodecs 
  -lopencv_features2d 
  -lopencv_flann
  -lopencv_xfeatures2d 
)

#Header files directory
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${opencv_LOCAL_INCLUDE_DIRS}
  path-to-your-json-installation/json/single_include
  path-to-your-aitoolbox-installation/AI-Toolbox/include
  path-to-your-sttree-installation/st_tree/include
)

link_directories(
  ${opencv_LOCAL_LIBS_DIR}
  path-to-your-aitoolbox-installation/AI-Toolbox/build
)

#List of source files
set(SOURCE
  ${CMAKE_SOURCE_DIR}/src/TreeHandle.cpp
  ${CMAKE_SOURCE_DIR}/src/Neighborhood.cpp
  ${CMAKE_SOURCE_DIR}/src/EnvGen.cpp
  ${CMAKE_SOURCE_DIR}/src/HPomdp.cpp
)

#Directory where executables are to be found
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#Executable for the demo program
add_executable(demo ${CMAKE_SOURCE_DIR}/src/demo.cpp ${SOURCE})
target_link_libraries(demo AIToolboxMDP AIToolboxPOMDP ${LPSOLVE_LIBRARIES} ${opencv_LOCAL_LIBS})

