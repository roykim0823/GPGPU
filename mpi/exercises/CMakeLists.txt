cmake_minimum_required(VERSION 3.5)
project(mpi)

find_package(MPI REQUIRED)
IF(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
ENDIF(MPI_FOUND)

#set(CMAKE_C_FLAGS "-g -Wall -Werror -pthread")
find_library(M_LIB m)   # math.h

# Exercise 1
add_executable(mpi_hello   "src/mpi_hello.c")
target_link_libraries(mpi_hello ${MPI_LIBRARIES})

# Exercise 2
add_executable(mpi_helloBsend   "src/mpi_helloBsend.c")
target_link_libraries(mpi_helloBsend ${MPI_LIBRARIES})

add_executable(mpi_helloNBsend   "src/mpi_helloNBsend.c")
target_link_libraries(mpi_helloNBsend ${MPI_LIBRARIES})

# Exercise 3
add_executable(ser_array   "src/ser_array.c")
add_executable(mpi_array   "src/mpi_array.c")
target_link_libraries(mpi_array ${MPI_LIBRARIES})

add_executable(ser_mm   "src/ser_mm.c")
add_executable(mpi_mm   "src/mpi_mm.c")
target_link_libraries(mpi_mm ${MPI_LIBRARIES})

add_executable(ser_prime   "src/ser_prime.c")
target_link_libraries(ser_prime ${M_LIB})
add_executable(mpi_prime   "src/mpi_prime.c")
target_link_libraries(mpi_prime ${M_LIB} ${MPI_LIBRARIES})

find_package(X11 REQUIRED)
IF(X11_FOUND)
include_directories(${X11_INCLUDE_DIR})
ENDIF(X11_FOUND)

add_executable(ser_heat2D   "src/ser_heat2D.c"  "src/draw_heat.c")     # Xwindows
target_link_libraries(ser_heat2D ${X11_LIBRARIES})
add_executable(mpi_heat2D   "src/mpi_heat2D.c"  "src/draw_heat.c")
target_link_libraries(mpi_heat2D ${X11_LIBRARIES} ${MPI_LIBRARIES})

add_executable(ser_wave   "src/ser_wave.c" "src/draw_wave.c")         # Xwindows
target_link_libraries(ser_wave ${M_LIB} ${X11_LIBRARIES})
add_executable(mpi_wave   "src/mpi_wave.c" "src/draw_wave.c")
target_link_libraries(mpi_wave ${M_LIB} ${X11_LIBRARIES} ${MPI_LIBRARIES})

add_executable(mpi_latency   "src/mpi_latency.c")
target_link_libraries(mpi_latency ${MPI_LIBRARIES})

add_executable(mpi_bandwidth   "src/mpi_bandwidth.c")
target_link_libraries(mpi_bandwidth ${MPI_LIBRARIES})
add_executable(mpi_bandwidth_nonblock   "src/mpi_bandwidth_nonblock.c")
target_link_libraries(mpi_bandwidth_nonblock ${MPI_LIBRARIES})
