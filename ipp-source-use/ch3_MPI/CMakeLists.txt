cmake_minimum_required(VERSION 3.5)
project(mpi)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

#set(CMAKE_C_FLAGS "-g -Wall -Werror -pthread")
find_library(M_LIB m)   # math.h

include_directories(include)

add_executable(mpi_hello   "src/mpi_hello.c")                           # 3.1.1
target_link_libraries(mpi_hello ${MPI_LIBRARIES})

add_executable(trap             "src/trap.c")                           # 3.2.1
add_executable(mpi_trap1        "src/mpi_trap1.c")                      # 3.3.1
add_executable(mpi_trap2        "src/mpi_trap2.c")                      # 3.3.2
add_executable(mpi_trap3        "src/mpi_trap3.c")                      # 3.4.5
add_executable(mpi_trap4        "src/mpi_trap4.c")                      # 3.5
target_link_libraries(mpi_trap1 ${MPI_LIBRARIES})
target_link_libraries(mpi_trap2 ${MPI_LIBRARIES})
target_link_libraries(mpi_trap3 ${MPI_LIBRARIES})
target_link_libraries(mpi_trap4 ${MPI_LIBRARIES})

add_executable(mpi_output       "src/mpi_output.c")                     # 3.3.1
target_link_libraries(mpi_output ${MPI_LIBRARIES})

add_executable(vector_add       "src/vector_add.c")                     # 3.4.6
add_executable(mpi_vector_add   "src/mpi_vector_add.c")                 # 3.4.6-8
target_link_libraries(mpi_vector_add ${MPI_LIBRARIES})

add_executable(mat_vect_mult        "src/mat_vect_mult.c")              # 3.4.9
add_executable(mpi_mat_vect_mult    "src/mpi_mat_vect_mult.c")          # 3.4.9
target_link_libraries(mpi_mat_vect_mult ${MPI_LIBRARIES})

add_executable(mpi_many_msgs    "src/mpi_many_msgs.c")                  # 3.5
target_link_libraries(mpi_many_msgs ${MPI_LIBRARIES})

add_executable(mpi_mat_vect_time    "src/mpi_mat_vect_time.c")          # 3.6
target_link_libraries(mpi_mat_vect_time ${MPI_LIBRARIES})

add_executable(bubble           "src/bubble.c")                         # 3.7
add_executable(odd_even         "src/odd_even.c")                       # 3.7
add_executable(mpi_odd_even     "src/mpi_odd_even.c")                   # 3.7
target_link_libraries(mpi_odd_even ${MPI_LIBRARIES})
