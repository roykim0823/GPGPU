cmake_minimum_required(VERSION 3.5)
project(mpi)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

#set(CMAKE_C_FLAGS "-g -Wall -Werror -pthread")
find_library(M_LIB m)   # math.h

add_executable(support   "src/support.c")
target_link_libraries(support ${MPI_LIBRARIES})

add_executable(environment   "src/environment.c")
target_link_libraries(environment ${MPI_LIBRARIES})

# Calculation Pi using Monte Carlo
add_executable(ser_pi_calc  "src/ser_pi_calc.c")        # Serial version
add_executable(mpi_pi_send   "src/mpi_pi_send.c")       # MPI by p2p comu
target_link_libraries(mpi_pi_send ${MPI_LIBRARIES})
add_executable(mpi_pi_reduce   "src/mpi_pi_reduce.c")   # MPI by collective comu
target_link_libraries(mpi_pi_reduce ${MPI_LIBRARIES})

add_executable(mpi_ping   "src/mpi_ping.c")             # Blocking msg passing
target_link_libraries(mpi_ping ${MPI_LIBRARIES})

add_executable(mpi_ringtopo   "src/mpi_ringtopo.c")     # Non-blocking msg passing
target_link_libraries(mpi_ringtopo ${MPI_LIBRARIES})

add_executable(mpi_scatter   "src/mpi_scatter.c")       # Collective Comm
target_link_libraries(mpi_scatter ${MPI_LIBRARIES})

# Derived Data Type
add_executable(mpi_contig   "src/mpi_contig.c")       
add_executable(mpi_vector   "src/mpi_vector.c")       
add_executable(mpi_indexed   "src/mpi_indexed.c")       
add_executable(mpi_struct   "src/mpi_struct.c")       
target_link_libraries(mpi_contig ${MPI_LIBRARIES})
target_link_libraries(mpi_vector ${MPI_LIBRARIES})
target_link_libraries(mpi_indexed ${MPI_LIBRARIES})
target_link_libraries(mpi_struct ${MPI_LIBRARIES})

# Group and Comm Management
add_executable(mpi_group   "src/mpi_group.c")       
target_link_libraries(mpi_group ${MPI_LIBRARIES})

# Virtual Topology
add_executable(mpi_cartesian   "src/mpi_cartesian.c")       
target_link_libraries(mpi_cartesian ${MPI_LIBRARIES})
