cmake_minimum_required(VERSION 3.5)
project(pthread)

set(CMAKE_C_FLAGS "-g -Wall -Werror -pthread")

find_library(M_LIB m)   # math.h

add_executable(hello "hello.c")
add_executable(hello_ex1 "hello_ex1.c")
add_executable(hello_ex2 "hello_ex2.c")
add_executable(hello_ex3 "hello_ex3.c")

add_executable(join "join.c")
target_link_libraries(join ${M_LIB})

add_executable(stack_manage "stack_manage.c")

add_executable(dotprod_serial "dotprod_serial.c")
add_executable(dotprod_mutex "dotprod_mutex.c")

add_executable(condvar "condvar.c")
