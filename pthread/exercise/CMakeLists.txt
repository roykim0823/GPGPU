cmake_minimum_required(VERSION 3.5)
project(pthread)

set(CMAKE_C_FLAGS "-g -Wall -Werror -pthread -lm")

find_library(M_LIB m)   # math.h

add_executable(hello32 "hello32.c")
target_link_libraries(hello32 ${M_LIB})

add_executable(hello_arg1 "hello_arg1.c")
add_executable(hello_arg2 "hello_arg2.c")

add_executable(bug3 "bug3.c")
add_executable(bug3fix "bug3fix.c")

add_executable(bug5 "bug5.c")
target_link_libraries(bug5 ${M_LIB})
add_executable(bug5fix "bug5fix.c")
target_link_libraries(bug5fix ${M_LIB})

add_executable(join_ex "join.c")
target_link_libraries(join_ex ${M_LIB})
add_executable(detached "detached.c")
target_link_libraries(detached ${M_LIB})

add_executable(bug2 "bug2.c")
add_executable(bug2fix "bug2fix.c")

add_executable(dotprod_serial_ex "dotprod_serial.c")
add_executable(dotprod_mutex_ex "dotprod_mutex.c")

add_executable(bug6 "bug6.c")
add_executable(bug6fix "bug6fix.c")
add_executable(arrayloops "arrayloops.c")

add_executable(condvar_ex "condvar.c")
add_executable(bug1 "bug1.c")
add_executable(bug1fix "bug1fix.c")

add_executable(bug4 "bug4.c")
target_link_libraries(bug4 ${M_LIB})
add_executable(bug4fix "bug4fix.c")
target_link_libraries(bug4fix ${M_LIB})
