cmake_minimum_required(VERSION 3.5)
project(Pthreads)

set(CMAKE_C_FLAGS "-Wall -Werror -pthread -g -O2")

include_directories(include)

#add_executable(pth_do_nothing   src/pth_do_nothing.c    )   # 4.2
#add_executable(pth_hello        src/pth_hello.c         )   

#add_executable(pth_mat_vect     src/pth_mat_vect.c      )   # 4.3

#add_executable(pth_pi           src/pth_pi.c            )   # 4.4

add_executable(pth_pi_busy1     src/pth_pi_busy1.c      )   # 4.5
#add_executable(pth_pi_busy2     src/pth_pi_busy2.c      )

#add_executable(pth_pi_mutex     src/pth_pi_mutex.c      )   # 4.6

#add_executable(pth_msg          src/pth_msg.c           )   # 4.7
#add_executable(pth_msg_sem      src/pth_msg_sem.c       )
#add_executable(pth_msg_sem_mac  src/pth_msg_sem_mac.c   )

#add_executable(pth_busy_bar     src/pth_busy_bar.c      )   # 4.8
#add_executable(pth_sem_bar      src/pth_sem_bar.c       )
#add_executable(pth_cond_bar     src/pth_cond_bar.c      )

#add_executable(pth_ll_one_mut   src/pth_ll_one_mut.c    src/my_rand.c)   # 4.9
#add_executable(pth_ll_mult_mut  src/pth_ll_mult_mut.c   src/my_rand.c)
#add_executable(pth_ll_rwl       src/pth_ll_rwl.c        src/my_rand.c)

#add_executable(pth_mat_vect_rand_split     src/pth_mat_vect_rand_split.c      )   # 4.10

#add_executable(pth_tokenize     src/pth_tokenize.c      )   # 4.11
#add_executable(pth_tokenize_r   src/pth_tokenize_r.c    )   
