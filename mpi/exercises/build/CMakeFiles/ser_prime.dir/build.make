# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hyunjin/codes_examples/mpi/exercises

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hyunjin/codes_examples/mpi/exercises/build

# Include any dependencies generated for this target.
include CMakeFiles/ser_prime.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ser_prime.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ser_prime.dir/flags.make

CMakeFiles/ser_prime.dir/src/ser_prime.c.o: CMakeFiles/ser_prime.dir/flags.make
CMakeFiles/ser_prime.dir/src/ser_prime.c.o: ../src/ser_prime.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/ser_prime.dir/src/ser_prime.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/ser_prime.dir/src/ser_prime.c.o   -c /home/hyunjin/codes_examples/mpi/exercises/src/ser_prime.c

CMakeFiles/ser_prime.dir/src/ser_prime.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ser_prime.dir/src/ser_prime.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/mpi/exercises/src/ser_prime.c > CMakeFiles/ser_prime.dir/src/ser_prime.c.i

CMakeFiles/ser_prime.dir/src/ser_prime.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ser_prime.dir/src/ser_prime.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/mpi/exercises/src/ser_prime.c -o CMakeFiles/ser_prime.dir/src/ser_prime.c.s

CMakeFiles/ser_prime.dir/src/ser_prime.c.o.requires:

.PHONY : CMakeFiles/ser_prime.dir/src/ser_prime.c.o.requires

CMakeFiles/ser_prime.dir/src/ser_prime.c.o.provides: CMakeFiles/ser_prime.dir/src/ser_prime.c.o.requires
	$(MAKE) -f CMakeFiles/ser_prime.dir/build.make CMakeFiles/ser_prime.dir/src/ser_prime.c.o.provides.build
.PHONY : CMakeFiles/ser_prime.dir/src/ser_prime.c.o.provides

CMakeFiles/ser_prime.dir/src/ser_prime.c.o.provides.build: CMakeFiles/ser_prime.dir/src/ser_prime.c.o


# Object files for target ser_prime
ser_prime_OBJECTS = \
"CMakeFiles/ser_prime.dir/src/ser_prime.c.o"

# External object files for target ser_prime
ser_prime_EXTERNAL_OBJECTS =

ser_prime: CMakeFiles/ser_prime.dir/src/ser_prime.c.o
ser_prime: CMakeFiles/ser_prime.dir/build.make
ser_prime: /usr/lib/x86_64-linux-gnu/libm.so
ser_prime: CMakeFiles/ser_prime.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ser_prime"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ser_prime.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ser_prime.dir/build: ser_prime

.PHONY : CMakeFiles/ser_prime.dir/build

CMakeFiles/ser_prime.dir/requires: CMakeFiles/ser_prime.dir/src/ser_prime.c.o.requires

.PHONY : CMakeFiles/ser_prime.dir/requires

CMakeFiles/ser_prime.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ser_prime.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ser_prime.dir/clean

CMakeFiles/ser_prime.dir/depend:
	cd /home/hyunjin/codes_examples/mpi/exercises/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/mpi/exercises /home/hyunjin/codes_examples/mpi/exercises /home/hyunjin/codes_examples/mpi/exercises/build /home/hyunjin/codes_examples/mpi/exercises/build /home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles/ser_prime.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ser_prime.dir/depend

