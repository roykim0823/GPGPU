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
CMAKE_SOURCE_DIR = /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI/build

# Include any dependencies generated for this target.
include CMakeFiles/mpi_odd_even.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mpi_odd_even.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpi_odd_even.dir/flags.make

CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o: CMakeFiles/mpi_odd_even.dir/flags.make
CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o: ../src/mpi_odd_even.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/ipp-source-use/ch3_MPI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o   -c /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI/src/mpi_odd_even.c

CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI/src/mpi_odd_even.c > CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.i

CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI/src/mpi_odd_even.c -o CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.s

CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o.requires:

.PHONY : CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o.requires

CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o.provides: CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o.requires
	$(MAKE) -f CMakeFiles/mpi_odd_even.dir/build.make CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o.provides.build
.PHONY : CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o.provides

CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o.provides.build: CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o


# Object files for target mpi_odd_even
mpi_odd_even_OBJECTS = \
"CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o"

# External object files for target mpi_odd_even
mpi_odd_even_EXTERNAL_OBJECTS =

mpi_odd_even: CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o
mpi_odd_even: CMakeFiles/mpi_odd_even.dir/build.make
mpi_odd_even: /usr/lib/openmpi/lib/libmpi_cxx.so
mpi_odd_even: /usr/lib/openmpi/lib/libmpi.so
mpi_odd_even: CMakeFiles/mpi_odd_even.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/ipp-source-use/ch3_MPI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable mpi_odd_even"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpi_odd_even.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpi_odd_even.dir/build: mpi_odd_even

.PHONY : CMakeFiles/mpi_odd_even.dir/build

CMakeFiles/mpi_odd_even.dir/requires: CMakeFiles/mpi_odd_even.dir/src/mpi_odd_even.c.o.requires

.PHONY : CMakeFiles/mpi_odd_even.dir/requires

CMakeFiles/mpi_odd_even.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpi_odd_even.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpi_odd_even.dir/clean

CMakeFiles/mpi_odd_even.dir/depend:
	cd /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI/build /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI/build /home/hyunjin/codes_examples/ipp-source-use/ch3_MPI/build/CMakeFiles/mpi_odd_even.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpi_odd_even.dir/depend

