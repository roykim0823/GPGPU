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
include CMakeFiles/mpi_array.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mpi_array.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpi_array.dir/flags.make

CMakeFiles/mpi_array.dir/src/mpi_array.c.o: CMakeFiles/mpi_array.dir/flags.make
CMakeFiles/mpi_array.dir/src/mpi_array.c.o: ../src/mpi_array.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/mpi_array.dir/src/mpi_array.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mpi_array.dir/src/mpi_array.c.o   -c /home/hyunjin/codes_examples/mpi/exercises/src/mpi_array.c

CMakeFiles/mpi_array.dir/src/mpi_array.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpi_array.dir/src/mpi_array.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/mpi/exercises/src/mpi_array.c > CMakeFiles/mpi_array.dir/src/mpi_array.c.i

CMakeFiles/mpi_array.dir/src/mpi_array.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpi_array.dir/src/mpi_array.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/mpi/exercises/src/mpi_array.c -o CMakeFiles/mpi_array.dir/src/mpi_array.c.s

CMakeFiles/mpi_array.dir/src/mpi_array.c.o.requires:

.PHONY : CMakeFiles/mpi_array.dir/src/mpi_array.c.o.requires

CMakeFiles/mpi_array.dir/src/mpi_array.c.o.provides: CMakeFiles/mpi_array.dir/src/mpi_array.c.o.requires
	$(MAKE) -f CMakeFiles/mpi_array.dir/build.make CMakeFiles/mpi_array.dir/src/mpi_array.c.o.provides.build
.PHONY : CMakeFiles/mpi_array.dir/src/mpi_array.c.o.provides

CMakeFiles/mpi_array.dir/src/mpi_array.c.o.provides.build: CMakeFiles/mpi_array.dir/src/mpi_array.c.o


# Object files for target mpi_array
mpi_array_OBJECTS = \
"CMakeFiles/mpi_array.dir/src/mpi_array.c.o"

# External object files for target mpi_array
mpi_array_EXTERNAL_OBJECTS =

mpi_array: CMakeFiles/mpi_array.dir/src/mpi_array.c.o
mpi_array: CMakeFiles/mpi_array.dir/build.make
mpi_array: /usr/lib/openmpi/lib/libmpi_cxx.so
mpi_array: /usr/lib/openmpi/lib/libmpi.so
mpi_array: CMakeFiles/mpi_array.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable mpi_array"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpi_array.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpi_array.dir/build: mpi_array

.PHONY : CMakeFiles/mpi_array.dir/build

CMakeFiles/mpi_array.dir/requires: CMakeFiles/mpi_array.dir/src/mpi_array.c.o.requires

.PHONY : CMakeFiles/mpi_array.dir/requires

CMakeFiles/mpi_array.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpi_array.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpi_array.dir/clean

CMakeFiles/mpi_array.dir/depend:
	cd /home/hyunjin/codes_examples/mpi/exercises/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/mpi/exercises /home/hyunjin/codes_examples/mpi/exercises /home/hyunjin/codes_examples/mpi/exercises/build /home/hyunjin/codes_examples/mpi/exercises/build /home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles/mpi_array.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpi_array.dir/depend
