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
CMAKE_SOURCE_DIR = /home/hyunjin/codes_examples/mpi/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hyunjin/codes_examples/mpi/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/environment.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/environment.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/environment.dir/flags.make

CMakeFiles/environment.dir/src/environment.c.o: CMakeFiles/environment.dir/flags.make
CMakeFiles/environment.dir/src/environment.c.o: ../src/environment.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/mpi/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/environment.dir/src/environment.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/environment.dir/src/environment.c.o   -c /home/hyunjin/codes_examples/mpi/examples/src/environment.c

CMakeFiles/environment.dir/src/environment.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/environment.dir/src/environment.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/mpi/examples/src/environment.c > CMakeFiles/environment.dir/src/environment.c.i

CMakeFiles/environment.dir/src/environment.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/environment.dir/src/environment.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/mpi/examples/src/environment.c -o CMakeFiles/environment.dir/src/environment.c.s

CMakeFiles/environment.dir/src/environment.c.o.requires:

.PHONY : CMakeFiles/environment.dir/src/environment.c.o.requires

CMakeFiles/environment.dir/src/environment.c.o.provides: CMakeFiles/environment.dir/src/environment.c.o.requires
	$(MAKE) -f CMakeFiles/environment.dir/build.make CMakeFiles/environment.dir/src/environment.c.o.provides.build
.PHONY : CMakeFiles/environment.dir/src/environment.c.o.provides

CMakeFiles/environment.dir/src/environment.c.o.provides.build: CMakeFiles/environment.dir/src/environment.c.o


# Object files for target environment
environment_OBJECTS = \
"CMakeFiles/environment.dir/src/environment.c.o"

# External object files for target environment
environment_EXTERNAL_OBJECTS =

environment: CMakeFiles/environment.dir/src/environment.c.o
environment: CMakeFiles/environment.dir/build.make
environment: /usr/local/lib/libmpi.so
environment: CMakeFiles/environment.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/mpi/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable environment"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/environment.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/environment.dir/build: environment

.PHONY : CMakeFiles/environment.dir/build

CMakeFiles/environment.dir/requires: CMakeFiles/environment.dir/src/environment.c.o.requires

.PHONY : CMakeFiles/environment.dir/requires

CMakeFiles/environment.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/environment.dir/cmake_clean.cmake
.PHONY : CMakeFiles/environment.dir/clean

CMakeFiles/environment.dir/depend:
	cd /home/hyunjin/codes_examples/mpi/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/mpi/examples /home/hyunjin/codes_examples/mpi/examples /home/hyunjin/codes_examples/mpi/examples/build /home/hyunjin/codes_examples/mpi/examples/build /home/hyunjin/codes_examples/mpi/examples/build/CMakeFiles/environment.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/environment.dir/depend

