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
include CMakeFiles/mpi_indexed.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mpi_indexed.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpi_indexed.dir/flags.make

CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o: CMakeFiles/mpi_indexed.dir/flags.make
CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o: ../src/mpi_indexed.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/mpi/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o   -c /home/hyunjin/codes_examples/mpi/examples/src/mpi_indexed.c

CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/mpi/examples/src/mpi_indexed.c > CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.i

CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/mpi/examples/src/mpi_indexed.c -o CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.s

CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o.requires:

.PHONY : CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o.requires

CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o.provides: CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o.requires
	$(MAKE) -f CMakeFiles/mpi_indexed.dir/build.make CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o.provides.build
.PHONY : CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o.provides

CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o.provides.build: CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o


# Object files for target mpi_indexed
mpi_indexed_OBJECTS = \
"CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o"

# External object files for target mpi_indexed
mpi_indexed_EXTERNAL_OBJECTS =

mpi_indexed: CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o
mpi_indexed: CMakeFiles/mpi_indexed.dir/build.make
mpi_indexed: /usr/local/lib/libmpi.so
mpi_indexed: CMakeFiles/mpi_indexed.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/mpi/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable mpi_indexed"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpi_indexed.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpi_indexed.dir/build: mpi_indexed

.PHONY : CMakeFiles/mpi_indexed.dir/build

CMakeFiles/mpi_indexed.dir/requires: CMakeFiles/mpi_indexed.dir/src/mpi_indexed.c.o.requires

.PHONY : CMakeFiles/mpi_indexed.dir/requires

CMakeFiles/mpi_indexed.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpi_indexed.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpi_indexed.dir/clean

CMakeFiles/mpi_indexed.dir/depend:
	cd /home/hyunjin/codes_examples/mpi/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/mpi/examples /home/hyunjin/codes_examples/mpi/examples /home/hyunjin/codes_examples/mpi/examples/build /home/hyunjin/codes_examples/mpi/examples/build /home/hyunjin/codes_examples/mpi/examples/build/CMakeFiles/mpi_indexed.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpi_indexed.dir/depend
