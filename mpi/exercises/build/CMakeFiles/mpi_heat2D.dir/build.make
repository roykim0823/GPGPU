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
include CMakeFiles/mpi_heat2D.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mpi_heat2D.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpi_heat2D.dir/flags.make

CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o: CMakeFiles/mpi_heat2D.dir/flags.make
CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o: ../src/mpi_heat2D.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o   -c /home/hyunjin/codes_examples/mpi/exercises/src/mpi_heat2D.c

CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/mpi/exercises/src/mpi_heat2D.c > CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.i

CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/mpi/exercises/src/mpi_heat2D.c -o CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.s

CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o.requires:

.PHONY : CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o.requires

CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o.provides: CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o.requires
	$(MAKE) -f CMakeFiles/mpi_heat2D.dir/build.make CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o.provides.build
.PHONY : CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o.provides

CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o.provides.build: CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o


CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o: CMakeFiles/mpi_heat2D.dir/flags.make
CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o: ../src/draw_heat.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o   -c /home/hyunjin/codes_examples/mpi/exercises/src/draw_heat.c

CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/mpi/exercises/src/draw_heat.c > CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.i

CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/mpi/exercises/src/draw_heat.c -o CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.s

CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o.requires:

.PHONY : CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o.requires

CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o.provides: CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o.requires
	$(MAKE) -f CMakeFiles/mpi_heat2D.dir/build.make CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o.provides.build
.PHONY : CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o.provides

CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o.provides.build: CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o


# Object files for target mpi_heat2D
mpi_heat2D_OBJECTS = \
"CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o" \
"CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o"

# External object files for target mpi_heat2D
mpi_heat2D_EXTERNAL_OBJECTS =

mpi_heat2D: CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o
mpi_heat2D: CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o
mpi_heat2D: CMakeFiles/mpi_heat2D.dir/build.make
mpi_heat2D: /usr/lib/x86_64-linux-gnu/libSM.so
mpi_heat2D: /usr/lib/x86_64-linux-gnu/libICE.so
mpi_heat2D: /usr/lib/x86_64-linux-gnu/libX11.so
mpi_heat2D: /usr/lib/x86_64-linux-gnu/libXext.so
mpi_heat2D: /usr/lib/openmpi/lib/libmpi_cxx.so
mpi_heat2D: /usr/lib/openmpi/lib/libmpi.so
mpi_heat2D: CMakeFiles/mpi_heat2D.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable mpi_heat2D"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpi_heat2D.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpi_heat2D.dir/build: mpi_heat2D

.PHONY : CMakeFiles/mpi_heat2D.dir/build

CMakeFiles/mpi_heat2D.dir/requires: CMakeFiles/mpi_heat2D.dir/src/mpi_heat2D.c.o.requires
CMakeFiles/mpi_heat2D.dir/requires: CMakeFiles/mpi_heat2D.dir/src/draw_heat.c.o.requires

.PHONY : CMakeFiles/mpi_heat2D.dir/requires

CMakeFiles/mpi_heat2D.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpi_heat2D.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpi_heat2D.dir/clean

CMakeFiles/mpi_heat2D.dir/depend:
	cd /home/hyunjin/codes_examples/mpi/exercises/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/mpi/exercises /home/hyunjin/codes_examples/mpi/exercises /home/hyunjin/codes_examples/mpi/exercises/build /home/hyunjin/codes_examples/mpi/exercises/build /home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles/mpi_heat2D.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpi_heat2D.dir/depend

