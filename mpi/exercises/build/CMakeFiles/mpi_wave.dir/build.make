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
include CMakeFiles/mpi_wave.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mpi_wave.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpi_wave.dir/flags.make

CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o: CMakeFiles/mpi_wave.dir/flags.make
CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o: ../src/mpi_wave.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o   -c /home/hyunjin/codes_examples/mpi/exercises/src/mpi_wave.c

CMakeFiles/mpi_wave.dir/src/mpi_wave.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpi_wave.dir/src/mpi_wave.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/mpi/exercises/src/mpi_wave.c > CMakeFiles/mpi_wave.dir/src/mpi_wave.c.i

CMakeFiles/mpi_wave.dir/src/mpi_wave.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpi_wave.dir/src/mpi_wave.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/mpi/exercises/src/mpi_wave.c -o CMakeFiles/mpi_wave.dir/src/mpi_wave.c.s

CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o.requires:

.PHONY : CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o.requires

CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o.provides: CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o.requires
	$(MAKE) -f CMakeFiles/mpi_wave.dir/build.make CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o.provides.build
.PHONY : CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o.provides

CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o.provides.build: CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o


CMakeFiles/mpi_wave.dir/src/draw_wave.c.o: CMakeFiles/mpi_wave.dir/flags.make
CMakeFiles/mpi_wave.dir/src/draw_wave.c.o: ../src/draw_wave.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/mpi_wave.dir/src/draw_wave.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mpi_wave.dir/src/draw_wave.c.o   -c /home/hyunjin/codes_examples/mpi/exercises/src/draw_wave.c

CMakeFiles/mpi_wave.dir/src/draw_wave.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mpi_wave.dir/src/draw_wave.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/mpi/exercises/src/draw_wave.c > CMakeFiles/mpi_wave.dir/src/draw_wave.c.i

CMakeFiles/mpi_wave.dir/src/draw_wave.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mpi_wave.dir/src/draw_wave.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/mpi/exercises/src/draw_wave.c -o CMakeFiles/mpi_wave.dir/src/draw_wave.c.s

CMakeFiles/mpi_wave.dir/src/draw_wave.c.o.requires:

.PHONY : CMakeFiles/mpi_wave.dir/src/draw_wave.c.o.requires

CMakeFiles/mpi_wave.dir/src/draw_wave.c.o.provides: CMakeFiles/mpi_wave.dir/src/draw_wave.c.o.requires
	$(MAKE) -f CMakeFiles/mpi_wave.dir/build.make CMakeFiles/mpi_wave.dir/src/draw_wave.c.o.provides.build
.PHONY : CMakeFiles/mpi_wave.dir/src/draw_wave.c.o.provides

CMakeFiles/mpi_wave.dir/src/draw_wave.c.o.provides.build: CMakeFiles/mpi_wave.dir/src/draw_wave.c.o


# Object files for target mpi_wave
mpi_wave_OBJECTS = \
"CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o" \
"CMakeFiles/mpi_wave.dir/src/draw_wave.c.o"

# External object files for target mpi_wave
mpi_wave_EXTERNAL_OBJECTS =

mpi_wave: CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o
mpi_wave: CMakeFiles/mpi_wave.dir/src/draw_wave.c.o
mpi_wave: CMakeFiles/mpi_wave.dir/build.make
mpi_wave: /usr/lib/x86_64-linux-gnu/libm.so
mpi_wave: /usr/lib/x86_64-linux-gnu/libSM.so
mpi_wave: /usr/lib/x86_64-linux-gnu/libICE.so
mpi_wave: /usr/lib/x86_64-linux-gnu/libX11.so
mpi_wave: /usr/lib/x86_64-linux-gnu/libXext.so
mpi_wave: /usr/lib/openmpi/lib/libmpi_cxx.so
mpi_wave: /usr/lib/openmpi/lib/libmpi.so
mpi_wave: CMakeFiles/mpi_wave.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable mpi_wave"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpi_wave.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpi_wave.dir/build: mpi_wave

.PHONY : CMakeFiles/mpi_wave.dir/build

CMakeFiles/mpi_wave.dir/requires: CMakeFiles/mpi_wave.dir/src/mpi_wave.c.o.requires
CMakeFiles/mpi_wave.dir/requires: CMakeFiles/mpi_wave.dir/src/draw_wave.c.o.requires

.PHONY : CMakeFiles/mpi_wave.dir/requires

CMakeFiles/mpi_wave.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpi_wave.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpi_wave.dir/clean

CMakeFiles/mpi_wave.dir/depend:
	cd /home/hyunjin/codes_examples/mpi/exercises/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/mpi/exercises /home/hyunjin/codes_examples/mpi/exercises /home/hyunjin/codes_examples/mpi/exercises/build /home/hyunjin/codes_examples/mpi/exercises/build /home/hyunjin/codes_examples/mpi/exercises/build/CMakeFiles/mpi_wave.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpi_wave.dir/depend

