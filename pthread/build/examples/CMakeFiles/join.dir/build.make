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
CMAKE_SOURCE_DIR = /home/hyunjin/codes_examples/pthread

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hyunjin/codes_examples/pthread/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/join.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/join.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/join.dir/flags.make

examples/CMakeFiles/join.dir/join.c.o: examples/CMakeFiles/join.dir/flags.make
examples/CMakeFiles/join.dir/join.c.o: ../examples/join.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/pthread/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object examples/CMakeFiles/join.dir/join.c.o"
	cd /home/hyunjin/codes_examples/pthread/build/examples && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/join.dir/join.c.o   -c /home/hyunjin/codes_examples/pthread/examples/join.c

examples/CMakeFiles/join.dir/join.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/join.dir/join.c.i"
	cd /home/hyunjin/codes_examples/pthread/build/examples && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/pthread/examples/join.c > CMakeFiles/join.dir/join.c.i

examples/CMakeFiles/join.dir/join.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/join.dir/join.c.s"
	cd /home/hyunjin/codes_examples/pthread/build/examples && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/pthread/examples/join.c -o CMakeFiles/join.dir/join.c.s

examples/CMakeFiles/join.dir/join.c.o.requires:

.PHONY : examples/CMakeFiles/join.dir/join.c.o.requires

examples/CMakeFiles/join.dir/join.c.o.provides: examples/CMakeFiles/join.dir/join.c.o.requires
	$(MAKE) -f examples/CMakeFiles/join.dir/build.make examples/CMakeFiles/join.dir/join.c.o.provides.build
.PHONY : examples/CMakeFiles/join.dir/join.c.o.provides

examples/CMakeFiles/join.dir/join.c.o.provides.build: examples/CMakeFiles/join.dir/join.c.o


# Object files for target join
join_OBJECTS = \
"CMakeFiles/join.dir/join.c.o"

# External object files for target join
join_EXTERNAL_OBJECTS =

examples/join: examples/CMakeFiles/join.dir/join.c.o
examples/join: examples/CMakeFiles/join.dir/build.make
examples/join: /usr/lib/x86_64-linux-gnu/libm.so
examples/join: examples/CMakeFiles/join.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/pthread/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable join"
	cd /home/hyunjin/codes_examples/pthread/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/join.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/join.dir/build: examples/join

.PHONY : examples/CMakeFiles/join.dir/build

examples/CMakeFiles/join.dir/requires: examples/CMakeFiles/join.dir/join.c.o.requires

.PHONY : examples/CMakeFiles/join.dir/requires

examples/CMakeFiles/join.dir/clean:
	cd /home/hyunjin/codes_examples/pthread/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/join.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/join.dir/clean

examples/CMakeFiles/join.dir/depend:
	cd /home/hyunjin/codes_examples/pthread/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/pthread /home/hyunjin/codes_examples/pthread/examples /home/hyunjin/codes_examples/pthread/build /home/hyunjin/codes_examples/pthread/build/examples /home/hyunjin/codes_examples/pthread/build/examples/CMakeFiles/join.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/join.dir/depend

