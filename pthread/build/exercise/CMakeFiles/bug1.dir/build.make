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
include exercise/CMakeFiles/bug1.dir/depend.make

# Include the progress variables for this target.
include exercise/CMakeFiles/bug1.dir/progress.make

# Include the compile flags for this target's objects.
include exercise/CMakeFiles/bug1.dir/flags.make

exercise/CMakeFiles/bug1.dir/bug1.c.o: exercise/CMakeFiles/bug1.dir/flags.make
exercise/CMakeFiles/bug1.dir/bug1.c.o: ../exercise/bug1.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/pthread/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object exercise/CMakeFiles/bug1.dir/bug1.c.o"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/bug1.dir/bug1.c.o   -c /home/hyunjin/codes_examples/pthread/exercise/bug1.c

exercise/CMakeFiles/bug1.dir/bug1.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/bug1.dir/bug1.c.i"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/pthread/exercise/bug1.c > CMakeFiles/bug1.dir/bug1.c.i

exercise/CMakeFiles/bug1.dir/bug1.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/bug1.dir/bug1.c.s"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/pthread/exercise/bug1.c -o CMakeFiles/bug1.dir/bug1.c.s

exercise/CMakeFiles/bug1.dir/bug1.c.o.requires:

.PHONY : exercise/CMakeFiles/bug1.dir/bug1.c.o.requires

exercise/CMakeFiles/bug1.dir/bug1.c.o.provides: exercise/CMakeFiles/bug1.dir/bug1.c.o.requires
	$(MAKE) -f exercise/CMakeFiles/bug1.dir/build.make exercise/CMakeFiles/bug1.dir/bug1.c.o.provides.build
.PHONY : exercise/CMakeFiles/bug1.dir/bug1.c.o.provides

exercise/CMakeFiles/bug1.dir/bug1.c.o.provides.build: exercise/CMakeFiles/bug1.dir/bug1.c.o


# Object files for target bug1
bug1_OBJECTS = \
"CMakeFiles/bug1.dir/bug1.c.o"

# External object files for target bug1
bug1_EXTERNAL_OBJECTS =

exercise/bug1: exercise/CMakeFiles/bug1.dir/bug1.c.o
exercise/bug1: exercise/CMakeFiles/bug1.dir/build.make
exercise/bug1: exercise/CMakeFiles/bug1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/pthread/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable bug1"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bug1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
exercise/CMakeFiles/bug1.dir/build: exercise/bug1

.PHONY : exercise/CMakeFiles/bug1.dir/build

exercise/CMakeFiles/bug1.dir/requires: exercise/CMakeFiles/bug1.dir/bug1.c.o.requires

.PHONY : exercise/CMakeFiles/bug1.dir/requires

exercise/CMakeFiles/bug1.dir/clean:
	cd /home/hyunjin/codes_examples/pthread/build/exercise && $(CMAKE_COMMAND) -P CMakeFiles/bug1.dir/cmake_clean.cmake
.PHONY : exercise/CMakeFiles/bug1.dir/clean

exercise/CMakeFiles/bug1.dir/depend:
	cd /home/hyunjin/codes_examples/pthread/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/pthread /home/hyunjin/codes_examples/pthread/exercise /home/hyunjin/codes_examples/pthread/build /home/hyunjin/codes_examples/pthread/build/exercise /home/hyunjin/codes_examples/pthread/build/exercise/CMakeFiles/bug1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : exercise/CMakeFiles/bug1.dir/depend

