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
include exercise/CMakeFiles/bug5.dir/depend.make

# Include the progress variables for this target.
include exercise/CMakeFiles/bug5.dir/progress.make

# Include the compile flags for this target's objects.
include exercise/CMakeFiles/bug5.dir/flags.make

exercise/CMakeFiles/bug5.dir/bug5.c.o: exercise/CMakeFiles/bug5.dir/flags.make
exercise/CMakeFiles/bug5.dir/bug5.c.o: ../exercise/bug5.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/pthread/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object exercise/CMakeFiles/bug5.dir/bug5.c.o"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/bug5.dir/bug5.c.o   -c /home/hyunjin/codes_examples/pthread/exercise/bug5.c

exercise/CMakeFiles/bug5.dir/bug5.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/bug5.dir/bug5.c.i"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/pthread/exercise/bug5.c > CMakeFiles/bug5.dir/bug5.c.i

exercise/CMakeFiles/bug5.dir/bug5.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/bug5.dir/bug5.c.s"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/pthread/exercise/bug5.c -o CMakeFiles/bug5.dir/bug5.c.s

exercise/CMakeFiles/bug5.dir/bug5.c.o.requires:

.PHONY : exercise/CMakeFiles/bug5.dir/bug5.c.o.requires

exercise/CMakeFiles/bug5.dir/bug5.c.o.provides: exercise/CMakeFiles/bug5.dir/bug5.c.o.requires
	$(MAKE) -f exercise/CMakeFiles/bug5.dir/build.make exercise/CMakeFiles/bug5.dir/bug5.c.o.provides.build
.PHONY : exercise/CMakeFiles/bug5.dir/bug5.c.o.provides

exercise/CMakeFiles/bug5.dir/bug5.c.o.provides.build: exercise/CMakeFiles/bug5.dir/bug5.c.o


# Object files for target bug5
bug5_OBJECTS = \
"CMakeFiles/bug5.dir/bug5.c.o"

# External object files for target bug5
bug5_EXTERNAL_OBJECTS =

exercise/bug5: exercise/CMakeFiles/bug5.dir/bug5.c.o
exercise/bug5: exercise/CMakeFiles/bug5.dir/build.make
exercise/bug5: /usr/lib/x86_64-linux-gnu/libm.so
exercise/bug5: exercise/CMakeFiles/bug5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/pthread/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable bug5"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bug5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
exercise/CMakeFiles/bug5.dir/build: exercise/bug5

.PHONY : exercise/CMakeFiles/bug5.dir/build

exercise/CMakeFiles/bug5.dir/requires: exercise/CMakeFiles/bug5.dir/bug5.c.o.requires

.PHONY : exercise/CMakeFiles/bug5.dir/requires

exercise/CMakeFiles/bug5.dir/clean:
	cd /home/hyunjin/codes_examples/pthread/build/exercise && $(CMAKE_COMMAND) -P CMakeFiles/bug5.dir/cmake_clean.cmake
.PHONY : exercise/CMakeFiles/bug5.dir/clean

exercise/CMakeFiles/bug5.dir/depend:
	cd /home/hyunjin/codes_examples/pthread/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/pthread /home/hyunjin/codes_examples/pthread/exercise /home/hyunjin/codes_examples/pthread/build /home/hyunjin/codes_examples/pthread/build/exercise /home/hyunjin/codes_examples/pthread/build/exercise/CMakeFiles/bug5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : exercise/CMakeFiles/bug5.dir/depend

