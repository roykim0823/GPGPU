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
include exercise/CMakeFiles/bug6fix.dir/depend.make

# Include the progress variables for this target.
include exercise/CMakeFiles/bug6fix.dir/progress.make

# Include the compile flags for this target's objects.
include exercise/CMakeFiles/bug6fix.dir/flags.make

exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o: exercise/CMakeFiles/bug6fix.dir/flags.make
exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o: ../exercise/bug6fix.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/pthread/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/bug6fix.dir/bug6fix.c.o   -c /home/hyunjin/codes_examples/pthread/exercise/bug6fix.c

exercise/CMakeFiles/bug6fix.dir/bug6fix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/bug6fix.dir/bug6fix.c.i"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/pthread/exercise/bug6fix.c > CMakeFiles/bug6fix.dir/bug6fix.c.i

exercise/CMakeFiles/bug6fix.dir/bug6fix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/bug6fix.dir/bug6fix.c.s"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && /usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/pthread/exercise/bug6fix.c -o CMakeFiles/bug6fix.dir/bug6fix.c.s

exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o.requires:

.PHONY : exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o.requires

exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o.provides: exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o.requires
	$(MAKE) -f exercise/CMakeFiles/bug6fix.dir/build.make exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o.provides.build
.PHONY : exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o.provides

exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o.provides.build: exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o


# Object files for target bug6fix
bug6fix_OBJECTS = \
"CMakeFiles/bug6fix.dir/bug6fix.c.o"

# External object files for target bug6fix
bug6fix_EXTERNAL_OBJECTS =

exercise/bug6fix: exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o
exercise/bug6fix: exercise/CMakeFiles/bug6fix.dir/build.make
exercise/bug6fix: exercise/CMakeFiles/bug6fix.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/pthread/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable bug6fix"
	cd /home/hyunjin/codes_examples/pthread/build/exercise && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bug6fix.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
exercise/CMakeFiles/bug6fix.dir/build: exercise/bug6fix

.PHONY : exercise/CMakeFiles/bug6fix.dir/build

exercise/CMakeFiles/bug6fix.dir/requires: exercise/CMakeFiles/bug6fix.dir/bug6fix.c.o.requires

.PHONY : exercise/CMakeFiles/bug6fix.dir/requires

exercise/CMakeFiles/bug6fix.dir/clean:
	cd /home/hyunjin/codes_examples/pthread/build/exercise && $(CMAKE_COMMAND) -P CMakeFiles/bug6fix.dir/cmake_clean.cmake
.PHONY : exercise/CMakeFiles/bug6fix.dir/clean

exercise/CMakeFiles/bug6fix.dir/depend:
	cd /home/hyunjin/codes_examples/pthread/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/pthread /home/hyunjin/codes_examples/pthread/exercise /home/hyunjin/codes_examples/pthread/build /home/hyunjin/codes_examples/pthread/build/exercise /home/hyunjin/codes_examples/pthread/build/exercise/CMakeFiles/bug6fix.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : exercise/CMakeFiles/bug6fix.dir/depend

