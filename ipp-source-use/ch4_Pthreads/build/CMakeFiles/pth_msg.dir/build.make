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
CMAKE_SOURCE_DIR = /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build

# Include any dependencies generated for this target.
include CMakeFiles/pth_msg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pth_msg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pth_msg.dir/flags.make

CMakeFiles/pth_msg.dir/src/pth_msg.c.o: CMakeFiles/pth_msg.dir/flags.make
CMakeFiles/pth_msg.dir/src/pth_msg.c.o: ../src/pth_msg.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/pth_msg.dir/src/pth_msg.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/pth_msg.dir/src/pth_msg.c.o   -c /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/src/pth_msg.c

CMakeFiles/pth_msg.dir/src/pth_msg.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pth_msg.dir/src/pth_msg.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/src/pth_msg.c > CMakeFiles/pth_msg.dir/src/pth_msg.c.i

CMakeFiles/pth_msg.dir/src/pth_msg.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pth_msg.dir/src/pth_msg.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/src/pth_msg.c -o CMakeFiles/pth_msg.dir/src/pth_msg.c.s

CMakeFiles/pth_msg.dir/src/pth_msg.c.o.requires:

.PHONY : CMakeFiles/pth_msg.dir/src/pth_msg.c.o.requires

CMakeFiles/pth_msg.dir/src/pth_msg.c.o.provides: CMakeFiles/pth_msg.dir/src/pth_msg.c.o.requires
	$(MAKE) -f CMakeFiles/pth_msg.dir/build.make CMakeFiles/pth_msg.dir/src/pth_msg.c.o.provides.build
.PHONY : CMakeFiles/pth_msg.dir/src/pth_msg.c.o.provides

CMakeFiles/pth_msg.dir/src/pth_msg.c.o.provides.build: CMakeFiles/pth_msg.dir/src/pth_msg.c.o


# Object files for target pth_msg
pth_msg_OBJECTS = \
"CMakeFiles/pth_msg.dir/src/pth_msg.c.o"

# External object files for target pth_msg
pth_msg_EXTERNAL_OBJECTS =

pth_msg: CMakeFiles/pth_msg.dir/src/pth_msg.c.o
pth_msg: CMakeFiles/pth_msg.dir/build.make
pth_msg: CMakeFiles/pth_msg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable pth_msg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pth_msg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pth_msg.dir/build: pth_msg

.PHONY : CMakeFiles/pth_msg.dir/build

CMakeFiles/pth_msg.dir/requires: CMakeFiles/pth_msg.dir/src/pth_msg.c.o.requires

.PHONY : CMakeFiles/pth_msg.dir/requires

CMakeFiles/pth_msg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pth_msg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pth_msg.dir/clean

CMakeFiles/pth_msg.dir/depend:
	cd /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build/CMakeFiles/pth_msg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pth_msg.dir/depend

