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
include CMakeFiles/pth_tokenize.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pth_tokenize.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pth_tokenize.dir/flags.make

CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o: CMakeFiles/pth_tokenize.dir/flags.make
CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o: ../src/pth_tokenize.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o   -c /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/src/pth_tokenize.c

CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/src/pth_tokenize.c > CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.i

CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/src/pth_tokenize.c -o CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.s

CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o.requires:

.PHONY : CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o.requires

CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o.provides: CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o.requires
	$(MAKE) -f CMakeFiles/pth_tokenize.dir/build.make CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o.provides.build
.PHONY : CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o.provides

CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o.provides.build: CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o


# Object files for target pth_tokenize
pth_tokenize_OBJECTS = \
"CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o"

# External object files for target pth_tokenize
pth_tokenize_EXTERNAL_OBJECTS =

pth_tokenize: CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o
pth_tokenize: CMakeFiles/pth_tokenize.dir/build.make
pth_tokenize: CMakeFiles/pth_tokenize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable pth_tokenize"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pth_tokenize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pth_tokenize.dir/build: pth_tokenize

.PHONY : CMakeFiles/pth_tokenize.dir/build

CMakeFiles/pth_tokenize.dir/requires: CMakeFiles/pth_tokenize.dir/src/pth_tokenize.c.o.requires

.PHONY : CMakeFiles/pth_tokenize.dir/requires

CMakeFiles/pth_tokenize.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pth_tokenize.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pth_tokenize.dir/clean

CMakeFiles/pth_tokenize.dir/depend:
	cd /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build /home/hyunjin/codes_examples/ipp-source-use/ch4_Pthreads/build/CMakeFiles/pth_tokenize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pth_tokenize.dir/depend

