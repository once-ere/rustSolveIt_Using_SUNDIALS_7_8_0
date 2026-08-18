# The posim video recorder.
#
# There is no compilation here: the recorder is Python, and the thing it
# drives is a cargo binary. CMake is used for what it is good at when a
# project has no compile step — locating the dependency, installing the
# entry points, and giving the checks a single command to run.
#
#   cmake -S recorder -B build/recorder
#   cmake --build build/recorder --target record-all   # re-record
#   ctest --test-dir build/recorder --output-on-failure
#   cmake --install build/recorder --prefix ~/.local
#
# LANGUAGES NONE: asking CMake for a C compiler it will never use only
# gives people without one a confusing failure.
cmake_minimum_required(VERSION 3.20)

project(posim_recorder
  VERSION 1.1.0
  DESCRIPTION "Record a posim run into a self-contained browser video"
  HOMEPAGE_URL "https://github.com/once-ere/rustSolveIt_Using_SUNDIALS_7_8_0"
  LANGUAGES NONE
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# GNUInstallDirs derives an architecture-specific library directory, and
# with LANGUAGES NONE there is no compiler to derive it from, so it warns.
# Nothing here installs a library — only scripts, data and docs — so the
# value is pinned rather than guessed, and the warning has nothing to say.
set(CMAKE_INSTALL_LIBDIR "lib" CACHE PATH "unused: this project installs no libraries")
include(GNUInstallDirs)

find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)

# Not REQUIRED: the package must still configure, install and run its
# offline checks on a machine with no Rust toolchain. Only the tests
# that actually drive posim are skipped, and they say so.
find_package(Posim)

set(RECORDER_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/record_video.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/record_all.py"
)
set(RECORDER_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/recordings.json")

if(Posim_FOUND)
  message(STATUS "posim: ${Posim_EXECUTABLE} (${Posim_PROFILE})")
  message(STATUS "posim workspace: ${Posim_WORKSPACE}")
else()
  message(STATUS "posim: not found — recording targets and the "
                 "end-to-end tests will be skipped")
endif()

# --- targets ----------------------------------------------------------

if(Posim_FOUND)
  # Re-record every entry in the manifest, in place.
  add_custom_target(record-all
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/record_all.py"
            --manifest "${RECORDER_MANIFEST}"
            --workspace "${Posim_WORKSPACE}"
    COMMENT "Recording every video in the manifest"
    VERBATIM
  )

  # Record to a temporary directory and compare with what is committed.
  # This is the check that matters: a recording is a pure function of
  # its scene, its parameters and the posim binary, so a difference
  # means one of those three moved.
  add_custom_target(check-recordings
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/record_all.py"
            --manifest "${RECORDER_MANIFEST}"
            --workspace "${Posim_WORKSPACE}"
            --check
    COMMENT "Checking every committed recording reproduces byte for byte"
    VERBATIM
  )
endif()

# --- install ----------------------------------------------------------

install(PROGRAMS ${RECORDER_SOURCES}
        DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES ${RECORDER_MANIFEST}
        DESTINATION "${CMAKE_INSTALL_DATADIR}/posim-recorder")
install(FILES
          "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
          "${CMAKE_CURRENT_SOURCE_DIR}/docs/FRAME_FORMAT.md"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}")

# --- tests ------------------------------------------------------------

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
