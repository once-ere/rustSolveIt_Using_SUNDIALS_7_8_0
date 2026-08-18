# FindPosim.cmake — locate a built posim binary and the workspace holding it.
#
# posim is built by cargo, not by CMake, so this module looks for the
# artefact rather than trying to compile anything.
#
# Input (any one is enough):
#   POSIM_WORKSPACE   cargo workspace root to look under
#   POSIM_EXECUTABLE  the binary itself, if it is somewhere unusual
#
# Result variables:
#   Posim_FOUND
#   Posim_EXECUTABLE  full path to the binary
#   Posim_WORKSPACE   the workspace root it was found under
#   Posim_PROFILE     "release" or "debug"
#
# A checkout may hold more than one posim workspace — a port beside the
# upstream it came from — and picking the wrong one is silent, because
# the two agree bit for bit on everything the older grammar can express.
# So candidate roots are taken in a stated order and never by globbing
# siblings.

set(_posim_roots "")
if(POSIM_WORKSPACE)
  list(APPEND _posim_roots "${POSIM_WORKSPACE}")
endif()
if(DEFINED ENV{POSIM_WORKSPACE})
  list(APPEND _posim_roots "$ENV{POSIM_WORKSPACE}")
endif()
# The manifest names the workspace these recordings belong to; it is the
# authority, so it is consulted before any convention.
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../recordings.json")
  file(READ "${CMAKE_CURRENT_LIST_DIR}/../recordings.json" _posim_manifest)
  string(JSON _posim_ws ERROR_VARIABLE _posim_err GET "${_posim_manifest}" workspace)
  if(NOT _posim_err)
    get_filename_component(_posim_ws
      "${CMAKE_CURRENT_LIST_DIR}/../${_posim_ws}" ABSOLUTE)
    list(APPEND _posim_roots "${_posim_ws}")
  endif()
endif()

foreach(_root IN LISTS _posim_roots)
  foreach(_profile release debug)
    if(NOT Posim_EXECUTABLE AND EXISTS "${_root}/target/${_profile}/posim")
      set(Posim_EXECUTABLE "${_root}/target/${_profile}/posim")
      set(Posim_WORKSPACE "${_root}")
      set(Posim_PROFILE "${_profile}")
    endif()
  endforeach()
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Posim
  REQUIRED_VARS Posim_EXECUTABLE Posim_WORKSPACE
  FAIL_MESSAGE "no built posim found. Build it with 'cargo build --release -p posim', or configure with -DPOSIM_WORKSPACE=/path/to/workspace"
)

mark_as_advanced(Posim_EXECUTABLE Posim_WORKSPACE Posim_PROFILE)
