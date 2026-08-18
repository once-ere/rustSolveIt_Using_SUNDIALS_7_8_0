# Two suites, registered separately so a failure names itself.
#
# recorder-units runs anywhere — no posim, no network. recorder-e2e
# drives a real posim; without one it reports SKIPPED rather than
# passing vacuously, which is the difference between "checked" and
# "did not check".

add_test(NAME recorder-units
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_units.py"
)

add_test(NAME recorder-e2e
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_end_to_end.py"
)

if(Posim_FOUND)
  set_tests_properties(recorder-e2e PROPERTIES
    ENVIRONMENT "POSIM_WORKSPACE=${Posim_WORKSPACE}"
    # Recording five videos is real integration work, not a unit test.
    TIMEOUT 900
  )
else()
  set_tests_properties(recorder-e2e PROPERTIES
    SKIP_REGULAR_EXPRESSION "no built posim"
  )
endif()

set_tests_properties(recorder-units PROPERTIES TIMEOUT 120)
