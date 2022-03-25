set(WRAP_FILE TorchIntegratorPluginWrapper.cpp)
set(MODULE_NAME torchintegratorplugin)

# Execute SWIG to generate source code for the Python module.

add_custom_command(
    OUTPUT "${WRAP_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
	#   "-I${TORCH_INCLUDE_DIRS}"
        "${CMAKE_CURRENT_SOURCE_DIR}/torchintegratorplugin.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/torchintegratorplugin.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# Compile the Python module.

add_custom_target(PythonInstall DEPENDS "${WRAP_FILE}")
set(TORCHINTEGRATORPLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(TORCHINTEGRATORPLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
#message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
#message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
#message(STATUS "TORCH_INCLUDE_DIR: ${TORCH_INCLUDE_DIRS}")
add_custom_command(TARGET PythonInstall
	#COMMAND "${PYTHON_EXECUTABLE}" -m pip install .
	COMMAND "${PYTHON_EXECUTABLE}" setup.py build
        COMMAND "${PYTHON_EXECUTABLE}" setup.py install
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
