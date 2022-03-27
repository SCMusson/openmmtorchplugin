OpenMM TorchIntegrator Plugin
=====================

This project is built from the OpenMM Example plugin and designed for [OpenMM](https://simtk.org/home/openmm)
It enables you to set the positions of simulation from a torch tensor and retrieve the forces of
those positions of a simulation into a torch tensor.
The primary benefit of this integrator is that it allows the copying of torch::Tensor (device =
cuda) directly into the cudaarrays of Openmm thus avoiding GPU -> CPU -> GPU operations. While
speed tests have yet to be carried out (and the plugin itself has not been fully tested yet) it
is hoped that this will enable Openmm to be used as a loss function for a neural network
predicting protein structure.

Building The Plugin
===================

This project uses [CMake](http://www.cmake.org) for its build system.  To build it, follow these
steps:

1. Create a directory in which to build the plugin.

2. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".

4. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.

5. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually,
this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.

6. PYTORCH_DIR to the directory where Libtorch is installed (Libtorch can be obtained from the [PyTorch website](https://pytorch.org))

7. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly
and that TORCHINTEGRATOR_BUILD_CUDA_LIB is selected.

8. Press "Configure" again if necessary, then press "Generate".

9. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install`.


Test Cases
==========

To run all the test cases build the "test" target, for example by typing `make test`.

This project contains several different directories for test cases: one for each platform, and
another for serialization related code.  Each of these directories contains a CMakeLists.txt file
that automatically creates a test from every file whose name starts with "Test" and ends with
".cpp".  To create new tests, just add a new file to any of these directories.  The file should
contain a `main()` function that executes any tests in the file and returns 0 if all tests were
successful or 1 if any of them failed.

Usually plugins are loaded dynamically at runtime, but that doesn't work well for test cases:
you want to be able to run the tests before the plugin has yet been installed into the plugins
directory.  Instead, the test cases directly link against the relevant plugin libraries.  But
that creates another problem: when a plugin is dynamically loaded at runtime, its platforms and
kernels are registered automatically, but that doesn't happen for code that statically links
against it.  Therefore, the very first line of each `main()` function typically invokes a method
to do the registration that _would_ have been done if the plugin were loaded automatically:

    registerExampleOpenCLKernelFactories();

The OpenCL and CUDA test directories create three tests from each source file: the program is
invoked three times while passing the strings "single", "mixed", and "double" as a command line
argument.  The `main()` function should take this value and set it as the default precision for
the platform:

    if (argc > 1)
        Platform::getPlatformByName("OpenCL").setPropertyDefaultValue("OpenCLPrecision", string(argv[1]));

This causes the plugin to be tested in all three of the supported precision modes every time you
run the test suite.


OpenCL and CUDA Kernels
=======================

The OpenCL and CUDA platforms compile all of their kernels from source at runtime.  This
requires you to store all your kernel source in a way that makes it accessible at runtime.  That
turns out to be harder than you might think: simply storing source files on disk is brittle,
since it requires some way of locating the files, and ordinary library files cannot contain
arbitrary data along with the compiled code.  Another option is to store the kernel source as
strings in the code, but that is very inconvenient to edit and maintain, especially since C++
doesn't have a clean syntax for multi-line strings.

This project (like OpenMM itself) uses a hybrid mechanism that provides the best of both
approaches.  The source code for the OpenCL and CUDA implementations each include a "kernels"
directory.  At build time, a CMake script loads every .cl (for OpenCL) or .cu (for CUDA) file
contained in the directory and generates a class with all the file contents as strings.  For
example, the OpenCL kernels directory contains a single file called exampleForce.cl.  You can
put anything you want into this file, and then C++ code can access the content of that file
as `OpenCLExampleKernelSources::exampleForce`.  If you add more .cl files to this directory,
correspondingly named variables will automatically be added to `OpenCLExampleKernelSources`.


Python API
==========

OpenMM uses [SWIG](http://www.swig.org) to generate its Python API.  SWIG takes an "interface
file", which is essentially a C++ header file with some extra annotations added, as its input.
It then generates a Python extension module exposing the C++ API in Python.

When building OpenMM's Python API, the interface file is generated automatically from the C++
API.  That guarantees the C++ and Python APIs are always synchronized with each other and avoids
the potential bugs that would come from have duplicate definitions.  It takes a lot of complex
processing to do that, though, and for a single plugin it's far simpler to just write the
interface file by hand.  You will find it in the "python" directory.

To build and install the Python API, build the "PythonInstall" target, for example by typing
"make PythonInstall".  (If you are installing into the system Python, you may need to use sudo.)
This runs SWIG to generate the C++ and Python files for the extension module
(ExamplePluginWrapper.cpp and exampleplugin.py), then runs a setup.py script to build and
install the module.  Once you do that, you can use the plugin from your Python scripts:

    from simtk.openmm import System
    from torchintegratorplugin import MyIntegrator

    integrator = MyIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)


