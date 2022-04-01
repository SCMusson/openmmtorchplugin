from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
torchintegratorplugin_header_dir = '@TORCHINTEGRATORPLUGIN_HEADER_DIR@'
torchintegratorplugin_library_dir = '@TORCHINTEGRATORPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_torchintegratorplugin',
                      sources=['TorchIntegratorPluginWrapper.cpp'],
                      libraries=['OpenMM', 'TorchIntegratorPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), torchintegratorplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), torchintegratorplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='torchintegratorplugin',
      version='1.0',
      py_modules=['torchintegratorplugin'],
      ext_modules=[extension],
     )
