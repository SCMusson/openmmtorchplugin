from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
torchexposedintegratorplugin_header_dir = '@TORCHEXPOSEDINTEGRATORPLUGIN_HEADER_DIR@'
torchexposedintegratorplugin_library_dir = '@TORCHEXPOSEDINTEGRATORPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_torchexposedintegratorplugin',
                      sources=['TorchExposedIntegratorPluginWrapper.cpp'],
                      libraries=['OpenMM', 'TorchExposedIntegratorPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), torchexposedintegratorplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), torchexposedintegratorplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='torchexposedintegratorplugin',
      version='1.1.3',
      py_modules=['torchexposedintegratorplugin'],
      ext_modules=[extension],
     )
