#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup, Extension
import os
import sys
import platform
from torch.utils import cpp_extension

openmm_dir = '@OPENMM_DIR@'
exampleplugin_header_dir = '@TORCHINTEGRATORPLUGIN_HEADER_DIR@'
exampleplugin_library_dir = '@TORCHINTEGRATORPLUGIN_LIBRARY_DIR@'
torch_dir, _  = os.path.split('@TORCH_LIBRARY@')
torch_libraries = '@TORCH_LIBRARIES@'
torch_include = '@TORCH_INCLUDE_DIRS@'
torch_include = torch_include.split(';')
print('')
print('torch_dir:', torch_dir)
print('LIB:', torch_libraries )
print('INCLUDE:', torch_include)


print('')

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++14']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

include_dirs=[os.path.join(openmm_dir, 'include'), exampleplugin_header_dir]+torch_include#+torch_include#+torch_include,#, torch_dir],
library_dirs=[os.path.join(openmm_dir, 'lib'), exampleplugin_library_dir]#, torch_dir],
print('include_dirs;: ', include_dirs)
print('library dirs;: ', library_dirs)
extension = Extension(name='_torchintegratorplugin',
                      sources=['TorchIntegratorPluginWrapper.cpp'],
                      libraries=['OpenMM', 'TorchIntegratorPlugin'],#, 'Torch'],
                      include_dirs=include_dirs,#[os.path.join(openmm_dir, 'include'), exampleplugin_header_dir],#+torch_include,#, torch_dir],
                      library_dirs=library_dirs,#[os.path.join(openmm_dir, 'lib'), exampleplugin_library_dir],#, torch_dir],
                      ##FROM TORCH OPENMM
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib'), torch_dir],
                      ###^
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

torch_extension = Extension(name = '_torchintegratorplugin',
                            sources = ['TorchIntegratorPluginWrapper.cpp'],
                      libraries=['OpenMM', 'TorchIntegratorPlugin'],
                            include_dirs = cpp_extension.include_paths()+[os.path.join(openmm_dir, 'include'), exampleplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), exampleplugin_library_dir],#, torch_dir],
                            language = 'c++',
                      extra_link_args=extra_link_args,
                      )

setup(name='torchintegratorplugin',
      version='1.0',
      py_modules=['torchintegratorplugin'],
      ext_modules=[extension],
      #ext_modules=[torch_extension]
     )
