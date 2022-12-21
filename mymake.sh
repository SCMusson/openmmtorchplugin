make && ^
sudo make install && ^
make PythonInstall && ^
cp libTorchExposedIntegratorPlugin.so ~/miniconda3/envs/torch_openmm/lib && ^
cp libTorchExposedIntegratorPluginReference.so ~/miniconda3/envs/torch_openmm/lib && ^
python ../test.py
