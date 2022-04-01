make && ^
sudo make install && ^
make PythonInstall && ^
cp libTorchIntegratorPlugin.so ~/miniconda3/envs/torch_openmm/lib && ^
cp libTorchIntegratorPluginReference.so ~/miniconda3/envs/torch_openmm/lib && ^
python ../test.py
