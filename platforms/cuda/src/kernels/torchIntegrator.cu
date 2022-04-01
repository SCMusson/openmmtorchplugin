extern "C" __global__
void setInputs(const real* __restrict__ input, real4* __restrict__ posq, int* __restrict__ atomIndex, int numAtoms){
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom +=blockDim.x*gridDim.x){
	//int index = atomIndex[atom];
	//real4 pos = posq[atom];
	int index = atomIndex[atom];
	//int index = atom;
	posq[atom].x = input[3*index];
	posq[atom].y = input[3*index+1];
	posq[atom].z = input[3*index+2];
    }
}


extern "C" __global__
void getForces(real* __restrict__ output, long long* __restrict__ forceBuffers, int* __restrict__ atomIndex, int numAtoms, int paddedNumAtoms, float scale){
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x){
	int index = atomIndex[atom];
	output[3*index] = forceBuffers[atom]*scale;
	output[3*index+1] = forceBuffers[atom+1*paddedNumAtoms]*scale;//paddedNumAtoms];
	output[3*index+2] = forceBuffers[atom+2*paddedNumAtoms]*scale;
    }
}


