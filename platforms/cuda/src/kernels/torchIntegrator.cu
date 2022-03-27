extern "C" __global__
void setInputs(const real* __restrict__ input, real4* __restrict__ posq, int numAtoms){
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom +=blockDim.x*gridDim.x){
	//int index = atomIndex[atom];
	//real4 pos = posq[atom];
	posq[atom].x = input[3*atom];
	posq[atom].y = input[3*atom+1];
	posq[atom].z = input[3*atom+2];
    }
}


extern "C" __global__
void getForces(real* __restrict__ output, long long* __restrict__ forceBuffers, int numAtoms, int paddedNumAtoms, float scale){
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x){
	output[3*atom] = forceBuffers[atom]*scale;
	output[3*atom+1] = forceBuffers[atom+1*paddedNumAtoms]*scale;//paddedNumAtoms];
	output[3*atom+2] = forceBuffers[atom+2*paddedNumAtoms]*scale;
    }
}


