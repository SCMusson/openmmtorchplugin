#include <torch/extension.h>
//#include <torch/torch.h>
#include <iostream>
//%#include <openmm>
//#include <OpenMM.h>
//using namespace OpenMM;
//#include <openmm/reference/ReferencePlatform.h>
#include <stdint.h>
using namespace at;
void mygetptr(unsigned long int in){
    //std::cout << i << std::endl;
    
    double * ptr = reinterpret_cast<double*>(in);
    //Tensor f = CPU(kFloat).tensorFromBlob(ptr, {5,});
    Tensor f = from_blob(ptr, {5,}, TensorOptions().dtype(kFloat));
    TensorAccessor<float, 1> f_a = f.accessor<float, 1>();
    for (int i = 0; i <5; i++){
        std::cout<< f_a[i] <<std::endl;
    }
    f_a[1]+=3.0;
    
}
/*
#Tensor& z){
    double * ptr =  z.data_ptr<double>();
    //return reinterpret_cast<int>(ptr);
    return reinterpret_cast<uintptr_t>(ptr);
}
*/


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mygetptr", &mygetptr, "my get pointer");
}
