// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////
// Implementation of the TCudaTensor class. //
/////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda/CudaTensor.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"

#include <cassert>

namespace TMVA {
namespace DNN  {


// Static members.
//____________________________________________________________________________
template<typename AFloat>
size_t                   TCudaTensor<AFloat>::fInstances        = 0;
/*template<typename AFloat>
cublasHandle_t           TCudaTensor<AFloat>::fCublasHandle     = nullptr;*/
template<typename AFloat>
cudnnHandle_t            TCudaTensor<AFloat>::fCudnnHandle      = nullptr;
template<typename AFloat>
cudnnTensorDescriptor_t  TCudaTensor<AFloat>::fTensorDescriptor = nullptr;
template<typename AFloat>
cudnnDataType_t          TCudaTensor<AFloat>::fDataType         = CUDNN_DATA_FLOAT;
/*template<typename AFloat>
AFloat                   * TCudaTensor<AFloat>::fDeviceReturn   = nullptr;*/
/*template<typename AFloat>
AFloat                   * TCudaTensor<AFloat>::fOnes           = nullptr;*/
/*template<typename AFloat>
curandState_t            * TCudaTensor<AFloat>::fCurandStates   = nullptr;*/
/*template<typename AFloat>
size_t                   TCudaTensor<AFloat>::fNCurandStates    = 0;*/
/*template<typename AFloat>
size_t                   TCudaTensor<AFloat>::fNOnes            = 0;*/

// Constructors.
//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor()
    : fShape(), fStrides(nullptr), fNDim(0), fSize(0), fElementBuffer()
{
   InitializeCuda();
}

//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(size_t size, size_t dim, const std::vector<size_t> shape)
    : fNDim(dim), fSize(size), fElementBuffer(size, 0)
{
   // Need a shape array with at least 4 entries for cuDNN tensors
   if (fNDim < 2) {
       std::puts("No matching cuDNN tensor description for given input dimension(s). "
                 "Inputs should be given as: batch size, no. channels, image dimensions.");
       exit(EXIT_FAILURE);
   }
   // fNDim contains only the spacial tensor dimensions, batchsize and #channels are
   // contained in the shape array
   size_t shape_size = fShape.size();
   // Reduce shape size afterwards for loop and direct array access
   fStrides = new size_t[shape_size--];
   for (int i = 0; i < shape_size; ++i) {
       fStrides[i] = shape[i+1];
       for (int j = 0; j < i; j++) {
          fStrides[j] *= shape[i+1];
       }
   }
   // Last stride should be one for cudnn
   fStrides[shape_size] = 1;
   
   assert(fSize == fStrides[0]*shape[0]);
   
   InitializeCuda();
}

//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(size_t size, const AFloat * host_data, size_t dim, const std::vector<size_t> shape)
    : TCudaTensor(size, dim, shape)
{
   // do I need to allocate this buffer ???? 
   // is not a mem leak
   // AFloat * buffer = new AFloat[fSize];
   // size_t index = 0;
   // for (size_t j = 0; j < fSize; ++j) {
   //       buffer[j] = static_cast<AFloat>(host_data[j]);
   //    }
   // }

   cudaMemcpy(fElementBuffer, host_data, fSize * sizeof(AFloat),
              cudaMemcpyHostToDevice);
}

//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::TCudaTensor(TCudaDeviceBuffer<AFloat> buffer, size_t dim, const std::vector<size_t> shape)
    : fNDim(dim), fElementBuffer(buffer), fShape(std::move(shape))
{   
   // Need a shape array with at least 4 entries for cuDNN tensors
   if (fNDim < 2) {
       std::puts("No matching cuDNN tensor description for given input dimension(s). "
                 "Inputs should be given as: batch size, no. channels, image dimensions.");
       exit(EXIT_FAILURE);
   }
   // fNDim contains only the spacial tensor dimensions, batchsize and #channels are
   // contained in the shape array
   size_t shape_size = fShape.size();
   // Reduce shape size afterwards for loop and direct array access
   fStrides = new size_t[shape_size--];
   for (int i = 0; i < shape_size; ++i) {
       fStrides[i] = shape[i+1];
       for (int j = 0; j < i; j++) {
          fStrides[j] *= shape[i+1];
       }
   }
   // Last stride should be one for cudnn
   fStrides[shape_size] = 1;
   
   fSize = fStrides[0]*shape[0];
   
   InitializeCuda();  
}

//____________________________________________________________________________
template <typename AFloat>
TCudaTensor<AFloat>::~TCudaTensor() {

   // Free resources of this instance
   if (fStrides) delete[] fStrides;
      
   if (--fInstances <= 0) {
      cudnnDestroyTensorDescriptor(fTensorDescriptor);
      cudnnDestroy(fCudnnHandle);
   }
}

//____________________________________________________________________________
template <typename AFloat>
inline void TCudaTensor<AFloat>::InitializeCuda()
{
   // add further initialization than done in TMatrixcPU::iNITIALIZEcUDA
   if (fInstances == 0) {
      //cublasCreate(&fCublasHandle);
      CUDNNCHECK(cudnnCreate(&fCudnnHandle));
   //     CUDACHECK(cudaMalloc(& fDeviceReturn, sizeof(AFloat)));
   //     CUDACHECK(cudaMalloc(& fCurandStates, TDevice::NThreads(*this)));
   
   
      CUDNNCHECK(cudnnCreateTensorDescriptor(&fTensorDescriptor));
   }
   // if (TDevice::NThreads(*this) > (int) fNCurandStates) {
   //     fNCurandStates = TDevice::NThreads(*this);
   //     if (fCurandStates) {
   //         cudaFree(fCurandStates);
   //     }
   //     cudaMalloc(&fCurandStates, TDevice::NThreads(*this) * sizeof(curandState_t));
   //     InitializeCurandStates();
   // }
   
   fInstances++;
      
   if      (std::is_same<AFloat, double>::value) { fDataType = CUDNN_DATA_DOUBLE; }
   else if (std::is_same<AFloat, float>::value)  { fDataType = CUDNN_DATA_FLOAT; }
   
   if (!fStrides) {
      return;
   }
   // cuDNN NdTensor format has a minsize of 3 tensor dimensions
   else if (fNDim == 2) {
      CUDNNCHECK(cudnnSetTensor4dDescriptor(fTensorDescriptor,
                                            CUDNN_TENSOR_NCHW,// Layout of the tensor in memory
                                            fDataType,
                                            (int)fShape[0],  // batch size
                                            (int)fShape[1],  // no. channels
                                            (int)fShape[2],  // image height
                                            (int)fShape[3]));// image width
   
   }
   // Evade case fNDim = 0
   else {
     CUDNNCHECK(cudnnSetTensorNdDescriptor(fTensorDescriptor,
                                            fDataType,
                                            (int)fNDim,
                                            (int *)fShape.data(),
                                            (int *)fStrides));
   }
}

//____________________________________________________________________________
template<typename AFloat>
void TCudaTensor<AFloat>::InitializeCurandStates()
{
   // dim3 blockDims = TDevice::BlockDims2D();
   // dim3 gridDims  = TDevice::GridDims2D(*this);
   // CurandInitializationKernel<<<gridDims, blockDims>>>(time(nullptr), fCurandStates);
}

#if 0
// Conversion to RTensor
//____________________________________________________________________________
template<typename AFloat>
TCudaTensor<AFloat>::operator Experimental::RTensor<AFloat>() const
{
   std::vector<size_t> shape(fNDims, fNDims + fDim)
   
   Experimental::RTensor<AFloat> hostTensor( shape)

   AFloat * buffer = new AFloat[fSize];
   cudaMemcpy(buffer, fElementBuffer, fSize * sizeof(AFloat),
              cudaMemcpyDeviceToHost);

   int index = 0;
   for (int j = 0; j < fSize; j++) {
         hostTensor.GetData()[j] = static_cast<AFloat>(buffer[j]);
      }
   }

   delete[] buffer;
   return hostTensor;
}
#endif
// Explicit Instantiations.

template class TCudaTensor<float>;
template class TCudaTensor<double>;

} // namespace DNN
} // namespace TMVA
