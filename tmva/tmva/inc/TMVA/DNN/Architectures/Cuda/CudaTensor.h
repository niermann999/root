// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////
// Contains the TCudaMatrix class for the representation of matrices //
// on CUDA devices as well as the TCudaDeviceReference class which   //
// is a helper class to emulate lvalue references to floating point  //
// values on the device.                                             //
///////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CUDA_CUDATENSOR
#define TMVA_DNN_ARCHITECTURES_CUDA_CUDATENSOR

//#include "cuda.h"
#include "cudnn.h"
/*#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand_kernel.h"*/
//#include "thrust/fill.h"
//#include "thrust/device_vector.h"

#include <vector>
#include <cstring>

#include "TMatrixT.h"
#include "CudaBuffers.h"
//#include "CudaMatrix.h"
//#include "TMVA/RTensor.hxx"

#define CUDNNCHECK(ans) {cudnnError((ans), __FILE__, __LINE__); }

namespace TMVA {
namespace DNN {

/**
 * Function to handle the status output of cuDNN function calls. See also
 * CUDACHECK in CudaMatrix.h.
 */
inline void cudnnError(cudnnStatus_t status, const char *file, int line, bool abort=true);


//____________________________________________________________________________
//
// Cuda Tensor
//____________________________________________________________________________

/** TCudaTensor Class
 *
 * The TCudaTGensor class extends the TCudaMatrix class for dimensions > 2. 
 *
 */
template<typename AFloat>
class TCudaTensor
{
public:

   using Shape_t = std::vector<size_t>;

private:

   //static size_t                         fInstances;        ///< Current number of matrix instances.
   static std::vector<cudnnHandle_t>     fCudnnHandle;      ///< Holds the cuddn library context (one for every CUDA stream and device)
   //static AFloat                         * fDeviceReturn;   ///< Buffer for kernel return values.
   //static AFloat                         * fOnes;           ///< Vector used for summations of columns.
   //static size_t                         fNOnes;            ///< Current length of the one vector.
   //static curandState_t                  * fCurandStates;
   //static size_t                         fNCurandStates;
   static cudnnDataType_t                fDataType;         ///< Cudnn datatype used for the tensor
   /** For each GPU device keep the CUDA streams in which tensors are used. 
     * Instances belonging to the same stream on the same deviceshare a 
     * cudnn library handel to keep cudnn contexts seperated */
   //static std::vector<std::vector<int> > fInstances;
   static std::vector<int> fInstances;

   /** The shape (size of dimensions) needs to be ordered as no. channels,
    *  image dimensions.
    */
   Shape_t      fShape;            ///< batch size, no. of channels and sizes of subdimensions
   Shape_t      fStrides;         ///< Strides between tensor dimensions (always assume dense, non overlapping tensor)
   size_t              fNDim;             ///< Dimension of the tensor (first dimension is the batch size, second is the no. channels)
   size_t              fSize;             ///< No. of elements
   int                 fDevice;           ///< Device associated with current tensor instance
   int                 fStreamIndx;           ///< Cuda stream associated with current instance
          
   cudnnTensorDescriptor_t   fTensorDescriptor;
   TCudaDeviceBuffer<AFloat> fElementBuffer;

public:

   //static AFloat * GetOnes() {return fOnes;}

   TCudaTensor();
   TCudaTensor(std::vector<TMatrixT<Double_t> >& inputTensor,
               const std::vector<size_t> & shape,
               int deviceIndx = 0, 
               int streamIndx = 0);
   TCudaTensor(size_t size, size_t ndim,
               const std::vector<size_t> & shape,
               int deviceIndx = 0, int streamIndx = 0);
   TCudaTensor(size_t size, const AFloat * data, size_t ndim, 
               const std::vector<size_t> & shape,
               int deviceIndx = 0, int streamIndx = 0);
   TCudaTensor(TCudaDeviceBuffer<AFloat> buffer, size_t ndim, 
               const std::vector<size_t> & shape,
               int deviceIndx = 0, int streamIndx = 0);

   TCudaTensor(const TCudaTensor  &);
   TCudaTensor(      TCudaTensor &&) = default;
   TCudaTensor & operator=(const TCudaTensor  &) = default;
   TCudaTensor & operator=(      TCudaTensor &&) = default;
   ~TCudaTensor();

   /** Convert cuda matrix to Root TMatrix. Performs synchronous data transfer. */
   //operator Experimental::RTensor<AFloat>() const;

   /*inline cudaStream_t GetComputeStream() const;
   inline void         SetComputeStream(cudaStream_t stream);*/
   /** Set the return buffer on the device to the specified value. This is
    * required for example for reductions in order to initialize the
    * accumulator. */
   //inline static void ResetDeviceReturn(AFloat value = 0.0);
   /** Transfer the value in the device return buffer to the host. This
    *  tranfer is synchronous */
   //inline static AFloat GetDeviceReturn();
   /** Return device pointer to the device return buffer */
   //inline static AFloat *        GetDeviceReturnPointer() {return fDeviceReturn;}
   //inline static curandState_t * GetCurandStatesPointer() {return fCurandStates;}

   /** Blocking synchronization with the associated compute stream, if it's
    * not the default stream. */
   //inline void Synchronize(const TCudaTensor &) const;

   const Shape_t & GetShape() const {return fShape;}
   const Shape_t & GetStrides() const {return fStrides;}
   size_t GetDimAt(size_t i) const {return fShape[i];}
   size_t GetNDim() const {return fNDim;}
   size_t GetSize() const {return fSize;}
    
   const AFloat * GetDataPointer() const {return fElementBuffer;}
   AFloat       * GetDataPointer()       {return fElementBuffer;}
   const TCudaDeviceBuffer<AFloat> & GetDeviceBuffer()     const {return fElementBuffer;}
   TCudaDeviceBuffer<AFloat>       & GetDeviceBuffer()           {return fElementBuffer;}
   const cudnnHandle_t             & GetCudnnHandle()      const {return fCudnnHandle[fStreamIndx];}
   const cudnnTensorDescriptor_t   & GetTensorDescriptor() const {return fTensorDescriptor;}

   /** Access to elements of device matrices provided through TCudaDeviceReference
    *  class. Note that access is synchronous end enforces device synchronization
    *  on all streams. Only used for testing. */
   //TCudaDeviceReference<AFloat> operator()(size_t i, size_t j) const;

   void Print() const { 
      //TMatrixT<AFloat> mat(*this); 
      //mat.Print(); 
   }

   void Zero() {
      cudaMemset(GetDataPointer(), 0, sizeof(AFloat) * GetSize());
   }
   
   void SetConstVal(const AFloat constVal) {
      TCudaHostBuffer<AFloat> hostBuffer(fSize);
      hostBuffer.SetConstVal(constVal);
      fElementBuffer.CopyFrom(hostBuffer);
   }

   // for size=3 tensors used so far in DNN
   size_t GetFirstSize() const { return fShape.back(); }  // CM order
   size_t GetFirstStride() const { return fStrides.back(); } // CM order 
   size_t GetHSize() const { return fShape[0];}
   size_t GetWSize() const { return fShape[1];}

   // for backward compatibility (assume column-major order
   size_t GetNrows() const { return fStrides.back(); }
   size_t GetNcols() const { return fShape.back(); }


      // Matrix conversion for tensors of shape 2
   TCudaMatrix<AFloat> GetMatrix() const  {
     assert(GetShape().size() == 2 || (GetShape().size() == 3 && GetFirstSize() == 1));
      // t.b.d should squeeze the tensor
      return TCudaMatrix<AFloat>(fBuffer, GetHSize(), GetWSize());
   }

   // return slice of tensor
   // return slices in the first dimension (if row wise) or last dimension if colun wise
   // so single event slides
   TCudaTensor<AFloat> At(size_t i) const {
      const Shape_t & shape = GetShape();
//      Shape_t sliced_shape = (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor)
//      ? Shape_t(shape.begin() + 1, shape.end()) :
//      : Shape_t(shape.begin(), shape.end() - 1);

      Shape_t sliced_shape = Shape_t(shape.begin(), shape.end() - 1); // assume column major


//      size_t buffsize = (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor) ? fTensor.GetStrides().front()
//      : fTensor.GetStrides().back();
      size_t buffsize = 1;
      for (size_t j = 0; j < sliced_shape.size(); ++j)
         buffsize *= sliced_shape[j];

      size_t offset = i * buffsize;

      return TCudaTensor<AFloat>(fBuffer.GetSubBuffer(offset, buffsize), sliced_shape.size(), sliced_shape, GetLayout());
   }


private:

   /** Initializes all shared devices resource and makes sure that a sufficient
    *  number of curand states are allocated on the device and initialized as
    *  well as that the one-vector for the summation over columns has the right
    *  size. */
   void InitializeCuda();
   void InitializeCurandStates();

};

//
// Inline Functions.
//______________________________________________________________________________
inline void cudnnError(cudnnStatus_t status, const char *file, int line, bool abort)
{
   if (status != CUDNN_STATUS_SUCCESS)
   {
      fprintf(stderr,"CUDNN Error: %s %s %d\n", cudnnGetErrorString(status), file, line);
      if (abort) exit(status);
   }
}

//______________________________________________________________________________
/*template<typename AFloat>
inline cudaStream_t TCudaTensor<AFloat>::GetComputeStream() const
{
   return fElementBuffer.GetComputeStream();
}

//______________________________________________________________________________
template<typename AFloat>
inline void TCudaTensor<AFloat>::SetComputeStream(cudaStream_t stream)
{
   return fElementBuffer.SetComputeStream(stream);
}

//______________________________________________________________________________
template<typename AFloat>
inline void TCudaTensor<AFloat>::Synchronize(const TCudaTensor &A) const
{
   cudaEvent_t event;
   cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
   cudaEventRecord(event, A.GetComputeStream());
   cudaStreamWaitEvent(fElementBuffer.GetComputeStream(), event, 0);
   cudaEventDestroy(event);
}*/

//______________________________________________________________________________
// template<typename AFloat>
// inline void TCudaTensor<AFloat>::ResetDeviceReturn(AFloat value)
// {
//    AFloat buffer = value;
//    cudaMemcpy(fDeviceReturn, & buffer, sizeof(AFloat), cudaMemcpyHostToDevice);
// }

// //______________________________________________________________________________
// template<typename AFloat>
// inline AFloat TCudaTensor<AFloat>::GetDeviceReturn()
// {
//    AFloat buffer;
//    cudaMemcpy(& buffer, fDeviceReturn, sizeof(AFloat), cudaMemcpyDeviceToHost);
//    return buffer;
// }

//______________________________________________________________________________
// template<typename AFloat>
// TCudaDeviceReference<AFloat> TCudaTensor<AFloat>::operator()(size_t i, size_t j) const
// {
//     AFloat * elementPointer = fElementBuffer;
//     elementPointer += j * fNRows + i;
//     return TCudaDeviceReference<AFloat>(elementPointer);
// }

} // namespace DNN
} // namespace TMVA

#endif
