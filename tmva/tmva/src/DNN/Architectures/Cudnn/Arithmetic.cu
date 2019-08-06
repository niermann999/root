// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Contains additional arithmetic functions required by the CUDA //
// neural network implementation.                                //
///////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cudnn.h"
#include "TMVA/DNN/Architectures/Cuda/CudaTensor.h"
//#include "TMVA/DNN/Architectures/Cuda/CudaMatrix.h"
//#include "TMVA/DNN/Architectures/Cuda/Device.h"
//#include "../Cuda/Kernels.cuh"

namespace TMVA
{
namespace DNN
{

//____________________________________________________________________________
template<>
void TCudnn<float>::Multiply(TCudaTensor<float> &C,
                             const TCudaTensor<float> &A,
                             const TCudaTensor<float> &B,
                             const float alpha,
                             const float beta,
                             const float gamma)
{                      
   // Descriptor for the Tensor Operation
   cudnnOpTensorDescriptor_t opTensorDescr;
   CUDNNCHECK(cudnnCreateOpTensorDescriptor(&opTensorDescr));
   
   CUDNNCHECK(cudnnSetOpTensorDescriptor(opTensorDescr,
                                         CUDNN_OP_TENSOR_MUL,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_PROPAGATE_NAN)); // NaN will be propagated
  
   // C = MUL(alpha*A, beta*B) + gamma*C                                          
   CUDNNCHECK(cudnnOpTensor(A.GetCudnnHandle(),
                            opTensorDescr,
                            (void*) &alpha,
                            A.GetTensorDescriptor(),
                            (void*) A.GetDataPointer(),
                            (void*) &beta,
                            B.GetTensorDescriptor(),
                            (void*) B.GetDataPointer(),
                            (void*) &gamma,           // gamma = 0: Don't add C
                            C.GetTensorDescriptor(),
                            (void*) C.GetDataPointer()));
                                                    
   CUDNNCHECK(cudnnDestroyOpTensorDescriptor(opTensorDescr));
}

//____________________________________________________________________________
template<>
void TCudnn<double>::Multiply(TCudaTensor<double> &C,
                             const TCudaTensor<double> &A,
                             const TCudaTensor<double> &B,                            
                             const double alpha,
                             const double beta,
                             const double gamma)
{                         
   // Descriptor for the Tensor Operation
   cudnnOpTensorDescriptor_t opTensorDescr;
   CUDNNCHECK(cudnnCreateOpTensorDescriptor(&opTensorDescr));
   
   CUDNNCHECK(cudnnSetOpTensorDescriptor(opTensorDescr,
                                         CUDNN_OP_TENSOR_MUL,
                                         CUDNN_DATA_DOUBLE,
                                         CUDNN_PROPAGATE_NAN)); // NaN will be propagated
  
   // C = MUL(alpha*A, beta*B) + gamma*C                                          
   CUDNNCHECK(cudnnOpTensor(A.GetCudnnHandle(),
                            opTensorDescr,
                            (void*) &alpha,
                            A.GetTensorDescriptor(),
                            (void*) A.GetDataPointer(),
                            (void*) &beta,
                            B.GetTensorDescriptor(),
                            (void*) B.GetDataPointer(),
                            (void*) &gamma,           // gamma = 0: Don't add C
                            C.GetTensorDescriptor(),
                            (void*) C.GetDataPointer()));
                                                    
   CUDNNCHECK(cudnnDestroyOpTensorDescriptor(opTensorDescr));
}

//____________________________________________________________________________
/*template<>
void TCudnn<float>::TransposeMultiply(TCudaTensor<float> & C,
                                      const TCudaTensor<float> & A,
                                      const TCudaTensor<float> & B)
{

}*/
//____________________________________________________________________________
/*template<>
void TCudnn<double>::TransposeMultiply(TCudaTensor<double> & C,
                                      const TCudaTensor<double> & A,
                                      const TCudaTensor<double> & B)
{

}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::Hadamard(TCudaTensor<AFloat> & B,
                             const TCudaTensor<AFloat> &A)
{

}*/

//____________________________________________________________________________
template<>
float TCudnn<float>::Sum(const TCudaTensor<float> & A, const float alpha, const float beta)
{
   cudnnHandle_t cudnnHandle = A.GetCudnnHandle();

   // The output tensor C, which has dimensions of a number
   TCudaHostBuffer<float>    hostBuffer (1);
   const std::vector<size_t> shapeVec {1,1,1,1};
   TCudaTensor<float>        C (1, hostBuffer.GetSubBuffer(0,1), 2, shapeVec);
                                         
   // Descriptor for the Tensor Reduction
   cudnnReduceTensorDescriptor_t reduceTensorDescr;
   CUDNNCHECK(cudnnCreateReduceTensorDescriptor(&reduceTensorDescr));
   CUDNNCHECK(cudnnSetReduceTensorDescriptor(reduceTensorDescr,
                                             CUDNN_REDUCE_TENSOR_ADD,
                                             CUDNN_DATA_FLOAT,
                                             CUDNN_PROPAGATE_NAN,                // NaN will be propagated
                                             CUDNN_REDUCE_TENSOR_NO_INDICES,     // Do not compute indices
                                             CUDNN_32BIT_INDICES));              // Type of the indices
                                             
   // Find the minimum size of the indices
   size_t indiceSizeInBytes;
   void*  indices = nullptr;
   CUDNNCHECK(cudnnGetReductionIndicesSize(cudnnHandle,
                                           reduceTensorDescr,
                                           A.GetTensorDescriptor(),
                                           C.GetTensorDescriptor(),
                                           &indiceSizeInBytes));
   cudaMalloc(&indices, indiceSizeInBytes);
   
   // Find the minimum size of the workspace needed for the reduction
   size_t workspaceSizeInBytes;
   void*  workspace = nullptr;
   CUDNNCHECK(cudnnGetReductionWorkspaceSize(cudnnHandle,
                                             reduceTensorDescr,
                                             A.GetTensorDescriptor(),
                                             C.GetTensorDescriptor(),
                                             &workspaceSizeInBytes));
   cudaMalloc(&workspace, workspaceSizeInBytes);
                                         
   // Tensor reduction to the dimensions of the tensor C set above
   // C = alpha * reduce op ( A ) + beta * C                                 
   CUDNNCHECK(cudnnReduceTensor(cudnnHandle,
                                reduceTensorDescr,
                                indices,
                                indiceSizeInBytes,
                                workspace,
                                workspaceSizeInBytes,
                                (void*) &alpha,
                                A.GetTensorDescriptor(),
                                (void*) A.GetDataPointer(),
                                (void*) &beta,
                                C.GetTensorDescriptor(),
                                (void*) C.GetDataPointer()));
                 
   cudaFree(indices);          
   cudaFree(workspace);
   CUDNNCHECK(cudnnDestroyReduceTensorDescriptor(reduceTensorDescr));
   
   return *(C.GetDataPointer());
}

//____________________________________________________________________________
template<>
double TCudnn<double>::Sum(const TCudaTensor<double> & A, const double alpha, const double beta)
{
   cudnnHandle_t cudnnHandle = A.GetCudnnHandle();

   // The output tensor C, which has dimensions of a number
   TCudaHostBuffer<double>   hostBuffer (1);
   const std::vector<size_t> shapeVec {1,1,1,1};
   TCudaTensor<double>       C (1, hostBuffer, 2, shapeVec);
                                         
   // Descriptor for the Tensor Reduction
   cudnnReduceTensorDescriptor_t reduceTensorDescr;
   CUDNNCHECK(cudnnCreateReduceTensorDescriptor(&reduceTensorDescr));
   CUDNNCHECK(cudnnSetReduceTensorDescriptor(reduceTensorDescr,
                                             CUDNN_REDUCE_TENSOR_ADD,
                                             CUDNN_DATA_DOUBLE,
                                             CUDNN_PROPAGATE_NAN,                // NaN will be propagated
                                             CUDNN_REDUCE_TENSOR_NO_INDICES,     // Do not compute indices
                                             CUDNN_32BIT_INDICES));              // Type of the indices
                                             
   // Find the minimum size of the indices
   size_t indiceSizeInBytes;
   void*  indices{nullptr};
   CUDNNCHECK(cudnnGetReductionIndicesSize(cudnnHandle,
                                           reduceTensorDescr,
                                           A.GetTensorDescriptor(),
                                           C.GetTensorDescriptor(),
                                           &indiceSizeInBytes));
   cudaMalloc(&indices, indiceSizeInBytes);
   
   // Find the minimum size of the workspace needed for the reduction
   size_t workspaceSizeInBytes;
   void*  workspace{nullptr};
   CUDNNCHECK(cudnnGetReductionWorkspaceSize(cudnnHandle,
                                             reduceTensorDescr,
                                             A.GetTensorDescriptor(),
                                             C.GetTensorDescriptor(),
                                             &workspaceSizeInBytes));
   cudaMalloc(&workspace, workspaceSizeInBytes);
                                         
   // Tensor reduction to the dimensions of the tensor C set above
   // C = alpha * reduce op ( A ) + beta * C                                 
   CUDNNCHECK(cudnnReduceTensor(cudnnHandle,
                                reduceTensorDescr,
                                indices,
                                indiceSizeInBytes,
                                workspace,
                                workspaceSizeInBytes,
                                (void*) &alpha,
                                A.GetTensorDescriptor(),
                                (void*) A.GetDataPointer(),
                                (void*) &beta,
                                C.GetTensorDescriptor(),
                                (void*) C.GetDataPointer()));
                 
   cudaFree(indices);          
   cudaFree(workspace);
   CUDNNCHECK(cudnnDestroyReduceTensorDescriptor(reduceTensorDescr));
   
   return *(C.GetDataPointer());
}

//____________________________________________________________________________
/*template<>
void TCudnn<float>::SumColumns(TCudaTensor<float> & B,
                              const TCudaTensor<float> & A)
{

}*/

//____________________________________________________________________________
/*template<>
void TCudnn<double>::SumColumns(TCudaTensor<double> & B,
                               const TCudaTensor<double> & A)
{

}

template<>
void TCudnn<float>::SumRows(TCudaTensor<float> & B,
                           const TCudaTensor<float> & A)
{

}*/

//____________________________________________________________________________
/*template<>
void TCudnn<double>::SumRows(TCudaTensor<double> & B,
                             const TCudaTensor<double> & A)
{

}*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Checks two matrices for element-wise equality.
/// \tparam AFloat An architecture-specific floating point number type.
/// \param A The first matrix.
/// \param B The second matrix.
/// \param epsilon Equality tolerance, needed to address floating point arithmetic.
/// \return Whether the two matrices can be considered equal element-wise
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*template<typename AFloat>
bool TCudnn<AFloat>::AlmostEquals(const TCudaTensor<AFloat> &A, const TCudaTensor<AFloat> &B, double epsilon)
{

}*/

//____________________________________________________________________________
template<>
void TCudnn<float>::ScaleAdd(TCudaTensor<float> & B,
                             const TCudaTensor<float> & A,
                             float alpha,
                             float beta)
{
   CUDNNCHECK(cudnnAddTensor(A.GetCudnnHandle(),
                             (void*) &alpha,
                             A.GetTensorDescriptor(),
                             (void*) A.GetDataPointer(),
                             (void*) &beta,
                             B.GetTensorDescriptor(),        // Destination Tensor
                             (void*) B.GetDataPointer()));
}

//____________________________________________________________________________
template<>
void TCudnn<double>::ScaleAdd(TCudaTensor<double> & B,
                              const TCudaTensor<double> & A,
                              double alpha,
                              double beta)
{
   CUDNNCHECK(cudnnAddTensor(A.GetCudnnHandle(),
                             (void*) &alpha,
                             A.GetTensorDescriptor(),
                             (void*) A.GetDataPointer(),
                             (void*) &beta,
                             B.GetTensorDescriptor(),        // Destination Tensor
                             (void*) B.GetDataPointer()));
}

//____________________________________________________________________________
template<>
void TCudnn<float>::ScaleAdd(std::vector<TCudaTensor<float>> & B,
                             const std::vector<TCudaTensor<float>> & A,
                             float alpha,
                             float beta)
{
   for (size_t i = 0; i < A.size(); ++i) {
      ScaleAdd(B[i], A[i], alpha, beta);
   }
}

//____________________________________________________________________________
template<>
void TCudnn<double>::ScaleAdd(std::vector<TCudaTensor<double>> & B,
                              const std::vector<TCudaTensor<double>> & A,
                              double alpha,
                              double beta)
{
   for (size_t i = 0; i < A.size(); ++i) {
      ScaleAdd(B[i], A[i], alpha, beta);
   }
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::ConstAdd(TCudaTensor<AFloat> &A, AFloat beta)
{
   TCudaHostBuffer<AFloat> hostBuffer (A.GetSize());
   hostBuffer.SetConstVal(beta);
   
   // tmp tensor that does the addition
   TCudaTensor<double> C (A);
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::ConstMult(TCudaTensor<AFloat> &A, AFloat beta)
{   
   CUDNNCHECK(cudnnScaleTensor(A.GetCudnnHandle(),
                               A.GetTensorDescriptor(),
                               (void*) A.GetDataPointer(),
                               (void*) &beta));
}

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::ReciprocalElementWise(TCudaTensor<AFloat> &A)
{

}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::SquareElementWise(TCudaTensor<AFloat> &A)
{

}*/

//____________________________________________________________________________
template<>
void TCudnn<float>::SqrtElementWise(TCudaTensor<float> &A, const float alpha, const float beta, const float gamma)
{
   // Descriptor for the Tensor Operation
   cudnnOpTensorDescriptor_t opTensorDescr;
   CUDNNCHECK(cudnnCreateOpTensorDescriptor(&opTensorDescr));
   
   CUDNNCHECK(cudnnSetOpTensorDescriptor(opTensorDescr,
                                         CUDNN_OP_TENSOR_SQRT,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_PROPAGATE_NAN)); // NaN will be propagated
                                         
   // C = MUL(alpha*A, beta*B) + gamma*C                                    
   CUDNNCHECK(cudnnOpTensor(A.GetCudnnHandle(),
                            opTensorDescr,
                            (void*) &alpha,
                            A.GetTensorDescriptor(),
                            (void*) A.GetDataPointer(),
                            (void*) &beta,
                            A.GetTensorDescriptor(),
                            (void*) A.GetDataPointer(),
                            (void*) &gamma,
                            A.GetTensorDescriptor(),  // Save result in A
                            (void*) A.GetDataPointer()));
                            
   CUDNNCHECK(cudnnDestroyOpTensorDescriptor(opTensorDescr));
}

//____________________________________________________________________________
template<>
void TCudnn<double>::SqrtElementWise(TCudaTensor<double> &A, const double alpha, const double beta, const double gamma)
{   
   // Descriptor for the Tensor Operation
   cudnnOpTensorDescriptor_t opTensorDescr;
   CUDNNCHECK(cudnnCreateOpTensorDescriptor(&opTensorDescr));
   
   CUDNNCHECK(cudnnSetOpTensorDescriptor(opTensorDescr,
                                         CUDNN_OP_TENSOR_SQRT,
                                         CUDNN_DATA_DOUBLE,
                                         CUDNN_PROPAGATE_NAN)); // NaN will be propagated
                                         
   // C = MUL(alpha*A, beta*B) + gamma*C                                    
   CUDNNCHECK(cudnnOpTensor(A.GetCudnnHandle(),
                            opTensorDescr,
                            (void*) &alpha,
                            A.GetTensorDescriptor(),
                            (void*) A.GetDataPointer(),
                            (void*) &beta,
                            A.GetTensorDescriptor(),
                            (void*) A.GetDataPointer(),
                            (void*) &gamma,
                            A.GetTensorDescriptor(),  // Save result in A
                            (void*) A.GetDataPointer()));
                            
   CUDNNCHECK(cudnnDestroyOpTensorDescriptor(opTensorDescr));
}

/// Adam updates 
//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::AdamUpdate(TCudaTensor<AFloat> &A, const TCudaTensor<AFloat> & M, const TCudaTensor<AFloat> & V, AFloat alpha, AFloat eps)
{

}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::AdamUpdateFirstMom(TCudaTensor<AFloat> &A, const TCudaTensor<AFloat> & B, AFloat beta)
{

}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::AdamUpdateSecondMom(TCudaTensor<AFloat> &A, const TCudaTensor<AFloat> & B, AFloat beta)
{

}*/
   
} // DNN
} // TMVA
