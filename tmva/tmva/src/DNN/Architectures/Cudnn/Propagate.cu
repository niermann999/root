// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
 // Implementation of the functions required for the forward and //
 // backward propagation of activations through a neural network //
 // for CUDA architectures using cuDNN library.                  //
 //////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cudnn.h"
/*#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"*/
#include <math.h>
#include <iostream>

namespace TMVA {
namespace DNN  {

//____________________________________________________________________________
/*template<>
void TCudnn<float>::MultiplyTranspose(TCudaTensor<float> &output,
                                      const TCudaTensor<float> &input,
                                      const TCudaTensor<float> &Weights)
{
   int m, n, k;
   k = input.GetNcols();
   m = input.GetNrows();
   n = Weights.GetNrows();
   float alpha = 1.0, beta = 0.0;

   // Compute C = beta * C + alpha * (A * B^T)
   cudaStream_t s = output.GetComputeStream();
   cublasSetStream(input.GetCublasHandle(), s);
   cublasSgemm(input.GetCublasHandle(),
               CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k, & alpha,
               input.GetDataPointer(), m,     // *A, lda
               Weights.GetDataPointer(), n,   // *B, ldb
               & beta,                        // beta
               output.GetDataPointer(), m);   // *C, ldc
}

//____________________________________________________________________________
template<>
void TCudnn<double>::MultiplyTranspose(TCudaTensor<double> &output,
                                       const TCudaTensor<double> &input,
                                       const TCudaTensor<double> &Weights)
{
   int m, n, k;
   k = input.GetNcols();
   m = input.GetNrows();
   n = Weights.GetNrows();
   double alpha = 1.0, beta = 0.0;

   // Compute C = beta * C + alpha * (A * B^T)
   cudaStream_t s = output.GetComputeStream();
   cublasSetStream(input.GetCublasHandle(), s);
   cublasDgemm(input.GetCublasHandle(),
               CUBLAS_OP_N, CUBLAS_OP_T,
               m, n, k, & alpha,
               input.GetDataPointer(), m,     // *A, lda
               Weights.GetDataPointer(), n,   // *B, ldb
               & beta,                        // beta
               output.GetDataPointer(), m);   // *C, ldc
}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::AddRowWise(TCudaTensor<AFloat> &Weights,
                                const TCudaTensor<AFloat> &theta)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(Weights);
   cudaStream_t s = Weights.GetComputeStream();
   ::TMVA::DNN::Cuda::AddRowWise<<<gridDims, blockDims, 0, s>>>(
       Weights.GetDataPointer(),
       theta.GetDataPointer(),
       Weights.GetNrows(),
       Weights.GetNcols());
}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::Backward(TCudaTensor<AFloat> & activation_gradients_backward,
                              TCudaTensor<AFloat> & weight_gradients,
                              TCudaTensor<AFloat> & bias_gradients,
                              TCudaTensor<AFloat> & df,
                              const TCudaTensor<AFloat> & activation_gradients,
                              const TCudaTensor<AFloat> & weights,
                              const TCudaTensor<AFloat> & activation_backward)
{
   // Compute element-wise product.
   TCudnn<AFloat>::Hadamard(df, activation_gradients);

   // Activation gradients.
   if (activation_gradients_backward.GetNoElements() > 0) {
      TCudnn<AFloat>::Multiply(activation_gradients_backward, df, weights);
   }

   // Weight gradients.
   if (weight_gradients.GetNoElements() > 0) {
      TCudnn<AFloat>::TransposeMultiply(weight_gradients, df, activation_backward);
   }

   // Bias gradients.
   if (bias_gradients.GetNoElements() > 0) {
      TCudnn<AFloat>::SumColumns(bias_gradients, df);
   }

}*/

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::Copy(TCudaTensor<AFloat> & B,
                          const TCudaTensor<AFloat> & A)
{
   size_t nElements = A.GetSize();
   cudaMemcpyAsync(B.GetDataPointer(), A.GetDataPointer(),
                   nElements * sizeof(AFloat), cudaMemcpyDeviceToDevice, 0);
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::Copy(std::vector<TCudaTensor<AFloat>> & B,
                         const std::vector<TCudaTensor<AFloat>> & A)
{
   for (size_t i = 0; i < B.size(); ++i) {
      Copy(B[i], A[i]);
   }
}

//____________________________________________________________________________
/*template<typename AFloat>
size_t TCudnn<AFloat>::calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride)
{
   size_t temp = imgDim - fltDim + 2 * padding;
   if (temp % stride || temp + stride <= 0) {
      Fatal("calculateDimension", "Not compatible hyper parameters for layer - (imageDim, filterDim, padding, stride)"
            " %zu , %zu , %zu , %zu", imgDim, fltDim, padding, stride);
   }
   return temp / stride + 1;
}*/


///////////////////////////////////////////////////////////////////////////////////
/// \brief A helper for image operations that rearranges image regions into
///        column vectors.
///
/// \param[out] A The output matrix. Each row corresponds to a receptive field.
/// \param[in] B The input matrix. Each row corresponds to a row in the image view.
/// \param[in] imgHeight The heigh of the input.
/// \param[in] imgWidth The output of the input.
/// \param[in] fltHeight Height of the kernel.
/// \param[in] fltWidth Width of the kernel.
/// \param[in] strideRows stride size in the horizontal dimension.
/// \param[in] strideCols stride size in the vertical dimension.
/// \param[in] zeroPaddingHeight The padding in the horizontal dimension.
/// \param[in] zeroPaddingWidth The padding in the vertical dimension.
///
/// This transformation allows us to express a 2D convolution as a matrix
/// multiplication. We can therefore harness the finely tuned GEMM
/// implementation of cuBLAS to achieve maximum performance. This function
/// can greatly speed-up propagation in TConvLayer.
///////////////////////////////////////////////////////////////////////////////////
/*template<typename AFloat>
void TCudnn<AFloat>::Im2col(TCudaTensor<AFloat> &A,
                           const TCudaTensor<AFloat> &B,
                           size_t imgHeight,
                           size_t imgWidth,
                           size_t fltHeight,
                           size_t fltWidth,
                           size_t strideRows,
                           size_t strideCols,
                           size_t zeroPaddingHeight,
                           size_t zeroPaddingWidth)
{
   size_t depth = B.GetNrows();

   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();

   ::TMVA::DNN::Cuda::Im2Col<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(), B.GetDataPointer(), depth, imgHeight, imgWidth,
                                                            fltHeight, fltWidth, strideRows, strideCols,
                                                            zeroPaddingHeight, zeroPaddingWidth);
}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::RotateWeights(TCudaTensor<AFloat> &A,
                                  const TCudaTensor<AFloat> &B,
                                  size_t filterDepth,
                                  size_t filterHeight,
                                  size_t filterWidth,
                                  size_t numFilters)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = B.GetComputeStream();

   ::TMVA::DNN::Cuda::RotateWeights<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(), B.GetDataPointer(), filterDepth,
                                                                   filterHeight, filterWidth, numFilters);
}*/

template <typename AFloat>
void TCudnn<AFloat>::PrepareInternals(std::vector<TCudaTensor<AFloat>> & inputPrime, 
                                      cudnnFilterDescriptor_t filterDescr,
                                      cudnnConvolutionDescriptor_t fConvolutionDescriptor)
{
   for (size_t event = 0; event < inputPrime.size(); event++) {
      cudaStream_t s;
      cudaStreamCreate(&s);
      inputPrime[event].SetComputeStream(s);
   }
}



template <>
void TCudnn<float>::ConvLayerForward(std::vector<TCudaTensor<float>> & output,
                                     std::vector<TCudaTensor<float>> & derivatives,
                                     const std::vector<TCudaTensor<float>> &input,
                                     const TCudaTensor<float> &weights, const TCudaTensor<float> & biases,
                                     const DNN::CNN::TConvParams & params, EActivationFunction activFunc,
                                     std::vector<TCudaTensor<float>> & inputPrime,
                                     const float alpha,
                                     const float beta)
{
   cudnnHandle_t cudnnHandle = input[0].GetCudnnHandle();
   
   cudnnFilterDescriptor_t fFilterDescriptor;           // Layout of the Kernel
   // Set the filter from TCudaMatrix instance first
   CUDNNCHECK(cudnnCreateFilterDescriptor(&fFilterDescriptor));
   CUDNNCHECK(cudnnSetFilter4dDescriptor(fFilterDescriptor,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW,
                                         params.numberFilters,
                                         params.inputDepth,
                                         params.filterHeight,
                                         params.filterWidth));
                                         
                                         
   
   // Set the convolution
   cudnnConvolutionDescriptor_t fConvolutionDescriptor;      // Params of the convolution (can be reused in backward pass)
   CUDNNCHECK(cudnnCreateConvolutionDescriptor(&fConvolutionDescriptor));
   
   // Use tensor ops
   /*cudnnStatus_t cudnnSetConvolutionMathType(
    cudnnConvolutionDescriptor_t    convDesc,
    cudnnMathType_t                 mathType)*/
    
   CUDNNCHECK(cudnnSetConvolution2dDescriptor(fConvolutionDescriptor,
                                              params.paddingHeight,
                                              params.paddingWidth,
                                              params.strideRows,
                                              params.strideCols,
                                              0,                 //Dilation height
                                              0,                 //Dilation width
                                              CUDNN_CONVOLUTION, // Convolution instead of cross correlation
                                              CUDNN_DATA_FLOAT));
                                              
   // Get the dimensions of the output tensor
   std::vector<size_t> outputShape {0,0,0,0};
   CUDNNCHECK(cudnnGetConvolution2dForwardOutputDim(fConvolutionDescriptor,
                                                    input[0].GetTensorDescriptor(),
                                                    fFilterDescriptor,
                                                    (int*)&outputShape[0],
                                                    (int*)&outputShape[1],
                                                    (int*)&outputShape[2],
                                                    (int*)&outputShape[3]));
   size_t size = 1;
   for (const auto& subDim: outputShape) size *= subDim;
   TCudaTensor<float> outputTensor (size, outputShape.size(), outputShape);
   
   // cuDNN decides on which algorithm to use
   cudnnConvolutionFwdAlgo_t algorithm;
   // More detailed alternative: cudnnFindConvolutionForwardAlgorithm
   CUDNNCHECK(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                  input[0].GetTensorDescriptor(),
                                                  fFilterDescriptor,
                                                  fConvolutionDescriptor,
                                                  outputTensor.GetTensorDescriptor(),
                                                  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                  0,     // Memory limit in bytes for mode CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
                                                  &algorithm));
                                                  
   // Allocate memory for the convolution
   size_t workSpaceSizeInBytes = 0;
   void   *workspace           = nullptr;
   CUDNNCHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                  input[0].GetTensorDescriptor(),
                                                  fFilterDescriptor,
                                                  fConvolutionDescriptor,
                                                  outputTensor.GetTensorDescriptor(),
                                                  algorithm,
                                                  &workSpaceSizeInBytes));
                                                  
   if (workSpaceSizeInBytes) cudaMalloc(&workspace, workSpaceSizeInBytes*sizeof(float));
   
   // Perform convolution
   CUDNNCHECK(cudnnConvolutionForward(cudnnHandle,
                                      &alpha,
                                      input[0].GetTensorDescriptor(),
                                      input[0].GetDataPointer(),
                                      fFilterDescriptor,
                                      weights.GetDataPointer(),
                                      fConvolutionDescriptor,
                                      algorithm,
                                      workspace,
                                      workSpaceSizeInBytes,
                                      &beta,
                                      outputTensor.GetTensorDescriptor(),
                                      outputTensor.GetDataPointer()));
                                      
   // Add biases
   ScaleAdd(outputTensor, biases);
                                      
   // Apply activation
   evaluate<TCudnn<float> >(outputTensor, activFunc);
   
   // Put cudnn tensor in output container of the layer
   output.push_back(outputTensor);
   
   cudaFree(&workspace);
   CUDNNCHECK(cudnnDestroyFilterDescriptor(fFilterDescriptor));
   CUDNNCHECK(cudnnDestroyConvolutionDescriptor(fConvolutionDescriptor));
}


template <>
void TCudnn<double>::ConvLayerForward(std::vector<TCudaTensor<double>> & output,
                                     std::vector<TCudaTensor<double>> & derivatives,
                                     const std::vector<TCudaTensor<double>> & input,
                                     const TCudaTensor<double> & weights, const TCudaTensor<double> & biases,
                                     const DNN::CNN::TConvParams & params, EActivationFunction activFunc,
                                     std::vector<TCudaTensor<double>> & inputPrime,
                                     const double alpha,
                                     const double beta)
{
   cudnnHandle_t cudnnHandle = input[0].GetCudnnHandle();

   cudnnFilterDescriptor_t fFilterDescriptor;           // Layout of the Kernel
   // Set the filter from TCudaMatrix instance first
   CUDNNCHECK(cudnnCreateFilterDescriptor(&fFilterDescriptor));
   CUDNNCHECK(cudnnSetFilter4dDescriptor(fFilterDescriptor,
                                         CUDNN_DATA_DOUBLE,
                                         CUDNN_TENSOR_NCHW,
                                         params.numberFilters,
                                         params.inputDepth,
                                         params.filterHeight,
                                         params.filterWidth));
                                    
   // Set the convolution
   cudnnConvolutionDescriptor_t fConvolutionDescriptor;      // Params of the convolution (can be reused in backward pass)
   CUDNNCHECK(cudnnCreateConvolutionDescriptor(&fConvolutionDescriptor));
   
   // Use tensor ops
   /*cudnnStatus_t cudnnSetConvolutionMathType(
    cudnnConvolutionDescriptor_t    convDesc,
    cudnnMathType_t                 mathType)*/
    
   CUDNNCHECK(cudnnSetConvolution2dDescriptor(fConvolutionDescriptor,
                                              params.paddingHeight,
                                              params.paddingWidth,
                                              params.strideRows,
                                              params.strideCols,
                                              1,                 //Dilation height
                                              1,                 //Dilation width
                                              CUDNN_CROSS_CORRELATION, // Convolution instead of cross correlation
                                              CUDNN_DATA_DOUBLE));
                                              
   // Get the dimensions of the output tensor
   std::vector<size_t> outputShape {0,0,0,0};
   CUDNNCHECK(cudnnGetConvolution2dForwardOutputDim(fConvolutionDescriptor,
                                                    input[0].GetTensorDescriptor(),
                                                    fFilterDescriptor,
                                                    (int*)&outputShape[0],
                                                    (int*)&outputShape[1],
                                                    (int*)&outputShape[2],
                                                    (int*)&outputShape[3]));
   //std::vector<size_t> outputShape {1,3,3,3};
   size_t size = 1;
   for (const auto& subDim: outputShape) size *= (size_t)subDim;
   output.emplace_back(size, outputShape.size(), outputShape);
   
   // cuDNN decides on which algorithm to use
   cudnnConvolutionFwdAlgo_t algorithm;
   // More detailed alternative: cudnnFindConvolutionForwardAlgorithm
   CUDNNCHECK(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                  input[0].GetTensorDescriptor(),
                                                  fFilterDescriptor,
                                                  fConvolutionDescriptor,
                                                  output[0].GetTensorDescriptor(),
                                                  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,    // make users choice
                                                  0,     // Memory limit in bytes for mode CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
                                                  &algorithm));
                                  
   // Allocate memory for the convolution
   size_t workSpaceSizeInBytes = 0;
   void   *workspace           = nullptr;
   CUDNNCHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                      input[0].GetTensorDescriptor(),
                                                      fFilterDescriptor,
                                                      fConvolutionDescriptor,
                                                      output[0].GetTensorDescriptor(),
                                                      algorithm,
                                                      &workSpaceSizeInBytes));
                                                  

   if (workSpaceSizeInBytes) cudaMalloc(&workspace, workSpaceSizeInBytes*sizeof(double));
   
   // Perform convolution
   CUDNNCHECK(cudnnConvolutionForward(cudnnHandle,
                                      &alpha,
                                      input[0].GetTensorDescriptor(),
                                      input[0].GetDataPointer(),
                                      fFilterDescriptor,
                                      weights.GetDataPointer(),
                                      fConvolutionDescriptor,
                                      algorithm,
                                      workspace,
                                      workSpaceSizeInBytes,
                                      &beta,
                                      output[0].GetTensorDescriptor(),
                                      output[0].GetDataPointer()));
                                      
   // Add biases
   ScaleAdd(output[0], biases);
                                      
   // Apply activation
   evaluate<TCudnn<double> >(output[0], activFunc);
   
   // Put cudnn tensor in output container of the layer

   
   //cudaFree(&workspace);
   CUDNNCHECK(cudnnDestroyFilterDescriptor(fFilterDescriptor));
   CUDNNCHECK(cudnnDestroyConvolutionDescriptor(fConvolutionDescriptor));
}


//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::ConvLayerBackward(std::vector<TCudaTensor<AFloat>> & activationGradientsBackward,
                                      TCudaTensor<AFloat> & weightGradients,
                                      TCudaTensor<AFloat> & biasGradients,
                                      std::vector<TCudaTensor<AFloat>> & df,
                                      const std::vector<TCudaTensor<AFloat>> & activationGradients,
                                      const TCudaTensor<AFloat> & weights,
                                      const std::vector<TCudaTensor<AFloat>> & activationBackward,
                                      size_t batchSize,
                                      size_t inputHeight,
                                      size_t inputWidth,
                                      size_t depth,
                                      size_t height,
                                      size_t width,
                                      size_t filterDepth,
                                      size_t filterHeight,
                                      size_t filterWidth,
                                      size_t nLocalViews)
{
    for (size_t i = 0; i < batchSize; i++) {
        // Compute element-wise product.
        Hadamard(df[i], activationGradients[i]);
    }

    // Calculate the activation gradients of the previous layer
    CalculateConvActivationGradients(activationGradientsBackward, df, weights, batchSize, inputHeight, inputWidth, depth,
                                     height, width, filterDepth, filterHeight, filterWidth);


    // Calculate the weight gradients
    CalculateConvWeightGradients(weightGradients, df, activationBackward, batchSize, inputHeight, inputWidth, depth,
                                 height, width, filterDepth, filterHeight, filterWidth, nLocalViews);

    // Calculate the bias gradients
    CalculateConvBiasGradients(biasGradients, df, batchSize, depth, nLocalViews);
}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::CalculateConvActivationGradients(
                                    std::vector<TCudaTensor<AFloat>> & activationGradientsBackward,
                                    std::vector<TCudaTensor<AFloat>> & df,
                                    const TCudaTensor<AFloat> & weights,
                                    size_t /* batchSize *//*,
                                    size_t inputHeight,
                                    size_t inputWidth,
                                    size_t depth,
                                    size_t height,
                                    size_t width,
                                    size_t filterDepth,
                                    size_t filterHeight,
                                    size_t filterWidth)
{
   if (activationGradientsBackward.size() == 0) return;

   TCudaTensor<AFloat> rotWeights(filterDepth, depth * filterHeight * filterWidth);
   RotateWeights(rotWeights, weights, filterDepth, filterHeight, filterWidth, weights.GetNrows());

   // Calculate the zero paddings.
   size_t tempZeroPaddingHeight = (size_t)(floor((inputHeight - height + filterHeight - 1) / 2));
   size_t tempZeroPaddingWidth = (size_t)(floor((inputWidth - width + filterWidth - 1) / 2));

   // Calculate the number of local views and the number of pixels in each view.
   size_t tempNLocalViews = inputHeight * inputWidth;
   size_t tempNLocalViewPixels = depth * filterHeight * filterWidth;

   // Problem here. We need to generalize!
   size_t tempStrideRows = 1;
   size_t tempStrideCols = 1;

   // Convolution.
   TCudaTensor<AFloat> dfPrime(tempNLocalViews, tempNLocalViewPixels);
   for(size_t event = 0; event < df.size(); event++) {
      Im2col(dfPrime, df[event], height, width, filterHeight, filterWidth, tempStrideRows, tempStrideCols,
             tempZeroPaddingHeight, tempZeroPaddingWidth);

      MultiplyTranspose(activationGradientsBackward[event], rotWeights, dfPrime);
   }
}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::CalculateConvWeightGradients(TCudaTensor<AFloat> & weightGradients,
                                                 std::vector<TCudaTensor<AFloat>> & df,
                                                 const std::vector<TCudaTensor<AFloat>> & activationsBackward,
                                                 size_t batchSize,
                                                 size_t inputHeight,
                                                 size_t inputWidth,
                                                 size_t depth,
                                                 size_t height,
                                                 size_t width,
                                                 size_t filterDepth,
                                                 size_t filterHeight,
                                                 size_t filterWidth,
                                                 size_t nLocalViews)
{
    // reinitialize the weight gradients to 0
    weightGradients.Zero();

    const size_t filterSize = filterHeight * filterWidth;
    const size_t nLocalViewPixels = filterDepth * filterSize;
    R__ASSERT( weightGradients.GetNcols() == nLocalViewPixels);
    R__ASSERT( weightGradients.GetNrows() == depth);
    R__ASSERT( df.size() ==  batchSize);



    const size_t tempStrideRows = 1;
    const size_t tempStrideCols = 1;

    // Calculate the zero paddings from the input height and width (assume stride = 1)
    const size_t tempZeroPaddingHeight = (height - inputHeight + filterHeight - 1) / 2;
    const size_t tempZeroPaddingWidth = (width - inputWidth + filterWidth - 1) / 2;

    // Convolution.
    TCudaTensor<AFloat> activationsPrime(nLocalViews, nLocalViewPixels);
    TCudaTensor<AFloat> resPrime(depth, nLocalViewPixels);
    for(size_t event = 0; event < df.size(); event++) {
        Im2col(activationsPrime, activationsBackward[event], inputHeight, inputWidth, filterHeight, filterWidth,
               tempStrideRows, tempStrideCols, tempZeroPaddingHeight, tempZeroPaddingWidth);

        Multiply(resPrime, df[event], activationsPrime);

        TCudnn<AFloat>::ScaleAdd(weightGradients, resPrime, 1.0); 
    }
}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::CalculateConvBiasGradients(TCudaTensor<AFloat> & biasGradients,
                                               std::vector<TCudaTensor<AFloat>> & df,
                                               size_t batchSize,
                                               size_t /* depth *//*,
                                               size_t /* nLocalViews *//*)
{
    biasGradients.Zero();
    TCudaTensor<AFloat> temp(biasGradients.GetNrows(), biasGradients.GetNcols());
    for (size_t event = 0; event < batchSize; event++) {
        TCudnn<AFloat>::SumRows(temp, df[event]);
        TCudnn<AFloat>::ScaleAdd(biasGradients, temp);
    }
}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::AddConvBiases(TCudaTensor<AFloat> &output,
                                  const TCudaTensor<AFloat> &biases)
{
    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(output);
    cudaStream_t s = output.GetComputeStream();
    ::TMVA::DNN::Cuda::AddBiases<<<gridDims, blockDims, 0, s>>>(
            output.GetDataPointer(),
            biases.GetDataPointer(),
            output.GetNrows(),
            output.GetNcols());
}*/


//____________________________________________________________________________
//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Downsampling function used as the forward propagation step of a
///        Max-Pooling layer.
///
/// \param[out] A The output matrix. Each row corresponds to a slice and each element
///             is the max within a receptive field.
/// \param[out] B The winning indices matrix. Each element is the index of the max element.
/// \param[in] C The input matrix. Each row is a slice.
/// \param[in] imgHeight The heigh of the input.
/// \param[in] imgWidth The output of the input.
/// \param[in] fltHeight Height of the kernel.
/// \param[in] fltWidth Width of the kernel.
/// \param[in] strideRows stride size in the horizontal dimension.
/// \param[in] strideCols stride size in the vertical dimension.
///
/// Each output element is the maximum of the receptive field. We also save the winning
/// indices to facilitate back-propagation - we need to know which input element influenced
/// the output and only apply the derivative correction to this particular element.
/// The slicing process is the same as in a convolutional layer, however padding is set to 0.
///////////////////////////////////////////////////////////////////////////////////////////////
/*template<typename AFloat>
void TCudnn<AFloat>::Downsample(TCudaTensor<AFloat> &A,
                               TCudaTensor<AFloat> &B,
                               const TCudaTensor<AFloat> &C,
                               size_t imgHeight,
                               size_t imgWidth,
                               size_t fltHeight,
                               size_t fltWidth,
                               size_t strideRows,
                               size_t strideCols)
{
   size_t depth = C.GetNrows();

   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Downsample<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(), B.GetDataPointer(),
                                                                C.GetDataPointer(), depth, imgHeight, imgWidth,
                                                                fltHeight, fltWidth, strideRows, strideCols);
}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::MaxPoolLayerBackward(TCudaTensor<AFloat> & activationGradientsBackward,
                                         const TCudaTensor<AFloat> & activationGradients,
                                         const TCudaTensor<AFloat> & indexMatrix,
                                         size_t imgHeight,
                                         size_t imgWidth,
                                         size_t fltHeight,
                                         size_t fltWidth,
                                         size_t strideRows,
                                         size_t strideCols,
                                         size_t /* nLocalViews *//*)
{
   size_t depth = activationGradientsBackward.GetNrows();

   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(activationGradientsBackward);
   cudaStream_t s = activationGradientsBackward.GetComputeStream();

   ::TMVA::DNN::Cuda::MaxPoolBackward<<<gridDims, blockDims, 0, s>>>(activationGradientsBackward.GetDataPointer(),
                                                                     activationGradients.GetDataPointer(),
                                                                     indexMatrix.GetDataPointer(),
                                                                     depth, imgHeight, imgWidth, fltHeight, fltWidth,
                                                                     strideRows, strideCols);
}*/

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::Reshape(TCudaTensor<AFloat> &A, const TCudaTensor<AFloat> &B)
{
    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(A);
    cudaStream_t s = A.GetComputeStream();

    ::TMVA::DNN::Cuda::Reshape<<<gridDims, blockDims>>>(A.GetDataPointer(), B.GetDataPointer(),
                                                        A.GetNrows(), A.GetNcols(), B.GetNrows(), B.GetNcols());
}*/

//______________________________________________________________________________
/*template <typename AReal>
void TCudnn<AReal>::Rearrange(std::vector<TCudaTensor<AReal>> &out, const std::vector<TCudaTensor<AReal>> &in)
{
   // B x T x D out --- T x B x D in*/
   /*size_t B = out.size();
   size_t T = out[0].GetNrows();
   size_t D = out[0].GetNcols();
   if ((T != in.size()) || (B != in[0].GetNrows()) 
       || (D != in[0].GetNcols())) {
      std::cout << "Incompatible Dimensions\n"
         << in.size() << "x" << in[0].GetNrows() << "x" << in[0].GetNcols() 
         << " --> " << B << "x" << T << "x" << D << "\n";
      return;
   }
   for (size_t i = 0; i < B; ++i) {
      for (size_t j = 0; j < T; ++j) {
         for (size_t k = 0; k < D; ++k) {
            out[i](j, k) = in[j](i, k);
         }
      }
   }
   return;
}*/

//____________________________________________________________________________
////////////////////////////////////////////////////////////////////////////////
/// \brief Flatten a vector of matrices into a single matrix.
///
/// \param[out] A Output matrix.
/// \param[in] B Input vector. Each element is a matrix to be concatenated.
/// \param[in] size Number of matrices in the input vector.
/// \param[in] nRows Number of rows in each matrix of the input vector.
/// \param[in] nCols Number of columns on each matrix of the input vector.
///
/// Each row in the output matrix is the concatenation of the same row in
/// each of the input matrices. Passing an std::vector to a CUDA kernel is
/// a non trivial task that requires manually allocating and copying to device
/// memory - details in comments within the function's body. Launching one
/// thread per output element.
//////////////////////////////////////////////////////////////////////////////////
/*template<typename AFloat>
void TCudnn<AFloat>::Flatten(TCudaTensor<AFloat> &A,
                            const std::vector<TCudaTensor<AFloat>> &B,
                            size_t size,
                            size_t nRows,
                            size_t nCols)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();

   // Get raw pointers from a vector of matrices - this is more challenging than it sounds.
   //
   // Attention: While `TCudaTensor.GetDataPointer() returns a pointer to device memory,
   //            std::vector (and its .data() raw pointer) resides on host memory. Therefore
   //            we need to manually copy these pointers to the device prior to invoking the kernel.

   const AFloat ** dB; // device pointer to device pointers.S
   const AFloat ** hB = new const AFloat * [size]; // host pointer to device pointers.

   cudaMalloc(&dB, sizeof(AFloat *) * size);
   for(size_t i = 0; i < size; ++i) {
      hB[i] = B[i].GetDataPointer();
   }

   cudaMemcpy(dB, hB, sizeof(AFloat *) * size, cudaMemcpyHostToDevice);

   // Launch the kernel using our device pointers.
   ::TMVA::DNN::Cuda::Flatten<<<gridDims, blockDims>>>(A.GetDataPointer(), dB, size, nRows, nCols);

   delete [] hB; 
   cudaFree(dB); 
}*/

//____________________________________________________________________________
////////////////////////////////////////////////////////////////////////////////
/// \brief Deflatten a matrix into a vector of matrices.
///
/// \param[out] A Output matrices. Each element will be a part of the input.
/// \param[in] B Input flat matrix.
/// \param[in] size Number of matrices in the output vector.
/// \param[in] nRows Number of rows in each matrix of the output vector.
/// \param[in] nCols Number of columns on each matrix of the output vector.
///
/// Each row in the input matrix is the concatenation of the same row in
/// each of the output matrices. Passing an std::vector to a CUDA kernel is
/// a non trivial task that requires manually allocating and copying to device
/// memory - details in comments within the function's body. Launching one
/// thread per input element.
//////////////////////////////////////////////////////////////////////////////////
/*template<typename AFloat>
void TCudnn<AFloat>::Deflatten(std::vector<TCudaTensor<AFloat>> &A,
                              const TCudaTensor<AFloat> &B,
                              size_t size,
                              size_t nRows,
                              size_t nCols)
{
    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(B);
    cudaStream_t s = B.GetComputeStream();

    // Get raw pointers from a vector of matrices - this is more challenging than it sounds.
    //
    // Attention: While `TCudaTensor.GetDataPointer() returns a pointer to device memory,
    //            std::vector (and its .data() raw pointer) resides on host memory. Therefore
    //            we need to manually copy these pointers to the device prior to invoking the kernel.

    AFloat ** dA; // device pointer to device pointers.
    AFloat ** hA = new AFloat * [size]; // host pointer to device pointers.

    cudaMalloc(&dA, sizeof(AFloat *) * size);

    for(size_t i = 0; i < size; ++i) {
        hA[i] = A[i].GetDataPointer();
    }

    cudaMemcpy(dA, hA, sizeof(AFloat *) * size, cudaMemcpyHostToDevice);

    // Launch the kernel using our device pointers.
    ::TMVA::DNN::Cuda::Deflatten<<<gridDims, blockDims>>>(dA, B.GetDataPointer(), size, nRows, nCols);

    cudaFree(dA); 
    delete [] hA; 
}*/

} // namespace DNN
} // namespace TMVA
