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

#include "TMVA/DNN/Architectures/TCudnn.h"

#include "TMVA/DNN/CNN/ConvLayer.h"

#include "TMVA/DNN/Architectures/Cuda.h"

// #include "TMVA/DNN/Architectures/Cuda/Device.h"
// #include "Kernels.cuh"*/
// #include <math.h>

namespace TMVA {
namespace DNN  {


//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::MultiplyTranspose(TCudaTensor<AFloat> &output,
                                       const TCudaTensor<AFloat> &input,
                                       const TCudaTensor<AFloat> &weights)
{
   TCuda<AFloat>::MultiplyTranspose(output, input, weights.GetMatrix());
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::AddRowWise(TCudaTensor<AFloat> &output,
                                const TCudaTensor<AFloat> &biases)
{
   TCuda<AFloat>::AddRowWise( output, biases.GetMatrix());
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::Backward(TCudaTensor<AFloat> & activation_gradients_backward,
                              TCudaTensor<AFloat> & weight_gradients,
                              TCudaTensor<AFloat> & bias_gradients,
                              TCudaTensor<AFloat> & df,
                              const TCudaTensor<AFloat> & activation_gradients,
                              const TCudaTensor<AFloat> & weights,
                              const TCudaTensor<AFloat> & activation_backward)
{
   // use implentation from TCuda 
   TCudaMatrix<AFloat> weightGradMatrix = weight_gradients.GetMatrix(); 
   TCudaMatrix<AFloat> biasGradMatrix = bias_gradients.GetMatrix(); 
   TCuda<AFloat>::Backward(activation_gradients_backward,
                              weightGradMatrix,
                              biasGradMatrix,
                              df,
                              activation_gradients,
                              weights.GetMatrix(), 
                              activation_backward);
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::Copy(Tensor_t & B, const Tensor_t & A)
{
   size_t nElements = A.GetSize();
   R__ASSERT(nElements == B.GetSize());
   
   cudaMemcpyAsync(B.GetDataPointer(), A.GetDataPointer(),
                   nElements * sizeof(AFloat), cudaMemcpyDeviceToDevice, 0);
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
void TCudnn<AFloat>::ConvLayerForward(Tensor_t & outputTensor,
                                      Tensor_t & inputActivation,
                                      const Tensor_t & input,
                                      const Matrix_t & weights, const Matrix_t & biases,
                                      const DNN::CNN::TConvParams & params, EActivationFunction activFunc,
                                      Tensor_t & inputPrime,
                                      const ConvDescriptors_t & descriptors,
                                      void * cudnnWorkspace)
//                                    const AFloat alpha,
//                                    const AFloat beta)
{

   ((Tensor_t & )input).Reshape( {params.batchSize, params.inputDepth, params.inputHeight, params.inputWidth}); 

   size_t outputHeight =  DNN::CNN::TConvLayer<TCudnn<AFloat>>::calculateDimension(params.inputHeight, params.filterHeight, params.paddingHeight, params.strideRows);
   size_t outputWidth =  DNN::CNN::TConvLayer<TCudnn<AFloat>>::calculateDimension(params.inputWidth, params.filterWidth, params.paddingWidth, params.strideCols);

   outputTensor.Reshape({params.batchSize, params.numberFilters, outputHeight, outputWidth});
   inputActivation.Reshape( {params.batchSize, params.numberFilters, outputHeight, outputWidth});

   ((Tensor_t & )weights).Reshape( { params.numberFilters, params.inputDepth, params.filterHeight, params.filterWidth } );
   //biases.Reshape ( { 1,params.numberFilters, 1, 1});

   AFloat alpha = 1.0; 
   AFloat beta  = 0.0; 
   cudnnHandle_t cudnnHandle = input.GetCudnnHandle();
   
   //FIXME: Move this to constructor
   cudnnDataType_t   cudnnDataType;
   if      (std::is_same<AFloat, double>::value) { cudnnDataType = CUDNN_DATA_DOUBLE;}
   else if (std::is_same<AFloat, float>::value)  { cudnnDataType = CUDNN_DATA_FLOAT;}

   PrintTensor(input ,"input tensor");
  
   // Set the  filter parameters
   CUDNNCHECK(cudnnSetFilter4dDescriptor(descriptors.WeightsDescriptor,
                                         cudnnDataType,
                                         CUDNN_TENSOR_NCHW,
                                         params.numberFilters,
                                         params.inputDepth,
                                         params.filterHeight,
                                         params.filterWidth));
                                         
   // Set the convolution parameters
   CUDNNCHECK(cudnnSetConvolution2dDescriptor(descriptors.LayerDescriptor,//descriptors.LayerDescriptor,
                                              params.paddingHeight,
                                              params.paddingWidth,
                                              params.strideRows,
                                              params.strideCols,
                                              1,                 //Dilation height
                                              1,                 //Dilation width
                                              CUDNN_CROSS_CORRELATION,
                                              cudnnDataType));
   
   // cuDNN decides on which algorithm to use
   cudnnConvolutionFwdAlgo_t algorithm;
   // More detailed alternative: cudnnFindConvolutionForwardAlgorithm
   CUDNNCHECK(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                  input.GetTensorDescriptor(),
                                                  descriptors.WeightsDescriptor,
                                                  descriptors.LayerDescriptor,
                                                  outputTensor.GetTensorDescriptor(),
                                                  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                  0,     // Memory limit in bytes for mode CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
                                                  &algorithm));
                                                  
   // Allocate memory for the convolution
   size_t workSpaceSizeInBytes = 0;
   CUDNNCHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                      input.GetTensorDescriptor(),
                                                      descriptors.WeightsDescriptor,
                                                      descriptors.LayerDescriptor,
                                                      outputTensor.GetTensorDescriptor(),
                                                      algorithm,
                                                      &workSpaceSizeInBytes));
                                                  
   if (workSpaceSizeInBytes) cudaMalloc(&cudnnWorkspace, workSpaceSizeInBytes*sizeof(AFloat));
   
   // Perform convolution
   CUDNNCHECK(cudnnConvolutionForward(cudnnHandle,
                                      &alpha,
                                      input.GetTensorDescriptor(),
                                      input.GetDataPointer(),
                                      descriptors.WeightsDescriptor,
                                      weights.GetDataPointer(),
                                      descriptors.LayerDescriptor,
                                      algorithm,
                                      cudnnWorkspace,
                                      workSpaceSizeInBytes,
                                      &beta,
                                      outputTensor.GetTensorDescriptor(),
                                      outputTensor.GetDataPointer()));
                                                                        
   // Apply biases
   AddConvBiases(outputTensor, biases);
   
   // Store the conv output before application of activation
   TCudnn<AFloat>::Copy(inputActivation, outputTensor);

   // Apply activation
   TCudnn<AFloat>::ActivationFunctionForward(outputTensor, activFunc, descriptors.HelperDescriptor, 0.0, 1.0, 0.0);
   
   //TCudnn<AFloat>::PrintTensor(outputTensor, "after activation");
   
   //cudaFree(cudnnWorkspace);
}

//____________________________________________________________________________
//#if 0
template<typename AFloat>
void TCudnn<AFloat>::ConvLayerBackward(Tensor_t &activationGradientsBackward,
                                       Matrix_t &weightGradients, Matrix_t &biasGradients,
                                       Tensor_t &inputActivation,
                                       Tensor_t &activationGradients,
                                       const Matrix_t &weights,
                                       const Tensor_t &activationBackward,
                                       const Tensor_t &outputTensor,
                                       EActivationFunction activFunc,
                                       const ConvDescriptors_t & descriptors,
                                       size_t /*batchSize*/,   size_t /*inputHeight*/, 
                                       size_t /*inputWidth*/,  size_t /*depth*/, 
                                       size_t /*height*/,      size_t /*width*/, 
                                       size_t /*filterDepth*/, size_t /*filterHeight*/, 
                                       size_t /*filterWidth*/, size_t /*nLocalViews*/,
                                       void * cudnnConvBwdWorkspaces, 
                                       void * cudnnFilterBwdWorkspace)
{


   activationGradients.Reshape( outputTensor.GetShape());
   weightGradients.Reshape( weights.GetShape());
   biasGradients.Reshape({ 1, outputTensor.GetShape()[1], 1, 1});   // second output dimension is number of filters
   // activationGradientsBackward.Reshape()
   // activationBackward.Reshape

   //--------------------------------------------------------------------------
   // Activation function gradient
   //--------------------------------------------------------------------------
   
   // x  : Output of previous layer without activation function             -> inputActivation
   // dx : Activation gradient to be computed                               -> activationGradients [in place op] 
   // y  : Ouput of this layer (activation applied)                         -> outputTensor
   // dy : Gradient of activation from the following layer (backpropagation)-> activationGradients
   
   //if (descriptors.HelperDescriptor)
   ActivationFunctionBackward(activationGradients, outputTensor, activationGradients, inputActivation, 
                              activFunc, descriptors.HelperDescriptor);  //y dy x dx
   
   //--------------------------------------------------------------------------
   // Network Activation gradient
   //--------------------------------------------------------------------------
   const AFloat alpha = 1.0;
   const AFloat beta  = 0.0;
   size_t  workSpaceSizeInBytes = 0;
   
   cudnnHandle_t cudnnHandle = outputTensor.GetCudnnHandle();
   cudnnConvolutionBwdDataAlgo_t backwardAlgorithm;
   
   // dx : Activation gradient to be computed                               -> activationGradients [in place op] 
   // dy : Gradient of activation from the following layer (backpropagation)-> activationGradients
   CUDNNCHECK(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
                                                       descriptors.WeightsDescriptor,
                                                       activationGradients.GetTensorDescriptor(),
                                                       descriptors.LayerDescriptor,
                                                       activationGradientsBackward.GetTensorDescriptor(),
                                                       CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                       0,
                                                       &backwardAlgorithm));
    
   CUDNNCHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                                                           descriptors.WeightsDescriptor,
                                                           activationGradients.GetTensorDescriptor(),
                                                           descriptors.LayerDescriptor,
                                                           activationGradientsBackward.GetTensorDescriptor(),
                                                           backwardAlgorithm,
                                                           &workSpaceSizeInBytes));
                                                           
   if (workSpaceSizeInBytes) cudaMalloc(&cudnnConvBwdWorkspaces, workSpaceSizeInBytes*sizeof(AFloat));
    
   CUDNNCHECK(cudnnConvolutionBackwardData(cudnnHandle,
                                           &alpha,
                                           descriptors.WeightsDescriptor,
                                           weights.GetDataPointer(),
                                           activationGradients.GetTensorDescriptor(),
                                           activationGradients.GetDataPointer(),
                                           descriptors.LayerDescriptor,
                                           backwardAlgorithm,
                                           cudnnConvBwdWorkspaces,
                                           workSpaceSizeInBytes,
                                           &beta,
                                           activationGradientsBackward.GetTensorDescriptor(),
                                           activationGradientsBackward.GetDataPointer()));
    
    //--------------------------------------------------------------------------
    // Filter gradient
    //--------------------------------------------------------------------------
    
    cudnnConvolutionBwdFilterAlgo_t backwardFilterAlgorithm;
    CUDNNCHECK(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
                                                          activationBackward.GetTensorDescriptor(),
                                                          activationGradients.GetTensorDescriptor(),
                                                          descriptors.LayerDescriptor,
                                                          descriptors.WeightsDescriptor,
                                                          CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                          0,
                                                          &backwardFilterAlgorithm));
                                                          
    CUDNNCHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                                                              activationBackward.GetTensorDescriptor(),
                                                              activationGradients.GetTensorDescriptor(),
                                                              descriptors.LayerDescriptor,
                                                              descriptors.WeightsDescriptor,
                                                              backwardFilterAlgorithm,
                                                              &workSpaceSizeInBytes));
                                                              
    if (workSpaceSizeInBytes) cudaMalloc(&cudnnFilterBwdWorkspace, workSpaceSizeInBytes*sizeof(AFloat));

    CUDNNCHECK(cudnnConvolutionBackwardFilter(cudnnHandle,
                                              &alpha,
                                              activationBackward.GetTensorDescriptor(),
                                              activationBackward.GetDataPointer(),
                                              activationGradients.GetTensorDescriptor(),
                                              activationGradients.GetDataPointer(),
                                              descriptors.LayerDescriptor,
                                              backwardFilterAlgorithm,
                                              cudnnFilterBwdWorkspace,
                                              workSpaceSizeInBytes,
                                              &beta,
                                              descriptors.WeightsDescriptor,
                                              weightGradients.GetDataPointer()));

                                              
    //--------------------------------------------------------------------------
    // Bias gradient
    //--------------------------------------------------------------------------
    
    
    CUDNNCHECK(cudnnConvolutionBackwardBias(cudnnHandle,
                                            &alpha,
                                            activationGradients.GetTensorDescriptor(),
                                            activationGradients.GetDataPointer(),
                                            &beta,
                                            biasGradients.GetTensorDescriptor(),
                                            biasGradients.GetDataPointer()));
    
   //PrintTensor(activationGradientsBackward, "dx before convolution");                                        
   /*cudnnConvolutionBackwardFilter ( weightGradients, activationBackward,  this->GetActivationGradients() )  // dw, x, dy 


    // Calculate the activation gradients of the previous layer
    CalculateConvActivationGradients(activationGradientsBackward, 
    , weights, batchSize, inputHeight, inputWidth, depth,
                                     height, width, filterDepth, filterHeight, filterWidth);


    // Calculate the weight gradients
    CalculateConvWeightGradients(weightGradients, df, activationBackward, batchSize, inputHeight, inputWidth, depth,
                                 height, width, filterDepth, filterHeight, filterWidth, nLocalViews);

    // Calculate the bias gradients
    CalculateConvBiasGradients(biasGradients, df, batchSize, depth, nLocalViews);*/
}
//# endif

//____________________________________________________________________________
/*template<typename AFloat>
void TCudnn<AFloat>::CalculateConvActivationGradients(Tensor_t &activationGradientsBackward,
                                                      const Tensor_t &df,
                                                      const Matrix_t &weights, 
                                                      size_t batchSize,
                                                      size_t inputHeight, 
                                                      size_t inputWidth, 
                                                      size_t depth, 
                                                      size_t height,
                                                      size_t width,
                                                      size_t filterDepth,
                                                      size_t filterHeight,
                                                      size_t filterWidth)
{

    
   /*if (activationGradientsBackward.size() == 0) return;

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
   }*/
//}

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
template<typename AFloat>
void TCudnn<AFloat>::AddConvBiases(Tensor_t &output,
                                   const Tensor_t &biases)
{
   TCudnn<AFloat>::ScaleAdd(output, biases);
}


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
void TCudnn<AFloat>::Downsample(Tensor_t &A,
                                Tensor_t &/*B*//*,
                                const Tensor_t &C,
                                size_t imgHeight,
                                size_t imgWidth,
                                size_t fltHeight,
                                size_t fltWidth,
                                size_t strideRows,
                                size_t strideCols)
{/*
   /*size_t depth = C.GetNrows();

   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Downsample<<<gridDims, blockDims, 0, s>>>(A.GetDataPointer(), B.GetDataPointer(),
                                                                C.GetDataPointer(), depth, imgHeight, imgWidth,
                                                                fltHeight, fltWidth, strideRows, strideCols);*/
                                                                
   /*CUDNNCHECK(cudnnSetPooling2dDescriptor(poolingDescr,
                                            CUDNN_POOLING_MAX,
    cudnnNanPropagation_t       maxpoolingNanOpt,
    int                         windowHeight,
    int                         windowWidth,
    int                         verticalPadding,
    int                         horizontalPadding,
    int                         verticalStride,
    int                         horizontalStride));                                                       
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
///
//////////////////////////////////////////////////////////////////////////////////
template<typename AFloat>
void TCudnn<AFloat>::Flatten(TCudaTensor<AFloat> &A,
                            const TCudaTensor<AFloat> &B)
{
   TCuda<AFloat>::Flatten(A,B); 
}

//____________________________________________________________________________
////////////////////////////////////////////////////////////////////////////////
/// \brief Deflatten a matrix into a vector of matrices.
///
/// \param[out] A Output matrices. Each element will be a part of the input.
/// \param[in] B Input flat matrix.
///
//////////////////////////////////////////////////////////////////////////////////
template<typename AFloat>
void TCudnn<AFloat>::Deflatten(TCudaTensor<AFloat> &A,
                              const TCudaTensor<AFloat> &B)
{
   TCuda<AFloat>::Deflatten(A,B); 
}

} // namespace DNN
} // namespace TMVA
