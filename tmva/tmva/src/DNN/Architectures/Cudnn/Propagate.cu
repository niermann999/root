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
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/CNN/ConvLayer.h"


namespace TMVA {
namespace DNN  {


   template<typename AFloat>
   TCudnn<AFloat>::TActivationFnctWorkspace::TActivationFnctWorkspace(const TParams & /*params*/, const TOptions & options) 
                                                                      : activOptions(static_cast<const typename TCudnn<AFloat>::ActivationOptions_t &>(options))
   {
      CUDNNCHECK(cudnnCreateActivationDescriptor(&activationFnctDescr));
   
       // Dont set activation function descriptor for identity function
       activationFunction = activOptions.activationFunction;
       isIdentity = activOptions.isIdentity;
       if (isIdentity) return;

       CUDNNCHECK(cudnnSetActivationDescriptor(activationFnctDescr, 
                                               activOptions.activationMode, 
                                               activOptions.nanPropagation, 
                                               activOptions.coef));
   }

   template<typename AFloat>
   TCudnn<AFloat>::TActivationFnctWorkspace::~TActivationFnctWorkspace() {
      CUDNNCHECK(cudnnDestroyActivationDescriptor(activationFnctDescr));
   }

   template<typename AFloat>
   void TCudnn<AFloat>::TActivationFnctWorkspace::DeepCopy(TWorkspace & A, const TWorkspace & B) {
      auto & newWorkspace = static_cast<TActivationFnctWorkspace &>(A);
      auto & oldWorkspace = static_cast<const TActivationFnctWorkspace &>(B);

      CUDNNCHECK(cudnnCreateActivationDescriptor(&newWorkspace.activationFnctDescr));

      newWorkspace.activationFunction = oldWorkspace.activationFunction;
      newWorkspace.isIdentity = oldWorkspace.isIdentity;
      // Dont set activation function descriptor for identity function
      if (newWorkspace.isIdentity) return;

      cudnnActivationMode_t activationMode;
      cudnnNanPropagation_t nanOpt;
      double coef;
      CUDNNCHECK(cudnnGetActivationDescriptor(oldWorkspace.activationFnctDescr,
                                              &activationMode,
                                              &nanOpt,
                                              &coef));

      CUDNNCHECK(cudnnSetActivationDescriptor(newWorkspace.activationFnctDescr, 
                                              activationMode, 
                                              nanOpt, 
                                              coef));
   }

   //____________________________________________________________________________
   template<typename AFloat>
   TCudnn<AFloat>::TBatchNormWorkspace::TBatchNormWorkspace(const TParams & params, const TOptions & options) 
                                                            : batchNormOptions(static_cast<const BatchNormOptions_t &>(options))
   {}
   template<typename AFloat>
   TCudnn<AFloat>::TBatchNormWorkspace::~TBatchNormWorkspace() {}

   template<typename AFloat>
   void TCudnn<AFloat>::TBatchNormWorkspace::DeepCopy(TWorkspace & A, const TWorkspace & B) {
      //auto & newWorkspace = static_cast<TBatchNormWorkspace &>(A);
      //auto & oldWorkspace = static_cast<const TBatchNormWorkspace &>(B);
   }

   //____________________________________________________________________________
   template<typename AFloat>
   TCudnn<AFloat>::TConvLayerWorkspace::TConvLayerWorkspace(const TParams & params, const TOptions & options)
                                                            : convOptions(static_cast<const ConvolutionOptions_t &>(options))
   {
      auto& convParams = static_cast<const DNN::CNN::TConvParams &>(params);

      CUDNNCHECK(cudnnCreateConvolutionDescriptor(&convolutionDescr));
      CUDNNCHECK(cudnnCreateFilterDescriptor(&filterDescr));
   
      // Set the convolution parameters
      CUDNNCHECK(cudnnSetConvolution2dDescriptor(convolutionDescr,
                                                 convParams.paddingHeight,
                                                 convParams.paddingWidth,
                                                 convParams.strideRows,
                                                 convParams.strideCols,
                                                 convParams.dilationHeight,
                                                 convParams.dilationWidth,
                                                 convOptions.convMode,
                                                 convOptions.cudnnDataType));

      // if using tensor math (cudnn version > 7)
      CUDNNCHECK(cudnnSetConvolutionMathType(convolutionDescr, convOptions.mathType));
   
      // Set the  filter parameters
      CUDNNCHECK(cudnnSetFilter4dDescriptor(filterDescr,
                                            convOptions.cudnnDataType,
                                            CUDNN_TENSOR_NCHW,         // Always assume NCHW memory layout
                                            convParams.numberFilters,
                                            convParams.inputDepth,
                                            convParams.filterHeight,
                                            convParams.filterWidth));
                                             
      Tensor_t inputTensor ({convParams.batchSize, convParams.inputDepth, convParams.inputHeight, convParams.inputWidth}, MemoryLayout::RowMajor, 0, 0);
                         
      size_t outputHeight = ConvLayer_t::calculateDimension(convParams.inputHeight, convParams.filterHeight, convParams.paddingHeight, convParams.strideRows);
      size_t outputWidth  = ConvLayer_t::calculateDimension(convParams.inputWidth, convParams.filterWidth, convParams.paddingWidth,  convParams.strideCols);

      cudnnTensorDescriptor_t outputTensorDescr;
      CUDNNCHECK(cudnnCreateTensorDescriptor(&outputTensorDescr));
      CUDNNCHECK(cudnnSetTensor4dDescriptor(outputTensorDescr,
                                            CUDNN_TENSOR_NCHW,         // Always assume NCHW memory layout
                                            convOptions.cudnnDataType,
                                            convParams.batchSize,
                                            convParams.numberFilters,
                                            outputHeight,
                                            outputWidth));
                                             
      // Get access to cudnn library handle, which is static for the CudaTensor class
      cudnnHandle_t cudnnHandle = inputTensor.GetCudnnHandle();
                             
      // User set algorithm
      if (convOptions.algorithmFwd > 0 && convOptions.algorithmBwd > 0 && convOptions.filterAlgorithmBwd > 0) {
         algorithmFwd = (cudnnConvolutionFwdAlgo_t) convOptions.algorithmFwd;
         algorithmBwd = (cudnnConvolutionBwdDataAlgo_t) convOptions.algorithmBwd;
         filterAlgorithmBwd = (cudnnConvolutionBwdFilterAlgo_t) convOptions.filterAlgorithmBwd;
         std::cout << " Using " << convOptions.algorithmFwd << "\t"
                   << convOptions.algorithmBwd << "\t"
                   << convOptions.filterAlgorithmBwd << std::endl;
      }
      else {
         // cuDNN decides which algorithm to use
         // More detailed alternative: cudnnFindConvolutionForwardAlgorithm
         CUDNNCHECK(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                      inputTensor.GetTensorDescriptor(),
                                                      filterDescr,
                                                      convolutionDescr,
                                                      outputTensorDescr,
                                                      convOptions.algorithmFwdPref,
                                                      convOptions.maxMemoryInBytes,  // Memory limit in bytes for mode CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
                                                      &algorithmFwd));
         //
         // Backward Algorithm
         //
         // dx : Activation gradient to be computed                               -> activationGradients [in place op] 
         // dy : Gradient of activation from the following layer (backpropagation)-> activationGradients
         CUDNNCHECK(cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
            filterDescr,
            outputTensorDescr,                  //activationGradients has same dimensions as output
            convolutionDescr,
            inputTensor.GetTensorDescriptor(),  //activationGradientsBackward has same dimensions as input tensor
            convOptions.algorithmBwdPref,
            convOptions.maxMemoryInBytes,
            &algorithmBwd));

         //
         // Filter gradient  
         //                              
         CUDNNCHECK(cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
            inputTensor.GetTensorDescriptor(),  //activationBackward has same dimensions as input tensor
            outputTensorDescr,                  //activationGradients has same dimensions as output
            convolutionDescr,
            filterDescr,
            convOptions.algorithmBwdFilterPref,
            convOptions.maxMemoryInBytes,
            &filterAlgorithmBwd));

      } 
        
      //
      // Allocate memory for the convolution algorithms
      //
      CUDNNCHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                         inputTensor.GetTensorDescriptor(),
                                                         filterDescr,
                                                         convolutionDescr,
                                                         outputTensorDescr,
                                                         algorithmFwd,
                                                         &fwdWorkspaceSize));
                                                                                            
                                            
      CUDNNCHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                                                              filterDescr,
                                                              outputTensorDescr,                  //activationGradients has same dimensions as output
                                                              convolutionDescr,
                                                              inputTensor.GetTensorDescriptor(),  //activationGradientsBackward has same dimensions as input tensor
                                                              algorithmBwd,
                                                              &bwdWorkspaceSize));
                                                                                                     
                                                                                                  
      CUDNNCHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                                                                inputTensor.GetTensorDescriptor(),  //activationBackward has same dimensions as input tensor
                                                                outputTensorDescr,                  //activationGradients has same dimensions as output
                                                                convolutionDescr,
                                                                filterDescr,
                                                                filterAlgorithmBwd,
                                                                &filterBwdWorkspaceSize));
      
      if (fwdWorkspaceSize) cudaMalloc(&fwdWorkspace, fwdWorkspaceSize*sizeof(AFloat));
      if (bwdWorkspaceSize) cudaMalloc(&bwdWorkspace, bwdWorkspaceSize*sizeof(AFloat));
      if (filterBwdWorkspaceSize) cudaMalloc(&filterBwdWorkspace, filterBwdWorkspaceSize*sizeof(AFloat));

      CUDNNCHECK(cudnnDestroyTensorDescriptor(outputTensorDescr));
                                             
   }

   template<typename AFloat>
   TCudnn<AFloat>::TConvLayerWorkspace::~TConvLayerWorkspace() {
   
      if(fwdWorkspace)       cudaFree(fwdWorkspace);
      if(bwdWorkspace)       cudaFree(bwdWorkspace);
      if(filterBwdWorkspace) cudaFree(filterBwdWorkspace);

      CUDNNCHECK(cudnnDestroyConvolutionDescriptor(convolutionDescr));
      CUDNNCHECK(cudnnDestroyFilterDescriptor(filterDescr));
      
   }

   template<typename AFloat>
   void TCudnn<AFloat>::TConvLayerWorkspace::DeepCopy(TWorkspace & A, const TWorkspace & B) {
      auto & newWorkspace = static_cast<TConvLayerWorkspace &>(A);
      auto & oldWorkspace = static_cast<const TConvLayerWorkspace &>(B);

      CUDNNCHECK(cudnnCreateConvolutionDescriptor(&newWorkspace.convolutionDescr));
      CUDNNCHECK(cudnnCreateFilterDescriptor(&newWorkspace.filterDescr));

      // Copy the convolution parameters
      int padHeight, padWidth, strideRows, strideCols, dilHeight, dilWidth;
      cudnnConvolutionMode_t convMode;
      cudnnDataType_t dataType;
      CUDNNCHECK(cudnnGetConvolution2dDescriptor(oldWorkspace.convolutionDescr, &padHeight, &padWidth, 
                                                 &strideRows, &strideCols, &dilHeight, &dilWidth,
                                                 &convMode, &dataType));
   
      CUDNNCHECK(cudnnSetConvolution2dDescriptor(newWorkspace.convolutionDescr,padHeight, padWidth,
                                                 strideRows, strideCols,dilHeight, dilWidth,
                                                 convMode, dataType));
   
      // Copy the  filter parameters
      cudnnTensorFormat_t format;
      int numFilters, inputDepth, fltHeight, fltWidth;                             
      CUDNNCHECK(cudnnGetFilter4dDescriptor(oldWorkspace.filterDescr,&dataType, &format,
                                            &numFilters,&inputDepth, &fltHeight, &fltWidth));

      CUDNNCHECK(cudnnSetFilter4dDescriptor(newWorkspace.filterDescr, dataType, format,
                                            numFilters, inputDepth, fltHeight, fltWidth));

      // Conditions should not have changed between the layer, so use the same algorithms
      newWorkspace.algorithmFwd = oldWorkspace.algorithmFwd;
      newWorkspace.fwdWorkspaceSize = oldWorkspace.fwdWorkspaceSize;
      if (newWorkspace.fwdWorkspaceSize) cudaMalloc(&newWorkspace.fwdWorkspace, 
                                                        newWorkspace.fwdWorkspaceSize*sizeof(AFloat));

      newWorkspace.algorithmBwd = oldWorkspace.algorithmBwd;
      newWorkspace.bwdWorkspaceSize = oldWorkspace.bwdWorkspaceSize;
      if (newWorkspace.bwdWorkspaceSize) cudaMalloc(&newWorkspace.bwdWorkspace, 
                                                         newWorkspace.bwdWorkspaceSize*sizeof(AFloat));

      newWorkspace.filterAlgorithmBwd = oldWorkspace.filterAlgorithmBwd;
      newWorkspace.filterBwdWorkspaceSize = oldWorkspace.filterBwdWorkspaceSize;
      if (newWorkspace.filterBwdWorkspaceSize) cudaMalloc(&newWorkspace.filterBwdWorkspace,
                                                               newWorkspace.filterBwdWorkspaceSize*sizeof(AFloat));
   }

   //____________________________________________________________________________
   template<typename AFloat>
   TCudnn<AFloat>::TDropoutWorkspace::TDropoutWorkspace(const TParams & params, const TOptions & /*options*/)
   {
      auto& dropoutParams = static_cast<const DNN::CNN::TPoolParams &>(params);
   
      Tensor_t inputTensor ({dropoutParams.batchSize, dropoutParams.inputDepth, dropoutParams.inputHeight, dropoutParams.inputWidth}, MemoryLayout::RowMajor, 0, 0);
      cudnnHandle_t cudnnHandle = inputTensor.GetCudnnHandle();
      
      // Space needed to execute forward and backward dropout pass
      CUDNNCHECK(cudnnDropoutGetReserveSpaceSize(inputTensor.GetTensorDescriptor(),
                                                 &dropoutReserveSpaceSize));
   
      if (dropoutReserveSpaceSize) cudaMalloc(&dropoutReserveSpace, dropoutReserveSpaceSize*sizeof(AFloat));
      
      // Space that contains random states                                           
      CUDNNCHECK(cudnnDropoutGetStatesSize(cudnnHandle, &dropoutStatesSpaceSize));
   
      if (dropoutStatesSpaceSize) cudaMalloc(&dropoutStatesSpace, dropoutStatesSpaceSize*sizeof(AFloat));
   
      // Fill the dropout workspace with random numbers and copy to device
      TRandom &  rand = TCudnn<AFloat>::GetRandomGenerator();
      unsigned long long seed = 5496729; //FIXME: rand.Uniform(0, ULLONG_MAX);

      CUDNNCHECK(cudnnCreateDropoutDescriptor(&dropoutDescr));
      // FIXME: Reset the descriptor at every forward pass, so that random states get newly initialized?
      CUDNNCHECK(cudnnSetDropoutDescriptor(dropoutDescr,
                                           cudnnHandle,
                                           dropoutParams.dropoutProbability,
                                           dropoutStatesSpace,
                                           dropoutStatesSpaceSize,
                                           seed));
   
   }


   template<typename AFloat>
   TCudnn<AFloat>::TDropoutWorkspace::~TDropoutWorkspace() {
      if(dropoutReserveSpace) cudaFree(dropoutReserveSpace);
      if(dropoutStatesSpace)  cudaFree(dropoutStatesSpace);

      CUDNNCHECK(cudnnDestroyDropoutDescriptor(dropoutDescr));
   }

   template<typename AFloat>
   void TCudnn<AFloat>::TDropoutWorkspace::DeepCopy(TWorkspace & A, const TWorkspace & B) {
      auto & newWorkspace = static_cast<TDropoutWorkspace &>(A);
      auto & oldWorkspace = static_cast<const TDropoutWorkspace &>(B);

      // Conditions should be the same, so use same space sizes
      newWorkspace.dropoutReserveSpaceSize = oldWorkspace.dropoutReserveSpaceSize;
   
      if (newWorkspace.dropoutReserveSpaceSize) cudaMalloc(&newWorkspace.dropoutReserveSpace, 
                                                           newWorkspace.dropoutReserveSpaceSize*sizeof(AFloat));
                                          
      newWorkspace.dropoutStatesSpaceSize = oldWorkspace.dropoutStatesSpaceSize;
   
      if (newWorkspace.dropoutStatesSpaceSize) cudaMalloc(&newWorkspace.dropoutStatesSpace, 
                                                          newWorkspace.dropoutStatesSpaceSize*sizeof(AFloat));

      // New descriptor
      CUDNNCHECK(cudnnCreateDropoutDescriptor(&newWorkspace.dropoutDescr));
      
      // Get the current cudnn handle. FIXME: Take care once multiple handles are used!!!
      Tensor_t dummy = Tensor_t();
      cudnnHandle_t cudnnHandle = dummy.GetCudnnHandle();
      float dropoutProb;
      void * statesSpace;
      unsigned long long seed;
      CUDNNCHECK(cudnnGetDropoutDescriptor(oldWorkspace.dropoutDescr, cudnnHandle,
                                           &dropoutProb, &statesSpace, &seed));

      // Prepare new random states
      TRandom &  rand = TCudnn<AFloat>::GetRandomGenerator();
      seed = 28957203958; //FIXME: rand.Uniform(0, ULLONG_MAX);

      CUDNNCHECK(cudnnSetDropoutDescriptor(dropoutDescr, cudnnHandle, dropoutProb,
                                           newWorkspace.dropoutStatesSpace,
                                           newWorkspace.dropoutStatesSpaceSize,
                                           seed));
   }

   //____________________________________________________________________________
   template<typename AFloat>
   TCudnn<AFloat>::TPoolingWorkspace::TPoolingWorkspace(const TParams & params, const TOptions & options)
                                                        : poolingOptions(static_cast<const PoolingOptions_t &>(options))
   {
      auto& poolingParams = static_cast<const DNN::CNN::TPoolParams &>(params);

      CUDNNCHECK(cudnnCreatePoolingDescriptor(&poolingDescr));
   
      CUDNNCHECK(cudnnSetPooling2dDescriptor(poolingDescr,
                                             poolingOptions.poolingMode,
                                             poolingOptions.nanPropagation,
                                             poolingParams.filterHeight,
                                             poolingParams.filterWidth,
                                             poolingParams.paddingHeight,
                                             poolingParams.paddingWidth,
                                             poolingParams.strideRows,
                                             poolingParams.strideCols));
   
   }


   template<typename AFloat>
   TCudnn<AFloat>::TPoolingWorkspace::~TPoolingWorkspace() {
      CUDNNCHECK(cudnnDestroyPoolingDescriptor(poolingDescr));
   }

   template<typename AFloat>
   void TCudnn<AFloat>::TPoolingWorkspace::DeepCopy(TWorkspace & A, const TWorkspace & B) {
      auto & newWorkspace = static_cast<TPoolingWorkspace &>(A);
      auto & oldWorkspace = static_cast<const TPoolingWorkspace &>(B);

      cudnnPoolingMode_t poolingMode;
      cudnnNanPropagation_t nanOpt;
      int fltHeight, fltWidth, padHeight, padWidth, strideRows, strideCols;
      CUDNNCHECK(cudnnGetPooling2dDescriptor(oldWorkspace.poolingDescr, &poolingMode, &nanOpt,
                                             &fltHeight, &fltWidth, &padHeight, &padWidth,
                                             &strideRows, &strideCols));
                                             
      CUDNNCHECK(cudnnCreatePoolingDescriptor(&newWorkspace.poolingDescr));
   
      CUDNNCHECK(cudnnSetPooling2dDescriptor(newWorkspace.poolingDescr, poolingMode, nanOpt,
                                             fltHeight, fltWidth, padHeight, padWidth,
                                             strideRows, strideCols));
   }

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::MultiplyTranspose(TCudaTensor<AFloat> &output,
                                       const TCudaTensor<AFloat> &input,
                                       const TCudaTensor<AFloat> &weights)
{
   //PrintTensor(input,"input to MultTrans");
   //PrintTensor(weights,"dense layer  weights");
   TCuda<AFloat>::MultiplyTranspose(output, input, weights.GetMatrix());
   //PrintTensor(input,"output of  MultTrans");
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

   //std::cout << "\n\n ------ Backward--------\n";
   //PrintTensor(activation_backward,"input to backward");
   //PrintTensor(weights,"dense layer  weights");
   //PrintTensor(activation_gradients,"input dy");
   //PrintTensor(activation_gradients,"df");

   TCudaMatrix<AFloat> weightGradMatrix = weight_gradients.GetMatrix(); 
   TCudaMatrix<AFloat> biasGradMatrix = bias_gradients.GetMatrix(); 

   TCuda<AFloat>::Backward(activation_gradients_backward,
                              weightGradMatrix,
                              biasGradMatrix,
                              df,
                              activation_gradients,
                              weights.GetMatrix(), 
                              activation_backward);

   //PrintTensor(activation_gradients_backward,"computed dx");
   //PrintTensor(weight_gradients,"computed dw");
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


///////////////////////////////////////////////////////////////////////////////
// Initialization of the cuDNN objects for the different layers
// ...
///////////////////////////////////////////////////////////////////////////////
#if 0
template<typename AFloat>
void TCudnn<AFloat>::InitializeBNormDescriptors(TDescriptors * & descriptors, ConvLayer_t *L) 
{
   auto bnormDescriptors = new CNN::TCNNDescriptors<typename TCudnn<AFloat>::BNormLayer_t> ();

   //FIXME: Move this to constructor
   cudnnDataType_t   cudnnDataType;
   if      (std::is_same<AFloat, double>::value) { cudnnDataType = CUDNN_DATA_DOUBLE;}
   else if (std::is_same<AFloat, float>::value)  { cudnnDataType = CUDNN_DATA_FLOAT;}

   Tensor_t inputTensor  ({L->GetBatchSize(), L->GetInputDepth(), L->GetInputHeight(), L->GetInputWidth()}, MemoryLayout::RowMajor, 0, 0);
   
   // derived BNormdescr
   CUDNNCHECK(cudnnCreateTensorDescriptor(&bnormDescriptors->HeplerDescriptor));

   CUDNNCHECK(cudnnDeriveBNTensorDescriptor(bnormDescriptors->HeplerDescriptor
                                            inputTensor.GetTensorDescriptor(),
                                            CUDNN_BATCHNORM_PER_ACTIVATION));

   descriptors = bnormDescriptors;
}

   // fix the weight tensor shapes 
   // by default the weights are columnmajor, set them to be row major . At this points 
   // they are not yet initialized 
   Tensor_t & filters = L->GetWeightsAt(0); 
   filters = Tensor_t (filters.GetDeviceBuffer(), {L->GetDepth(),L->GetInputDepth(), L->GetFilterHeight(),L->GetFilterWidth()}, MemoryLayout::RowMajor, 0, 0 );
   //PrintTensor(L->GetWeightsAt(0)); 
   Tensor_t & biases = L->GetBiasesAt(0);
   biases = Tensor_t (biases.GetDeviceBuffer(), {1, L->GetDepth(),1,1}, GetTensorLayout(), 0, 0 );

   Tensor_t & outputTensor = L->GetOutput(); 
   outputTensor = Tensor_t(outputTensor.GetDeviceBuffer(),{ L->GetBatchSize(), L->GetDepth(), L->GetHeight(), L->GetWidth() },GetTensorLayout(),0,0 );
   Tensor_t & inputActivation = L->GetInputActivation(); 
   inputActivation = Tensor_t(inputActivation.GetDeviceBuffer(),outputTensor.GetShape() ,GetTensorLayout(),0,0 );

   Tensor_t &  activationGradients = L->GetActivationGradients();
   activationGradients =  Tensor_t(activationGradients.GetDeviceBuffer(),outputTensor.GetShape() ,GetTensorLayout(),0,0 );
   
   Tensor_t & weightGradients = L->GetWeightGradientsAt(0); 
   weightGradients = Tensor_t( weightGradients.GetDeviceBuffer(), filters.GetShape(), GetTensorLayout(), 0, 0 ); 
   
   Tensor_t & biasGradients = L->GetBiasGradientsAt(0); 
   biasGradients = Tensor_t( biasGradients.GetDeviceBuffer(), biases.GetShape(), GetTensorLayout(), 0, 0 ); 
   

   // FIXME: Use descriptors instead (Tensor device memory is otherwise allocated during initialization)
   Tensor_t inputTensor  ({L->GetBatchSize(), L->GetInputDepth(), L->GetInputHeight(), L->GetInputWidth()}, MemoryLayout::RowMajor, 0, 0);

   // size_t outputHeight = ConvLayer_t::calculateDimension(L->GetInputHeight(), L->GetFilterHeight(), L->GetPaddingHeight(), L->GetStrideRows());
   // size_t outputWidth  = ConvLayer_t::calculateDimension(L->GetInputWidth(), L->GetFilterWidth(), L->GetPaddingWidth(),  L->GetStrideCols());
   //Tensor_t outputTensor ({L->GetBatchSize(), L->GetDepth(), outputHeight, outputWidth}, MemoryLayout::RowMajor, 0, 0);
   
   // Get access to cudnn library handle, which is static for the CudaTensor class

#endif
//____________________________________________________________________________
template <typename AFloat>
void TCudnn<AFloat>::BatchNormLayerForwardTraining(Matrix_t input,
                                          Matrix_t & gamma,
                                          Matrix_t & beta,
                                          Matrix_t outputActivation,
                                          Matrix_t & Xmu,
                                          Matrix_t & output,
                                          Matrix_t & Variance,
                                          Matrix_t & IVariance,
                                          # if 0
                                          const BNormDescriptors_t & descriptors,
                                          BNormWorkspace_t & workspace,
                                          # endif
                                          std::vector<Scalar_t> & RunningMeans,
                                          std::vector<Scalar_t> & RunningVars,
                                          Scalar_t nTrainedBatches,
                                          Scalar_t momentum,
                                          Scalar_t epsilon)
{
   /*AFloat a = 1.0;
   AFloat b = 0.0;
   CUDNNCHECK(cudnnBatchNormalizationForwardTraining(input.GetCudnnHandle(),
      cudnnBatchNormMode_t             mode,
                                                     &alpha,
                                                     &beta,
                                                     input.GetTensorDescriptor(),
                                                     input.GetDataPointer(),
      const cudnnTensorDescriptor_t    yDesc,
      void                            *y,
      const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,
      const void                      *bnScale,
      const void                      *bnBias,
      double                           exponentialAverageFactor,
      void                            *resultRunningMean,
      void                            *resultRunningVariance,
                                                      epsilon,
      void                            *resultSaveMean,
      void                            *resultSaveInvVariance));*/
}

//____________________________________________________________________________

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
template<typename AFloat>
void TCudnn<AFloat>::PrepareInternals(Tensor_t & output, Tensor_t & inputActivation, Matrix_t & weights,
                                      Tensor_t & biases, Tensor_t & weightGradients, Tensor_t & biasGradients, 
                                      Tensor_t & activationGradients, const DNN::CNN::TConvParams & params) 
{
   // fix the weight tensor shapes 
   // by default the weights are columnmajor, set them to be row major . At this points 
   // they are not yet initialized 
   weights = Matrix_t (weights.GetDeviceBuffer(), {params.numberFilters, params.inputDepth, params.filterHeight, params.filterWidth}, MemoryLayout::RowMajor, 0, 0 );

   biases = Tensor_t (biases.GetDeviceBuffer(), {1, params.numberFilters, 1, 1}, GetTensorLayout(), 0, 0 );
   
   size_t outputHeight = ConvLayer_t::calculateDimension(params.inputHeight, params.filterHeight, params.paddingHeight, params.strideRows);
   size_t outputWidth  = ConvLayer_t::calculateDimension(params.inputWidth, params.filterWidth, params.paddingWidth, params.strideCols);
   output = Tensor_t(output.GetDeviceBuffer(),{params.batchSize, params.numberFilters, outputHeight, outputWidth },GetTensorLayout(),0,0 );

   inputActivation = Tensor_t(inputActivation.GetDeviceBuffer(), output.GetShape(), GetTensorLayout(),0,0 );
                                          
   activationGradients = Tensor_t(activationGradients.GetDeviceBuffer(), output.GetShape(), GetTensorLayout(),0,0 );
                                             
   weightGradients = Tensor_t(weightGradients.GetDeviceBuffer(), weights.GetShape(), GetTensorLayout(), 0, 0 ); 

   biasGradients = Tensor_t(biasGradients.GetDeviceBuffer(), biases.GetShape(), GetTensorLayout(), 0, 0 );
}

//____________________________________________________________________________
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
/*template <typename AFloat>
using ConvDescriptors_t       =  CNN::TCNNDescriptors<CNN::TConvLayer<TCudnn<AFloat>>>;*/

template <typename AFloat>
void TCudnn<AFloat>::ConvLayerForward(Tensor_t & outputTensor,
                                      Tensor_t & inputActivation,
                                      const Tensor_t & input,
                                      const Matrix_t & weights, const Matrix_t & biases,
                                      const DNN::CNN::TConvParams & params, 
                                      Tensor_t & inputPrime,
                                      const ActivationWorkspace_t & activWorkspace,
                                      const ConvolutionWorkspace_t & convWorkspace)
//                                    const AFloat alpha,
//                                    const AFloat beta)
{
   //((Tensor_t & )input).Reshape( {params.batchSize, params.inputDepth, params.inputHeight, params.inputWidth});
   assert( input.GetLayout() == GetTensorLayout()); 

   size_t outputHeight =  DNN::CNN::TConvLayer<TCudnn<AFloat>>::calculateDimension(params.inputHeight, params.filterHeight, params.paddingHeight, params.strideRows);
   size_t outputWidth =  DNN::CNN::TConvLayer<TCudnn<AFloat>>::calculateDimension(params.inputWidth, params.filterWidth, params.paddingWidth, params.strideCols);

   // PrintTensor(input,"input");
   // PrintTensor(outputTensor,"output");
   // PrintTensor(weights,"weights"); 
   // PrintTensor(biases,"biases");
   //((Tensor_t & )weights).Reshape( { params.numberFilters, params.inputDepth, params.filterHeight, params.filterWidth } );
   //((Tensor_t & )biases).Reshape(  { 1,params.numberFilters, 1, 1});
   //biases.Reshape ( { 1,params.numberFilters, 1, 1});

   AFloat alpha = 1.0; 
   AFloat beta  = 0.0; 
   cudnnHandle_t cudnnHandle = input.GetCudnnHandle();

   // check descriptors 
   int n,c,h,w = 0; 
   int s1,s2,s3,s4 = 0; 
   cudnnDataType_t  dataType; 
   cudnnGetTensor4dDescriptor( input.GetTensorDescriptor(), &dataType,&n,&c,&h,&w,&s1,&s2,&s3,&s4 );
   std::vector<size_t>  shape_input = {n,c,h,w}; 
   assert (shape_input == input.GetShape());

   cudnnGetTensor4dDescriptor( outputTensor.GetTensorDescriptor(), &dataType,&n,&c,&h,&w,&s1,&s2,&s3,&s4 );
   std::vector<size_t>  shape_output = {n,c,h,w}; 
   assert (shape_output == outputTensor.GetShape());

   // Perform convolution
   CUDNNCHECK(cudnnConvolutionForward(cudnnHandle,
                                      &alpha,
                                      input.GetTensorDescriptor(),
                                      input.GetDataPointer(),
                                      convWorkspace.filterDescr,
                                      weights.GetDataPointer(),
                                      convWorkspace.convolutionDescr,
                                      convWorkspace.algorithmFwd,
                                      convWorkspace.fwdWorkspace,
                                      convWorkspace.fwdWorkspaceSize,
                                      &beta,
                                      outputTensor.GetTensorDescriptor(),
                                      outputTensor.GetDataPointer()));

   // Apply biases
   AddConvBiases(outputTensor, biases);

   // Store the conv output before application of activation to use in the backward pass
   TCudnn<AFloat>::Copy(inputActivation, outputTensor);

   // Apply activation
   TCudnn<AFloat>::ActivationFunctionForward(outputTensor, activWorkspace, 0.0, 1.0, 0.0);
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::ConvLayerBackward(Tensor_t &activationGradientsBackward,
                                       Matrix_t &weightGradients, Matrix_t &biasGradients,
                                       Tensor_t &inputActivation,
                                       Tensor_t &activationGradients,
                                       const Matrix_t &weights,
                                       const Tensor_t &activationBackward,
                                       const Tensor_t &outputTensor,
                                       const ActivationWorkspace_t & activWorkspace,
                                       const ConvolutionWorkspace_t & convWorkspace,
                                       size_t /*batchSize*/,   size_t /*inputHeight*/, 
                                       size_t /*inputWidth*/,  size_t /*depth*/, 
                                       size_t /*height*/,      size_t /*width*/, 
                                       size_t /*filterDepth*/, size_t /*filterHeight*/, 
                                       size_t /*filterWidth*/, size_t /*nLocalViews*/)
{
   // activationGradients.Reshape( outputTensor.GetShape());
   // weightGradients.Reshape( weights.GetShape());
   // biasGradients.Reshape({ 1, outputTensor.GetShape()[1], 1, 1});   // second output dimension is number of filters
   // // activationGradientsBackward.Reshape()
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
                              activWorkspace);  //y dy x dx
   
   //--------------------------------------------------------------------------
   // Network Activation gradient
   //--------------------------------------------------------------------------
   const AFloat alpha = 1.0;
   const AFloat beta  = 0.0;
   
   cudnnHandle_t cudnnHandle = outputTensor.GetCudnnHandle();
    
   // do not compute activation gradients for first layer (i.e. when input activationGradientBackward is a dummy empty tensor)
   if (activationGradientsBackward.GetSize() > 0)
      CUDNNCHECK(cudnnConvolutionBackwardData(cudnnHandle,
                                              &alpha,
                                              convWorkspace.filterDescr,
                                              weights.GetDataPointer(),
                                              activationGradients.GetTensorDescriptor(),
                                              activationGradients.GetDataPointer(),
                                              convWorkspace.convolutionDescr,
                                              convWorkspace.algorithmBwd,
                                              convWorkspace.bwdWorkspace,
                                              convWorkspace.bwdWorkspaceSize,
                                              &beta,
                                              activationGradientsBackward.GetTensorDescriptor(),
                                              activationGradientsBackward.GetDataPointer()));
    
    //--------------------------------------------------------------------------
    // Filter gradient
    //--------------------------------------------------------------------------

    CUDNNCHECK(cudnnConvolutionBackwardFilter(cudnnHandle,
                                              &alpha,
                                              activationBackward.GetTensorDescriptor(),
                                              activationBackward.GetDataPointer(),
                                              activationGradients.GetTensorDescriptor(),
                                              activationGradients.GetDataPointer(),
                                              convWorkspace.convolutionDescr,
                                              convWorkspace.filterAlgorithmBwd,
                                              convWorkspace.filterBwdWorkspace,
                                              convWorkspace.filterBwdWorkspaceSize,
                                              &beta,
                                              convWorkspace.filterDescr,
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
}

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
template<typename AFloat>
void TCudnn<AFloat>::Downsample(Tensor_t &A, Tensor_t &/*B*/, const Tensor_t &C,
                                const PoolingWorkspace_t & poolingWorkspace,
                                size_t /*imgHeight*/, size_t /*imgWidth*/,
                                size_t /*fltHeight*/, size_t /*fltWidth*/,
                                size_t /*strideRows*/, size_t /*strideCols*/)
{
   const AFloat alpha = 1.0;
   const AFloat beta = 0.0;

   cudnnDataType_t dataTypeC;
   int nC, cC, hC, wC, nStrideC, cStrideC, hStrideC, wStrideC;
   CUDNNCHECK(cudnnGetTensor4dDescriptor(C.GetTensorDescriptor(),
                                         &dataTypeC,
                                         &nC,
                                         &cC,
                                         &hC,
                                         &wC,
                                         &nStrideC,
                                         &cStrideC,
                                         &hStrideC,
                                         &wStrideC));

   std::cout << dataTypeC << "\t" << nC << "\t" << cC << "\t" << hC << "\t" << wC << "\t" << nStrideC << "\t"
    << cStrideC << "\t" << hStrideC << "\t" << wStrideC << std::endl;

    cudnnDataType_t dataTypeA;
   int nA, cA, hA, wA, nStrideA, cStrideA, hStrideA, wStrideA;
   CUDNNCHECK(cudnnGetTensor4dDescriptor(A.GetTensorDescriptor(),
                                         &dataTypeA,
                                         &nA,
                                         &cA,
                                         &hA,
                                         &wA,
                                         &nStrideA,
                                         &cStrideA,
                                         &hStrideA,
                                         &wStrideA));

   std::cout << dataTypeA << "\t" << nA << "\t" << cA << "\t" << hA << "\t" << wA << "\t" << nStrideA << "\t"
    << cStrideA << "\t" << hStrideA << "\t" << wStrideA << std::endl;

   CUDNNCHECK(cudnnPoolingForward(C.GetCudnnHandle(),
                                  poolingWorkspace.poolingDescr,
                                  &alpha,
                                  C.GetTensorDescriptor(),
                                  C.GetDataPointer(),
                                  &beta,
                                  A.GetTensorDescriptor(),
                                  A.GetDataPointer()));
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::MaxPoolLayerBackward(Tensor_t & activationGradientsBackward, //dx
                                         const Tensor_t & activationGradients, // dy
                                         const Tensor_t & indexMatrix,
                                         const Tensor_t & activationBackward,  //X
                                         const Tensor_t & outputTensor,        //Y
                                         const PoolingWorkspace_t & poolingWorkspace,
                                         size_t imgHeight,
                                         size_t imgWidth,
                                         size_t fltHeight,
                                         size_t fltWidth,
                                         size_t strideRows,
                                         size_t strideCols,
                                         size_t /* nLocalViews */)
{
   const AFloat alpha = 1.0;
   const AFloat beta = 0.0;
   // x  : Output of previous layer without                                 -> inputActivation
   // dx : Activation gradient to be computed                               -> activationGradientsBackward
   // y  : Ouput of this layer (activation applied)                         -> outputTensor
   // dy : Gradient of activation from the following layer (backpropagation)-> activationGradients
   CUDNNCHECK(cudnnPoolingBackward(outputTensor.GetCudnnHandle(),
                                   poolingWorkspace.poolingDescr,
                                   &alpha,
                                   outputTensor.GetTensorDescriptor(),
                                   outputTensor.GetDataPointer(),
                                   activationGradients.GetTensorDescriptor(),
                                   activationGradients.GetDataPointer(),
                                   activationBackward.GetTensorDescriptor(),
                                   activationBackward.GetDataPointer(),
                                   &beta,
                                   activationGradientsBackward.GetTensorDescriptor(),
                                   activationGradientsBackward.GetDataPointer()));
}

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

//_________________________________________TCudaMatrix<AFloat> weightGradMatrix = weight_gradients.GetMatrix(); ____________________________
///////////////////////////////////////////TCudaMatrix<AFloat> weightGradMatrix = weight_gradients.GetMatrix(); //////////////////////////////
/// \brief Deflatten a matrix into a vectorTCudaMatrix<AFloat> weightGradMatrix = weight_gradients.GetMatrix(); rices.
///
/// \param[out] A Output matrices. Each eleTCudaMatrix<AFloat> weightGradMatrix = weight_gradients.GetMatrix(); ll be a part of the input.
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
