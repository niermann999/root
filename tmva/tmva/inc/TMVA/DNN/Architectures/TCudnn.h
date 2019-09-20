// @(#)root/tmva/tmva/dnn:$Id$
// Author: Joana Niermann 23/07/19

/*************************************************************************
 * Copyright (C) 2019, Joana Niermann                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Definition of the TCudnn architecture class, which provides   //
// a wrapping of the low-level functionality for neural networks //
// in the cuDNN library.                                         //
///////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CUDNN
#define TMVA_DNN_ARCHITECTURES_CUDNN

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/CNN/ContextHandles.h"
//#include "TMVA/DNN/CNN/Descriptors.h"
//#include "TMVA/DNN/BatchNormLayer.h"
#include "TMVA/DNN/CNN/ConvLayer.h"
#include "TMVA/DNN/CNN/MaxPoolLayer.h"

#include "cudnn.h"
#include "Cuda/CudaBuffers.h"
#include "Cuda/CudaTensor.h"
#include "TMVA/DNN/TensorDataLoader.h"
#include <utility>
#include <vector>

class TRandom;

namespace TMVA
{
namespace DNN
{
 
/** The TCudnn architecture class.
 *
 * Low-level interface class for CUDA computing architectures using the cuDNN
 * library as backend. Contains as public types the declaration of the scalar, 
 * matrix and buffer types for this architecture, as well as the remaining 
 * functions in the low-level interface in the form of static members.
 */
template<typename AFloat>
class TCudnn
{
private:
   static TRandom * fgRandomGen;
public:

   using Scalar_t       = AFloat;
   using Matrix_t       = TCudaTensor<AFloat>;
   using Tensor_t       = TCudaTensor<AFloat>;
   using DeviceBuffer_t = TCudaDeviceBuffer<AFloat>;
   using HostBuffer_t   = TCudaHostBuffer<AFloat>;

   //____________________________________________________________________________
   //
   // Architecture Initialization
   //____________________________________________________________________________
 

   typedef struct TActivationOptions : public TOptions {
      cudnnActivationMode_t activationMode;
      EActivationFunction   activationFunction;

      bool isIdentity = false;
      double coef;

      cudnnNanPropagation_t nanPropagation;
      
      TActivationOptions(EActivationFunction _activationFunction = EActivationFunction::kTanh,
                         double _coef = 0.0, cudnnNanPropagation_t _nanPropagation = CUDNN_PROPAGATE_NAN) 
                         : activationFunction(_activationFunction), coef(_coef), nanPropagation(_nanPropagation)
      {
         switch(activationFunction) {
            // cuDNN identity activation only works for cudnnConvolutionBiasActivationForward()
            case EActivationFunction::kIdentity: isIdentity = true; return; 
            case EActivationFunction::kRelu:     activationMode = CUDNN_ACTIVATION_RELU;    break;
            case EActivationFunction::kSigmoid:  activationMode = CUDNN_ACTIVATION_SIGMOID; break;
            case EActivationFunction::kTanh:     activationMode = CUDNN_ACTIVATION_TANH;    break;
            // The activations otherwise used are not supported by cuDNN
            default:   activationFunction = EActivationFunction::kTanh;
                       activationMode = CUDNN_ACTIVATION_TANH; 
         };
      }

   } TActivationOptions;


   typedef struct TBatchNormOptions : public TOptions {} TBatchNormOptions;


   typedef struct TConvOptions       : public TOptions {
      cudnnDataType_t        cudnnDataType;
      cudnnMathType_t        mathType;
      cudnnNanPropagation_t  nanPropagation;
      cudnnConvolutionMode_t convMode;

      cudnnConvolutionFwdPreference_t       algorithmFwdPref;
      cudnnConvolutionBwdDataPreference_t   algorithmBwdPref;
      cudnnConvolutionBwdFilterPreference_t algorithmBwdFilterPref;

      // Set algorithms directly
      cudnnConvolutionFwdAlgo_t       algorithmFwd;
      cudnnConvolutionBwdDataAlgo_t   algorithmBwd;
      cudnnConvolutionBwdFilterAlgo_t filterAlgorithmBwd;

      size_t maxMemoryInBytes;

      TConvOptions(cudnnDataType_t _cudnnDataType = CUDNN_DATA_DOUBLE, cudnnMathType_t _mathType = CUDNN_TENSOR_OP_MATH,
                   cudnnNanPropagation_t _nanPropagation = CUDNN_PROPAGATE_NAN,
                   cudnnConvolutionMode_t _convMode = CUDNN_CROSS_CORRELATION,  int _maxMemoryInBytes = 0,
                   cudnnConvolutionFwdPreference_t _algorithmFwdPref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                   cudnnConvolutionBwdDataPreference_t _algorithmBwdPref = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 
                   cudnnConvolutionBwdFilterPreference_t _algorithmBwdFilterPref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                   cudnnConvolutionFwdAlgo_t _algorithmFwd = (cudnnConvolutionFwdAlgo_t)0,
                   cudnnConvolutionBwdDataAlgo_t _algorithmBwd = (cudnnConvolutionBwdDataAlgo_t )0, 
                   cudnnConvolutionBwdFilterAlgo_t _filterAlgorithmBwd = (cudnnConvolutionBwdFilterAlgo_t)0)
                   : cudnnDataType(_cudnnDataType), mathType(_mathType), nanPropagation(_nanPropagation), convMode(_convMode),
                     algorithmFwdPref(_algorithmFwdPref), algorithmBwdPref(_algorithmBwdPref),
                     algorithmBwdFilterPref(_algorithmBwdFilterPref), algorithmFwd(_algorithmFwd),
                     algorithmBwd(_algorithmBwd), filterAlgorithmBwd(_filterAlgorithmBwd)
      {
         if (_maxMemoryInBytes < 0) {
            algorithmFwdPref       = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
            algorithmBwdPref       = CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
            algorithmBwdFilterPref = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
         }
         if (_maxMemoryInBytes > 0) {
            algorithmFwdPref       = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
            algorithmBwdPref       = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
            algorithmBwdFilterPref = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
         }
                                                   
         maxMemoryInBytes = (_maxMemoryInBytes > 0) ? (size_t) _maxMemoryInBytes / 3 : 0;
      }

   } TConvOptions;


   typedef struct TPoolingOptions    : public TOptions {
      cudnnNanPropagation_t nanPropagation;
      cudnnPoolingMode_t    poolingMode;

      //TPoolingOptions() = delete;
      TPoolingOptions(cudnnNanPropagation_t _nanPropagation = CUDNN_PROPAGATE_NAN,
                      cudnnPoolingMode_t _poolingMode = CUDNN_POOLING_MAX)
                      : nanPropagation(_nanPropagation), poolingMode(_poolingMode)
      {}

   } TPoolingOptions;


   using ActivationOptions_t  = TActivationOptions;
   using BatchNormOptions_t   = TBatchNormOptions;
   using ConvolutionOptions_t = TConvOptions;
   using PoolingOptions_t     = TPoolingOptions;


   typedef struct TActivationFnctWorkspace : public TWorkspace {
      const ActivationOptions_t &  activOptions;
      cudnnActivationDescriptor_t  activationFnctDescr;
      EActivationFunction          activationFunction;
      bool isIdentity;
   
      TActivationFnctWorkspace() = default;
      TActivationFnctWorkspace(const TParams & params,  const TOptions & userOptions);
      ~TActivationFnctWorkspace();

      void DeepCopy(TWorkspace & A, const TWorkspace & B) override;
   } TActivationFnctWorkspace;


   typedef struct TBatchNormWorkspace : public TWorkspace {
      const BatchNormOptions_t & batchNormOptions;

      TBatchNormWorkspace(const TParams & params,  const TOptions & userOptions);
      ~TBatchNormWorkspace();

      void DeepCopy(TWorkspace & A, const TWorkspace & B) override;
   } TBatchNormWorkspace;


   typedef struct TConvLayerWorkspace : public TWorkspace {
      const ConvolutionOptions_t & convOptions;

      cudnnConvolutionDescriptor_t convolutionDescr;
      cudnnFilterDescriptor_t      filterDescr;

      cudnnConvolutionFwdAlgo_t       algorithmFwd;
      cudnnConvolutionBwdDataAlgo_t   algorithmBwd;
      cudnnConvolutionBwdFilterAlgo_t filterAlgorithmBwd;
   
      size_t * fwdWorkspace;
      size_t * bwdWorkspace;
      size_t * filterBwdWorkspace;
   
      size_t fwdWorkspaceSize;
      size_t bwdWorkspaceSize;
      size_t filterBwdWorkspaceSize;

      TConvLayerWorkspace(const TParams & params,  const TOptions & userOptions);
      ~TConvLayerWorkspace();

      void DeepCopy(TWorkspace & A, const TWorkspace & B) override;
   } TConvLayerWorkspace;


   typedef struct TDropoutWorkspace : public TWorkspace {
      cudnnDropoutDescriptor_t dropoutDescr;

      size_t * dropoutReserveSpace;
      size_t * dropoutStatesSpace;

      size_t dropoutReserveSpaceSize;
      size_t dropoutStatesSpaceSize;
   
      TDropoutWorkspace(const TParams & params,  const TOptions & userOptions);
      ~TDropoutWorkspace();

      void DeepCopy(TWorkspace & A, const TWorkspace & B) override;
   } TDropoutWorkspace;
   

   typedef struct TPoolingWorkspace : public TWorkspace {
      const PoolingOptions_t & poolingOptions;

      cudnnPoolingDescriptor_t poolingDescr;
   
      TPoolingWorkspace(const TParams & params,  const TOptions & userOptions);
      ~TPoolingWorkspace();

      void DeepCopy(TWorkspace & A, const TWorkspace & B) override;
   } TPoolingWorkspace;

   using ActivationWorkspace_t  = TActivationFnctWorkspace;
   using BatchNormWorkspace_t   = TBatchNormWorkspace;
   using ConvolutionWorkspace_t = TConvLayerWorkspace;
   using DropoutWorkspace_t     = TDropoutWorkspace;
   using PoolingWorkspace_t     = TPoolingWorkspace;

   using ConvLayer_t = CNN::TConvLayer<TCudnn<AFloat>>;

   static TMVA::Experimental::MemoryLayout GetTensorLayout() { return TMVA::Experimental::MemoryLayout::RowMajor; }


   static Tensor_t CreateTensor(size_t n, size_t c, size_t h, size_t w) { 
      return Tensor_t( {n,c,h,w}, GetTensorLayout(), 0, 0); 
   }

   //____________________________________________________________________________
   //
   // Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */
      ///@{
   /** Matrix-multiply \p input with the transpose of \pweights and
    *  write the results into \p output. */
   static void MultiplyTranspose(Tensor_t &output, const Tensor_t &input, const Matrix_t &weights);

   /** Add the vectors biases row-wise to the matrix output */
   static void AddRowWise(Tensor_t &output,const Matrix_t &biases);

   /** @name Backward Propagation (Dense Layers)
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */
      ///@{
   /** Perform the complete backward propagation step. If the provided
    *  \p activationGradientsBackward matrix is not empty, compute the
    *  gradients of the objective function with respect to the activations
    *  of the previous layer (backward direction).
    *  Also compute the weight and the bias gradients. Modifies the values
    *  in \p df and thus produces only a valid result, if it is applied the
    *  first time after the corresponding forward propagation has been per-
    *  formed. */
   static void Backward(Tensor_t & activationGradientsBackward,
                        Matrix_t & weightGradients,
                        Matrix_t & biasGradients,
                        Tensor_t & df,
                        const Tensor_t & activationGradients,
                        const Matrix_t & weights,
                        const Tensor_t & activationBackward); 

   /** Above functions extended to vectors */
   static void ScaleAdd(Tensor_t & A, const Tensor_t & B,
                        Scalar_t alpha = 1.0,
                        Scalar_t beta = 1.0);

   /** Deep copy from B to A. */
   static void Copy(Tensor_t & A, const Tensor_t & B);

   // copy from another tensor
   /*template<typename ATensor_t>
   static void CopyDiffArch(Tensor_t & A,
                            const ATensor_t & B);*/

   //____________________________________________________________________________
   //
   // Activation Functions
   //____________________________________________________________________________

   /** @name Activation Functions
    * For each activation function, the low-level interface contains two routines.
    * One that applies the acitvation function to a matrix and one that evaluate
    * the derivatives of the activation function at the elements of a given matrix
    * and writes the results into the result matrix.
    */
   ///@{

   /** Computes the forward pass for the activation function. */
   static void ActivationFunctionForward(Tensor_t & X,
                                         const ActivationWorkspace_t & activationWorkspace,
                                         const double coef = 0.0, const AFloat alpha = 1, 
                                         const AFloat beta = 0);
                          
   /** Computes the gradient of the activation function. */
   static void ActivationFunctionBackward(Tensor_t & dX, const Tensor_t & Y,
                                          const Tensor_t & dY,  const Tensor_t & X, 
                                          const ActivationWorkspace_t & activationWorkspace,
                                          const AFloat alpha = 1, 
                                          const AFloat beta = 0);

   //
   // No cudnn implementation for the following activation functions
   //
   #if 0
   static void Identity(Tensor_t & X) {}
   static void IdentityDerivative(Tensor_t & dX, Tensor_t& X, 
                                  Tensor_t & Y,  Tensor_t & dY, 
                                  const TActivationFnctWorkspace & activationWorkspace,
                                  const AFloat alpha = 1, 
                                  const AFloat beta = 1) {}

   static void SymmetricRelu(Tensor_t & B);
   static void SymmetricReluDerivative(Tensor_t & B,
                                       const Tensor_t & A) {}

   static void SoftSign(Tensor_t & B);
   static void SoftSignDerivative(Tensor_t & B,
                                  const Tensor_t & A) {}

   static void Gauss(Tensor_t & B);
   static void GaussDerivative(Tensor_t & B,
                               const Tensor_t & A) {}
   #endif
   ///@}

   //____________________________________________________________________________
   //
   // Loss Functions
   //____________________________________________________________________________

   /** @name Loss Functions
    * Loss functions compute a scalar value given the \p output of the network
    * for a given training input and the expected network prediction \p Y that
    * quantifies the quality of the prediction. For each function also a routing
    * that computes the gradients (suffixed by Gradients) must be provided for
    * the starting of the backpropagation algorithm.
    */
      ///@{

   static Scalar_t MeanSquaredError(const Matrix_t &Y, const Matrix_t &output,
                                    const Matrix_t &weights);
   static void MeanSquaredErrorGradients(Matrix_t &dY, const Matrix_t &Y,
                                         const Matrix_t &output, const Matrix_t &weights);

   /** Sigmoid transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   static Scalar_t CrossEntropy(const Matrix_t &Y, const Matrix_t &output,
                                const Matrix_t &weights);

   static void CrossEntropyGradients(Matrix_t &dY, const Matrix_t &Y,
                                     const Matrix_t &output, const Matrix_t &weights);

   /** Softmax transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   static Scalar_t SoftmaxCrossEntropy(const Matrix_t &Y, const Matrix_t &output,
                                       const Matrix_t &weights);
   static void SoftmaxCrossEntropyGradients(Matrix_t &dY, const Matrix_t &Y,
                                            const Matrix_t &output, const Matrix_t &weights);
   ///@}

   //____________________________________________________________________________
   //
   // Output Functions
   //____________________________________________________________________________

   /** @name Output Functions
    * Output functions transform the activations \p output of the
    * output layer in the network to a valid prediction \p YHat for
    * the desired usage of the network, e.g.  the identity function
    * for regression or the sigmoid transformation for two-class
    * classification.
    */
   ///@{
   static void Sigmoid(Matrix_t &YHat,
                       const Matrix_t & );
   static void Softmax(Matrix_t &YHat,
                       const Matrix_t & );
   ///@}

   //____________________________________________________________________________
   //
   // Regularization
   //____________________________________________________________________________

   /** @name Regularization
    * For each regularization type two functions are required, one named
    * <tt><Type>Regularization</tt> that evaluates the corresponding
    * regularization functional for a given weight matrix and the
    * <tt>Add<Type>RegularizationGradients</tt>, that adds the regularization
    * component in the gradients to the provided matrix.
    */
   ///@{

   static Scalar_t L1Regularization(const Matrix_t & /*W*/) { return 0;}
   static void AddL1RegularizationGradients(Matrix_t & /*A*/,
                                            const Matrix_t & /*W*/,
                                            Scalar_t /*weightDecay*/) {}

   static Scalar_t L2Regularization(const Matrix_t & /*W*/) { return 0; }
   static void AddL2RegularizationGradients(Matrix_t & /*A*/,
                                            const Matrix_t & /*W*/,
                                            Scalar_t /*weightDecay*/) {}
      ///@}

   //____________________________________________________________________________
   //
   // Initialization
   //____________________________________________________________________________

   /** @name Initialization
    * For each initialization method, one function in the low-level interface
    * is provided. The naming scheme is <p>Initialize<Type></p> for a given
    * initialization method Type.
    */
   ///@{

   static void InitializeGauss(Matrix_t & A);
   static void InitializeUniform(Matrix_t & A);
   static void InitializeIdentity(Matrix_t & A);
   static void InitializeZero(Matrix_t & A);
   static void InitializeGlorotNormal(Matrix_t & A); 
   static void InitializeGlorotUniform(Matrix_t & A);

      // return static instance of random generator used for initialization
      // if generator does not exist it is created the first time with a random seed (e.g. seed = 0)
   static TRandom & GetRandomGenerator();
      // set random seed for the static geenrator
      // if the static geneerator does not exists it is created
   static void SetRandomSeed(size_t seed); 
      ///@}

      //____________________________________________________________________________
      //
      // Dropout
      //____________________________________________________________________________

   /** @name Dropout
    */
      ///@{

   /** Apply dropout with activation probability \p p to the given
    *  tensor \p A and scale the result by reciprocal of \p p. */
   static void DropoutForward(Tensor_t & A,
                              const TDropoutWorkspace & dropoutWorkspace,
                              Scalar_t p);

   static void DropoutBackward(Tensor_t & A,
                               const TDropoutWorkspace & dropoutWorkspace);

      ///@}

   //____________________________________________________________________________
   //
   // Batch Normalization
   //____________________________________________________________________________

   /** @name Batch Normalization Layer Propagation
    */
   ///@{

   /** The input from each batch are normalized during training to have zero mean and unit variance 
     * and they are then scaled by two parameter, different for each input variable: 
     *  - a scale factor \gamma gamma
     *  - an offset \beta beta */
   static void BatchNormLayerForwardTraining(Matrix_t input,                        // input
                                             Matrix_t & gamma,                      // gamma
                                             Matrix_t & beta,                       // beta
                                             Matrix_t outputActivation,             // out
                                             Matrix_t & Xmu,                        // Xmu
                                             Matrix_t & output,                     // Xhat
                                             Matrix_t & Variance,                   // Var
                                             Matrix_t & IVariance,                  // Ivar
                                             #if 0
                                             const BNormDescriptors_t & descriptors,
                                             BNormWorkspace_t & workspace,
                                             #endif
                                             std::vector<Scalar_t> & RunningMeans,  // Mu_Training
                                             std::vector<Scalar_t> & RunningVars,   // Var_Training
                                             Scalar_t nTrainedBatches,              // TrainedBatches
                                             Scalar_t momentum,                     // GD momentum
                                             Scalar_t epsilon);                     // epsilon

   /** During inference the inputs are not normalized using the batch mean but the previously computed 
     * at  running mean and variance */
   static void BatchNormLayerForwardInference(Matrix_t input,
                                              Matrix_t & gamma,
                                              Matrix_t & beta,
                                              Matrix_t outputActivation,
                                              #if 0
                                              const BNormDescriptors_t & descriptors,
                                              BNormWorkspace_t & workspace,
                                              #endif
                                              std::vector<Scalar_t> & RunningMeans,
                                              std::vector<Scalar_t> & RunningVars,
                                              Scalar_t nTrainedBatches,
                                              Scalar_t epsilon) {}

   /**
    * */
   static void BatchNormLayerForwardBackward(const Matrix_t & outputGrad,
                                             const Matrix_t & gamma,
                                             Matrix_t &dgamma,
                                             Matrix_t &dbeta,
                                             Matrix_t dx,
                                             Matrix_t & output,
                                             Matrix_t & Xmu,
                                             Matrix_t & IVariance,
                                             Matrix_t & Variance,
                                             #if 0
                                             const BNormDescriptors_t & descriptors,
                                             BNormWorkspace_t & workspace,
                                             #endif
                                             Scalar_t epsilon) {}

   ///@}

      //____________________________________________________________________________
      //
      //  Convolutional Layer Propagation
      //____________________________________________________________________________

   /** @name Forward Propagation in Convolutional Layer
    */
      ///@{

   /** Calculate how many neurons "fit" in the output layer, given the input as well as the layer's hyperparameters. */
   //static size_t calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride) {}

   /** Transform the matrix B in local view format, suitable for
    *  convolution, and store it in matrix A */
   /*static void Im2col(Matrix_t &A,
                      const Matrix_t &B,
                      size_t imgHeight,
                      size_t imgWidth,
                      size_t fltHeight,
                      size_t fltWidth,
                      size_t strideRows,
                      size_t strideCols,
                      size_t zeroPaddingHeight,
                      size_t zeroPaddingWidth) {}

   static void Im2colIndices(std::vector<int> &V, const Matrix_t &B, size_t nLocalViews,
                             size_t imgHeight, size_t imgWidth, size_t fltHeight,
                             size_t fltWidth, size_t strideRows, size_t strideCols, size_t zeroPaddingHeight,
                             size_t zeroPaddingWidth) {}
   static void Im2colFast(Matrix_t &A, const Matrix_t &B, const std::vector<int> & V) {}*/

   /** Rotates the matrix \p B, which is representing a weights,
    *  and stores them in the matrix \p A. */
   /*static void RotateWeights(Matrix_t &A, const Matrix_t &B, size_t filterDepth, size_t filterHeight,
                             size_t filterWidth, size_t numFilters) {}*/

   /** Add the biases in the Convolutional Layer.  */
   static void AddConvBiases(Matrix_t &output, const Matrix_t &biases);
      ///@}

   /** Dummy placeholder - preparation is currently only required for the CUDA architecture. */
   static void PrepareInternals(Tensor_t & output, Tensor_t & inputActivation, Matrix_t & weights,
                                Matrix_t & biases, Matrix_t & weightGrads, Matrix_t & biasGrads, 
                                Tensor_t & activationGrads, const DNN::CNN::TConvParams & params);

   /** Forward propagation in the Convolutional layer */
   static void ConvLayerForward(Tensor_t & output,
                                Tensor_t & inputActivationFunc, // this is output conv w/o activ func.
                                const Tensor_t &input,
                                const Matrix_t &weights, const Matrix_t & biases,
                                const DNN::CNN::TConvParams & params,
                                Tensor_t & /* inputPrime */,
                                const ActivationWorkspace_t & activWorkspace,
                                const ConvolutionWorkspace_t & convWorkspace);
                                //const AFloat alpha = 1,
                                //const AFloat beta  = 1);

   /** @name Backward Propagation in Convolutional Layer
    */
      ///@{


   /** Perform the complete backward propagation step in a Convolutional Layer.
    *  If the provided \p activationGradientsBackward matrix is not empty, compute the
    *  gradients of the objective function with respect to the activations
    *  of the previous layer (backward direction).
    *  Also compute the weight and the bias gradients. Modifies the values
    *  in \p df and thus produces only a valid result, if it is applied the
    *  first time after the corresponding forward propagation has been per-
    *  formed. */
   static void ConvLayerBackward(Tensor_t &activationGradientsBackward,
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
                                 size_t /*filterWidth*/, size_t /*nLocalViews*/ );

   
   
   ///@}

      //____________________________________________________________________________
      //
      //  Max Pooling Layer Propagation
      //____________________________________________________________________________
   /** @name Forward Propagation in Max Pooling Layer
    */
      ///@{

   /** Downsample the matrix \p C to the matrix \p A, using max
    * operation, such that the winning indices are stored in matrix
    * \p B. No winning indices needed for cuDNN. */
   static void Downsample(Tensor_t & A, Tensor_t & /*B*/, const Tensor_t & C,
                          const PoolingWorkspace_t & poolingWorkspace,
                          size_t imgHeight, size_t imgWidth, size_t fltHeight, 
                          size_t fltWidth, size_t strideRows, size_t strideCols);

      ///@}

   /** @name Backward Propagation in Max Pooling Layer
    */
      ///@{
   /** Perform the complete backward propagation step in a Pooling Layer. Based on the
    *  input to and output from the MaxPoolLayer, the gradients for the winning pixels 
    *  are computed. */
   static void MaxPoolLayerBackward(Tensor_t & activationGradientsBackward,
                                    const Tensor_t & activationGradients,
                                    const Tensor_t & /*indexMatrix*/,
                                    const Tensor_t & inputActivation,
                                    const Tensor_t & outputTensor,
                                    const PoolingWorkspace_t & poolingWorkspace,
                                    size_t imgHeight,
                                    size_t imgWidth,
                                    size_t fltHeight,
                                    size_t fltWidth,
                                    size_t strideRows,
                                    size_t strideCols,
                                    size_t nLocalViews);

      ///@}

      //____________________________________________________________________________
      //
      //  Reshape Layer Propagation
      //____________________________________________________________________________
   /** @name Forward and Backward Propagation in Reshape Layer
    */
      ///@{

   /** Transform the matrix \p B to a matrix with different dimensions \p A */
   //static void Reshape(Matrix_t &A, const Matrix_t &B);

   /** Flattens the tensor \p B, such that each matrix, is stretched in
    *  one row, resulting with a matrix \p A. */
   static void Flatten(Tensor_t &A, const Tensor_t &B); 

   /** Transforms each row of \p B to a matrix and stores it in the
    *  tensor \p B. */
   static void Deflatten(Tensor_t &A, const Tensor_t &B); // size_t index, size_t nRows,size_t nCols);

   /** Rearrage data accoring to time fill B x T x D out with T x B x D matrix in*/
   //static void Rearrange(Tensor_t &out, const Tensor_t &in);


   /** Backward pass for Recurrent Networks */
   /*static Matrix_t & RecurrentLayerBackward(Matrix_t & state_gradients_backward, // BxH
                                            Matrix_t & input_weight_gradients,
                                            Matrix_t & state_weight_gradients,
                                            Matrix_t & bias_gradients,
                                            Matrix_t & df, //DxH
                                            const Matrix_t & state, // BxH
                                            const Matrix_t & weights_input, // HxD
                                            const Matrix_t & weights_state, // HxH
                                            const Matrix_t & input,  // BxD
                                            Matrix_t & input_gradient);*/


      ///@}
      
      //____________________________________________________________________________
      //
      // Additional Arithmetic Functions
      //____________________________________________________________________________

   /** @name Additional Arithmetic Functions
    *
    * Additional arithmetic on CUDA matrices  used to implement the low-level
    * interface.
    */
      ///@{

   /** Standard multiplication of two matrices \p A and \p B with the result being
    *  written into C.
    */
   /*static void Multiply(Matrix_t &C,
                        const Matrix_t &A,
                        const Matrix_t &B);*/
   /** Matrix multiplication of two matrices \p A and \p B^T (transposed) with the
    *  result being written into C.
    */
   /*static void TransposeMultiply(Matrix_t &output,
                                 const Matrix_t &input,
                                 const Matrix_t &Weights,
                                 Scalar_t alpha = 1.0, Scalar_t beta = 0.);*/
   /** In-place Hadamard (element-wise) product of matrices \p A and \p B
    *  with the result being written into \p A.
    */
   /*static void Hadamard(Tensor_t &A,
                        const Tensor_t &B);
   static void Hadamard(Matrix_t &A,
                        const Matrix_t &B);*/
      // {
      //    Tensor_t tA(A);
      //    Hadamard( tA, Tensor_t(B));
      // }

   /** Sum columns of (m x n) matrixx \p A and write the results into the first
    * m elements in \p A.
    */
   /*static void SumColumns(Matrix_t &B,
                          const Matrix_t &A,
                          Scalar_t alpha = 1.0, Scalar_t beta = 0.);*/

   /** Compute the sum of all elements in \p A */
   static Scalar_t Sum(const Matrix_t &A, Scalar_t alpha = 1.0, Scalar_t beta = 0.0);

   /** Check two matrices for equality, taking floating point arithmetic errors into account. */
   //static bool AlmostEquals(const Matrix_t &A, const Matrix_t &B, double epsilon = 0.1);

   /** Add the constant \p beta to all the elements of matrix \p A and write the
    * result into \p A.
    */
   static void ConstAdd(Matrix_t &A, Scalar_t beta);

   /** Multiply the constant \p beta to all the elements of matrix \p A and write the
    * result into \p A.
    */
   static void ConstMult(Matrix_t &A, Scalar_t beta);

   /** Reciprocal each element of the matrix \p A and write the result into
    * \p A
    */
   //static void ReciprocalElementWise(Matrix_t &A);

   /** Square each element of the matrix \p A and write the result into
    * \p A
    */
   //static void SquareElementWise(Matrix_t &A);

   /** Square root each element of the matrix \p A and write the result into
    * \p A
    */
   static void SqrtElementWise(Matrix_t &A, Scalar_t alpha = 1, Scalar_t beta = 0, Scalar_t gamma = 0);

      // optimizer functions
   /*static void AdamUpdate(Matrix_t & A, const Matrix_t & M, const Matrix_t & V, Scalar_t alpha, Scalar_t eps);
   static void AdamUpdateFirstMom(Matrix_t & A, const Matrix_t & B, Scalar_t beta);
   static void AdamUpdateSecondMom(Matrix_t & A, const Matrix_t & B, Scalar_t beta);*/

      // printing of tensor
   static void PrintTensor( const Tensor_t & A, const std::string name = "tensor");



   ///////////////////////////////////////////////////////////////////////////////
   /// extra functions defined only for CPU architecture !!!
   //////////////////////////////////////////////////////////////////////////////
   
   /** Sum rows of (m x n) matrix \p A and write the results into the first
   * m elements in \p B.
   */
   static void SumRows(Matrix_t & B, const Matrix_t & A);


};

//____________________________________________________________________________
/*template <typename AFloat>
template <typename AMatrix_t>
void TCuda<AFloat>::CopyDiffArch(TCudaMatrix<AFloat> &B,
                        const AMatrix_t &A)
{
   // copy from another architecture using the reference one
   // this is not very efficient since creates temporary objects
   TMatrixT<AFloat> tmp = A;
   Copy(B, TCudaMatrix<AFloat>(tmp) ); 
}

//____________________________________________________________________________
template <typename AFloat>
template <typename AMatrix_t>
void TCuda<AFloat>::CopyDiffArch(std::vector<TCudaMatrix<AFloat>> &B,
                            const std::vector<AMatrix_t> &A)
{
   for (size_t i = 0; i < B.size(); ++i) {
      CopyDiffArch(B[i], A[i]);
   }
}*/

template <typename Real_t>
void TCudnn<Real_t>::PrintTensor(const typename TCudnn<Real_t>::Tensor_t & A, const std::string name ) 
{
   std::cout << name << "  size = " << A.GetSize() << " shape = { "; 
   auto shape = A.GetShape(); 
   for (size_t k = 0; k < shape.size()-1; ++k)
      std::cout << shape[k] << " , ";
   std::cout << shape.back() << " } ";
   std::cout << " strides = { ";
   auto strides = A.GetStrides(); 
   for (size_t k = 0; k < strides.size()-1; ++k)
      std::cout << strides[k] << " , ";
   std::cout << strides.back() << " }\n ";

   if (A.GetShape().size() == 2 ) { 
      for (size_t i = 0; i < A.GetShape()[0]; ++i) {
         std::cout << "{ ";
         for (size_t j = 0; j < A.GetShape()[1]; ++j) {
            std::cout << A(i,j) << " ";
         }
         std::cout << " } " << std::endl;
      }
   } else if  (A.GetShape().size() == 3 ) {
      for (size_t i = 0; i < A.GetFirstSize(); ++i) {
         std::cout << "{ ";
         for (size_t j = 0; j < A.GetHSize(); ++j) {
            std::cout << "{ ";
            for (size_t k = 0; k < A.GetWSize(); ++k) {
               std::cout << A(i,j,k) << " ";
            }
            std::cout << " } " << std::endl;
         }
         std::cout << " } " << std::endl;
      }
   } else if  (A.GetShape().size() == 4 ) {
      for (size_t i = 0; i < A.GetShape()[0]; ++i) {
         std::cout << "{ ";
         for (size_t j = 0; j < A.GetShape()[1]; ++j) {
            std::cout << "{ ";
            for (size_t k = 0; k < A.GetShape()[2]; ++k) {
               for (size_t l = 0; l < A.GetShape()[3]; ++l) {
                  std::cout << A(i,j,k,l) << " ";
               }  
               std::cout << " } " << std::endl;
            }
            std::cout << " } " << std::endl;
         }
         std::cout << " } " << std::endl;
      }
   }
   else {  
      for (size_t l = 0; l < A.GetSize(); ++l) {
         std::cout << A.GetData()[l] << " ";
      }
      std::cout << "\n";
   }  
}


} // namespace DNN
} // namespace TMVA

#endif
