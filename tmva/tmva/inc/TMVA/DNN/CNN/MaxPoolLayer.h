// @(#)root/tmva/tmva/dnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMaxPoolLayer                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Max Pool Deep Neural Network Layer                                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef MAXPOOLLAYER_H_
#define MAXPOOLLAYER_H_

#include "TMatrix.h"

#include "TMVA/DNN/CNN/ConvLayer.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/CNN/ContextHandles.h"

#include <iostream>

namespace TMVA {
namespace DNN {
namespace CNN {

typedef struct TPoolParams : public TParams {

public:
   float dropoutProbability;   ///< Probability for a single element of the activation to be set to zero

   TPoolParams(size_t _batchSize, size_t _inputDepth, size_t _inputHeight, size_t _inputWidth, 
               size_t _filterHeight, size_t _filterWidth, size_t _strideRows, size_t _strideCols,
               size_t _paddingHeight = 0, size_t _paddingWidth = 0, float _dropoutProbability = 0.0)      //FIXME: Remove default values
               : TParams(_batchSize, _inputDepth, _inputHeight, _inputWidth, _filterHeight, _filterWidth, _strideRows,
                        _strideCols, _paddingHeight, _paddingWidth),
                 dropoutProbability(_dropoutProbability)
   {}
} TPoolParams;

/** \class TMaxPoolLayer

    Generic Max Pooling Layer class.

    This generic Max Pooling Layer Class represents a pooling layer of
    a CNN. It inherits all of the properties of the convolutional layer
    TConvLayer, but it overrides the propagation methods. In a sense, max pooling
    can be seen as non-linear convolution: a filter slides over the input and produces
    one element as a function of the the elements within the receptive field.
    In addition to that, it contains a matrix of winning units.

    The height and width of the weights and biases is set to 0, since this
    layer does not contain any weights.
 */
template <typename Architecture_t>
class TMaxPoolLayer : public VGeneralLayer<Architecture_t> {

public:
   using Tensor_t = typename Architecture_t::Tensor_t;
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

   using PoolingOptions_t = typename Architecture_t::PoolingOptions_t;
   using DropoutWorkspace_t = typename Architecture_t::DropoutWorkspace_t;
   using PoolingWorkspace_t = typename Architecture_t::PoolingWorkspace_t;

   /* Calculate the output dimension of the convolutional layer */
   static size_t calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride);

   /* Calculate the number of pixels in a single receptive field */
   static size_t inline calculateNLocalViewPixels(size_t depth, size_t height, size_t width) { return depth * height * width; }

   /* Calculate the number of receptive fields in an image given the filter and image sizes */
   static size_t calculateNLocalViews(size_t inputHeight, size_t filterHeight, size_t paddingHeight, size_t strideRows,
                                      size_t inputWidth, size_t filterWidth, size_t paddingWidth, size_t strideCols);

private:
   size_t fFilterHeight; ///< The height of the downsampling window.
   size_t fFilterWidth;  ///< The width of the downsampling window.

   size_t fStrideRows;   ///< The number of row pixels to slide the window each step.
   size_t fStrideCols;   ///< The number of column pixels to slide the window each step.

   size_t fNLocalViews;          ///< The number of local views in one image.

   Scalar_t fDropoutProbability; ///< Probability that an input is active.

   size_t fPaddingHeight;        ///< The number of zero layers added top and bottom of the input.
   size_t fPaddingWidth;         ///< The number of zero layers left and right of the input.
   
   DropoutWorkspace_t     * fDropoutWorkspace = nullptr;   ///< Contains descriptors and pointers to on-device memory needed for the dropout operation
   PoolingWorkspace_t     * fPoolingWorkspace = nullptr;   ///< Contains descriptors and pointers to on-device memory needed for the pooling operation

   Tensor_t fIndexTensor; ///< Matrix of indices for the backward pass.

public:
   /*! Constructor. */
   TMaxPoolLayer(size_t BatchSize, size_t InputDepth, size_t InputHeight, size_t InputWidth, size_t FilterHeight,
                 size_t FilterWidth, size_t StrideRows, size_t StrideCols, Scalar_t DropoutProbability,
                 size_t paddingHeight = 0, size_t paddingWidth = 0,                    // pass padding params and options only if using cudnn
                 const PoolingOptions_t & poolingOptions = PoolingOptions_t());

   /*! Copy the max pooling layer provided as a pointer */
   TMaxPoolLayer(TMaxPoolLayer<Architecture_t> *layer);

   /*! Copy constructor. */
   TMaxPoolLayer(const TMaxPoolLayer &);

   /*! Destructor. */
   ~TMaxPoolLayer();

   /*! Computes activation of the layer for the given input. The input
    *  must be in 3D tensor form with the different matrices corresponding to
    *  different events in the batch. It spatially downsamples the input
    *  matrices. */
   void Forward(Tensor_t &input, bool applyDropout = true);

   /*! Depending on the winning units determined during the Forward pass,
    *  it only forwards the derivatives to the right units in the previous
    *  layer. Must only be called directly at the corresponding call
    *  to Forward(...). */
   void Backward(Tensor_t &gradients_backward, const Tensor_t &activations_backward);
    //             Tensor_t &inp1, Tensor_t &inp2);

   /*! Writes the information and the weights about the layer in an XML node. */
   virtual void AddWeightsXMLTo(void *parent);

   /*! Read the information and the weights about the layer from XML node. */
   virtual void ReadWeightsFromXML(void *parent);

   /*! Prints the info about the layer. */
   void Print() const;

   /*! Getters */
   size_t GetFilterHeight() const { return fFilterHeight; }
   size_t GetFilterWidth() const { return fFilterWidth; }

   size_t GetStrideRows() const { return fStrideRows; }
   size_t GetStrideCols() const { return fStrideCols; }

   size_t GetPaddingHeight() const { return fPaddingHeight; }
   size_t GetPaddingWidth() const { return fPaddingWidth; }

   size_t GetNLocalViews() const { return fNLocalViews; }

   Scalar_t GetDropoutProbability() const { return fDropoutProbability; }

   const Tensor_t & GetIndexTensor() const { return fIndexTensor; }
   Tensor_t & GetIndexTensor() { return fIndexTensor; }

   /** Get the descriptors and device memory used for the dropout operation. */
   const DropoutWorkspace_t & GetDropoutWorkspace() const { return *fDropoutWorkspace; }
   DropoutWorkspace_t & GetDropoutWorkspace() { return *fDropoutWorkspace; }

   /** Get the descriptors used for the pooling operation. */
   const PoolingWorkspace_t & GetPoolingWorkspace() const { return *fPoolingWorkspace; }
   PoolingWorkspace_t & GetPoolingWorkspace() { return *fPoolingWorkspace; }

};

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::TMaxPoolLayer(size_t batchSize, size_t inputDepth, size_t inputHeight, size_t inputWidth,
                                             size_t filterHeight, size_t filterWidth, size_t strideRows, size_t strideCols,
                                             Scalar_t dropoutProbability, size_t paddingHeight, size_t paddingWidth,
                                             const PoolingOptions_t & poolingOptions)

         : VGeneralLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, inputDepth,
                                         calculateDimension(inputHeight, filterHeight, paddingHeight, strideRows),
                                         calculateDimension(inputWidth, filterWidth, paddingWidth, strideCols),
                                         1, inputDepth, calculateNLocalViewPixels(inputDepth, filterHeight, filterWidth),
                                         1, inputDepth, 1,/*0,0,0,0,0,0,*/ batchSize, inputDepth,
                                         calculateNLocalViews(inputHeight, filterHeight, paddingHeight, strideRows,
                                                              inputWidth, filterWidth, paddingWidth, strideCols),
                                         EInitialization::kZero),

         /*: TConvLayer<Architecture_t>(batchSize, inputDepth, inputHeight, inputWidth, inputDepth, EInitialization::kZero,
                                      filterHeight, filterWidth, strideRows, strideCols, 0, 0, dropoutProbability,
                                      EActivationFunction::kIdentity, ERegularization::kNone, 0, {}, {}),*/
         fFilterHeight(filterHeight), fFilterWidth(filterWidth), fStrideRows(strideRows), fStrideCols(strideCols),
         fNLocalViews(calculateNLocalViews(inputHeight, filterHeight, paddingHeight, strideRows,
                                       inputWidth, filterWidth, paddingWidth, strideCols)),
         fDropoutProbability(dropoutProbability), fPaddingHeight(paddingHeight), fPaddingWidth(paddingWidth),
         fIndexTensor( this->GetBatchSize(), this->GetDepth(), this->GetNLocalViews() )
{
   TPoolParams params (this->GetBatchSize(), this->GetInputDepth(), this->GetInputHeight(), this->GetInputWidth(),
                       this->GetFilterHeight(), this->GetFilterWidth(), this->GetStrideRows(), this->GetStrideCols(), 
                       0, 0, this->GetDropoutProbability());;

   fDropoutWorkspace = new DropoutWorkspace_t(params, TOptions());
   fPoolingWorkspace = new PoolingWorkspace_t(params, poolingOptions);
}

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::TMaxPoolLayer(TMaxPoolLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer),
     fFilterHeight(layer->GetFilterHeight()), fFilterWidth(layer->GetFilterWidth()),
     fStrideRows(layer->GetStrideRows()), fStrideCols(layer->GetStrideCols()),
     fNLocalViews(layer->GetNLocalViews()), fDropoutProbability(layer->GetDropoutProbability()),
     fPaddingHeight(layer->GetPaddingHeight()), fPaddingWidth(layer->GetPaddingWidth()),
   //: TConvLayer<Architecture_t>(layer), 
     fIndexTensor( layer->GetIndexTensor().GetShape() )
{
   fDropoutWorkspace = new DropoutWorkspace_t();
   fPoolingWorkspace = new PoolingWorkspace_t();
   DropoutWorkspace_t::DeepCopy(fDropoutWorkspace, layer->GetDropoutWorkspace());
   PoolingWorkspace_t::DeepCopy(fPoolingWorkspace, layer->GetPoolingWorkspace());
}

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::TMaxPoolLayer(const TMaxPoolLayer &layer)
   : VGeneralLayer<Architecture_t>(layer),
     fFilterHeight(layer.GetFilterHeight()), fFilterWidth(layer.GetFilterWidth()),
     fStrideRows(layer.GetStrideRows()), fStrideCols(layer.GetStrideCols()),
     fNLocalViews(layer.GetNLocalViews()), fDropoutProbability(layer.GetDropoutProbability()),
     fPaddingHeight(layer.GetPaddingHeight()), fPaddingWidth(layer.GetPaddingWidth()),
   //: TConvLayer<Architecture_t>(layer),
   fIndexTensor( layer.GetIndexTensor().GetShape() )
{
   fDropoutWorkspace = new DropoutWorkspace_t();
   fPoolingWorkspace = new PoolingWorkspace_t();
   DropoutWorkspace_t::DeepCopy(fDropoutWorkspace, layer.GetDropoutWorkspace());
   PoolingWorkspace_t::DeepCopy(fPoolingWorkspace, layer.GetPoolingWorkspace());
}

//______________________________________________________________________________
template <typename Architecture_t>
TMaxPoolLayer<Architecture_t>::~TMaxPoolLayer()
{
   if (fDropoutWorkspace) delete fDropoutWorkspace;
   if (fPoolingWorkspace) delete fPoolingWorkspace;
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TMaxPoolLayer<Architecture_t>::Forward(Tensor_t &input, bool applyDropout) -> void
{
   if (applyDropout && (this->GetDropoutProbability() != 1.0)) {
         Architecture_t::DropoutForward(input, 
                                        this->GetDropoutWorkspace(),
                                        this->GetDropoutProbability());
   }

   //size_t outputHeight = calculateDimension(this->GetInputHeight(), this->GetFilterHeight(), this->GetPaddingHeight(), this->GetStrideRows());
   //size_t outputWidth  = calculateDimension(this->GetInputWidth(), this->GetFilterWidth(), this->GetPaddingWidth(), this->GetStrideCols());
   //Tensor_t & output = this->GetOutput();
   //output = Tensor_t(output.GetDeviceBuffer(),{this->GetBatchSize(), this->GetInputDepth(), outputHeight, outputWidth }, Architecture_t::GetTensorLayout(),0,0 );

   Architecture_t::Downsample(this->GetOutput(), fIndexTensor, input, 
                              this->GetPoolingWorkspace(),
                              this->GetInputHeight(), this->GetInputWidth(), 
                              this->GetFilterHeight(), this->GetFilterWidth(),
                              this->GetStrideRows(), this->GetStrideCols());
   
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TMaxPoolLayer<Architecture_t>::Backward(Tensor_t &gradients_backward,
                                             const Tensor_t & activations_backward) -> void
//                                             Tensor_t & /*inp1*/, Tensor_t &
{
   if (this->GetDropoutProbability() != 1.0) {
      Architecture_t::DropoutBackward(this->GetActivationGradients(), 
                                      this->GetDropoutWorkspace());
   }

   Architecture_t::MaxPoolLayerBackward(gradients_backward, this->GetActivationGradients(), fIndexTensor, activations_backward, this->GetOutput(),
                                        this->GetPoolingWorkspace(),
                                        this->GetInputHeight(), this->GetInputWidth(),      
                                        this->GetFilterHeight(), this->GetFilterWidth(),
                                        this->GetStrideRows(), this->GetStrideCols(), this->GetNLocalViews());
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TMaxPoolLayer<Architecture_t>::Print() const -> void
{
   std::cout << " POOL Layer: \t";
   std::cout << "( W = " << this->GetWidth() << " , ";
   std::cout << " H = " << this->GetHeight() << " , ";
   std::cout << " D = " << this->GetDepth() << " ) ";

   std::cout << "\t Filter ( W = " << this->GetFilterWidth() << " , ";
   std::cout << " H = " << this->GetFilterHeight() << " ) ";

   if (this->GetOutput().GetSize() > 0) {
      std::cout << "\tOutput = ( " << this->GetOutput().GetFirstSize() << " , " << this->GetOutput().GetHSize() << " , " << this->GetOutput().GetWSize() << " ) ";
   }
   std::cout << std::endl;
}

//______________________________________________________________________________
template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::AddWeightsXMLTo(void *parent)
{
   auto layerxml = gTools().xmlengine().NewChild(parent, 0, "MaxPoolLayer");

   // write  maxpool layer info
   gTools().xmlengine().NewAttr(layerxml, 0, "FilterHeight", gTools().StringFromInt(this->GetFilterHeight()));
   gTools().xmlengine().NewAttr(layerxml, 0, "FilterWidth", gTools().StringFromInt(this->GetFilterWidth()));
   gTools().xmlengine().NewAttr(layerxml, 0, "StrideRows", gTools().StringFromInt(this->GetStrideRows()));
   gTools().xmlengine().NewAttr(layerxml, 0, "StrideCols", gTools().StringFromInt(this->GetStrideCols()));

}

//______________________________________________________________________________
template <typename Architecture_t>
void TMaxPoolLayer<Architecture_t>::ReadWeightsFromXML(void * /*parent */)
{
   // all info is read before - nothing to do 
}

//______________________________________________________________________________
template <typename Architecture_t>
size_t TMaxPoolLayer<Architecture_t>::calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride)
{
   size_t temp = imgDim - fltDim + 2 * padding;
   if (temp % stride || temp + stride <= 0) {
      Fatal("calculateDimension", "Not compatible hyper parameters for layer - (imageDim, filterDim, padding, stride) "
            "%zu, %zu, %zu, %zu", imgDim, fltDim, padding, stride);
   }
   return temp / stride + 1;
}

template <typename Architecture_t>
size_t TMaxPoolLayer<Architecture_t>::calculateNLocalViews(size_t inputHeight, size_t filterHeight, size_t paddingHeight,
                                                        size_t strideRows, size_t inputWidth, size_t filterWidth,
                                                        size_t paddingWidth, size_t strideCols)
{
    int height = calculateDimension(inputHeight, filterHeight, paddingHeight, strideRows);
    int width = calculateDimension(inputWidth, filterWidth, paddingWidth, strideCols);

    return height * width;
}

//______________________________________________________________________________

} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif
