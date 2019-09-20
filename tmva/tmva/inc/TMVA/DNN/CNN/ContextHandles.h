// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/06/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Contains function enums for activation and output functions, as //
// well as generic evaluation functions, that delegate the call to //
// the corresponding evaluation kernel.                            //
/////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_CNN_DESCRIPTORS
#define TMVA_DNN_CNN_DESCRIPTORS

#include <stddef.h>
#include <set>
#include <string>

namespace TMVA
{
namespace DNN
{ 

   /** Contains parameters for the DNN layers.*/
   struct TParams {
      size_t batchSize; ///< Batch size used for training and evaluation

      // FIXME: Move these to CNN params:
      size_t inputDepth;  ///< The depth of the previous layer or input.
      size_t inputHeight; ///< The height of the previous layer or input.
      size_t inputWidth;  ///< The width of the previous layer or input.

      size_t filterHeight;  ///< The height of the filter.
      size_t filterWidth;   ///< The width of the filter.

      size_t strideRows;    ///< The number of row pixels to slid the filter each step.
      size_t strideCols;    ///< The number of column pixels to slid the filter each step
      size_t paddingHeight; ///< The number of zero layers added top and bottom of the input.
      size_t paddingWidth;  ///< The number of zero layers left and right of the input.

      TParams() = default;
      TParams(const TParams  &) = default;
      TParams(      TParams &&) = default;

      TParams(size_t _batchSize, size_t _inputDepth, size_t _inputHeight, size_t _inputWidth,
              size_t _filterHeight, size_t _filterWidth, size_t _strideRows, size_t _strideCols,
              size_t _paddingHeight, size_t _paddingWidth) 
              : batchSize(_batchSize), inputDepth(_inputDepth), inputHeight(_inputHeight), 
                inputWidth(_inputWidth), filterHeight(_filterHeight), filterWidth(_filterWidth),
                strideRows(_strideRows), strideCols(_strideCols), paddingHeight(_paddingHeight),
                paddingWidth(_paddingWidth)
      {}

      TParams & operator=(const TParams  &) = default;
      TParams & operator=(      TParams &&) = default;

      virtual ~TParams() {};
   };

   /** Contains the options that are set at library call level (cuDNN). */
   struct TOptions {
      TOptions() = default;
      TOptions(const TOptions  &) = default;
      TOptions(      TOptions &&) = default;

      TOptions & operator=(const TOptions  &) = default;
      TOptions & operator=(      TOptions &&) = default;

      virtual ~TOptions() {};
   };

   /** Contains descriptors and pointers to allocated memory, which are used
    *  by the CNN/RNN layers during fwd/bwd propagation
    */
   struct TWorkspace {
      TWorkspace() = default;
      TWorkspace(const TParams & /*params*/,  const TOptions & /*userOptions*/) {};

      TWorkspace(const TWorkspace  &) = default;     // When moving the workspace around, don't make deep copies
      TWorkspace(      TWorkspace &&) = default;
      TWorkspace & operator=(const TWorkspace  &) = default;
      TWorkspace & operator=(      TWorkspace &&) = default;

      // Avoid multiple definition for pure virtual destructor -> protected constructor instead
      virtual ~TWorkspace() {};

      /** Allows for deep copies of a workspace, also including device memory. */
      virtual void DeepCopy(TWorkspace & A, const TWorkspace & B) = 0;
   };

} // namespace DNN
} // namespace TMVA

#endif
