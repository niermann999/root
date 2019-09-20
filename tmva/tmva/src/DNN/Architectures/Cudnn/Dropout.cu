// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 14/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMVA/DNN/Architectures/TCudnn.h"
/*#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"*/

/////////////////////////////////////////////////////////////////////
// Implementation of the Dropout function for TCudnn architectures.//
/////////////////////////////////////////////////////////////////////

namespace TMVA {
namespace DNN  {

// FIXME: Do testing!!!
//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::DropoutForward(TCudaTensor<AFloat> &A,
                                    const DropoutWorkspace_t & dropoutWorkspace, 
                                    AFloat /*dropoutProbability*/)
{
    //TCudaTensor<AFloat> tmp (A);

    // Write the output into A      
    CUDNNCHECK(cudnnDropoutForward(A.GetCudnnHandle(),
                                   dropoutWorkspace.dropoutDescr,
                                   A.GetTensorDescriptor(),// use tmp, if inplace op fails
                                   A.GetDataPointer(),
                                   A.GetTensorDescriptor(),
                                   A.GetDataPointer(),
                                   dropoutWorkspace.dropoutReserveSpace,
                                   dropoutWorkspace.dropoutReserveSpaceSize));
}

//____________________________________________________________________________
template<typename AFloat>
void TCudnn<AFloat>::DropoutBackward(TCudaTensor<AFloat> &A,
                                     const DropoutWorkspace_t & dropoutWorkspace)
{
    //TCudaTensor<AFloat> tmp (A);

    // Write the output into A
    CUDNNCHECK(cudnnDropoutBackward(A.GetCudnnHandle(),
                                    dropoutWorkspace.dropoutDescr,
                                    A.GetTensorDescriptor(),// use tmp, if inplace op fails
                                    A.GetDataPointer(),
                                    A.GetTensorDescriptor(),
                                    A.GetDataPointer(),
                                    dropoutWorkspace.dropoutStatesSpace,
                                    dropoutWorkspace.dropoutStatesSpaceSize));
}

} // namespace DNN
} // namespace TMVA
