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

#ifndef TMVA_DNN_FUNCTIONS
#define TMVA_DNN_FUNCTIONS

namespace TMVA
{
namespace DNN
{
//______________________________________________________________________________
//
//  Enum Definitions
//______________________________________________________________________________

/*! Enum that represents layer activation functions. */
/*enum class EActivationFunction
{
   kIdentity = 0,
   kRelu     = 1,
   kSigmoid  = 2,
   kTanh     = 3,
   kSymmRelu = 4,
   kSoftSign = 5,
   kGauss    = 6
};


//______________________________________________________________________________
//
//  
//______________________________________________________________________________

/

//______________________________________________________________________________
//
// Initialization
//______________________________________________________________________________

/*template<typename Architecture_t>
inline void initialize(typename Architecture_t::Matrix_t & A,
                       EInitialization m)
{
   switch(m) {
   case EInitialization::kGauss    : Architecture_t::InitializeGauss(A);
       break;
   case EInitialization::kUniform  : Architecture_t::InitializeUniform(A);
       break;
   case EInitialization::kIdentity : Architecture_t::InitializeIdentity(A);
       break;
   case EInitialization::kZero     : Architecture_t::InitializeZero(A);
       break;
   case EInitialization::kGlorotNormal    : Architecture_t::InitializeGlorotNormal(A);
       break;
   case EInitialization::kGlorotUniform  : Architecture_t::InitializeGlorotUniform(A);
       break;
   }
}*/

} // namespace DNN
} // namespace TMVA

#endif
