/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2017-2018, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the ISO/IEC nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "RAHT.h"

#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>
#include <stdio.h>

#include "PCCTMC3Common.h"
#include "PCCMisc.h"

using namespace pcc::RAHT;

namespace pcc {

//============================================================================
int8_t
PCCRAHTComputeLCP::computeLCPCoeff(int m, int64_t coeffs[][3])
{
  int64_t sumk1k22 = 0;
  int64_t sumk1k11 = 0;
  for (size_t coeffIdx = 0; coeffIdx < m; ++coeffIdx) {
    auto& attr = coeffs[coeffIdx];
    sumk1k22 += attr[1] * attr[2];
    sumk1k11 += attr[1] * attr[1];
  }

  if (window1.size() < 128) {
    window1.push(sumk1k22);
    window2.push(sumk1k11);
    sumk1k2 += sumk1k22;
    sumk1k1 += sumk1k11;
  } else {
    int removedValue1 = window1.front();
    window1.pop();
    int removedValue2 = window2.front();
    window2.pop();
    sumk1k2 -= removedValue1;
    sumk1k1 -= removedValue2;

    window1.push(sumk1k22);
    window2.push(sumk1k11);

    sumk1k2 += sumk1k22;
    sumk1k1 += sumk1k11;
  }

  int scale = 0;
  if (sumk1k2 && sumk1k1) {
    scale = divApprox(sumk1k2, sumk1k1, 4);
  }

  // NB: coding range is limited to +-16
  return PCCClip(scale, -16, 16);

}

//============================================================================

struct PCCRAHTACCoefficientEntropyEstimate {
  PCCRAHTACCoefficientEntropyEstimate() { init(); }

  PCCRAHTACCoefficientEntropyEstimate(
    const PCCRAHTACCoefficientEntropyEstimate& other) = default;

  PCCRAHTACCoefficientEntropyEstimate&
  operator=(const PCCRAHTACCoefficientEntropyEstimate&) = default;

  void resStatUpdate(int32_t values, int k);
  void init();
  void updateCostBits(int32_t values, int k);
  double costBits() { return sumCostBits; }
  void resetCostBits() { sumCostBits = 0.; }

private:
  // Encoder side residual cost calculation
  static constexpr unsigned scaleRes = 1 << 20;
  static constexpr unsigned windowLog2 = 6;
  int probResGt0[3];  //prob of residuals larger than 0: 1 for each component
  int probResGt1[3];  //prob of residuals larger than 1: 1 for each component
  double sumCostBits;
};

//============================================================================

void
PCCRAHTACCoefficientEntropyEstimate::init()
{
  for (int k = 0; k < 3; k++)
    probResGt0[k] = probResGt1[k] = (scaleRes >> 1);
  sumCostBits = 0.;
}

//---------------------------------------------------------------------------

void
PCCRAHTACCoefficientEntropyEstimate::updateCostBits(int32_t value, int k)
{
  int log2scaleRes = ilog2(uint32_t(scaleRes));
  double bits = 0;
  bits += value ? log2scaleRes - log2(probResGt0[k])
                : log2scaleRes - log2(scaleRes - probResGt0[k]);  //Gt0
  int mag = abs(value);
  if (mag) {
    bits += mag > 1 ? log2scaleRes - log2(probResGt1[k])
                    : log2scaleRes - log2(scaleRes - probResGt1[k]);  //Gt1
    bits += 1;  //sign bit.
    if (mag > 1)
      bits += 2.0 * log2(mag - 1.0) + 1.0;  //EG0 approximation.
  }
  sumCostBits += bits;
}

//----------------------------------------------------------------------------

void
PCCRAHTACCoefficientEntropyEstimate::resStatUpdate(int32_t value, int k)
{
  probResGt0[k] += value ? (scaleRes - probResGt0[k]) >> windowLog2
                         : -((probResGt0[k]) >> windowLog2);
  if (value)
    probResGt1[k] += abs(value) > 1 ? (scaleRes - probResGt1[k]) >> windowLog2
                                    : -((probResGt1[k]) >> windowLog2);
}

//============================================================================
// Generate the spatial prediction of a block.

template<bool haarFlag, int numAttrs, bool rahtExtension, typename It, typename It2>
void
intraDcPred(
  const int parentNeighIdx[19],
  const int childNeighIdx[12][8],
  int occupancy,
  It first,
  It firstChild,
  It2 intraFirstChild,
  FixedPoint predBuf[][8],
  FixedPoint intraPredBuf[][8],
  const RahtPredictionParams &rahtPredParams, 
  int64_t& limitLow,
  int64_t& limitHigh,
  const bool& enableACInterPred)
{
  static const uint8_t predMasks[19] = {255, 240, 204, 170, 192, 160, 136,
                                        3,   5,   15,  17,  51,  85,  10,
                                        34,  12,  68,  48,  80};

  const auto& predWeightParent = rahtPredParams.predWeightParent;
  const auto& predWeightChild = rahtPredParams.predWeightChild;

  static const int kDivisors[64] = {
    32768, 16384, 10923, 8192, 6554, 5461, 4681, 4096, 3641, 3277, 2979,
    2731,  2521,  2341,  2185, 2048, 1928, 1820, 1725, 1638, 1560, 1489,
    1425,  1365,  1311,  1260, 1214, 1170, 1130, 1092, 1057, 1024, 993,
    964,   936,   910,   886,  862,  840,  819,  799,  780,  762,  745,
    728,   712,   697,   683,  669,  655,  643,  630,  618,  607,  596,
    585,   575,   565,   555,  546,  537,  529,  520,  512};

  int weightSum[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

  
  int64_t neighValue[3];
  int64_t childNeighValue[3];
  int64_t intraChildNeighValue[3];

  const auto parentOnlyCheckMaxIdx =
    rahtPredParams.raht_subnode_prediction_enabled_flag ? 7 : 19;
  for (int i = 0; i < parentOnlyCheckMaxIdx; i++) {
    if (parentNeighIdx[i] == -1)
      continue;

    auto neighValueIt = std::next(first, numAttrs * parentNeighIdx[i]);
    for (int k = 0; k < numAttrs; k++)
      neighValue[k] = *neighValueIt++;

    // skip neighbours that are outside of threshold limits
    if (i) {
      if (10 * neighValue[0] <= limitLow || 10 * neighValue[0] >= limitHigh)
        continue;
    } else {
      constexpr int ratioThreshold1 = 2;
      constexpr int ratioThreshold2 = 25;
      limitLow = ratioThreshold1 * neighValue[0];
      limitHigh = ratioThreshold2 * neighValue[0];
    }

    // apply weighted neighbour value to masked positions
    for (int k = 0; k < numAttrs; k++)
      if (rahtExtension)
        neighValue[k] *= predWeightParent[i];
      else
        neighValue[k] *= predWeightParent[i] << pcc::FixedPoint::kFracBits;

    int mask = predMasks[i] & occupancy;
    for (int j = 0; mask; j++, mask >>= 1) {
      if (mask & 1) {
        weightSum[j] += predWeightParent[i];
        for (int k = 0; k < numAttrs; k++) {
          predBuf[k][j].val += neighValue[k];
          if (enableACInterPred)
            intraPredBuf[k][j].val += neighValue[k];
        }
      }
    }
  }
  if (rahtPredParams.raht_subnode_prediction_enabled_flag) {
    for (int i = 0; i < 12; i++) {
      if (parentNeighIdx[7 + i] == -1)
        continue;

      auto neighValueIt = std::next(first, numAttrs * parentNeighIdx[7 + i]);
      for (int k = 0; k < numAttrs; k++)
        neighValue[k] = *neighValueIt++;

      // skip neighbours that are outside of threshold limits
      if (10 * neighValue[0] <= limitLow || 10 * neighValue[0] >= limitHigh)
        continue;

      // apply weighted neighbour value to masked positions
      for (int k = 0; k < numAttrs; k++)
        if (rahtExtension)
          neighValue[k] *= predWeightParent[7 + i];
        else
          neighValue[k] *= predWeightParent[7 + i] << pcc::FixedPoint::kFracBits;

      int mask = predMasks[7 + i] & occupancy;
      for (int j = 0; mask; j++, mask >>= 1) {
        if (mask & 1) {
          if (childNeighIdx[i][j] != -1) {
            weightSum[j] += predWeightChild[i];
            auto childNeighValueIt =
              std::next(firstChild, numAttrs * childNeighIdx[i][j]);
            for (int k = 0; k < numAttrs; k++)
              if (rahtExtension)
                childNeighValue[k] = (*childNeighValueIt++)
                  * predWeightChild[i];
              else
                childNeighValue[k] = (*childNeighValueIt++)
                  * (predWeightChild[i] << pcc::FixedPoint::kFracBits);

            for (int k = 0; k < numAttrs; k++)
              predBuf[k][j].val += childNeighValue[k];

            if (enableACInterPred) {
              auto intraChildNeighValueIt =
                std::next(intraFirstChild, numAttrs * childNeighIdx[i][j]);
              for (int k = 0; k < numAttrs; k++)
                if (rahtExtension)
                  intraChildNeighValue[k] =
                    (*intraChildNeighValueIt++) * predWeightChild[i];
                else
                  intraChildNeighValue[k] = (*intraChildNeighValueIt++)
                    * (predWeightChild[i] << pcc::FixedPoint::kFracBits);
              for (int k = 0; k < numAttrs; k++)
                intraPredBuf[k][j].val += intraChildNeighValue[k];
            }
          } else {
            weightSum[j] += predWeightParent[7 + i];
            for (int k = 0; k < numAttrs; k++) {
              predBuf[k][j].val += neighValue[k];
              if (enableACInterPred)
                intraPredBuf[k][j].val += neighValue[k];
            }
          }
        }
      }
    }
  }

  // normalise
  FixedPoint div;
  for (int i = 0; i < 8; i++, occupancy >>= 1) {
    if (occupancy & 1) {
      div.val = kDivisors[weightSum[i]];
      for (int k = 0; k < numAttrs; k++) {
        predBuf[k][i] *= div;
        if (enableACInterPred)
          intraPredBuf[k][i] *= div;
      }
      if (haarFlag) {
        for (int k = 0; k < numAttrs; k++) {
          predBuf[k][i].val = (predBuf[k][i].val >> predBuf[k][i].kFracBits)
            << predBuf[k][i].kFracBits;
          if (enableACInterPred)
            intraPredBuf[k][i].val =
              (intraPredBuf[k][i].val >> intraPredBuf[k][i].kFracBits)
              << intraPredBuf[k][i].kFracBits;
        }
      }
    }
  }
}

//============================================================================
int getRate(int trainZeros)
{
  static const int LUTbins[11] = { 1,2,3, 5,5, 7,7, 9,9 ,11 ,11 };
  int Rate = LUTbins[trainZeros > 10 ? 10 : trainZeros];
  if (trainZeros > 10) {
    int temp = trainZeros - 11;
    // prefix k =2
    temp += 1;
    int a = 0;
    while (temp) {
      a++;
      temp >>= 1;
    }
    Rate += 2 * a - 1;
    // suffix  k=2
    Rate += 2;
  }
  return Rate;
}

//============================================================================
// estimate filter tap by a binary search

int getFilterTap (int64_t autocorr, int64_t crosscorr)
{
  //binary search to replace divison. ( returns 128*crosscorr/autocorr)
  if (crosscorr == 0)
    return 0;
  
  bool isneg = crosscorr < 0;
  crosscorr = abs(crosscorr);
  
  if (crosscorr == autocorr)
    return (isneg ? -128 : 128);
  
  int tapint = 0, tapfrac = 0;
  
  //determine integer part by repeated subtraction
  while (crosscorr >= autocorr) {
    crosscorr -= autocorr;
    tapint += 128;
  }
  if (crosscorr == 0) {
    return (isneg ? -tapint : tapint);
  }
  
  int min = 0, max = 128;
  
  while (min < (max - 1)) {
    int mid = (min+max) >> 1;
    int64_t midval = (mid*autocorr)>>7;
    if (crosscorr == midval) {
      tapfrac = mid; 
      return (isneg ? -(tapint + tapfrac) : (tapint + tapfrac));
    }
    else if (crosscorr < midval) {
      max = mid;
    }
    else {
      min = mid;
    }
  }
  tapfrac = min;
  return  (isneg ? -(tapint+tapfrac) : (tapint+tapfrac));
  
}

//============================================================================
template<int numAttrs>
int
estimate_layer_filter(
  const std::vector<UrahtNode>& weightsLf,
  const std::vector<UrahtNode>& weightsLf_ref,
  const std::vector<int64_t>& attrRecParentUs,
  int level, int level_ref, bool inheritDc,  bool rahtExtension
){
  int64_t autocorr = 0, crosscorr = 0;
  int layerFilter = 128;
  int pit=0;
  for (int i = 0, j = 0, iLast, jLast, iEnd = weightsLf.size(), jEnd = weightsLf_ref.size(); i < iEnd; i = iLast) {
    FixedPoint transformBuf[6][8] = {};
    FixedPoint transformInterPredBuf[3][8] = {};
    int weights[8 + 8 + 8 + 8] = {};
    Qps nodeQp[8] = {};
    uint8_t occupancy = 0;
    int nodeCnt = 0;
    int64_t InheritedDCs[3]={};
    FixedPoint finterDC[3] = {0}, interParentMean[3] = {0};
    for(int k=0; k<numAttrs; k++)
    {
      if(inheritDc)
      {
        InheritedDCs[k]= attrRecParentUs[pit];
        pit++;
      }
    }
    int weights_ref[8 + 8 + 8 + 8] = {};
    uint64_t sumWeights_ref = 0;
    bool interNode = false;
    
    const auto cur_pos = weightsLf[i].pos >> (level + 3);
    auto ref_pos = j < jEnd-1 ? weightsLf_ref[j].pos >> (level_ref + 3) : 0x7FFFFFFFFFFFFFFFLL;
    while ((j < jEnd-1) && (cur_pos > ref_pos)) {
      j++;
      ref_pos = weightsLf_ref[j].pos >> (level_ref + 3);
    }
    if (cur_pos == ref_pos) {
      interNode = true;
    }

    for (iLast = i; iLast < iEnd; iLast++) {
      int nextNode = iLast > i
      && !isSibling(weightsLf[iLast].pos, weightsLf[i].pos, level + 3);
      if (nextNode)
        break;
      
      int nodeIdx = (weightsLf[iLast].pos >> level) & 0x7;
      weights[nodeIdx] = weightsLf[iLast].weight;
      
      occupancy |= 1 << nodeIdx;
      
      if (rahtExtension)
        nodeCnt++;
  
      for (int k = 0; k < numAttrs; k++)
        transformBuf[k][nodeIdx] = weightsLf[iLast].sumAttr[k];
    }

    if (rahtExtension && nodeCnt == 1) {
      interNode = false;
    }
    if (interNode == false)
      continue;

    if (interNode) {
      for (jLast = j; jLast < jEnd; jLast++) {
        int nextNode = jLast > j
        && !isSibling(weightsLf_ref[jLast].pos, weightsLf_ref[j].pos, level_ref + 3);
        if (nextNode)
          break;
        int nodeIdx = (weightsLf_ref[jLast].pos >> level_ref) & 0x7;
        weights_ref[nodeIdx] = weightsLf_ref[jLast].weight;
        sumWeights_ref += (uint64_t)weights_ref[nodeIdx];
        for (int k = 0; k < numAttrs; k++){
          transformInterPredBuf[k][nodeIdx] = weightsLf_ref[jLast].sumAttr[k];
          finterDC[k] += transformInterPredBuf[k][nodeIdx];
        }
      }

      FixedPoint rsqrtWeightSumRef(0);
      uint64_t w = sumWeights_ref;
      int shiftBits = 5 * ((w > 1024) + (w > 1048576));
      rsqrtWeightSumRef.val = fastIrsqrt(w) >> (40 - shiftBits - FixedPoint::kFracBits);
      for (int k = 0; k < numAttrs; k++) {
        finterDC[k].val >>= shiftBits;
        finterDC[k] *= rsqrtWeightSumRef;
        interParentMean[k].val = finterDC[k].val;
        interParentMean[k].val >>= shiftBits;
        interParentMean[k] *= rsqrtWeightSumRef;
	    }

      int64_t curinheritDC = (inheritDc)? InheritedDCs[0] : 0;
      int64_t interDC = finterDC[0].val;
      if(curinheritDC > 0 && (interDC > 0)){
        bool condition1 = 10 * interDC < ((curinheritDC) * 5);
        bool condition2 = 10 * interDC > ((curinheritDC) * 20);
        if(condition1 || condition2){
          interNode = false;
        }
      }
    }
 
    mkWeightTree(weights);
    
    if (interNode) {
      for (int childIdx = 0; childIdx < 8; childIdx++) {
        if(weights[childIdx] == 0){
          for(int k = 0; k < numAttrs; k++){
            transformInterPredBuf[k][childIdx].val = 0;
          }
          continue;
        } else if(weights_ref[childIdx] == 0){
          for(int k = 0; k < numAttrs; k++){
            transformInterPredBuf[k][childIdx].val = interParentMean[k].val;
          }
        }
        if(weights_ref[childIdx] > 1) {
          FixedPoint rsqrtWeight;
          uint64_t w = weights_ref[childIdx];
          int shift = 5 * ((w > 1024) + (w > 1048576));
          rsqrtWeight.val = fastIrsqrt(w) >> (40 - shift - FixedPoint::kFracBits);
          for (int k = 0; k < numAttrs; k++) {
            transformInterPredBuf[k][childIdx].val >>= shift;
            transformInterPredBuf[k][childIdx] *= rsqrtWeight;
            transformInterPredBuf[k][childIdx].val >>= shift;
            transformInterPredBuf[k][childIdx] *= rsqrtWeight; //mean attribute
          }
        }
        if(weights[childIdx] > 1){
          FixedPoint sqrtWeight;
          sqrtWeight.val = fastIsqrt(weights[childIdx]);
          for (int k = 0; k < numAttrs; k++) {
            transformInterPredBuf[k][childIdx] *= sqrtWeight; //sum of attribute
          }
        } 
      }
    }
    
    for (int childIdx = 0; childIdx < 8; childIdx++) {
      if (weights[childIdx] <= 1)
        continue;
      
      // Summed attribute values
      if (true) {
        FixedPoint rsqrtWeight;
        uint64_t w = weights[childIdx];
        int shift = 5 * ((w > 1024) + (w > 1048576));
        rsqrtWeight.val = fastIrsqrt(w) >> (40 - shift - FixedPoint::kFracBits);
        for (int k = 0; k < numAttrs; k++) {
          transformBuf[k][childIdx].val >>= shift;
          transformBuf[k][childIdx] *= rsqrtWeight;
        }
      }
    }
    
    if (interNode) {
      
      fwdTransformBlock222<RahtKernel>(numAttrs, transformBuf, weights);
      fwdTransformBlock222<RahtKernel>(numAttrs, transformInterPredBuf, weights);

      scanBlock(weights, [&](int idx) {
        if (inheritDc && !idx)
          return;
          
        int shiftbits = transformBuf[0][idx].kFracBits;
        int64_t refVal = transformInterPredBuf[0][idx].val;
        if (refVal) {
          autocorr += (refVal*refVal) >> shiftbits;
          crosscorr += (refVal*transformBuf[0][idx].val) >> shiftbits;
        }
      });  
    }
  }
  if (autocorr) {
    layerFilter = getFilterTap(autocorr, crosscorr);
  }
  return layerFilter;
}

const int FLOOR = 0;
const int CEIL = 1;
const int nQ = 2;
static void getRDOData(
  FixedPoint& transCoeffFP,
  const Quantizer& q,
  int64_t(&qCoeffs)[nQ],
  int64_t(&dqCoeffs)[nQ],
  int64_t(&dist)[nQ],
  int64_t predCoeffVal = 0
)
{
  int64_t coeff;
  if (predCoeffVal) { // Cr coeff is predicted from Cb coeff, need to subtract prediction get pre-quantized coeff
    FixedPoint residCoeff = 0;
    residCoeff.val = transCoeffFP.val - predCoeffVal;
    coeff = residCoeff.round();
  }
  else {
    coeff = transCoeffFP.round();
    assert(coeff <= INT_MAX && coeff >= INT_MIN);
  }
  // prepare quantized candidate coeffs (to compute rate)
  qCoeffs[FLOOR] = q.quantize_no_deadzone(coeff << kFixedPointAttributeShift);
  qCoeffs[CEIL] = qCoeffs[FLOOR] + ((coeff > 0) ? 1 : -1);
  // prepare dequantized candidate coeffs (to compute distortion)
  if (predCoeffVal) { // Cr coeff is predicted Cr coeff, need to add prediction to get final dequantized Cr coeff
    FixedPoint tmp = divExp2RoundHalfUp(q.scale(qCoeffs[FLOOR]), kFixedPointAttributeShift); // FixedPoint = int, operator=(int) does conversion
    tmp.val += predCoeffVal;
    dqCoeffs[FLOOR] = tmp.round();
    tmp = divExp2RoundHalfUp(q.scale(qCoeffs[CEIL]), kFixedPointAttributeShift); // FixedPoint = int, operator=(int) does conversion
    tmp.val += predCoeffVal;
    dqCoeffs[CEIL] = tmp.round();
  }
  else {
    dqCoeffs[FLOOR] = divExp2RoundHalfUp(q.scale(qCoeffs[FLOOR]), kFixedPointAttributeShift);
    dqCoeffs[CEIL] = divExp2RoundHalfUp(q.scale(qCoeffs[CEIL]), kFixedPointAttributeShift);
  }
  // compute distortion
  int64_t diff = dqCoeffs[FLOOR] - transCoeffFP.round();
  dist[FLOOR] = diff * diff;
  diff = dqCoeffs[CEIL] - transCoeffFP.round();
  dist[CEIL] = diff * diff;
}
inline bool allZero(int64_t(&arr)[3])
{
  return (arr[0] == 0) && (arr[1] == 0) && (arr[2] == 0);
}

template <int numCh>
static void doRDOQ(
  int numAttrs,
  Quantizers& quantizers,
  int idx,
  int8_t LcpCoeff,
  FixedPoint (&transformBuf)[numCh][8],
  int64_t (&bestQCoeffs)[3],
  bool enableLCPPred,
  bool haarFlag,
  const std::array<double, 4>& rdoqFactors,
  int trainZeros,
  PCCResidualsEncoder* encoder
)
 {
  int64_t distForZTDecision[3] = {};
  FixedPoint dqCoeffOrig = {};
  const int Cb = 1;
  const int Cr = 2;

  // Quantize-dequantize naturally
  for (int k = 0; k < numAttrs; k++) {
    const auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];
    if (!enableLCPPred || k != Cr) {
      // prepare and store quantized/dequantized original coeff (to bypass RDO decision)
      const auto qCoeffOrig = q.quantize(transformBuf[k][idx].round() << kFixedPointAttributeShift);
      bestQCoeffs[k] = qCoeffOrig;
      dqCoeffOrig = divExp2RoundHalfUp(q.scale(qCoeffOrig), kFixedPointAttributeShift);
      const int64_t diff = transformBuf[k][idx].round() - dqCoeffOrig.round();
      distForZTDecision[k] = diff * diff;
    }
    else { // Cr coeff for "raht_last_component_prediction_enabled_flag" == true
      // prepare and store quantized original Cr coeff (to bypass RDO decision)
      FixedPoint tmp = {};
      tmp.val = transformBuf[Cr][idx].val - ((LcpCoeff * dqCoeffOrig.val) >> 4);
      auto coeff = tmp.round();
      const auto qCoeffOrig = q.quantize(coeff << kFixedPointAttributeShift);
      bestQCoeffs[Cr] = qCoeffOrig;
      tmp = divExp2RoundHalfUp(q.scale(qCoeffOrig), kFixedPointAttributeShift);
      tmp.val += LcpCoeff * dqCoeffOrig.val >> 4;;
      const int64_t diff = transformBuf[k][idx].round() - tmp.round();
      distForZTDecision[Cr] = diff * diff;
    }
  }

  if (!haarFlag && numAttrs == 3) {
    // omit RDOQ is all the coeffs are less than qStep/2
    auto& q = quantizers[std::min(0, int(quantizers.size()) - 1)];
    const int64_t stepSz = q.scale(1);
    const double qStep = static_cast<double>(stepSz) / (1ll << kFixedPointAttributeShift);

    const double threshold = qStep / 2;
    const bool omitRDOQ = (std::abs(transformBuf[0][idx].round()) < threshold) && (std::abs(transformBuf[1][idx].round()) < threshold) && (std::abs(transformBuf[2][idx].round()) < threshold);

    // do RDOQ
    if (!omitRDOQ) {
      const int numComb = 1 << 3;

      int64_t qCoeffsBuf[numComb + 1][3] = {};
      int64_t distBuf[numComb + 1][3] = {};
      int64_t rateBuf[numComb][3] = {};
      FixedPoint dqCoeffsBuf[numComb] = {}; // needed only for raht_last_component_prediction_enabled_flag to store Cb dequantized candidate

      // collect distortions for zero triplet
      for (int k = 0; k < numAttrs; k++) {
        qCoeffsBuf[numComb][k] = 0; // zero triplet
        auto coeff = transformBuf[k][idx].round();
        distBuf[numComb][k] = coeff * coeff;
      }

      const bool doRDOQNTZ = true;
      if (doRDOQNTZ) {
        // collect data for RDO NZT (Non-Zero Triplet) decision
        for (int k = 0; k < numAttrs; k++) {
          const auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];
          if (!enableLCPPred || k != Cr) {
            int64_t qCoeffs[nQ] = {};
            int64_t dqCoeffs[nQ] = {};
            int64_t dist[nQ] = {};
            int64_t rate[nQ] = {};
            getRDOData(transformBuf[k][idx], q, qCoeffs, dqCoeffs, dist);
            if (k == Cb) {
              rate[FLOOR] = encoder->encodegetbitvalueCb(0, qCoeffs[FLOOR], 0);
              rate[CEIL] = encoder->encodegetbitvalueCb(0, qCoeffs[CEIL], 0);
            }
            for (int i = 0; i < numComb; i++) {
              const int idx = (i & (1 << k)) >> k; // mix FLOOR and CEIL to get all possible combinations of candidates for all 3 components
              // store quantized cantidate coeffs and distortions (to later compute RD cost)
              qCoeffsBuf[i][k] = qCoeffs[idx];
              distBuf[i][k] = dist[idx];
              rateBuf[i][k] = rate[idx];
              // store dequantized Cb candidate coeff to predict Cr candidate coeff
              if (enableLCPPred && k == Cb)
                dqCoeffsBuf[i] = dqCoeffs[idx]; // FixedPoint = int, operator=(int) does conversion
            }
          }
          else { // Cr coeff for "raht_last_component_prediction_enabled_flag" == true
            int64_t qCoeffs[nQ][nQ] = {};
            int64_t dqCoeffs[nQ][nQ] = {};
            int64_t dist[nQ][nQ] = {};
            int64_t rate[nQ][nQ] = {};
            const int FROM_FLOOR = 0; // predicted from FLOOR dequantized Cb candidate
            const int FROM_CEIL = 1; // predicted from CEIL dequantized Cb candidate
            // predict Cr coeff from dequantized Cb candidate
            int64_t predCoeffVal[nQ] = {};
            predCoeffVal[FROM_FLOOR] = (LcpCoeff * dqCoeffsBuf[0].val) >> 4;
            predCoeffVal[FROM_CEIL] = (LcpCoeff * dqCoeffsBuf[2].val) >> 4;
            // process Cr coeff predicted from FLOOR dequantized Cb candidate
            getRDOData(transformBuf[k][idx], q, qCoeffs[FROM_FLOOR], dqCoeffs[FROM_FLOOR], dist[FROM_FLOOR], predCoeffVal[FROM_FLOOR]);
            rate[FROM_FLOOR][FLOOR] = encoder->encodegetbitvalueCr(0, qCoeffsBuf[0][Cb], qCoeffs[FROM_FLOOR][FLOOR]);
            rate[FROM_FLOOR][CEIL] = encoder->encodegetbitvalueCr(0, qCoeffsBuf[0][Cb], qCoeffs[FROM_FLOOR][CEIL]);
            // process Cr coeff predicted from CEIL dequantized Cb candidate
            getRDOData(transformBuf[k][idx], q, qCoeffs[FROM_CEIL], dqCoeffs[FROM_CEIL], dist[FROM_CEIL], predCoeffVal[FROM_CEIL]);
            rate[FROM_CEIL][FLOOR] = encoder->encodegetbitvalueCr(0, qCoeffsBuf[2][Cb], qCoeffs[FROM_CEIL][FLOOR]);
            rate[FROM_CEIL][CEIL] = encoder->encodegetbitvalueCr(0, qCoeffsBuf[2][Cb], qCoeffs[FROM_CEIL][CEIL]);

            for (int i = 0; i < numComb; i++) {
              const int idxFrom = (i & (1 << Cb)) >> Cb;
              const int idx = (i & (1 << Cr)) >> Cr;
              // store quantized cantidate coeffs and distortions (to later compute RD cost)
              qCoeffsBuf[i][Cr] = qCoeffs[idxFrom][idx];
              distBuf[i][Cr] = dist[idxFrom][idx];
              rateBuf[i][Cr] = rate[idxFrom][idx];
            }
          }
        }

        // do NZT decision
        int64_t rdCosts[numComb] = {};
        auto& q = quantizers[std::min(0, int(quantizers.size()) - 1)]; // so far use only Luma QP for lambda computation
        const int64_t stepSz = q.scale(1);
        const double qStep = static_cast<double>(stepSz) / (1ll << kFixedPointAttributeShift);
        const double alphaY = rdoqFactors[0];
        const double alphaCb = rdoqFactors[1];
        const double alphaCr = rdoqFactors[2];
        double lambdaY = (alphaY * qStep * qStep) / 32768;
        double lambdaCb = (alphaCb * qStep * qStep) / 32768;
        double lambdaCr = (alphaCr * qStep * qStep) / 32768;
        int64_t bestCost = std::numeric_limits<int64_t>::max();
        int bestComb = 0;

        for (int i = 0; i < numComb; i++) {
          if (allZero(qCoeffsBuf[i]))
            continue;
          const uint64_t dist = distBuf[i][0] + distBuf[i][1] + distBuf[i][2]; // SSE(orig_coeffs - recon_coeffs)
          const int64_t rateY = encoder->encodegetbitvalueY(qCoeffsBuf[i][0], qCoeffsBuf[i][1], qCoeffsBuf[i][2]);
          const int64_t rateCb = rateBuf[i][1];
          const int64_t rateCr = rateBuf[i][2];

          const int64_t cost = dist + lambdaY * rateY + lambdaCb * rateCb + lambdaCr * rateCr;
          if (cost < bestCost) {
            bestCost = cost;
            bestComb = i;
          }
        }
        bestQCoeffs[0] = qCoeffsBuf[bestComb][0];
        bestQCoeffs[1] = qCoeffsBuf[bestComb][1];
        bestQCoeffs[2] = qCoeffsBuf[bestComb][2];
        distForZTDecision[0] = distBuf[bestComb][0];
        distForZTDecision[1] = distBuf[bestComb][1];
        distForZTDecision[2] = distBuf[bestComb][2];
      }

      // do ZT (Zero Triplet) decition
      const int sumCoeff = std::abs(bestQCoeffs[0]) + std::abs(bestQCoeffs[1]) + std::abs(bestQCoeffs[2]);
      const bool doRDOQZT = !allZero(bestQCoeffs) && (sumCoeff < 3);
      if (doRDOQZT) {
        auto& q = quantizers[std::min(0, int(quantizers.size()) - 1)];
        const int64_t stepSz = q.scale(1);
        const double qStep = static_cast<double>(stepSz) / (1ll << kFixedPointAttributeShift);
        const double alpha = rdoqFactors[3];
        double lambda = (alpha * qStep * qStep) / 32768;
        const int64_t distZT = distBuf[numComb][0] + distBuf[numComb][1] + distBuf[numComb][2];
        const int64_t costZT = distZT; // assume zero triplet doesn't cost any bits
        const int64_t rateNZT = encoder->encodeRunLengthgetbit(trainZeros)
          + encoder->encodegetbitvalue(bestQCoeffs[0], bestQCoeffs[1], bestQCoeffs[2], 3); // assume non-zero triplet costs all the bits including bits for zero-run
        const int64_t costNZT = distForZTDecision[0] + distForZTDecision[1] + distForZTDecision[2] + lambda * rateNZT;
        if (costZT < costNZT) {
          bestQCoeffs[0] = bestQCoeffs[1] = bestQCoeffs[2] = 0;
        }
      }
    }
  }
}

//============================================================================
// Core transform process (for encoder)
template<bool haarFlag, int numAttrs, bool rahtExtension, class ModeCoder>
void
uraht_process_encoder(
  const RahtPredictionParams& rahtPredParams,
  AttributeBrickHeader& abh,
  const QpSet& qpset,
  const Qps* pointQpOffsets,
  int numPoints,
  int64_t* positions,
  int* attributes,
  int32_t* coeffBufIt,
  AttributeInterPredParams& attrInterPredParams,
  ModeCoder& coder,
  PCCResidualsEncoder* encoder = nullptr,
  const std::array<double, 4>* rdoqFactors = nullptr
)
{
  int zeroRun = 0;
  const bool newRDOQ = (rdoqFactors != nullptr);
  const bool is420 = abh.is420;
  int& first420 = abh.first420;
  first420 = -1;

  // coefficients are stored in three planar arrays.  coeffBufItK is a set
  // of iterators to each array.
  int32_t* coeffBufItK[3] = {
    coeffBufIt, coeffBufIt + numPoints, coeffBufIt + numPoints * 2};

  // early termination only one point
  if (numPoints == 1) {
    auto quantizers = qpset.quantizers(0, pointQpOffsets[0]);
    for (int k = 0; k < numAttrs; k++) {
      auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];
      auto coeff = attributes[k];
      assert(coeff <= INT_MAX && coeff >= INT_MIN);
      *coeffBufItK[k]++ = coeff =
        q.quantize(coeff << kFixedPointAttributeShift);
      attributes[k] =
        divExp2RoundHalfUp(q.scale(coeff), kFixedPointAttributeShift);
    }
    return;
  }

  bool enableLCPPred = rahtPredParams.raht_last_component_prediction_enabled_flag;

  bool enableACInterPred = attrInterPredParams.enableAttrInterPred;

  bool enableACRDOInterPred =
    attrInterPredParams.paramsForInterRAHT.raht_enable_inter_intra_layer_RDO
    && enableACInterPred;

  bool enableACRDONonPred = rahtPredParams.raht_enable_intraPred_nonPred_code_layer
    && rahtPredParams.raht_prediction_enabled_flag;

  int treeDepth = 0;
  int treeDepthLimit = 1 + attrInterPredParams.paramsForInterRAHT.raht_inter_prediction_depth_minus1;

  bool enableFilterEstimation = attrInterPredParams.paramsForInterRAHT.enableFilterEstimation;
  std::vector<int64_t> fixedFilterTaps = {128, 128, 128, 127, 125, 121, 115};
  int skipInitLayersForFiltering = attrInterPredParams.paramsForInterRAHT.skipInitLayersForFiltering;

  int regionQpShift = 4;
  const int maxAcCoeffQpOffsetLayers = qpset.rahtAcCoeffQps.size() - 1;

  std::vector<UrahtNode> weightsHf;
  std::vector<std::vector<UrahtNode>> weightsLfStack;

  weightsLfStack.emplace_back();
  weightsLfStack.back().reserve(numPoints);
  auto weightsLf = &weightsLfStack.back();

  // copy positions into internal form
  for (int i = 0; i < numPoints; i++) {
    UrahtNode node;
    node.pos = positions[i];
	node.weight = 1;
	node.qp = {
      int16_t(pointQpOffsets[i][0] << regionQpShift),
      int16_t(pointQpOffsets[i][1] << regionQpShift)};   
    for (int k = 0; k < numAttrs; k++) {
      node.sumAttr[k] = attributes[i * numAttrs + k];
    }
    weightsLf->emplace_back(node);
  }

  std::vector<UrahtNode> weightsHf_ref;
  std::vector<std::vector<UrahtNode>> weightsLfStack_ref;

  weightsLfStack_ref.emplace_back();
  weightsLfStack_ref.back().reserve(attrInterPredParams.paramsForInterRAHT.voxelCount);
  auto weightsLf_ref = &weightsLfStack_ref.back();

  if (enableACInterPred) {

    for (int i = 0; i < attrInterPredParams.paramsForInterRAHT.voxelCount; i++) {
      UrahtNode node_ref;
      node_ref.pos = attrInterPredParams.paramsForInterRAHT.mortonCode[i];
      node_ref.weight = attrInterPredParams.paramsForInterRAHT.dupPoints_available ? attrInterPredParams.paramsForInterRAHT.dupPoints[i]+1: 1;
      node_ref.qp = {0, 0};
      for (int k = 0; k < numAttrs; k++) {
        node_ref.sumAttr[k] =
          attrInterPredParams.paramsForInterRAHT.attributes[i * numAttrs + k];
        if (!haarFlag)
          node_ref.sumAttr[k] *= node_ref.weight;
      }
      weightsLf_ref->emplace_back(node_ref);
    }
  }
  
  // ascend tree
  int numNodes = weightsLf->size();
  // process any duplicate points
  numNodes = reduceUnique<haarFlag, numAttrs>(numNodes, weightsLf, &weightsHf);
  weightsLfStack[0].resize(numNodes);  //shrink

  const bool flagNoDuplicate = weightsHf.size() == 0;
  
  int numDepth = 0;
  for (int levelD = 3; numNodes > 1; levelD += 3) {
    // one depth reduction
    weightsLfStack.emplace_back();
    weightsLfStack.back().reserve(numNodes / 3);
    weightsLf = &weightsLfStack.back();

    auto weightsLfRefold = &weightsLfStack[weightsLfStack.size() - 2];
    numNodes = reduceDepth<haarFlag, numAttrs>(levelD, numNodes, weightsLfRefold, weightsLf);
    numDepth++;
  }

  int numDepth_ref = 0;
  if (enableACInterPred) {
    int numNodes = weightsLf_ref->size();
    numNodes = reduceUnique<haarFlag, numAttrs>(numNodes, weightsLf_ref, &weightsHf_ref);
    weightsLfStack_ref[0].resize(numNodes);

    for (int levelD = 3; numNodes > 1; levelD += 3) {
      // one depth reduction
      weightsLfStack_ref.emplace_back();
      weightsLfStack_ref.back().reserve(numNodes / 3);
      weightsLf_ref = &weightsLfStack_ref.back();

      auto weightsLfRefold_ref = &weightsLfStack_ref[weightsLfStack_ref.size() - 2];
      numNodes = reduceDepth<haarFlag, numAttrs>(
        levelD, numNodes, weightsLfRefold_ref, weightsLf_ref);
      numDepth_ref++;
    }
  }

  auto& rootNode = weightsLfStack.back()[0];
  assert(rootNode.weight == numPoints);

  // reconstruction buffers
  std::vector<int32_t> attrRec, attrRecParent, intraAttrRec, nonPredAttrRec;
  attrRec.resize(numPoints * numAttrs);
  attrRecParent.resize(numPoints * numAttrs);
  if (enableACRDOInterPred)
    intraAttrRec.resize(numPoints * numAttrs);
  if (enableACRDONonPred)
    nonPredAttrRec.resize(numPoints * numAttrs);

  std::vector<int64_t> attrRecUs, attrRecParentUs, intraAttrRecUs, nonPredAttrRecUs;
  attrRecUs.resize(numPoints * numAttrs);
  attrRecParentUs.resize(numPoints * numAttrs);
  if (enableACRDOInterPred)
    intraAttrRecUs.resize(numPoints * numAttrs);
  if (enableACRDONonPred)
    nonPredAttrRecUs.resize(numPoints * numAttrs);

  std::vector<UrahtNode> weightsParent;
  weightsParent.reserve(numPoints);

  std::vector<UrahtNode> weightsParent_ref;
  weightsParent_ref.reserve(weightsLfStack_ref[0].size());

  std::vector<int8_t> numParentNeigh, numGrandParentNeigh;
  numParentNeigh.resize(numPoints);
  numGrandParentNeigh.resize(numPoints);

  // quant layer selection
  auto qpLayer = 0;
  // AC coeff QP offset laer
  auto acCoeffQpLayer = -1;

  // For inter, intra and non-pred level RD-cost estimation
  PCCRAHTACCoefficientEntropyEstimate intraEstimate;
  PCCRAHTACCoefficientEntropyEstimate curEstimate;
  PCCRAHTACCoefficientEntropyEstimate nonPredEstimate;
  std::vector<int> intraACCoeffcients, nonPredACCoeffcients;
  if (enableACRDOInterPred) {
    intraACCoeffcients.resize(numPoints * numAttrs);
  }
  if (enableACRDONonPred) {
    nonPredACCoeffcients.resize(numPoints * numAttrs);
  }

  // For RDOQ
  int trainZeros = 0;
  int intraTrainZeros = 0;
  int nonPredTrainZeros = 0;
  // For LCP coeff computation
  int8_t LcpCoeff = 0;
  int8_t nonPredLcpCoeff = 0;
  int8_t intraPredLcpCoeff = 0;

  bool sameNumNodes = 0;

  double dlambda = 1.0;
  int filterIdx = 0;

  // ----------------------------------- descend tree, loop on depth ------------------------------------
  for (int levelD = numDepth, levelD_ref = numDepth_ref, isFirst = 1; levelD > 0; /*nop*/) {
    std::vector<UrahtNode>& weightsParent = weightsLfStack[levelD];
    std::vector<UrahtNode>& weightsLf = weightsLfStack[levelD - 1];

    sameNumNodes = (weightsLf.size() == weightsParent.size());
    int sumNodes = weightsLf.size() - weightsParent.size();

    levelD--;
    int level = 3 * levelD;

    if (levelD_ref <= 0) {
      enableACInterPred = false;
    }

    if (is420 && level == 0) {
      enableLCPPred = false;
    }

    if (treeDepth >= treeDepthLimit)
      enableACInterPred = false;

    std::vector<UrahtNode>& weightsParent_ref = enableACInterPred
      ? weightsLfStack_ref[levelD_ref]  // to avoid copying and use reference
      : weightsLfStack_ref[0];          // will not be used
	
    std::vector<UrahtNode>& weightsLf_ref = enableACInterPred
      ? weightsLfStack_ref[levelD_ref - 1]  // to avoid copying and use reference
      : weightsLfStack_ref[0]; // will not be used

    int level_ref = 0;
    if (enableACInterPred) {
      levelD_ref--;
      level_ref = 3 * levelD_ref;
    }

    //if current level nodes number is equal to previous nodes level, skip current level
    if (sameNumNodes)
      continue;

    // initial scan position of the coefficient buffer
    //  -> first level = all coeffs
    //  -> otherwise = ac coeffs only
    bool inheritDc = !isFirst;
    bool enablePredictionInLvl = inheritDc && rahtPredParams.raht_prediction_enabled_flag;
    isFirst = 0;
	
	//--------------- initialize parameters of layer RDO for current level ------------
    enableACRDOInterPred =
      attrInterPredParams.paramsForInterRAHT.raht_enable_inter_intra_layer_RDO
      && enableACInterPred && enablePredictionInLvl;

	enableACRDONonPred = enablePredictionInLvl
      ? rahtPredParams.raht_enable_intraPred_nonPred_code_layer
      : false;

    const bool& enableRDOCodingLayer = enableACRDOInterPred || enableACRDONonPred;

    const bool& hybridPredLayer = rahtPredParams.raht_hybrid_prediction_enabled_flag && !haarFlag &&
    (treeDepth >= rahtPredParams.raht_hybrid_prediction_lower_depth_minus1 + 1) &&
    (treeDepth <= rahtPredParams.raht_hybrid_prediction_lower_depth_minus1 + rahtPredParams.raht_hybrid_prediction_num_enabled_layers);

	//For intra layer
    int32_t* intraCoeffBufItK[3] = {
      intraACCoeffcients.data(),
      intraACCoeffcients.data() + sumNodes,
      intraACCoeffcients.data() + sumNodes * 2,
    };
    int32_t* intraCoeffBufItBeginK[3] = {
      intraCoeffBufItK[0],
      intraCoeffBufItK[1],
      intraCoeffBufItK[2],
    };
    //For non-pred layer
    int32_t* nonPredCoeffBufItK[3] = {
      nonPredACCoeffcients.data(),
      nonPredACCoeffcients.data() + sumNodes,
      nonPredACCoeffcients.data() + sumNodes * 2,
    };
    int32_t* nonPredCoeffBufItBeginK[3] = {
      nonPredCoeffBufItK[0],
      nonPredCoeffBufItK[1],
      nonPredCoeffBufItK[2],
    };
    //for current layer
    int32_t* coeffBufItBeginK[3] = {
      coeffBufItK[0],
      coeffBufItK[1],
      coeffBufItK[2],
    };

    // Control whether current level enables inter and intra rdo
    bool curLevelEnableACInterPred = false;
    // Control whether current level enables intra and non-pred rdo
    bool curLevelEnableACIntraPred = false;
    if (enableRDOCodingLayer) {
      curLevelEnableACInterPred = enableACRDOInterPred;
      curLevelEnableACIntraPred = enableACRDONonPred;
    }

	double distinter = 0;
    double distintra = 0;
    double distnonPred = 0;
    FixedPoint origsamples[3][8] = {0};

	//--------------- initialize LCP coeff for current level ------------ 
    LcpCoeff = 0;
    nonPredLcpCoeff = 0;
    intraPredLcpCoeff = 0;
    PCCRAHTComputeLCP curlevelLcp;
    PCCRAHTComputeLCP curlevelLcpIntra;
    PCCRAHTComputeLCP curlevelLcpNonPred;

	//--------------- initialize parent node information for current level ------------ 
    if (enablePredictionInLvl) {
      for (auto& ele : weightsParent)
        ele.occupancy = 0;
    }
    
    //--------------- select quantiser according to transform layer ------------ 
    qpLayer = std::min(qpLayer + 1, int(qpset.layers.size()) - 1);
    acCoeffQpLayer++;

    //--------------- prepare reconstruction buffers ------------ 
    // previous reconstruction -> attrRecParent
    std::swap(attrRec, attrRecParent);
    std::swap(attrRecUs, attrRecParentUs);
    std::swap(numParentNeigh, numGrandParentNeigh);
    auto attrRecParentUsIt = attrRecParentUs.cbegin();
    auto attrRecParentIt = attrRecParent.cbegin();
    auto numGrandParentNeighIt = numGrandParentNeigh.cbegin();

    std::vector<int32_t> nonZeroData;
    if (levelD == 0) {    
     nonZeroData.resize(attrRecParent.size());
     nonZeroData.clear();
         for (int32_t value : attrRecParent) {
             if (value != 0) {
                 nonZeroData.push_back(value);
        //if (enableACInterPred)
        //std::cout << " value: "
        //         << value << std::endl;
             }
         }
    }
    
	//---------------- estimate inter filter of current level if enable---------------------
    int64_t interFilterTap = 128;
    if ((!enableFilterEstimation) && (enableACInterPred) && (treeDepth < treeDepthLimit)) {
      int filtexidx = treeDepth < fixedFilterTaps.size() ? treeDepth : (fixedFilterTaps.size()-1);
      interFilterTap = fixedFilterTaps[filtexidx];
    }

    bool enableEstimateLayer = enableFilterEstimation && enableACInterPred &&
	  (treeDepth < treeDepthLimit) && (treeDepth >= skipInitLayersForFiltering);

    //begin filter estimation at encoder
    int64_t quantizedResFilterTap = 0;
    if (enableEstimateLayer ) {
      int origFilterTap = estimate_layer_filter<numAttrs>(weightsLf, weightsLf_ref,
		attrRecParentUs, level, level_ref, inheritDc, rahtExtension);
      attrRecParentUsIt = attrRecParentUs.cbegin();
      int residueFilterTap = 128 - origFilterTap;
      auto quantizers = qpset.quantizers(qpLayer, {0,0});
      auto& q = quantizers[0];
      quantizedResFilterTap = q.quantize(residueFilterTap << kFixedPointAttributeShift);
      int64_t recResidueFilterTap = divExp2RoundHalfUp(q.scale(quantizedResFilterTap), kFixedPointAttributeShift);
	  interFilterTap = 128 - recResidueFilterTap;
    } //end filter estimation

	// ----------------------------- loop on nodes of current level -----------------------------------
    int i = 0;
    int j = 0, jLast = 0, jEnd = weightsLf_ref.size();

    if (enableACInterPred && (levelD == 0)) {
    int cousincount = 0;
    int per_cousin = 0;
         for (auto weightsParentIt_ref = weightsParent_ref.begin();
         weightsParentIt_ref < weightsParent_ref.end();
         /*nop*/) {
             //uint8_t occupancy_ref = 0;
             //int weightsParent_ref_childcount = 0;
             for (auto it = weightsParentIt_ref->firstChild;
                  it != weightsParentIt_ref->lastChild; it++) {
                 //int nodeIdx_ref = (it->pos >> level_ref) & 0x7;
                 //occupancy_ref |= 1 << nodeIdx_ref;
                 //weightsParent_ref_childcount++;
                 cousincount++;
             }
             //std::cout << "weightsParent_ref_childcount:"
             //<< weightsParent_ref_childcount << std::endl;
             //std::cout << "per_cousin:" << per_cousin << std::endl;
             //for (int m = per_cousin; m < cousincount; m++) {
             //    //weightsLf_ref[m].cousincount = weightsParent_ref_childcount;
             //    //weightsLf_ref[m].cousin_occupancy = occupancy_ref;
             //    for (int k = 0; k < numAttrs; k++) {
             //        weightsLf_ref[m].parent_sumAttr[k] =
             //            weightsParentIt_ref->sumAttr[k];
             //    }
             //}
             for (int k = 0; k < numAttrs; k++) {
                  weightsLf_ref[per_cousin].parent_sumAttr[k] =
                    weightsParentIt_ref->sumAttr[k];
             }
             per_cousin = cousincount;
             //std::bitset<8> binaryOccupancy(occupancy_ref);
             //std::cout << "binaryOccupancy:" << binaryOccupancy << std::endl;
             //weightsParentIt_ref->occupancy = occupancy_ref;
             //weightsParentIt_ref->childcount = weightsParent_ref_childcount;
             weightsParentIt_ref++;
         }    

    
    }

    int rec_cur_attr_count;
    rec_cur_attr_count = 0;

	for (auto weightsParentIt = weightsParent.begin(); weightsParentIt < weightsParent.end(); /*nop*/){

      if (enableACInterPred && (levelD == 0)) {
         for (int k = 0; k < numAttrs; k++) {
             weightsParentIt->parent_sumAttr[k] = nonZeroData[rec_cur_attr_count * numAttrs + k];
         }
      }
      rec_cur_attr_count++;

      FixedPoint SampleBuf[6][8] = {0}, transformBuf[6][8] = {0};
      FixedPoint (*SamplePredBuf)[8] = &SampleBuf[numAttrs], (*transformPredBuf)[8] = &transformBuf[numAttrs];
      FixedPoint SampleInterPredBuf[3][8] = {0}, transformInterPredBuf[3][8] = {0};
      FixedPoint PredDC[3] = {0};
      FixedPoint NodeRecBuf[3][8] = {0};
      FixedPoint normalizedSqrtBuf[8] = {0};

      // For intra layer prediction
      FixedPoint SampleIntraPredBuf[3][8] = {0}, transformIntraPredBuf[3][8] = {0};
      FixedPoint transformIntraBuf[3][8] = {0};
      FixedPoint PredAllIntraDC[3] = {0};
      FixedPoint NodeAllIntraRecBuf[3][8] = {0};

      // For Non pred layer prediction
      FixedPoint transformNonPredBuf[3][8] = {0};
      FixedPoint NodeNonPredRecBuf[3][8] = {0};

	  // For Lcp prediction
	  int64_t CoeffRecBuf[8][3] = {0};
      int64_t nonPredCoeffRecBuf[8][3] = {0};
      int64_t intraPredCoeffRecBuf[8][3] = {0};
      FixedPoint transformResidueRecBuf[3] = {0};
      FixedPoint transformNonPredResRecBuf[3] = {0};
      FixedPoint transformIntraPredResRecBuf[3] = {0}; 
	  int nodelvlSum = 0;

	  // ---- now compute information of current node ----
      int weights[8 + 8 + 8 + 8] = {};
      uint64_t sumWeights_cur = 0; 
      Qps nodeQp[8] = {};
      uint8_t occupancy = 0;   
      int nodeCnt = 0;
      int nodeCnt_real = 0;

	  for (auto it = weightsParentIt->firstChild; it != weightsParentIt->lastChild; it++) {
        int nodeIdx = (it->pos >> level) & 0x7;
        weights[nodeIdx] = it->weight;
        sumWeights_cur += (uint64_t)weights[nodeIdx];
        nodeQp[nodeIdx][0] = it->qp[0] >> regionQpShift;
        nodeQp[nodeIdx][1] = it->qp[1] >> regionQpShift;

        occupancy |= 1 << nodeIdx;
        nodeCnt_real++;

        if (rahtExtension)
          nodeCnt++;

        for (int k = 0; k < numAttrs; k++)
          SampleBuf[k][nodeIdx] = it->sumAttr[k];
      }

      if (!inheritDc) {
        for (int j = i, nodeIdx = 0; nodeIdx < 8; nodeIdx++) {
          if (!weights[nodeIdx])
            continue;
          numParentNeigh[j++] = 19;
        }
      }

      // --- now compute inter reference node information ---
	  int weights_ref[8 + 8 + 8 + 8] = {};
      FixedPoint SampleInterPredBuf_origin[3][8] = {0};

      uint64_t sumWeights_ref = 0;
      FixedPoint finterDC[3] = {0}, interParentMean[3] = {0};

      bool interNode = false;
      bool testInterNode = !(rahtExtension && nodeCnt == 1);
      bool checkInterNode = enableACInterPred && testInterNode;
      if (enableACRDOInterPred)
        checkInterNode = curLevelEnableACInterPred && testInterNode;

      if (checkInterNode) {
        const auto cur_pos = weightsLf[i].pos >> (level + 3);
        auto ref_pos = weightsLf_ref[j].pos >> (level_ref + 3);

        int cur_Luma;
        int minus_Luma;
        int condition;
        condition = 0;

        while ((j < weightsLf_ref.size() - 1) && (cur_pos > ref_pos)) {
          j++;
          ref_pos = weightsLf_ref[j].pos >> (level_ref + 3);
        }

        //if (levelD == 0) {
        //    cur_Luma = (weightsParentIt->parent_sumAttr[0] * weightsParentIt->weight) >> 15;
        //    minus_Luma = abs(weightsLf_ref[j].parent_sumAttr[0] - cur_Luma);
        //    condition3 = levelD < 1 && cur_Luma < 20 * minus_Luma;
        //}
        if (cur_pos == ref_pos) {
            if (levelD < 1) {
                cur_Luma =
              (weightsParentIt->parent_sumAttr[0] * weightsParentIt->weight)
              >> 15;
                minus_Luma = abs(weightsLf_ref[j].parent_sumAttr[0] - cur_Luma); 
                condition = cur_Luma >= 20 * minus_Luma;
                if (condition) {
                    interNode = true;
                }
            }else
               interNode = true;
        }
        //if (cur_pos == ref_pos && (!condition3)) {
        //  interNode = true;
        //}
      }

      if (interNode) {
        for (jLast = j; jLast < jEnd; jLast++) {
          int nextNode = jLast > j
            && !isSibling(weightsLf_ref[jLast].pos, weightsLf_ref[j].pos, level_ref + 3);
          if (nextNode)
            break;
          int nodeIdx = (weightsLf_ref[jLast].pos >> level_ref) & 0x7;
          weights_ref[nodeIdx] = weightsLf_ref[jLast].weight;
          sumWeights_ref += (uint64_t) weights_ref[nodeIdx];
          for (int k = 0; k < numAttrs; k++){
            SampleInterPredBuf[k][nodeIdx] = weightsLf_ref[jLast].sumAttr[k];
            SampleInterPredBuf_origin[k][nodeIdx] =
              weightsLf_ref[jLast].sumAttr[k];
            finterDC[k] += SampleInterPredBuf[k][nodeIdx];
          }
        }

        if(haarFlag) {
          mkWeightTree(weights_ref);
          std::copy_n(&SampleInterPredBuf[0][0], numAttrs * 8, &transformInterPredBuf[0][0]);
          ComputeDCfor222Block<HaarKernel>(numAttrs, transformInterPredBuf, weights_ref);
          for (int k = 0; k < numAttrs; k++) {
            finterDC[k].val = transformInterPredBuf[k][0].val;
            interParentMean[k].val = finterDC[k].val;
          }
        } 
		else{
          FixedPoint rsqrtWeightSumRef(0);
          uint64_t w = sumWeights_ref;
          int shiftBits = 5 * ((w > 1024) + (w > 1048576));
          rsqrtWeightSumRef.val = fastIrsqrt(w) >> (40 - shiftBits - FixedPoint::kFracBits);
          for (int k = 0; k < numAttrs; k++) {
            finterDC[k].val >>= shiftBits;
            finterDC[k] *= rsqrtWeightSumRef;
            interParentMean[k].val = finterDC[k].val;
            interParentMean[k].val >>= shiftBits;
            interParentMean[k] *= rsqrtWeightSumRef;
          }
        }

        int64_t curinheritDC = (inheritDc) ? *attrRecParentUsIt : 0;
        int64_t interDC = finterDC[0].val;

        if ((curinheritDC > 0) && (interDC > 0) && (!haarFlag)) {
          bool condition1 = 10 * interDC < ((curinheritDC)* 5);
          bool condition2 = 10 * interDC > ((curinheritDC)* 20);
          if (condition1 || condition2) {
            interNode = false;
          }
        }
      }

	  // --- now compute intra prediction information ---
      // Inter-level prediction:
      //  - Find the parent neighbours of the current node
      //  - Generate prediction for all attributes into transformPredBuf
      //  - Subtract transformed coefficients from forward transform
      //  - The transformPredBuf is then used for reconstruction
      bool enablePrediction = enablePredictionInLvl;

      bool eligiblePrediction = enablePrediction;
          
      if (enablePredictionInLvl) {
        weightsParentIt->occupancy = occupancy;
        // indexes of the neighbouring parents
        int parentNeighIdx[19];
        int childNeighIdx[12][8];

        int parentNeighCount = 0;
        if (rahtExtension && nodeCnt == 1) {
          enablePrediction = false;
          eligiblePrediction = false;
          parentNeighCount = 19;
        } else if (
          *numGrandParentNeighIt < rahtPredParams.raht_prediction_threshold0) {
          enablePrediction = false;
          eligiblePrediction = false;
        } else {
          findNeighbours(
            weightsParent.begin(), weightsParent.end(), weightsParentIt,
            weightsLf.begin(), weightsLf.begin() + i, level + 3, occupancy,
            parentNeighIdx, childNeighIdx,
            rahtPredParams.raht_subnode_prediction_enabled_flag, rahtPredParams.raht_prediction_search_range);
          for (int i = 0; i < 19; i++) {
            parentNeighCount += (parentNeighIdx[i] != -1);
          }
          if (parentNeighCount < rahtPredParams.raht_prediction_threshold1) {
            enablePrediction = false;
          } 
        }

        if (enableACRDONonPred)
          enablePrediction = enablePrediction || eligiblePrediction;

        if (enablePrediction) {
          int64_t limitLow = 0;
          int64_t limitHigh = 0;
          intraDcPred<haarFlag, numAttrs, rahtExtension>(
            parentNeighIdx, childNeighIdx, occupancy,
            attrRecParent.begin(), attrRec.begin(), intraAttrRec.begin(),
            SamplePredBuf, SampleIntraPredBuf, rahtPredParams,
            limitLow, limitHigh, curLevelEnableACInterPred);
        }

        for (int j = i, nodeIdx = 0; nodeIdx < 8; nodeIdx++) {
          if (!weights[nodeIdx])
            continue;
          numParentNeigh[j++] = parentNeighCount;
        }
      }

      if (inheritDc) {
        numGrandParentNeighIt++;
      }
      weightsParentIt++;

      bool enableIntraPrediction =
      curLevelEnableACInterPred && enablePrediction;
      bool enableInterPrediction = curLevelEnableACInterPred;

      int indices[8][3] = {{1, 2, 4}, {0, 3, 5}, {0, 3, 6}, {1, 2, 7},
                           {0, 5, 6}, {1, 4, 7}, {2, 4, 7}, {3, 5, 6}};
      uint64_t sumWeights_ref_local = 0;
      FixedPoint sumattr_ref_local[3] = {0};
      
      // --- now resample the reference inter node according to the weights ---
      if (haarFlag) {
        if (interNode) {
          for (int childIdx = 0; childIdx < 8; childIdx++) {
            if (weights[childIdx] == 0) {
                    for (int k = 0; k < numAttrs; k++) {
                SampleInterPredBuf[k][childIdx].val = 0;
                    }
                    continue;
            } else if (weights_ref[childIdx] == 0) {
                    for (int k = 0; k < numAttrs; k++) {
                SampleInterPredBuf[k][childIdx].val = interParentMean[k].val;
                    }
            }
            for (int k = 0; k < numAttrs; k++)
                    SampleInterPredBuf[k][childIdx].val =
                      (SampleInterPredBuf[k][childIdx].val
                       >> FixedPoint::kFracBits)
                      << (FixedPoint::kFracBits);
          }
          enablePrediction = true;
          std::copy_n(
            &SampleInterPredBuf[0][0], numAttrs * 8, &SamplePredBuf[0][0]);
        }
      } else {
        // normalise coefficients
        if (interNode) {
          for (int childIdx = 0; childIdx < 8; childIdx++) {
            if (weights[childIdx] == 0) {
                    for (int k = 0; k < numAttrs; k++) {
                SampleInterPredBuf[k][childIdx].val = 0;
                    }
                    continue;
            } else if (weights_ref[childIdx] == 0) {
                    if (
                      weights_ref[indices[childIdx][0]] == 0
                      && weights_ref[indices[childIdx][1]] == 0
                      && weights_ref[indices[childIdx][2]] == 0) {
                for (int k = 0; k < numAttrs; k++) {
                  SampleInterPredBuf[k][childIdx].val = interParentMean[k].val;
                }
                    } else {
                sumWeights_ref_local =
                  (uint64_t)weights_ref[indices[childIdx][0]]
                  + (uint64_t)weights_ref[indices[childIdx][1]]
                  + (uint64_t)weights_ref[indices[childIdx][2]];
                for (int k = 0; k < numAttrs; k++) {
                  sumattr_ref_local[k].val =
                    SampleInterPredBuf_origin[k][indices[childIdx][0]].val
                    + SampleInterPredBuf_origin[k][indices[childIdx][1]].val
                    + SampleInterPredBuf_origin[k][indices[childIdx][2]].val;
                }
                FixedPoint rsqrtWeightSumRef_local(0);
                uint64_t w_local = sumWeights_ref_local;
                int shiftBits_local =
                  5 * ((w_local > 1024) + (w_local > 1048576));
                rsqrtWeightSumRef_local.val = fastIrsqrt(w_local)
                  >> (40 - shiftBits_local - FixedPoint::kFracBits);
                for (int k = 0; k < numAttrs; k++) {
                  sumattr_ref_local[k].val >>= shiftBits_local;
                  sumattr_ref_local[k] *= rsqrtWeightSumRef_local;
                  //interParentMean[k].val = finterDC[k].val;
                  sumattr_ref_local[k].val >>= shiftBits_local;
                  sumattr_ref_local[k] *= rsqrtWeightSumRef_local;
                  SampleInterPredBuf[k][childIdx].val =
                    sumattr_ref_local[k].val;
                }
                    }
            }

            //if (weights_ref[childIdx] > 1 && weights[childIdx] != 0) {
            if (weights_ref[childIdx] > 1) {
                    //if (weights[childIdx] == 0)
                    // std::cout << "pay attention: " << std::endl;
                    FixedPoint rsqrtWeight;
                    uint64_t w = weights_ref[childIdx];
                    int shift = 5 * ((w > 1024) + (w > 1048576));
                    rsqrtWeight.val =
                      fastIrsqrt(w) >> (40 - shift - FixedPoint::kFracBits);
                    for (int k = 0; k < numAttrs; k++) {
                SampleInterPredBuf[k][childIdx].val >>= shift;
                SampleInterPredBuf[k][childIdx] *=
                  rsqrtWeight;  //sqrt normalized: DC
                SampleInterPredBuf[k][childIdx].val >>= shift;
                SampleInterPredBuf[k][childIdx] *=
                  rsqrtWeight;  //mean attribute
                    }
            }
          }
          enablePrediction = true;
          std::copy_n(
            &SampleInterPredBuf[0][0], numAttrs * 8, &SamplePredBuf[0][0]);
        }

        // --- normalise coefficients in lossy case ---
        for (int childIdx = 0; childIdx < 8; childIdx++) {
          if (weights[childIdx] <= 1)
            continue;
          // Summed attribute values
          FixedPoint rsqrtWeight;
          uint64_t w = weights[childIdx];
          int shift = 5 * ((w > 1024) + (w > 1048576));
          rsqrtWeight.val =
            fastIrsqrt(w) >> (40 - shift - FixedPoint::kFracBits);
          for (int k = 0; k < numAttrs; k++) {
            SampleBuf[k][childIdx].val >>= shift;
            SampleBuf[k][childIdx] *= rsqrtWeight;
          }

          // Predicted attribute values
          FixedPoint sqrtWeight;
          if (enablePrediction) {
            sqrtWeight.val = fastIsqrt(weights[childIdx]);
            for (int k = 0; k < numAttrs; k++) {
                    SamplePredBuf[k][childIdx] *= sqrtWeight;
            }
          }
          if (enableIntraPrediction) {
            for (int k = 0; k < numAttrs; k++) {
                    SampleIntraPredBuf[k][childIdx] *= sqrtWeight;
            }
          }
        }
      }

      const bool& enableHybridPred = hybridPredLayer && enablePrediction;

      if(hybridPredLayer){
        if(interNode && enableIntraPrediction){
          for(int childIdx = 0; childIdx < 8; childIdx++){
            if(weights[childIdx])
            for(int k = 0; k < numAttrs; k++){
              SampleIntraPredBuf[k][childIdx].val = (SampleIntraPredBuf[k][childIdx].val + SamplePredBuf[k][childIdx].val) >> 1;
            }
          }
        }else if(interNode){
          for (int childIdx = 0; childIdx < 8; childIdx++){
              for(int k = 0; k < numAttrs; k++){
                SampleIntraPredBuf[k][childIdx].val = SamplePredBuf[k][childIdx].val;
              }
          }
        }
      }

      // forward transform:
      //  - encoder: transform both attribute sums and prediction 
      // TODO: sample domain prediction only transform the prediction residuals
      mkWeightTree(weights);
      if (haarFlag) {
        if (enablePrediction){
          std::copy_n(&SampleBuf[0][0], 2 * numAttrs * 8, &transformBuf[0][0]);
          fwdTransformBlock222<HaarKernel>(2 * numAttrs, transformBuf, weights);
        }
        else {
          std::copy_n(&SampleBuf[0][0], numAttrs * 8, &transformBuf[0][0]);
          fwdTransformBlock222<HaarKernel>(numAttrs, transformBuf, weights);
        }

        if (enableIntraPrediction){
          std::copy_n(&SampleIntraPredBuf[0][0], numAttrs * 8, &transformIntraPredBuf[0][0]);
          fwdTransformBlock222<HaarKernel>(numAttrs, transformIntraPredBuf, weights);
        }        
      }
      else {
        if (enablePrediction){
          std::copy_n(&SampleBuf[0][0], 2 * numAttrs * 8, &transformBuf[0][0]);
          fwdTransformBlock222<RahtKernel>(2 * numAttrs, transformBuf, weights);
        }
        else {
          std::copy_n(&SampleBuf[0][0], numAttrs * 8, &transformBuf[0][0]);
          fwdTransformBlock222<RahtKernel>(numAttrs, transformBuf, weights);
        }
        
        if(interNode) //temporal filtering
        {
          for (int childIdx = 0; childIdx < 8; childIdx++) {
            for (int k = 0; k < numAttrs; k++){
              int64_t refVal = 0, filteredVal = 0;
              refVal = transformPredBuf[k][childIdx].val;
              filteredVal = (treeDepth < skipInitLayersForFiltering) ? refVal
                : (refVal * interFilterTap) >> 7;
              transformPredBuf[k][childIdx].val = filteredVal;              

              refVal = SamplePredBuf[k][childIdx].val;
              filteredVal = (treeDepth < skipInitLayersForFiltering) ? refVal
			    : (refVal * interFilterTap ) >> 7;
              SamplePredBuf[k][childIdx].val = filteredVal;
            }
          }
        }
        
        if (enableIntraPrediction || enableHybridPred){
          std::copy_n(&SampleIntraPredBuf[0][0], numAttrs * 8, &transformIntraPredBuf[0][0]);
          fwdTransformBlock222<RahtKernel>(numAttrs, transformIntraPredBuf, weights);
        }
      }

      if (curLevelEnableACInterPred)
      {
        std::copy_n(&transformBuf[0][0], 8 * numAttrs, &transformIntraBuf[0][0]);
        std::copy_n(&transformBuf[0][0], 8 * numAttrs, &origsamples[0][0]);
      }
      if (curLevelEnableACIntraPred)
      {
        std::copy_n(&transformBuf[0][0], 8 * numAttrs, &origsamples[0][0]);
        std::copy_n(&transformBuf[0][0], 8 * numAttrs, &transformNonPredBuf[0][0]);
      }
        
      //compute DC of the predictions: Done in the same way at the encoder and decoder to avoid drifting
      if((enablePrediction || enableIntraPrediction || enableHybridPred) && !haarFlag){
        FixedPoint rsqrtweightsum;
        rsqrtweightsum.val = fastIrsqrt(sumWeights_cur);
        for (int childIdx = 0; childIdx < 8; childIdx++) {
          if (weights[childIdx] == 0)
            continue;
          FixedPoint normalizedsqrtweight;
          if (weights[childIdx] == 1) {
            normalizedsqrtweight.val = rsqrtweightsum.val >> (40 - FixedPoint::kFracBits);
          } else {
            FixedPoint sqrtWeight;
            sqrtWeight.val = fastIsqrt(weights[childIdx]);
            normalizedsqrtweight.val = sqrtWeight.val * rsqrtweightsum.val >> 40;
          }
          normalizedSqrtBuf[childIdx] = normalizedsqrtweight;
          for (int k = 0; k < numAttrs; k++){
            FixedPoint prod;
            prod.val = normalizedsqrtweight.val; prod *= SamplePredBuf[k][childIdx];
            PredDC[k].val += prod.val;
            if (enableIntraPrediction || enableHybridPred) {
              prod.val = normalizedsqrtweight.val; prod *= SampleIntraPredBuf[k][childIdx];
              PredAllIntraDC[k].val += prod.val;
			}
          }
        }
      }
      
      //flags for skiptransform
      bool skipTransform = enablePrediction;
	  bool skipAllIntraTransform = enableIntraPrediction || enableHybridPred;
      
      // per-coefficient operations:
      //  - subtract transform domain prediction (encoder)
      //  - subtract the prediction between chroma channel components
      //  - write out/read in quantised coefficients
      //  - inverse quantise + add transform domain prediction
      scanBlock(weights, [&](int idx) {
        // skip the DC coefficient unless at the root of the tree
        if (inheritDc && !idx)
          return;

        if (enablePrediction) {
          // subtract transformed prediction (skipping DC)
          for (int k = 0; k < numAttrs; k++) {
            transformBuf[k][idx] -= transformPredBuf[k][idx];
          }
        }

        if (enableIntraPrediction || enableHybridPred) {
          for (int k = 0; k < numAttrs; k++) {
            transformIntraBuf[k][idx] -= transformIntraPredBuf[k][idx];
          }
        }

        // decision for RDOQ
        static const int LUTlog[16] = {0,   256, 406, 512, 594, 662, 719,  768,
                                812, 850, 886, 918, 947, 975, 1000, 1024};
        bool flagRDOQ = false;
        bool intraFlagRDOQ = false;
        bool nonPredFlagRDOQ = false;        
        
        int64_t sumCoeff = 0;
        int64_t intraSumCoeff = 0;
        int64_t nonPredSumCoeff = 0;

        int64_t Qcoeff; 
        int64_t intraQcoeff; 
        int64_t nonPredQcoeff;

        FixedPoint reconCurCoeff = 0, reconAllIntraCoeff = 0, reconNonPredCoeff = 0;

        if (!haarFlag && !newRDOQ) {

          int64_t lambda0;        

          int64_t coeff = 0;
          int64_t recDist2 = 0;
          int64_t Dist2 = 0;
          int Ratecoeff = 0;

          int64_t intraCoeff = 0;
          int64_t intraRecDist2 = 0;
          int64_t intraDist2 = 0;
          int intraRatecoeff = 0;

          int64_t nonPredCoeff = 0;
          int64_t nonPredRecDist2 = 0;
          int64_t nonPredDist2 = 0;
          int nonPredRatecoeff = 0;

          for (int k = 0; k < numAttrs; k++) {

            auto quantizers = qpset.quantizers(qpLayer, nodeQp[idx]);
            auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];
            coeff = transformBuf[k][idx].round();
            Dist2 += coeff * coeff;

            if (enableLCPPred) {
              if (k != 2) {
                Qcoeff = q.quantize(coeff << kFixedPointAttributeShift);
                transformResidueRecBuf[k] =
                  divExp2RoundHalfUp(q.scale(Qcoeff), kFixedPointAttributeShift);
              } else if (k == 2) {
                transformResidueRecBuf[k].val =
                  transformBuf[k][idx].val - ((LcpCoeff * transformResidueRecBuf[1].val) >> 4);
                coeff = transformResidueRecBuf[k].round();
                Qcoeff = q.quantize((coeff) << kFixedPointAttributeShift);
              }
            }else
              Qcoeff = q.quantize(coeff << kFixedPointAttributeShift);

			auto recCoeff = divExp2RoundHalfUp(
			  q.scale(Qcoeff), kFixedPointAttributeShift);
            recDist2 += (coeff - recCoeff) * (coeff - recCoeff);

            sumCoeff += std::abs(Qcoeff);
            //Ratecoeff += !!Qcoeff; // sign
            Ratecoeff +=
              std::abs(Qcoeff) < 15 ? LUTlog[std::abs(Qcoeff)] : LUTlog[15];
            if (!k)
              lambda0 = q.scale(1);
            if (curLevelEnableACInterPred) {
              intraCoeff = transformIntraBuf[k][idx].round();
              intraDist2 += intraCoeff * intraCoeff;

              if (enableLCPPred) {
                if (k != 2) {
                  intraQcoeff = q.quantize(intraCoeff << kFixedPointAttributeShift);
                  transformIntraPredResRecBuf[k] =
                    divExp2RoundHalfUp(q.scale(intraQcoeff), kFixedPointAttributeShift);
                }
                else if (k == 2) {
                  transformIntraPredResRecBuf[k].val =
                    transformIntraBuf[k][idx].val - ((intraPredLcpCoeff * transformIntraPredResRecBuf[1].val) >> 4);
                  intraCoeff = transformIntraPredResRecBuf[k].round();
                  intraQcoeff = q.quantize((intraCoeff) << kFixedPointAttributeShift);
                }
              }
              else
                intraQcoeff = q.quantize(intraCoeff << kFixedPointAttributeShift);

			  auto recIntraCoeff = divExp2RoundHalfUp(
                q.scale(intraQcoeff), kFixedPointAttributeShift);
              intraRecDist2 += (intraCoeff - recIntraCoeff) * (intraCoeff - recIntraCoeff);

              intraSumCoeff += std::abs(intraQcoeff);
              //Ratecoeff += !!Qcoeff; // sign
              intraRatecoeff += std::abs(intraQcoeff) < 15
                ? LUTlog[std::abs(intraQcoeff)]
                : LUTlog[15];
            }

            if (curLevelEnableACIntraPred) {

              nonPredCoeff = transformNonPredBuf[k][idx].round();
              nonPredDist2 += nonPredCoeff * nonPredCoeff;

              if (enableLCPPred) {
                if (k != 2) {
                  nonPredQcoeff = q.quantize(nonPredCoeff << kFixedPointAttributeShift);
                  transformNonPredResRecBuf[k] =
                    divExp2RoundHalfUp(q.scale(nonPredQcoeff), kFixedPointAttributeShift);
                }
                else if (k == 2) {
                  transformNonPredResRecBuf[k].val =
                    transformNonPredBuf[k][idx].val - ((nonPredLcpCoeff * transformNonPredResRecBuf[1].val) >> 4);
                  nonPredCoeff = transformNonPredResRecBuf[k].round();
                  nonPredQcoeff = q.quantize((nonPredCoeff) << kFixedPointAttributeShift);
                }
              }
              else
                nonPredQcoeff = q.quantize(nonPredCoeff << kFixedPointAttributeShift);

			  auto recNonPredCoeff = divExp2RoundHalfUp(
                q.scale(nonPredQcoeff), kFixedPointAttributeShift);
              nonPredRecDist2 +=
                (nonPredCoeff - recNonPredCoeff) * (nonPredCoeff - recNonPredCoeff);
              
              nonPredSumCoeff += std::abs(nonPredQcoeff);
              //Ratecoeff += !!Qcoeff; // sign
              nonPredRatecoeff += std::abs(nonPredQcoeff) < 15
                ? LUTlog[std::abs(nonPredQcoeff)]
                : LUTlog[15];
            }
          }
          const int64_t lambda = lambda0 * lambda0 * (numAttrs == 1 ? 25 : 35);
          dlambda = (double) lambda;
          if (sumCoeff < 3) {
            int Rate = getRate(trainZeros);
            Rate += (Ratecoeff + 128) >> 8;
            flagRDOQ = (Dist2 << 26) < (lambda * Rate + (recDist2 << 26));
          }
          if (curLevelEnableACInterPred && intraSumCoeff < 3) {
            int intraRate = getRate(intraTrainZeros);
            intraRate += (intraRatecoeff + 128) >> 8;
            intraFlagRDOQ = (intraDist2 << 26) < (lambda * intraRate + (intraRecDist2 << 26));
          }
          if (curLevelEnableACIntraPred && nonPredSumCoeff < 3) {
            int nonPredRate = getRate(nonPredTrainZeros);
            nonPredRate += (nonPredRatecoeff + 128) >> 8;
            nonPredFlagRDOQ = (nonPredDist2 << 26) < (lambda * nonPredRate + (nonPredRecDist2 << 26));
          }

          // Track RL for RDOQ
          if (flagRDOQ || sumCoeff == 0)
            trainZeros++;
          else
            trainZeros = 0;

          if (curLevelEnableACInterPred) {
            if (intraFlagRDOQ || intraSumCoeff == 0)
              intraTrainZeros++;
            else
              intraTrainZeros = 0;
          }
          if (curLevelEnableACIntraPred) {
            if (nonPredFlagRDOQ || nonPredSumCoeff == 0)
              nonPredTrainZeros++;
            else
              nonPredTrainZeros = 0;
          }
        }

        // Check if QP offset is to be applied to AC coeffiecients
        Qps coeffQPOffset = (acCoeffQpLayer <= maxAcCoeffQpOffsetLayers && idx)
          ? qpset.rahtAcCoeffQps[acCoeffQpLayer][idx - 1]
          : Qps({0, 0});

        Qps nodeQPOffset = {
          nodeQp[idx][0] + coeffQPOffset[0],
          nodeQp[idx][1] + coeffQPOffset[1]};

        // The RAHT transform
        int64_t bestQCoeffs[3] = {};
        auto quantizers = qpset.quantizers(qpLayer, nodeQPOffset);
        if (newRDOQ) {
          doRDOQ(numAttrs, quantizers, idx, LcpCoeff, transformBuf, bestQCoeffs, enableLCPPred, haarFlag, *rdoqFactors, trainZeros, encoder);

          if (allZero(bestQCoeffs))
            trainZeros++;
          else
            trainZeros = 0;
        }

        int values[3] = { 0, 0, 0 };

        for (int k = 0; k < numAttrs; k++) {

          auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];

          int64_t iresidueInter = 0;
          int64_t iresidueIntra = 0;
          int64_t iresidueNonPred = 0;

          

          if (is420 && k > 0 && level == 0 && !haarFlag) {
            transformBuf[k][idx] = 0;
            if (first420 == -1)
              first420 = std::distance(coeffBufIt, coeffBufItK[0]) - 1;
          }

          int64_t coeff = 0;
          if (newRDOQ) {
            coeff = bestQCoeffs[k];
            if(first420!=-1&&k>0)
              coeff=0;
          } else {
            if (flagRDOQ) {  // apply RDOQ
              transformBuf[k][idx].val = 0;
              transformResidueRecBuf[k].val = 0;
            }

            if (intraFlagRDOQ) {  // apply RDOQ
              transformIntraBuf[k][idx].val = 0;
              transformIntraPredResRecBuf[k].val = 0;
            }

            if (nonPredFlagRDOQ) {
              transformNonPredBuf[k][idx].val = 0;
              transformNonPredResRecBuf[k].val = 0;
            }

            coeff = transformBuf[k][idx].round();
            assert(coeff <= INT_MAX && coeff >= INT_MIN);
            coeff = q.quantize(coeff << kFixedPointAttributeShift);
          }

          
          if (!haarFlag && enableLCPPred) {
            if (k != 2) {
              if (newRDOQ) {
                transformResidueRecBuf[k] =
                  divExp2RoundHalfUp(q.scale(coeff), kFixedPointAttributeShift);
              }

              transformPredBuf[k][idx] += transformResidueRecBuf[k];
            }
            else if (k == 2) {
              if (!newRDOQ) {
                coeff = transformResidueRecBuf[k].round();
                coeff = q.quantize(coeff << kFixedPointAttributeShift);
            }

              transformResidueRecBuf[k] = divExp2RoundHalfUp(
				  q.scale(coeff), kFixedPointAttributeShift);
              transformResidueRecBuf[k].val += LcpCoeff * transformResidueRecBuf[1].val >> 4;
              transformPredBuf[k][idx] += transformResidueRecBuf[k];
            }
            CoeffRecBuf[nodelvlSum][k] = transformResidueRecBuf[k].round();
            NodeRecBuf[k][idx] = transformResidueRecBuf[k];
          }
		  else {
            reconCurCoeff = divExp2RoundHalfUp(q.scale(coeff), kFixedPointAttributeShift);
            transformPredBuf[k][idx] += reconCurCoeff;
            NodeRecBuf[k][idx] = reconCurCoeff;
          }

          skipTransform = skipTransform && (NodeRecBuf[k][idx].val == 0);

          *coeffBufItK[k]++ = coeff;

          values[k] = coeff;

          if (enableRDOCodingLayer)
            curEstimate.updateCostBits(coeff, k);

          FixedPoint fOrgResidue, fIntraResidue, fNonPredResidue;
          fOrgResidue.val = origsamples[k][idx].val - transformPredBuf[k][idx].val;

          if (enableRDOCodingLayer)
            curEstimate.resStatUpdate(coeff, k);

          if (curLevelEnableACInterPred) {  //< estimate
            auto intraCoeff = transformIntraBuf[k][idx].round();
            assert(intraCoeff <= INT_MAX && intraCoeff >= INT_MIN);
            intraCoeff = q.quantize(intraCoeff << kFixedPointAttributeShift);

            if (!haarFlag && enableLCPPred) {
              if (k != 2) {
                transformIntraPredBuf[k][idx] += transformIntraPredResRecBuf[k];
              }
			  else if (k == 2) {
                intraCoeff = transformIntraPredResRecBuf[k].round();
                intraCoeff = q.quantize(intraCoeff << kFixedPointAttributeShift);
                transformIntraPredResRecBuf[k] = divExp2RoundHalfUp(
                  q.scale(intraCoeff), kFixedPointAttributeShift);
                transformIntraPredResRecBuf[k].val += intraPredLcpCoeff * transformIntraPredResRecBuf[1].val >> 4;
                transformIntraPredBuf[k][idx] += transformIntraPredResRecBuf[k];
              }
              intraPredCoeffRecBuf[nodelvlSum][k] = transformIntraPredResRecBuf[k].round();
              NodeAllIntraRecBuf[k][idx] = transformIntraPredResRecBuf[k];
            }
			else {
              reconAllIntraCoeff = divExp2RoundHalfUp(q.scale(intraCoeff), kFixedPointAttributeShift);
              transformIntraPredBuf[k][idx] += reconAllIntraCoeff;
              NodeAllIntraRecBuf[k][idx] = reconAllIntraCoeff;
            }

            intraEstimate.updateCostBits(intraCoeff, k);
            *intraCoeffBufItK[k]++ = intraCoeff;

            intraEstimate.resStatUpdate(intraCoeff, k);

            fIntraResidue.val = origsamples[k][idx].val - transformIntraPredBuf[k][idx].val;
            iresidueIntra = fIntraResidue.round();

            iresidueInter = fOrgResidue.round();
            int64_t idistinter = (iresidueInter) * (iresidueInter);
			int64_t	idistintra = (iresidueIntra) * (iresidueIntra);
            distinter += (double)idistinter; 
			distintra += (double)idistintra;

            skipAllIntraTransform = skipAllIntraTransform && (NodeAllIntraRecBuf[k][idx].val == 0);
          }

          if (curLevelEnableACIntraPred) {

            auto nonPredCoeff = transformNonPredBuf[k][idx].round();
            assert(nonPredCoeff <= INT_MAX && nonPredCoeff >= INT_MIN);
            nonPredCoeff = q.quantize(nonPredCoeff << kFixedPointAttributeShift);

            if (!haarFlag && enableLCPPred) {
              if (k != 2) {
                transformNonPredBuf[k][idx] = transformNonPredResRecBuf[k];
              }
			  else if (k == 2) {
                nonPredCoeff = transformNonPredResRecBuf[k].round();
                nonPredCoeff = q.quantize(nonPredCoeff << kFixedPointAttributeShift);
                transformNonPredResRecBuf[k] = divExp2RoundHalfUp(
                  q.scale(nonPredCoeff), kFixedPointAttributeShift);
                transformNonPredResRecBuf[k].val += nonPredLcpCoeff * transformNonPredResRecBuf[1].val >> 4;
                transformNonPredBuf[k][idx] = transformNonPredResRecBuf[k];
              }
              nonPredCoeffRecBuf[nodelvlSum][k] = transformNonPredResRecBuf[k].round();
              NodeNonPredRecBuf[k][idx] = transformNonPredResRecBuf[k];
            }
			else {
              reconNonPredCoeff = divExp2RoundHalfUp(q.scale(nonPredCoeff), kFixedPointAttributeShift);
              transformNonPredBuf[k][idx] = reconNonPredCoeff;
              NodeNonPredRecBuf[k][idx] = reconNonPredCoeff;
            }

            nonPredEstimate.updateCostBits(nonPredCoeff, k);
            *nonPredCoeffBufItK[k]++ = nonPredCoeff;
            nonPredEstimate.resStatUpdate(nonPredCoeff, k);

            if (!curLevelEnableACInterPred) {
              iresidueIntra = fOrgResidue.round();
              int64_t idistintra = (iresidueIntra) * (iresidueIntra);
              distintra += (double)idistintra;
            }

            fNonPredResidue.val = origsamples[k][idx].val - transformNonPredBuf[k][idx].val;
            iresidueNonPred = fNonPredResidue.round();
            int64_t idistnonPred = (iresidueNonPred) * (iresidueNonPred);
            distnonPred += (double)idistnonPred;
          }
        }
        nodelvlSum++;

        if (newRDOQ) {
          if (!values[0] && !values[1] && !values[2]) {
            ++zeroRun;
          }
          else {
            encoder->encodeRunLength(zeroRun);
           //encoder.encodeRunLength(zeroRun);
          if (first420!=-1)
              encoder->encode420(values[0]);
          else
              encoder->encode(values[0], values[1], values[2]);
              zeroRun = 0;
          }
        }
      });

      // compute last component coefficient
      if (numAttrs == 3 && nodeCnt > 1 && !haarFlag && inheritDc && enableLCPPred) {

        LcpCoeff = curlevelLcp.computeLCPCoeff(nodelvlSum, CoeffRecBuf);

		if (enableACRDOInterPred)
          intraPredLcpCoeff = curlevelLcpIntra.computeLCPCoeff(nodelvlSum, intraPredCoeffRecBuf);

		if (enableACRDONonPred)
          nonPredLcpCoeff = curlevelLcpNonPred.computeLCPCoeff(nodelvlSum, nonPredCoeffRecBuf);
      }
      
      if(haarFlag){
        std::copy_n(&transformPredBuf[0][0], numAttrs * 8, &NodeRecBuf[0][0]);
        if (curLevelEnableACInterPred) {
          std::copy_n(&transformIntraPredBuf[0][0], numAttrs * 8, &NodeAllIntraRecBuf[0][0]);
        }
        if (curLevelEnableACIntraPred){
          std::copy_n(&transformNonPredBuf[0][0], numAttrs * 8, &NodeNonPredRecBuf[0][0]);
        }   
      }
      // replace DC coefficient with parent if inheritable
      if (inheritDc) {
        for (int k = 0; k < numAttrs; k++) {
          attrRecParentIt++;
          int64_t val = *attrRecParentUsIt++;
          if (rahtExtension){
            NodeRecBuf[k][0].val = (haarFlag)? val: val - PredDC[k].val;
          }
          else if (val > 0)
            transformPredBuf[k][0].val = val << (15 - 2);
          else
            transformPredBuf[k][0].val = -((-val) << (15 - 2));
          if (curLevelEnableACInterPred) {
            ///< inherit the parent DC coefficients
            NodeAllIntraRecBuf[k][0].val = haarFlag? val: val - PredAllIntraDC[k].val;  
          }
          if (curLevelEnableACIntraPred) {
            ///< inherit the parent DC coefficients
            NodeNonPredRecBuf[k][0].val = val;
          }
        }
      }

      if (haarFlag) {
        invTransformBlock222<HaarKernel>(numAttrs, NodeRecBuf, weights);
        if (curLevelEnableACInterPred)
          invTransformBlock222<HaarKernel>(numAttrs, NodeAllIntraRecBuf, weights);
        if (curLevelEnableACIntraPred)
          invTransformBlock222<HaarKernel>(numAttrs, NodeNonPredRecBuf, weights);
      } else {
        // apply skip transform here
        if (skipTransform) {
          FixedPoint DCerror[3];
          for (int k = 0; k < numAttrs; k++) {
            DCerror[k] = NodeRecBuf[k][0]; NodeRecBuf[k][0].val = 0;
          }
          for (int cidx = 0; cidx < 8; cidx++) {
            if (!weights[cidx])
              continue;
            
            for(int k = 0; k < numAttrs; k++) {
              FixedPoint Correctionterm = normalizedSqrtBuf[cidx];  Correctionterm *= DCerror[k];
              NodeRecBuf[k][cidx] = Correctionterm;
            }
          }
          
        } 
        else{
          invTransformBlock222<RahtKernel>(numAttrs, NodeRecBuf, weights);
        }
        if (curLevelEnableACInterPred){
          if (skipAllIntraTransform) {
            FixedPoint DCerror[3];
            for (int k = 0; k < numAttrs; k++) {
              DCerror[k] = NodeAllIntraRecBuf[k][0]; NodeAllIntraRecBuf[k][0].val = 0;
            }
            for (int cidx = 0; cidx < 8; cidx++) {
              if (!weights[cidx])
                continue;
              
              for(int k = 0; k < numAttrs; k++) {
                FixedPoint Correctionterm = normalizedSqrtBuf[cidx];  Correctionterm *= DCerror[k];
                NodeAllIntraRecBuf[k][cidx] = Correctionterm;
              }
            }
          }
          else{
            invTransformBlock222<RahtKernel>(numAttrs, NodeAllIntraRecBuf, weights);
          }  
        }
        if (curLevelEnableACIntraPred){
          invTransformBlock222<RahtKernel>(numAttrs, NodeNonPredRecBuf, weights);
        }
      }

      for (int j = i, nodeIdx = 0; nodeIdx < 8; nodeIdx++) {
        if (!weights[nodeIdx])
          continue;

        for (int k = 0; k < numAttrs; k++)
          if (rahtExtension) {
            if(!haarFlag){
              NodeRecBuf[k][nodeIdx].val += SamplePredBuf[k][nodeIdx].val;
            }
            attrRecUs[j * numAttrs + k] = NodeRecBuf[k][nodeIdx].val;            
            if (curLevelEnableACInterPred) {
              if (!haarFlag) {
                NodeAllIntraRecBuf[k][nodeIdx].val += SampleIntraPredBuf[k][nodeIdx].val;
              }
              intraAttrRecUs[j * numAttrs + k] =
              NodeAllIntraRecBuf[k][nodeIdx].val;
            }
            if (curLevelEnableACIntraPred) {
              nonPredAttrRecUs[j * numAttrs + k] =
                NodeNonPredRecBuf[k][nodeIdx].val;
            }
          }
          else {
            FixedPoint temp = transformPredBuf[k][nodeIdx];
            temp.val <<= 2;
            attrRecUs[j * numAttrs + k] = temp.round();
            if (curLevelEnableACInterPred) {
              temp = transformIntraPredBuf[k][nodeIdx];
              temp.val <<= 2;
              intraAttrRecUs[j * numAttrs + k] = temp.round();
            }
            if (curLevelEnableACIntraPred) {
              temp = transformNonPredBuf[k][nodeIdx];
              temp.val <<= 2;
              nonPredAttrRecUs[j * numAttrs + k] = temp.round();
            }
          }

        // scale values for next level
        if (!haarFlag) {
          if (weights[nodeIdx] > 1) {
            FixedPoint rsqrtWeight;
            uint64_t w = weights[nodeIdx];
            int shift = 5 * ((w > 1024) + (w > 1048576));
            rsqrtWeight.val = fastIrsqrt(w) >> (40 - shift - FixedPoint::kFracBits);
            for (int k = 0; k < numAttrs; k++) {
              NodeRecBuf[k][nodeIdx].val >>= shift;
              NodeRecBuf[k][nodeIdx] *= rsqrtWeight;
              if (curLevelEnableACInterPred) {
                NodeAllIntraRecBuf[k][nodeIdx].val >>= shift;
                NodeAllIntraRecBuf[k][nodeIdx] *= rsqrtWeight;
              }
              if (curLevelEnableACIntraPred) {
                NodeNonPredRecBuf[k][nodeIdx].val >>= shift;
                NodeNonPredRecBuf[k][nodeIdx] *= rsqrtWeight;
              }
            }
          }
        }

        for (int k = 0; k < numAttrs; k++) {
          attrRec[j * numAttrs + k] = rahtExtension
              ? NodeRecBuf[k][nodeIdx].val
            : NodeRecBuf[k][nodeIdx].round();
          if (curLevelEnableACInterPred) {
            intraAttrRec[j * numAttrs + k] = rahtExtension
                ? NodeAllIntraRecBuf[k][nodeIdx].val
              : NodeAllIntraRecBuf[k][nodeIdx].round();
          }
          if (curLevelEnableACIntraPred) {
            nonPredAttrRec[j * numAttrs + k] = rahtExtension
              ? NodeNonPredRecBuf[k][nodeIdx].val
              : NodeNonPredRecBuf[k][nodeIdx].round();
          }
        }
        j++;
      }
      i += nodeCnt_real;
  }  //end loop on nodes of current level


    if (enableRDOCodingLayer) {
      int64_t ifactor = 1 << 24;
      double dfactor = (double)(ifactor);
      if (enableACRDOInterPred) {
        double curCost = curEstimate.costBits();
        double intraCost = intraEstimate.costBits();
        if (enableACRDONonPred) {
          double nonPredCost = nonPredEstimate.costBits();
          double rdcostinter = distinter * dfactor + dlambda * curCost;
          double rdcostintra = distintra * dfactor + dlambda * intraCost;
          double rdcostnonPred = distnonPred * dfactor + dlambda * nonPredCost;
          if (rdcostintra <= rdcostinter && rdcostintra <= rdcostnonPred) {
            for (int k = 0; k < numAttrs; ++k)
              std::copy_n(intraCoeffBufItBeginK[k], sumNodes, coeffBufItBeginK[k]);
            std::swap(intraAttrRec, attrRec);
            std::swap(intraAttrRecUs, attrRecUs);
            curEstimate = intraEstimate;
            nonPredEstimate = intraEstimate;
            coder._encodeMode(0,enableACRDOInterPred, enableACRDONonPred);
            trainZeros = intraTrainZeros;
            nonPredTrainZeros = intraTrainZeros;
          }
          else if (rdcostnonPred <= rdcostinter && rdcostnonPred <= rdcostintra) {
            for (int k = 0; k < numAttrs; ++k)
              std::copy_n(nonPredCoeffBufItBeginK[k], sumNodes, coeffBufItBeginK[k]);
            std::swap(nonPredAttrRec, attrRec);
            std::swap(nonPredAttrRecUs, attrRecUs);
            curEstimate = nonPredEstimate;
            intraEstimate = nonPredEstimate;
            coder._encodeMode(2,enableACRDOInterPred, enableACRDONonPred);
            trainZeros = nonPredTrainZeros;
            intraTrainZeros = nonPredTrainZeros;
          }
          else {
            intraEstimate = curEstimate;
            nonPredEstimate = curEstimate;
            coder._encodeMode(1,enableACRDOInterPred, enableACRDONonPred);
            if (enableEstimateLayer)
              attrInterPredParams.paramsForInterRAHT.FilterTaps.push_back(
                quantizedResFilterTap);

            intraTrainZeros = trainZeros;
            nonPredTrainZeros = trainZeros;
          }
        }
        else {
          double rdcostinter = distinter * dfactor + dlambda * curCost;
          double rdcostintra = distintra * dfactor + dlambda * intraCost;
          bool newdecision = rdcostintra < rdcostinter;

          if (newdecision) {
            for (int k = 0; k < numAttrs; ++k)
              std::copy_n(intraCoeffBufItBeginK[k], sumNodes, coeffBufItBeginK[k]);
            std::swap(intraAttrRec, attrRec);
            std::swap(intraAttrRecUs, attrRecUs);
            curEstimate = intraEstimate;
            coder._encodeMode(0,enableACRDOInterPred, enableACRDONonPred);
            trainZeros = intraTrainZeros;
          }
          else {
            intraEstimate = curEstimate;
            coder._encodeMode(1,enableACRDOInterPred, enableACRDONonPred);
            if (enableEstimateLayer)
              attrInterPredParams.paramsForInterRAHT.FilterTaps.push_back(
                quantizedResFilterTap);

            intraTrainZeros = trainZeros;
          }
        }
      }
      else if (enableACRDONonPred) {
        double curCost = curEstimate.costBits();
        double nonPredCost = nonPredEstimate.costBits();
        double rdcostintra = distintra * dfactor + dlambda * curCost;
        double rdcostnonPred = distnonPred * dfactor + dlambda * nonPredCost;
        bool newdecision = rdcostnonPred < rdcostintra;
        if (newdecision) {
          for (int k = 0; k < numAttrs; ++k)
            std::copy_n(nonPredCoeffBufItBeginK[k], sumNodes, coeffBufItBeginK[k]);
          std::swap(nonPredAttrRec, attrRec);
          std::swap(nonPredAttrRecUs, attrRecUs);
          curEstimate = nonPredEstimate;
          coder._encodeMode(1,enableACRDOInterPred, enableACRDONonPred);
          trainZeros = nonPredTrainZeros;
        }
        else {
          nonPredEstimate = curEstimate;
          coder._encodeMode(0,enableACRDOInterPred, enableACRDONonPred);
          nonPredTrainZeros = trainZeros;
        }

      }
      
      curEstimate.resetCostBits();
      intraEstimate.resetCostBits();
      nonPredEstimate.resetCostBits();
    } 
	else if (enableEstimateLayer) 
	{//case 1: skip = 0; case 2: RDO coding layer is disabled
      attrInterPredParams.paramsForInterRAHT.FilterTaps.push_back(
        quantizedResFilterTap);
    }

    sameNumNodes = 0;
    // increment tree depth
    treeDepth++;
  }//end loop on level

  // -------------- process duplicate points at level 0 if exists--------------
  if (!flagNoDuplicate) {
    std::swap(attrRec, attrRecParent);
    auto attrRecParentIt = attrRecParent.cbegin();
    auto attrsHfIt = weightsHf.cbegin();

  std::vector<UrahtNode>& weightsLf = weightsLfStack[0];
    for (int i = 0, out = 0, iEnd = weightsLf.size(); i < iEnd; i++) {
      int weight = weightsLf[i].weight;
      // unique points have weight = 1
      if (weight == 1) {
        for (int k = 0; k < numAttrs; k++)
          attrRec[out++] = *attrRecParentIt++;
        continue;
      }
      Qps nodeQp = {
        weightsLf[i].qp[0] >> regionQpShift,
        weightsLf[i].qp[1] >> regionQpShift};

      // duplicates
      FixedPoint attrSum[3];
      FixedPoint attrRecDc[3];
      FixedPoint sqrtWeight;
      sqrtWeight.val = fastIsqrt(weight);

      int64_t sumCoeff = 0;
      for (int k = 0; k < numAttrs; k++) {
        attrSum[k] = weightsLf[i].sumAttr[k];
        if (rahtExtension)
          attrRecDc[k].val = *attrRecParentIt++;
        else
          attrRecDc[k] = *attrRecParentIt++;
        if (!haarFlag) {
          attrRecDc[k] *= sqrtWeight;
        }
      }

      FixedPoint rsqrtWeight;
      for (int w = weight - 1; w > 0; w--) {
        RahtKernel kernel(w, 1);
        HaarKernel haarkernel(w, 1);
        int shift = 5 * ((w > 1024) + (w > 1048576));
        rsqrtWeight.val = fastIrsqrt(w) >> (40 - shift - FixedPoint::kFracBits);

        int values[3] = { 0, 0, 0 };

        auto quantizers = qpset.quantizers(qpLayer, nodeQp);
        for (int k = 0; k < numAttrs; k++) {
          auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];

          FixedPoint transformBuf[2];

          // invert the initial reduction (sum)
          // NB: read from (w-1) since left side came from attrsLf.
          transformBuf[1] = attrsHfIt[w - 1].sumAttr[k];
          if (haarFlag) {
            attrSum[k].val -= transformBuf[1].val >> 1;
            transformBuf[1].val += attrSum[k].val;
            transformBuf[0] = attrSum[k];
          } else {
            attrSum[k] -= transformBuf[1];
            transformBuf[0] = attrSum[k];

            // NB: weight of transformBuf[1] is by construction 1.
            transformBuf[0].val >>= shift;
            transformBuf[0] *= rsqrtWeight;
          }

          if (haarFlag) {
            haarkernel.fwdTransform(
              transformBuf[0], transformBuf[1], &transformBuf[0],
              &transformBuf[1]);
          } else {
            kernel.fwdTransform(
              transformBuf[0], transformBuf[1], &transformBuf[0],
              &transformBuf[1]);
          }

          auto coeff = transformBuf[1].round();
          assert(coeff <= INT_MAX && coeff >= INT_MIN);
          *coeffBufItK[k]++ = coeff =
            q.quantize(coeff << kFixedPointAttributeShift);
          transformBuf[1] =
            divExp2RoundHalfUp(q.scale(coeff), kFixedPointAttributeShift);

          sumCoeff +=
			std::abs(q.quantize(coeff << kFixedPointAttributeShift));            

          values[k] = coeff;

          // inherit the DC value
          transformBuf[0] = attrRecDc[k];

          if (haarFlag) {
            haarkernel.invTransform(
              transformBuf[0], transformBuf[1], &transformBuf[0],
              &transformBuf[1]);
          } else {
            kernel.invTransform(
              transformBuf[0], transformBuf[1], &transformBuf[0],
              &transformBuf[1]);
          }

          attrRecDc[k] = transformBuf[0];
          attrRec[out + w * numAttrs + k] =
            rahtExtension ? transformBuf[1].val : transformBuf[1].round();
          if (w == 1)
            attrRec[out + k] =
              rahtExtension ? transformBuf[0].val : transformBuf[0].round();
        }

        // Track RL for RDOQ
        if (sumCoeff == 0)
          trainZeros++;
        else
          trainZeros = 0;

        if (newRDOQ) {
          if (!values[0] && !values[1] && !values[2]) {
            ++zeroRun;
          }
          else {
            encoder->encodeRunLength(zeroRun);
          if (first420!=-1)
              encoder->encode420(values[0]);
          else
              encoder->encode(values[0], values[1], values[2]);
            zeroRun = 0;
          }
        }
      }

      attrsHfIt += (weight - 1);
      out += weight * numAttrs;
    }
  }
  

  // -------------- write-back reconstructed attributes --------------
  if (zeroRun && newRDOQ)
    encoder->encodeRunLength(zeroRun);
  assert(attrRec.size() == numAttrs * numPoints);
  if(rahtExtension)
    for (auto& attr : attrRec) {
      attr += FixedPoint::kOneHalf;
      *(attributes++) = attr >> FixedPoint::kFracBits;
    }
  else
    std::copy(attrRec.begin(), attrRec.end(), attributes);
}

//============================================================================
/*
 * RAHT Fixed Point
 *
 * Inputs:
 * quantStepSizeLuma = Quantization step
 * mortonCode = list of 'voxelCount' Morton codes of voxels, sorted in ascending Morton code order
 * attributes = 'voxelCount' x 'attribCount' array of attributes, in row-major order
 * attribCount = number of attributes (e.g., 3 if attributes are red, green, blue)
 * voxelCount = number of voxels
 *
 * Outputs:
 * weights = list of 'voxelCount' weights associated with each transform coefficient
 * coefficients = quantized transformed attributes array, in column-major order
 * binaryLayer = binary layer where each coefficient was generated
 *
 * Note output weights are typically used only for the purpose of
 * sorting or bucketing for entropy coding.
 */
void
regionAdaptiveHierarchicalTransform(
  const RahtPredictionParams &rahtPredParams,
  AttributeBrickHeader& abh,
  const QpSet& qpset,
  const Qps* pointQpOffsets,
  int64_t* mortonCode,
  int* attributes,
  const int attribCount,
  const int voxelCount,
  int* coefficients,
  const bool rahtExtension,
  AttributeInterPredParams& attrInterPredParams,
  ModeEncoder& encoder,
  PCCResidualsEncoder& resencoder,
  const std::array<double, 4>* rdoqFactors
)
{
  if (rahtPredParams.integer_haar_enable_flag || attribCount != 3)
    abh.is420 = false;  

  switch (attribCount) {
  case 3:
    if (rahtPredParams.integer_haar_enable_flag) {
      if (rahtExtension)
        uraht_process_encoder<true, 3, true>(
          rahtPredParams, abh,qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, encoder, &resencoder, rdoqFactors);
      else
        uraht_process_encoder<true, 3, false>(
          rahtPredParams, abh,qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, encoder, &resencoder, rdoqFactors);
    } 
	else {
      if (rahtExtension)
        uraht_process_encoder<false, 3, true>(
          rahtPredParams, abh,qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, encoder, &resencoder, rdoqFactors);
      else
        uraht_process_encoder<false, 3, false>(
          rahtPredParams,abh, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, encoder, &resencoder, rdoqFactors);
    }
    break;

  case 1: 
    if (rahtPredParams.integer_haar_enable_flag) {
      if (rahtExtension)
        uraht_process_encoder<true, 1, true>(
          rahtPredParams,abh, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, encoder, &resencoder, rdoqFactors);
      else
        uraht_process_encoder<true, 1, false>(
          rahtPredParams, abh,qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, encoder, &resencoder, rdoqFactors);
	}
    else {
	  if (rahtExtension)
        uraht_process_encoder<false, 1, true>(
          rahtPredParams, abh,qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, encoder, &resencoder, rdoqFactors);
      else
        uraht_process_encoder<false, 1, false>(
          rahtPredParams, abh,qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, encoder, &resencoder, rdoqFactors);
	}
    break;
  default: throw std::runtime_error("attribCount only support 1 or 3");
  }
       
}

//============================================================================

}  // namespace pcc
