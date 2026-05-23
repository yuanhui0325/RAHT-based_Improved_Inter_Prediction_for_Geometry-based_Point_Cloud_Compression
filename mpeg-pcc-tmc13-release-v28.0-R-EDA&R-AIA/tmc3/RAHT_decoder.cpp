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

using namespace pcc::RAHT;

namespace pcc {
//============================================================================
// remove any non-unique leaves from a level in the uraht tree

int
reduceUnique_decoder(
  int numNodes,
  std::vector<UrahtNode>* weightsIn,
  std::vector<UrahtNode>* weightsOut)
{
  // process a single level of the tree
  int64_t posPrev = -1;
  auto weightsInWrIt = weightsIn->begin();
  auto weightsInRdIt = weightsIn->cbegin();
  for (int i = 0; i < numNodes; i++) {
    const auto& node = *weightsInRdIt++;

    // copy across unique nodes
    if (node.pos != posPrev) {
      posPrev = node.pos;
      *weightsInWrIt++ = node;
      continue;
    }

    // duplicate node
    (weightsInWrIt - 1)->weight += node.weight;
    weightsOut->push_back(node);
  }

  // number of nodes in next level
  return std::distance(weightsIn->begin(), weightsInWrIt);
}

//============================================================================
// Split a level of values into sum and difference pairs.

int
reduceLevel_decoder(
  int level,
  int numNodes,
  std::vector<UrahtNode>* weightsIn,
  std::vector<UrahtNode>* weightsOut)
{
  // process a single level of the tree
  int64_t posPrev = -1;
  auto weightsInWrIt = weightsIn->begin();
  auto weightsInRdIt = weightsIn->cbegin();

  for (int i = 0; i < numNodes; i++) {
    auto& node = *weightsInRdIt++;
    bool newPair = (posPrev ^ node.pos) >> level != 0;
    posPrev = node.pos;
    if (newPair) {
      *weightsInWrIt++ = node;
    } 
	else {
      auto& left = *(weightsInWrIt - 1);
      left.weight += node.weight;
      left.qp[0] = (left.qp[0] + node.qp[0]) >> 1;
      left.qp[1] = (left.qp[1] + node.qp[1]) >> 1;
      weightsOut->push_back(node);
    }
  }

  // number of nodes in next level
  return std::distance(weightsIn->begin(), weightsInWrIt);
}

int
reduceDepthDecoder(
  int level,
  int numNodes,
  std::vector<UrahtNode>* weightsIn,
  std::vector<UrahtNode>* weightsOut)
{
  // process a single level of the tree
  int64_t posPrev = -1;
  auto weightsInRdIt = weightsIn->begin();
  auto it = weightsIn->begin();
  for (int i = 0; i < weightsIn->size(); ) {
    // this is a new node
    auto last = weightsInRdIt[i];
    posPrev = last.pos;
    last.firstChild = it++;

    // look for same node
    int i2 = i + 1;
    for (; i2 < weightsIn->size(); i2++,it++)
      if ((posPrev ^ weightsInRdIt[i2].pos) >> level)
        break;

    // process same nodes
    for (int j = i + 1; j < i2; j++) {
      const auto node = weightsInRdIt[j];
      last.weight += node.weight;
      // TODO: fix local qp to be same in encoder and decoder
      last.qp[0] = (last.qp[0] + node.qp[0]) >> 1;
      last.qp[1] = (last.qp[1] + node.qp[1]) >> 1;
    }

    last.lastChild = it;
    weightsOut->push_back(last);
    i = i2;
  }

  // number of nodes in next level
  return weightsOut->size();
}


//============================================================================
// Merge sum and difference values to form a tree.

void
expandLevel_decoder(
  int level,
  int numNodes,
  std::vector<UrahtNode>* weightsIn,  // expand by numNodes before expand
  std::vector<UrahtNode>* weightsOut  // shrink after expand
)
{
  if (numNodes == 0)
    return;

  // process a single level of the tree
  auto weightsInWrIt = weightsIn->rbegin();
  auto weightsInRdIt = std::next(weightsIn->crbegin(), numNodes);
  auto weightsOutRdIt = weightsOut->crbegin();

  for (int i = 0; i < numNodes;) {
    bool isPair = (weightsOutRdIt->pos ^ weightsInRdIt->pos) >> level == 0;
    if (!isPair) {
      *weightsInWrIt++ = *weightsInRdIt++;
      
      continue;
    }

    // going to process a pair
    i++;

    // Out node is inserted before In node.
    const auto& nodeDelta = *weightsInWrIt++ = *weightsOutRdIt++;

    // move In node to correct position, subtracting delta
    *weightsInWrIt = *weightsInRdIt++;
    (weightsInWrIt++)->weight -= nodeDelta.weight;
  }
}

//============================================================================
// Generate the spatial prediction of a block.

template<bool haarFlag, int numAttrs, bool rahtExtension, typename It>
void
intraDcPred_decoder(
  const int parentNeighIdx[19],
  const int childNeighIdx[12][8],
  int occupancy,
  It first,
  It firstChild,
  FixedPoint predBuf[][8],
  const RahtPredictionParams &rahtPredParams, 
  int64_t& limitLow,
  int64_t& limitHigh)
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

          } else {
            weightSum[j] += predWeightParent[7 + i];
            for (int k = 0; k < numAttrs; k++) {
              predBuf[k][j].val += neighValue[k];
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
      }
      if (haarFlag) {
        for (int k = 0; k < numAttrs; k++) {
          predBuf[k][i].val = (predBuf[k][i].val >> predBuf[k][i].kFracBits)
            << predBuf[k][i].kFracBits;
        }
      }
    }
  }
}

//============================================================================
// Core transform process (for decoder)
template<bool haarFlag, int numAttrs, bool rahtExtension, class ModeCoder>
void
uraht_process_decoder(
  const RahtPredictionParams& rahtPredParams,
  const AttributeBrickHeader& abh,
  const QpSet& qpset,
  const Qps* pointQpOffsets,
  int numPoints,
  int64_t* positions,
  int* attributes,
  int32_t* coeffBufIt,
  AttributeInterPredParams& attrInterPredParams,
  ModeCoder& coder)
{
  const bool is420 = abh.is420;

  // coefficients are stored in three planar arrays.  coeffBufItK is a set
  // of iterators to each array.
  int32_t* coeffBufItK[3] = {
    coeffBufIt, coeffBufIt + numPoints, coeffBufIt + numPoints * 2};

  // early termination only one point
  if (numPoints == 1) {
    auto quantizers = qpset.quantizers(0, pointQpOffsets[0]);
    for (int k = 0; k < numAttrs; k++) {
      auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];
	  int64_t coeff = *coeffBufItK[k]++;
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
  // no need to copy attribute at decoder side
  for (int i = 0; i < numPoints; i++) {
    UrahtNode node;
    node.pos = positions[i];
    node.weight = 1;
    node.qp = {
		int16_t(pointQpOffsets[i][0] << regionQpShift),
        int16_t(pointQpOffsets[i][1] << regionQpShift)};
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
  numNodes = reduceUnique_decoder(numNodes, weightsLf, &weightsHf);
  weightsLfStack[0].resize(numNodes);//shrink

  const bool flagNoDuplicate = weightsHf.size() == 0;

  int numDepth = 0;
  for (int levelD = 3; numNodes > 1; levelD += 3) {
    // one depth reduction
    weightsLfStack.emplace_back();
    weightsLfStack.back().reserve(numNodes / 3);
    weightsLf = &weightsLfStack.back();

    auto weightsLfRefold = &weightsLfStack[weightsLfStack.size() - 2];
    numNodes =
      reduceDepthDecoder(levelD, numNodes, weightsLfRefold, weightsLf);
    numDepth++;
  }

  int numDepth_ref = 0;  
  if (enableACInterPred){

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
  std::vector<int32_t> attrRec, attrRecParent;
  attrRec.resize(numPoints * numAttrs);
  attrRecParent.resize(numPoints * numAttrs);

  std::vector<int64_t> attrRecUs, attrRecParentUs;
  attrRecUs.resize(numPoints * numAttrs);
  attrRecParentUs.resize(numPoints * numAttrs);

  std::vector<UrahtNode> weightsParent;
  weightsParent.reserve(numPoints);

  std::vector<int8_t> numParentNeigh, numGrandParentNeigh;
  numParentNeigh.resize(numPoints);
  numGrandParentNeigh.resize(numPoints);

  // quant layer selection
  auto qpLayer = 0;
  // AC coeff QP offset laer
  auto acCoeffQpLayer = -1;

  // descend tree
  bool sameNumNodes = 0;
  int8_t LcpCoeff = 0;
  int filterIdx = 0;

  // ----------------------------------- descend tree, loop on depth ------------------------------------
  for (int levelD = numDepth, levelD_ref = numDepth_ref, isFirst = 1; levelD > 0; /*nop*/) {
    std::vector<UrahtNode>& weightsParent = weightsLfStack[levelD];
    std::vector<UrahtNode>& weightsLf = weightsLfStack[levelD - 1];

    sameNumNodes = (weightsLf.size() == weightsParent.size());
    
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

    const bool& hybridPredLayer = rahtPredParams.raht_hybrid_prediction_enabled_flag && !haarFlag &&
    (treeDepth >= rahtPredParams.raht_hybrid_prediction_lower_depth_minus1 + 1) &&
    (treeDepth <= rahtPredParams.raht_hybrid_prediction_lower_depth_minus1 + rahtPredParams.raht_hybrid_prediction_num_enabled_layers);

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

	bool curLevelEnableACInterPred = false;
    bool curLevelEnableACIntraPred = false;
    int predMode = 0;
    if (enableRDOCodingLayer) {
      predMode = coder.decodeMode(enableACRDOInterPred, enableACRDONonPred);
      if (enableACRDOInterPred) {
        if (enableACRDONonPred) {  // 0: intraPred 1:interPred 2:nonPred
          curLevelEnableACIntraPred = predMode == 0;
          curLevelEnableACInterPred = (hybridPredLayer ? (predMode < 2) : (predMode == 1));
        } else {  // 0: intraPred 1:interPred
          curLevelEnableACIntraPred = predMode == 0;
          curLevelEnableACInterPred = (hybridPredLayer ? 1 : (predMode == 1));
        }
      } else if (enableACRDONonPred) {
        curLevelEnableACIntraPred = predMode == 0;  // 0: intraPred 1:nonPred
      }
    }
    const bool& enableHybridPred = hybridPredLayer && predMode == 0;
    const bool& interMode = enableACRDOInterPred && predMode == 1;
	//--------------- initialize LCP coeff for current level ------------
    LcpCoeff = 0;
    PCCRAHTComputeLCP curlevelLcp;

	//--------------- initialize parent node information for current level ------------
    if (enablePredictionInLvl) {
      for (auto& ele : weightsParent)
        ele.occupancy = 0;
    }   
      
    //--------------- select quantiser according to transform layer ------------
    qpLayer = std::min(qpLayer + 1, int(qpset.layers.size()) - 1);
    acCoeffQpLayer++;
   
    //--------------- prepare reconstruction buffers ------------
    //  previous reconstruction -> attrRecParent
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
    
	//---------------- get inter filter of current level ---------------------
    int64_t interFilterTap = 128;
    int64_t quantizedResFilterTap = 0;

    if ((!enableFilterEstimation) && (enableACInterPred) && (treeDepth < treeDepthLimit)) {
      int filtexidx = treeDepth < fixedFilterTaps.size() ? treeDepth : (fixedFilterTaps.size()-1);
      interFilterTap = fixedFilterTaps[filtexidx];
    }    

    //get filter tap at the decoder 
    bool enableDecoderParsing = false;

    if (enableFilterEstimation && enableACRDOInterPred && curLevelEnableACInterPred && (predMode > 0) &&
      (treeDepth >= skipInitLayersForFiltering))
    {
      enableDecoderParsing = true;
    }
    else if (enableFilterEstimation && !enableACRDOInterPred && enableACInterPred && 
      (treeDepth >= skipInitLayersForFiltering)) 
    {
      enableDecoderParsing = true;
    }

    if (enableDecoderParsing) {
      auto quantizers = qpset.quantizers(qpLayer, {0,0});
      auto& q = quantizers[0];
      quantizedResFilterTap = attrInterPredParams.paramsForInterRAHT.FilterTaps[filterIdx];
      filterIdx++;
      int64_t recResidueFilterTap = divExp2RoundHalfUp(q.scale(quantizedResFilterTap), kFixedPointAttributeShift);
      interFilterTap = 128 - recResidueFilterTap;
    }

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
        //  //weightsLf_ref[m].cousincount = weightsParent_ref_childcount;
        //  //weightsLf_ref[m].cousin_occupancy = occupancy_ref;
        //  for (int k = 0; k < numAttrs; k++) {
        //    weightsLf_ref[m].parent_sumAttr[k] =
        //      weightsParentIt_ref->sumAttr[k];
        //  }
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
          weightsParentIt->parent_sumAttr[k] =
            nonZeroData[rec_cur_attr_count * numAttrs + k];
        }
      }
      rec_cur_attr_count++;

      FixedPoint SampleBuf[6][8] = {0}, transformBuf[6][8] = {0};
      FixedPoint (*SamplePredBuf)[8] = &SampleBuf[numAttrs], (*transformPredBuf)[8] = &transformBuf[numAttrs];
      FixedPoint SampleInterPredBuf[3][8] = {0}, SamplePredTmpBuf[3][8] = {0}, transformInterPredBuf[3][8] = {0};
      FixedPoint PredDC[3] = {0};
      FixedPoint NodeRecBuf[3][8] = {0};
      FixedPoint normalizedSqrtBuf[8] = {0};

	  // For Lcp prediction
	  int64_t CoeffRecBuf[8][3] = {0};     
      FixedPoint transformResidueRecBuf[3] = {0};
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
        sumWeights_cur += (uint64_t) weights[nodeIdx];
        nodeQp[nodeIdx][0] = it->qp[0] >> regionQpShift;
        nodeQp[nodeIdx][1] = it->qp[1] >> regionQpShift;

        occupancy |= 1 << nodeIdx;        
        nodeCnt_real++;

		if (rahtExtension)
		  nodeCnt++;
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
      if(enableACRDOInterPred)
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
        //  cur_Luma =
        //    (weightsParentIt->parent_sumAttr[0] * weightsParentIt->weight)
        //    >> 15;
        //  minus_Luma = abs(weightsLf_ref[j].parent_sumAttr[0] - cur_Luma);
        //  condition3 = levelD < 1 && cur_Luma < 20 * minus_Luma;
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
          } else
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

        if(haarFlag){
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

      if (enableACRDONonPred)
        enablePrediction = curLevelEnableACIntraPred;

      if (enableACInterPred) {
        if (curLevelEnableACInterPred && interMode && !interNode)
          enablePrediction = enablePredictionInLvl;
        else if (curLevelEnableACIntraPred)
          enablePrediction = enablePredictionInLvl;
      }

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
          intraDcPred_decoder<haarFlag, numAttrs, rahtExtension>(
            parentNeighIdx, childNeighIdx, occupancy,
            attrRecParent.begin(), attrRec.begin(),
            SamplePredBuf, rahtPredParams, limitLow, limitHigh);
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
                (SampleInterPredBuf[k][childIdx].val >> FixedPoint::kFracBits)
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
                  //std::cout << "interParentMean[k].val: "
                  //           << interParentMean[k].val << std::endl;
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
                  //std::cout
                  //  << "sumattr_ref_local[k].val: " << sumattr_ref_local[k].val
                  //  << std::endl;
                  //std::cout
                  //  << "interParentMean[k].val: " << interParentMean[k].val
                  //  << std::endl;
                }
              }
              //for(int k = 0; k < numAttrs; k++){
              //  SampleInterPredBuf[k][childIdx].val = interParentMean[k].val;
              //}
            }
            //if (weights_ref[childIdx] > 1 && weights[childIdx] != 0) {
            if (weights_ref[childIdx] > 1) {
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
          if (enableHybridPred)
            std::copy_n(
              &SamplePredBuf[0][0], numAttrs * 8, &SamplePredTmpBuf[0][0]);
          std::copy_n(
            &SampleInterPredBuf[0][0], numAttrs * 8, &SamplePredBuf[0][0]);
        }

        // --- normalise coefficients in lossy case ---
        for (int childIdx = 0; childIdx < 8; childIdx++) {
          if (weights[childIdx] <= 1)
            continue;

          // Predicted attribute values
          FixedPoint sqrtWeight;
          if (enablePrediction) {
            sqrtWeight.val = fastIsqrt(weights[childIdx]);
            for (int k = 0; k < numAttrs; k++) {
              SamplePredBuf[k][childIdx] *= sqrtWeight;
            }
          }
        }
      }

      if(enableHybridPred){
        if(interNode && enableIntraPrediction){
          for (int childIdx = 0; childIdx < 8; childIdx++){
            if(weights[childIdx]){
              if(weights[childIdx] > 1){
                FixedPoint sqrtWeight;
                sqrtWeight.val =
                  isqrt(uint64_t(weights[childIdx]) << (2 * FixedPoint::kFracBits));
                for(int k = 0; k < numAttrs; k++)
                  SamplePredTmpBuf[k][childIdx] *= sqrtWeight;
              }
              for(int k = 0; k < numAttrs; k++)
                SamplePredTmpBuf[k][childIdx].val = (SamplePredBuf[k][childIdx].val + SamplePredTmpBuf[k][childIdx].val) >> 1;
            }
            for(int k = 0; k < numAttrs; k++)
              SamplePredBuf[k][childIdx].val = SamplePredTmpBuf[k][childIdx].val;
          }
        }
      }


      // forward transform:
      //  - decoder: just transform prediction
      mkWeightTree(weights);
      if (haarFlag) {
        if (enablePrediction){
          std::copy_n(&SamplePredBuf[0][0], numAttrs * 8, &transformPredBuf[0][0]);
          fwdTransformBlock222<HaarKernel>(numAttrs, transformPredBuf, weights);
        }    
      }
      else {      
        if(interNode && !enableHybridPred) //temporal filtering
        {
          for (int childIdx = 0; childIdx < 8; childIdx++) {
            for (int k = 0; k < numAttrs; k++){
              int64_t refVal = 0, filteredVal = 0;
              refVal = SamplePredBuf[k][childIdx].val;
              filteredVal = (treeDepth < skipInitLayersForFiltering) ? refVal
			    : (refVal * interFilterTap ) >> 7;
              SamplePredBuf[k][childIdx].val = filteredVal;
            }
          }
        }
      }
        
      //compute DC of the predictions: Done in the same way at the encoder and decoder to avoid drifting
      if(enablePrediction && !haarFlag){
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
          }
        }
      }
      
      //flags for skiptransform
      bool skipTransform = enablePrediction;
      
      // per-coefficient operations:
      //  - subtract transform domain prediction (encoder)
      //  - subtract the prediction between chroma channel components
      //  - write out/read in quantised coefficients
      //  - inverse quantise + add transform domain prediction
      scanBlock(weights, [&](int idx) {
        // skip the DC coefficient unless at the root of the tree
        if (inheritDc && !idx)
          return;

        // Check if QP offset is to be applied to AC coeffiecients
        Qps coeffQPOffset = (acCoeffQpLayer <= maxAcCoeffQpOffsetLayers && idx)
          ? qpset.rahtAcCoeffQps[acCoeffQpLayer][idx - 1]
          : Qps({0, 0});

        Qps nodeQPOffset = {
          nodeQp[idx][0] + coeffQPOffset[0],
          nodeQp[idx][1] + coeffQPOffset[1]};

        // The RAHT transform
        auto quantizers = qpset.quantizers(qpLayer, nodeQPOffset);
        for (int k = 0; k < numAttrs; k++) {

          auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];

          int64_t coeff = *coeffBufItK[k]++;
          transformResidueRecBuf[k] = CoeffRecBuf[nodelvlSum][k] =
            divExp2RoundHalfUp(q.scale(coeff), kFixedPointAttributeShift);

          if (!haarFlag && enableLCPPred) {
            if (k != 2) {
              NodeRecBuf[k][idx] = transformResidueRecBuf[k];
            } 
			else if (k == 2) {
              transformResidueRecBuf[k].val +=
                (LcpCoeff * transformResidueRecBuf[1].val) >> 4;
              NodeRecBuf[k][idx] = transformResidueRecBuf[k];
              CoeffRecBuf[nodelvlSum][k] = transformResidueRecBuf[k].round();
            }
          } 
		  else {
            transformPredBuf[k][idx] += transformResidueRecBuf[k];
            NodeRecBuf[k][idx] = transformResidueRecBuf[k];
          }
          skipTransform = skipTransform && (NodeRecBuf[k][idx].val == 0);          
        }
        nodelvlSum++;
      });

      // compute last component coefficient
      if (numAttrs == 3 && nodeCnt > 1 && !haarFlag && inheritDc && enableLCPPred) {
        LcpCoeff = curlevelLcp.computeLCPCoeff(nodelvlSum, CoeffRecBuf);
      }
      
      if(haarFlag){
        std::copy_n(&transformPredBuf[0][0], numAttrs * 8, &NodeRecBuf[0][0]);   
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
        }
      }

      if (haarFlag) {
        invTransformBlock222<HaarKernel>(numAttrs, NodeRecBuf, weights);
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
          }
          else {
            FixedPoint temp = transformPredBuf[k][nodeIdx];
            temp.val <<= 2;
            attrRecUs[j * numAttrs + k] = temp.round();
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
            }
          }
        }

        for (int k = 0; k < numAttrs; k++) {
          attrRec[j * numAttrs + k] = rahtExtension
              ? NodeRecBuf[k][nodeIdx].val
            : NodeRecBuf[k][nodeIdx].round();
        }
        j++;
      }      
      i += nodeCnt_real;
    }//end loop on nodes of current level

    sameNumNodes = 0;
    // increment tree depth
    treeDepth++;
  }//end loop on level

  // -------------- process duplicate points at level 0 if exists --------------
  if (!flagNoDuplicate) {
    std::swap(attrRec, attrRecParent);
    auto attrRecParentIt = attrRecParent.cbegin();

    std::vector<int64_t> attrsHf;
    attrsHf.resize(weightsHf.size() * numAttrs);
    auto attrsHfIt = attrsHf.cbegin();

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
      FixedPoint attrRecDc[3];
      FixedPoint sqrtWeight;
      sqrtWeight.val = fastIsqrt(weight);

      for (int k = 0; k < numAttrs; k++) {
        if (rahtExtension)
          attrRecDc[k].val = *attrRecParentIt++;
        else
          attrRecDc[k] = *attrRecParentIt++;
        if (!haarFlag) {
          attrRecDc[k] *= sqrtWeight;
        }
      }

      for (int w = weight - 1; w > 0; w--) {
        RahtKernel kernel(w, 1);
        HaarKernel haarkernel(w, 1);

        auto quantizers = qpset.quantizers(qpLayer, nodeQp);
        for (int k = 0; k < numAttrs; k++) {
          auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];

          FixedPoint transformBuf[2];

          int64_t coeff = *coeffBufItK[k]++;
          transformBuf[1] =
            divExp2RoundHalfUp(q.scale(coeff), kFixedPointAttributeShift);
    
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
      }

      attrsHfIt += (weight - 1) * numAttrs;
      out += weight * numAttrs;
    }
  }
  

  // -------------- write-back reconstructed attributes --------------
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
 * inverse RAHT Fixed Point
 *
 * Inputs:
 * quantStepSizeLuma = Quantization step
 * mortonCode = list of 'voxelCount' Morton codes of voxels, sorted in ascending Morton code order
 * attribCount = number of attributes (e.g., 3 if attributes are red, green, blue)
 * voxelCount = number of voxels
 * coefficients = quantized transformed attributes array, in column-major order
 *
 * Outputs:
 * attributes = 'voxelCount' x 'attribCount' array of attributes, in row-major order
 *
 * Note output weights are typically used only for the purpose of
 * sorting or bucketing for entropy coding.
 */
void
regionAdaptiveHierarchicalInverseTransform(
  const RahtPredictionParams &rahtPredParams,
  const AttributeBrickHeader& abh,
  const QpSet& qpset,
  const Qps* pointQpOffsets,
  int64_t* mortonCode,
  int* attributes,
  const int attribCount,
  const int voxelCount,
  int* coefficients,
  const bool rahtExtension,
  AttributeInterPredParams& attrInterPredParams,
  ModeDecoder& decoder)
{
  switch (attribCount) {
  case 3:
    if (rahtPredParams.integer_haar_enable_flag) {
      if (rahtExtension)
        uraht_process_decoder<true, 3, true>(
          rahtPredParams, abh, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
      else
        uraht_process_decoder<true, 3, false>(
          rahtPredParams, abh, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
    } else {
      if (rahtExtension)
        uraht_process_decoder<false, 3, true>(
          rahtPredParams, abh, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
      else
        uraht_process_decoder<false, 3, false>(
          rahtPredParams, abh, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
    }
    break;

  case 1:
    if (rahtPredParams.integer_haar_enable_flag) {
      if (rahtExtension)
        uraht_process_decoder<true, 1, true>(
          rahtPredParams, abh, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
      else
        uraht_process_decoder<true, 1, false>(
          rahtPredParams, abh, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
    } else {
      if (rahtExtension)
        uraht_process_decoder<false, 1, true>(
          rahtPredParams, abh, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
      else
        uraht_process_decoder<false, 1, false>(
          rahtPredParams, abh, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
    }
    break;
  default: throw std::runtime_error("attribCount only support 1 or 3");
  }
}

//============================================================================

}  // namespace pcc
