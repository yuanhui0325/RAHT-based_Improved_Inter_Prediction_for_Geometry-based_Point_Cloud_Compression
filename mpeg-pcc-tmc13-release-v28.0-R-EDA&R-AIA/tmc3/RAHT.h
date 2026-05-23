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

#pragma once
#include <cstdint>

#include "FixedPoint.h"
#include "quantization.h"
#include "hls.h"
#include "PCCTMC3Common.h"
#include <vector>
#include <queue>

#include "AttributeCommon.h"

namespace pcc {
  class PCCResidualsEncoder : protected AttributeContexts {
  public:
    PCCResidualsEncoder(
      const AttributeParameterSet& aps,
      const AttributeBrickHeader& abh,
      const AttributeContexts& ctxtMem);

    EntropyEncoder arithmeticEncoder;

    const AttributeContexts& getCtx() const { return *this; }

    void start(const SequenceParameterSet& sps, int numPoints);
    int stop();

    void encodeRunLength(int runLength);
    void encodeSymbol(uint32_t value, int k1, int k2, int k3);
    int encodeSymbolgetbit(uint32_t value, int k1, int k2, int k3);
    int encodeRunLengthgetbit(int runLength);
    int encodegetbitvalue(int32_t value0, int32_t value1, int32_t value2, int number);
    int encodegetbitvalueY(int32_t value0, int32_t value1, int32_t value2);
    int encodegetbitvalueCb(int32_t value0, int32_t value1, int32_t value2);
    int encodegetbitvalueCr(int32_t value0, int32_t value1, int32_t value2);
    
    void encode420(int32_t value0);
    void encode(int32_t value0, int32_t value1, int32_t value2);
    void encode(int32_t value);

    int availPredModes;
    double bitsPtColor(Vec3<int32_t> value, int parity);
    double bitsPtRefl(int32_t value, int parity);

    // Encoder side residual cost calculation
    const int scaleRes = 1 << 20;
    const int windowLog2 = 6;
    int probResGt0[3];  //prob of residuals larger than 0: 1 for each component
    int probResGt1[3];  //prob of residuals larger than 1: 1 for each component
    void resStatUpdateColor(Vec3<int32_t> values);
    void resStatUpdateRefl(int32_t values);
    void resStatReset();
  };

class ModeCoder
{
public:
  void updateInterIntraEnabled(bool flag, bool flag2, int flag3) {
  
  }
  int decodeMode(const bool& enableInter, const bool& enableIntra) { throw std::runtime_error("not implemented"); }
  void _encodeMode(int predMode, const bool& enableInter, const bool& enableIntra,const bool& code = true) {
    throw std::runtime_error("not implemented");
  }
};

class ModeEncoder: public ModeCoder 
{
  typedef decltype(AdaptiveBitModel::probability) probaType;
  EntropyEncoder* arith;
  std::deque<int> modeBuffer;
  std::deque<bool> curLayerEnableInter;
  std::deque<bool> curLayerEnableIntra;
public:
  void set(EntropyEncoder* coder) {
    arith = coder;
    modeBuffer.resize(0);
    curLayerEnableInter.resize(0);
    curLayerEnableIntra.resize(0);
  }
  ModeEncoder()
    : ModeCoder()
    , arith(nullptr)
  {
    
  }
  ~ModeEncoder() { if (arith) flush(); }
  void flush()
  {
    for (int depth = 0; depth < modeBuffer.size(); ++depth) {
      _encodeMode(modeBuffer[depth], curLayerEnableInter[depth], curLayerEnableIntra[depth], true);
    }
    modeBuffer.clear();
    curLayerEnableInter.clear();
    curLayerEnableIntra.clear();
  }
  
  void _encodeMode(int predMode,const bool& enableInter,const bool& enableIntra,const bool& writeOut = false)
  {
    if (!writeOut) {
      modeBuffer.push_back(predMode);
      curLayerEnableInter.push_back(enableInter);
      curLayerEnableIntra.push_back(enableIntra);
    }
    else {
      if (enableInter) {
        const bool& isInter = predMode == 1;
        arith->encode(isInter);
        if (enableIntra && !isInter) {
          const bool& isIntraPred = predMode == 0;
          arith->encode(isIntraPred);
        }
      }
      else if (enableIntra) {
        const bool& isIntraPredMode = predMode == 0;
        arith->encode(isIntraPredMode);
      }
    }
  }
};


class ModeDecoder : public ModeCoder {
  EntropyDecoder* arith;

public:
  ModeDecoder() : ModeCoder(), arith(nullptr) {}

  void set(EntropyDecoder* coder) { arith = coder; }

  int decodeMode(const bool& enableInter, const bool& enableIntra)
  {
    if (enableInter) {
      bool isInter;
      isInter = arith->decode();
      if (isInter)
        return 1;
      if (!enableIntra)
        return 0;
      bool isIntraPred = arith->decode();
      if (isIntraPred)
        return 0;
      return 2;
    }
    else if (enableIntra) {
      bool isIntraPred = arith->decode();
      if (isIntraPred)
        return 0;
      return 1;
    } 
	else
      return -1;
  }
};

void regionAdaptiveHierarchicalTransform(
  const RahtPredictionParams& rahtPredParams,
  AttributeBrickHeader& abh,
  const QpSet& qpset,
  const Qps* pointQPOffset,
  int64_t* mortonCode,
  int* attributes,
  const int attribCount,
  const int voxelCount,
  int* coefficients,
  const bool removeRoundingOps,
  AttributeInterPredParams& attrInterPredParam,
  ModeEncoder& encoder,
  PCCResidualsEncoder& resencoder,
  const std::array<double, 4>* rdoqFactors = nullptr
);

void regionAdaptiveHierarchicalInverseTransform(
  const RahtPredictionParams &rahtPredParams,
  const AttributeBrickHeader& abh,
  const QpSet& qpset,
  const Qps* pointQpOffset,
  int64_t* mortonCode,
  int* attributes,
  const int attribCount,
  const int voxelCount,
  int* coefficients,
  const bool removeRoundingOps,
  AttributeInterPredParams& attrInterPredParams,
  ModeDecoder& decoder);

namespace RAHT {

//============================================================================

struct UrahtNode {
  int64_t pos;
  int weight;
  std::array<int16_t, 2> qp;

  int64_t sumAttr[3];
  int64_t parent_sumAttr[3];
  uint8_t occupancy;
  std::vector<UrahtNode>::iterator firstChild;
  std::vector<UrahtNode>::iterator lastChild;
};

struct HaarNode {
  int64_t pos;
  int32_t attr[3];
};

//============================================================================
struct PCCRAHTComputeLCP {

  int8_t computeLCPCoeff(int m, int64_t coeffs[][3]);

private:
  int64_t sumk1k2 = 0;
  int64_t sumk1k1 = 0;
  std::queue<int64_t> window1;
  std::queue<int64_t> window2;
};

//============================================================================
// Search for neighbour with @value in the ordered list [first, last).
//
// If distance is positive, search [from, from+distance].
// If distance is negative, search [from-distance, from].

template<typename It, typename T, typename T2, typename Cmp>
It findNeighbour(It first, It last, It from, T value, T2 distance, Cmp compare)
{
  It start = first;
  It end = last;

  if (distance >= 0) {
    start = from;
    if ((distance + 1) < std::distance(from, last))
      end = std::next(from, distance + 1);
  } else {
    end = from;
    if ((-distance) < std::distance(first, from))
      start = std::prev(from, -distance);
  }

  auto found = std::lower_bound(start, end, value, compare);
  if (found == end)
    return last;
  return found;
}

//============================================================================
// Find the neighbours of the node indicated by @t between @first and @last.
// The position weight of each found neighbour is stored in two arrays.

template<typename It>
void findNeighbours(
  It first,
  It last,
  It it,
  It firstChild,
  It lastChild,
  int level,
  uint8_t occupancy,
  int parentNeighIdx[19],
  int childNeighIdx[12][8],
  const bool rahtSubnodePredictionEnabled,
  const int& raht_prediction_search_range)
{
  static const uint8_t neighMasks[19] = {255, 240, 204, 170, 192, 160, 136,
                                         3,   5,   15,  17,  51,  85,  10,
                                         34,  12,  68,  48,  80};

  // current position (discard extra precision)
  int64_t cur_pos = it->pos >> level;

  // the position of the parent, offset by (-1,-1,-1)
  int64_t base_pos = morton3dAdd(cur_pos, -1ll);

  // these neighbour offsets are relative to base_pos
  static const uint8_t neighOffset[19] = {0, 35, 21, 14, 49, 42, 28, 1,  2, 3,
                                          4, 5,  6,  10, 12, 17, 20, 33, 34};

  // special case for the direct parent (no need to search);
  parentNeighIdx[0] = std::distance(first, it);

  for (int i = 1; i < 19; i++) {
    // Only look for neighbours that have an effect
    if (!(occupancy & neighMasks[i])) {
      parentNeighIdx[i] = -1;
      continue;
    }

    // compute neighbour address to look for
    // the delta between it and the current position is
    int64_t neigh_pos = morton3dAdd(base_pos, neighOffset[i]);

    int64_t delta = neigh_pos - cur_pos;
    ///< in there will limit the prediction nearset neighbors searchRange
    if (delta >= 0)
      delta = delta >= raht_prediction_search_range
        ? raht_prediction_search_range
        : delta;
    else
      delta = (-delta) >= raht_prediction_search_range
        ? -raht_prediction_search_range
        : delta;
    // find neighbour
    auto found = findNeighbour(
      first, last, it, neigh_pos, delta,
      [=](decltype(*it)& candidate, int64_t neigh_pos) {
        return (candidate.pos >> level) < neigh_pos;
      });

    if (found == last) {
      parentNeighIdx[i] = -1;
      continue;
    }

    if ((found->pos >> level) != neigh_pos) {
      parentNeighIdx[i] = -1;
      continue;
    }

    parentNeighIdx[i] = std::distance(first, found);
  }

  if (rahtSubnodePredictionEnabled) {
    //initialize the childNeighIdx
    for (int *p = (int*)childNeighIdx, i = 0; i < 96; p++, i++)
      *p = -1;

    static const uint8_t occuMasks[12] = {3,  5,  15, 17, 51, 85,
                                          10, 34, 12, 68, 48, 80};
    static const uint8_t occuShift[12] = {6, 5, 4, 3, 2, 1, 3, 1, 2, 1, 2, 3};

    int curLevel = level - 3;
    for (int i = 0; i < 9; i++) {
      if (parentNeighIdx[7 + i] == -1)
        continue;

      auto neiIt = first + parentNeighIdx[7 + i];
      uint8_t mask =
        (neiIt->occupancy >> occuShift[i]) & occupancy & occuMasks[i];
      if (!mask)
        continue;

      for (auto it = neiIt->firstChild; it != neiIt->lastChild; it++) {
        int nodeIdx = ((it->pos >> curLevel) & 0x7) - occuShift[i];
        if ((nodeIdx >= 0) && ((mask >> nodeIdx) & 1)) {
          childNeighIdx[i][nodeIdx] = std::distance(firstChild, it);
        }
      }
    }

    for (int i = 9; i < 12; i++) {
      if (parentNeighIdx[7 + i] == -1)
        continue;

      auto neiIt = first + parentNeighIdx[7 + i];
      uint8_t mask =
        (neiIt->occupancy << occuShift[i]) & occupancy & occuMasks[i];
      if (!mask)
        continue;

      for (auto it = neiIt->firstChild; it != neiIt->lastChild; it++) {
        int nodeIdx = ((it->pos >> curLevel) & 0x7) + occuShift[i];
        if ((nodeIdx < 8) && ((mask >> nodeIdx) & 1)) {
          childNeighIdx[i][nodeIdx] = std::distance(firstChild, it);
        }
      }
    }
  }
}

//============================================================================
// Encapsulation of a RAHT transform stage.

class RahtKernel {
public:
  inline
  RahtKernel(int weightLeft, int weightRight)
  {
    uint64_t w = weightLeft + weightRight;
    uint64_t isqrtW = fastIrsqrt(w);
    _a.val = fastIsqrt(weightLeft) * isqrtW >> 40;
    _b.val = fastIsqrt(weightRight) * isqrtW >> 40;
  }

  void fwdTransform(
    FixedPoint left, FixedPoint right, FixedPoint* lf, FixedPoint* hf)
  {
    FixedPoint a = _a, b = _b;
    // lf = left * a + right * b
    // hf = right * a - left * b

    *lf = right;
    *lf *= b;
    *hf = right;
    *hf *= a;

    a *= left;
    b *= left;

    *lf += a;
    *hf -= b;
  }

  void invTransform(
    FixedPoint lf, FixedPoint hf, FixedPoint* left, FixedPoint* right)
  {
    FixedPoint a = _a, b = _b;

    *left = lf;
    *left *= a;
    *right = lf;
    *right *= b;

    b *= hf;
    a *= hf;

    *left -= b;
    *right += a;
  }

private:
  FixedPoint _a, _b;
};

//============================================================================
// Encapsulation of an Integer Haar transform stage.

class HaarKernel {
public:
  inline
  HaarKernel(int weightLeft, int weightRight) {}

  void fwdTransform(
    FixedPoint left, FixedPoint right, FixedPoint* lf, FixedPoint* hf)
  {
    hf->val = right.val - left.val;
    lf->val = left.val + ((hf->val >> (1 + hf->kFracBits)) << hf->kFracBits);
  }

  void invTransform(
    FixedPoint lf, FixedPoint hf, FixedPoint* left, FixedPoint* right)
  {
    left->val = lf.val - (((hf.val >> (1 + hf.kFracBits)) << hf.kFracBits));
    right->val = hf.val + left->val;
  }
};

//============================================================================
// In-place transform a set of sparse 2x2x2 blocks each using the same weights

template<class Kernel>
void fwdTransformBlock222(
  int numBufs, FixedPoint buf[][8], int weights[8 + 8 + 8 + 8])
{
  static const int a[4 + 4 + 4] = {0, 2, 4, 6, 0, 4, 1, 5, 0, 1, 2, 3};
  static const int b[4 + 4 + 4] = {1, 3, 5, 7, 2, 6, 3, 7, 4, 5, 6, 7};
  for (int i = 0, iw = 0; i < 12; i++, iw += 2) {
    int i0 = a[i];
    int i1 = b[i];

    if (weights[iw] + weights[iw + 1] == 0)
      continue;

    // only one occupied, propagate to next level
    if (!weights[iw] || !weights[iw + 1]) {
      if (!weights[iw]) {
        for (int k = 0; k < numBufs; k++)
          std::swap(buf[k][i0], buf[k][i1]);
      }
      continue;
    }

    // actual transform
    Kernel kernel(weights[iw], weights[iw + 1]);
    for (int k = 0; k < numBufs; k++) {
      auto& bufk = buf[k];
      kernel.fwdTransform(bufk[i0], bufk[i1], &bufk[i0], &bufk[i1]);
    }
  }
}

//============================================================================

// In-place inverse transform a set of sparse 2x2x2 blocks each using the
// same weights

template<class Kernel>
void invTransformBlock222(
  int numBufs, FixedPoint buf[][8], int weights[8 + 8 + 8 + 8])
{
  static const int a[4 + 4 + 4] = {0, 2, 4, 6, 0, 4, 1, 5, 0, 1, 2, 3};
  static const int b[4 + 4 + 4] = {1, 3, 5, 7, 2, 6, 3, 7, 4, 5, 6, 7};
  for (int i = 11, iw = 22; i >= 0; i--, iw -= 2) {
    int i0 = a[i];
    int i1 = b[i];

    if (weights[iw] + weights[iw + 1] == 0)
      continue;

    // only one occupied, propagate to next level
    if (!weights[iw] || !weights[iw + 1]) {
      if (!weights[iw]) {
        for (int k = 0; k < numBufs; k++)
          std::swap(buf[k][i0], buf[k][i1]);
      }
      continue;
    }

    // actual transform
    Kernel kernel(weights[iw], weights[iw + 1]);
    for (int k = 0; k < numBufs; k++) {
      auto& bufk = buf[k];
      kernel.invTransform(bufk[i0], bufk[i1], &bufk[i0], &bufk[i1]);
    }
  }
}

//============================================================================
// In-place transform a set of sparse 2x2x2 blocks each using the same weights

template<class Kernel>
void ComputeDCfor222Block(
  int numBufs, FixedPoint buf[][8], int weights[8 + 8 + 8 + 8])
{
  static const int a[4 + 4 + 4] = {0, 2, 4, 6, 0, 4, 1, 5, 0, 1, 2, 3};
  static const int b[4 + 4 + 4] = {1, 3, 5, 7, 2, 6, 3, 7, 4, 5, 6, 7};
  static const bool skip[4 + 4 + 4] = {false, false, false, false,
                                       false, false, true,  true,
                                       false, true,  true,  true};
  for (int i = 0, iw = 0; i < 12; i++, iw += 2) {
    if (skip[i])
      continue;

    int i0 = a[i];
    int i1 = b[i];

    if (weights[iw] + weights[iw + 1] == 0)
      continue;

    // only one occupied, propagate to next level
    if (!weights[iw] || !weights[iw + 1]) {
      if (!weights[iw]) {
        for (int k = 0; k < numBufs; k++)
          std::swap(buf[k][i0], buf[k][i1]);
      }
      continue;
    }

    // actual transform
    Kernel kernel(weights[iw], weights[iw + 1]);
    for (int k = 0; k < numBufs; k++) {
      auto& bufk = buf[k];
      kernel.fwdTransform(bufk[i0], bufk[i1], &bufk[i0], &bufk[i1]);
    }
  }
}

//============================================================================
// Invoke mapFn(coefIdx) for each present coefficient in the transform

template<class T>
void scanBlock(int weights[8 + 8 + 8 + 8], T mapFn)
{
  static const int8_t kRahtScanOrder[] = {0, 4, 2, 1, 6, 5, 3, 7};

  // there is always the DC coefficient (empty blocks are not transformed)
  mapFn(0);

  for (int i = 1; i < 8; i++) {
    if (!weights[24 + kRahtScanOrder[i]])
      continue;

    mapFn(kRahtScanOrder[i]);
  }
}

//============================================================================
// Tests if two positions are siblings at the given tree level

inline static bool isSibling(int64_t pos0, int64_t pos1, int level)
{
  return ((pos0 ^ pos1) >> level) == 0;
}

//============================================================================
// expand a set of eight weights into three levels

inline void mkWeightTree(int weights[8 + 8 + 8 + 8])
{
  int* in = &weights[0];
  int* out = &weights[8];

  for (int i = 0; i < 4; i++) {
    out[0] = out[4] = in[0] + in[1];
    if (!in[0] || !in[1])
      out[4] = 0;  // single node, no high frequencies
    in += 2;
    out++;
  }
  out += 4;
  for (int i = 0; i < 4; i++) {
    out[0] = out[4] = in[0] + in[1];
    if (!in[0] || !in[1])
      out[4] = 0;  // single node, no high frequencies
    in += 2;
    out++;
  }
  out += 4;
  for (int i = 0; i < 4; i++) {
    out[0] = out[4] = in[0] + in[1];
    if (!in[0] || !in[1])
      out[4] = 0;  // single node, no high frequencies
    in += 2;
    out++;
  }
}

//============================================================================

// remove any non-unique leaves from a level in the uraht tree
template<bool haarFlag, int numAttrs>
int reduceUnique(
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
    if (haarFlag) {
      for (int k = 0; k < numAttrs; k++) {
        auto temp = node.sumAttr[k] - (weightsInWrIt - 1)->sumAttr[k];
        (weightsInWrIt - 1)->sumAttr[k] += temp >> 1;
        weightsOut->back().sumAttr[k] = temp;
      }
    } 
	else {
      for (int k = 0; k < numAttrs; k++)
        (weightsInWrIt - 1)->sumAttr[k] += node.sumAttr[k];
    }
  }

  // number of nodes in next level
  return std::distance(weightsIn->begin(), weightsInWrIt);
}

//============================================================================
// Split a level of values into sum and difference pairs.
template<bool haarFlag, int numAttrs>
int reduceLevel(
  int level,
  int numNodes,
  std::vector<UrahtNode>* weightsIn,
  std::vector<UrahtNode>* weightsOut,
  std::vector<int64_t>* attrsIn,
  std::vector<int64_t>* attrsOut)
{
  // process a single level of the tree
  int64_t posPrev = -1;
  auto weightsInWrIt = weightsIn->begin();
  auto weightsInRdIt = weightsIn->cbegin();
  auto attrsInWrIt = attrsIn->begin();
  auto attrsInRdIt = attrsIn->begin();

  for (int i = 0; i < numNodes; i++) {
    auto& node = *weightsInRdIt++;
    bool newPair = (posPrev ^ node.pos) >> level != 0;
    posPrev = node.pos;
    if (newPair) {
      *weightsInWrIt++ = node;
      for (int k = 0; k < numAttrs; k++)
        *attrsInWrIt++ = *attrsInRdIt++;
    } else {
      auto& left = *(weightsInWrIt - 1);
      left.weight += node.weight;
      left.qp[0] = (left.qp[0] + node.qp[0]) >> 1;
      left.qp[1] = (left.qp[1] + node.qp[1]) >> 1;
      weightsOut->push_back(node);

      for (int k = 0; k < numAttrs; k++) {
        if (haarFlag) {
          attrsOut->push_back(*attrsInRdIt++ - *(attrsInWrIt - numAttrs + k));
          *(attrsInWrIt - numAttrs + k) += attrsOut->back() >> 1;
        } else {
          *(attrsInWrIt - numAttrs + k) += *attrsInRdIt;
          attrsOut->push_back(*attrsInRdIt++);
        }
      }
    }
  }

  // number of nodes in next level
  return std::distance(weightsIn->begin(), weightsInWrIt);
}

template<bool haarFlag, int numAttrs>
int reduceDepth(
  int level,
  int numNodes,
  std::vector<UrahtNode>* weightsIn,
  std::vector<UrahtNode>* weightsOut)
{
  int64_t posPrev = -1;
  auto weightsInRdIt = weightsIn->begin();
  auto it = weightsIn->begin();
  for (int i = 0; i < weightsIn->size();) {
    // this is a new node
    UrahtNode last = weightsInRdIt[i];
    posPrev = last.pos;
    last.firstChild = it++;

    // look for same node
    int i2 = i + 1;
    for (; i2 < weightsIn->size(); i2++, it++)
      if ((posPrev ^ weightsInRdIt[i2].pos) >> level)
        break;

    // process same nodes
    for (int j = i + 1; j < i2; j++) {
      const auto node = weightsInRdIt[j];
      last.weight += node.weight;
      // TODO: fix local qp to be same in encoder and decoder
      last.qp[0] = (last.qp[0] + node.qp[0]) >> 1;
      last.qp[1] = (last.qp[1] + node.qp[1]) >> 1;

      if (!haarFlag) {
        for (int k = 0; k < numAttrs; k++)
          last.sumAttr[k] += node.sumAttr[k];
      }
    }

    //attribute processign for Haar per direction in the interval [i, i2]
    if (haarFlag) {
      HaarNode haarNode[4];
      // first direction (at most 8 nodes)
      int numNode = 0;
      int64_t posPrevH = -1;
      for (int j = i; j < i2; j++) {
        const auto node = weightsInRdIt[j];
        bool newPair = (posPrevH ^ node.pos) >> (level - 2) != 0;
        posPrevH = node.pos;

        if (newPair) {
          haarNode[numNode].pos = node.pos;
          for (int k = 0; k < numAttrs; k++) {
            haarNode[numNode].attr[k] = node.sumAttr[k];
          }
          numNode++;
        } else {
          auto& lastH = haarNode[numNode - 1];
          for (int k = 0; k < numAttrs; k++) {
            auto temp = node.sumAttr[k] - lastH.attr[k];
            lastH.attr[k] += temp >> 1;
          }
        }
      }

      // second direction (at most 4 nodes)
      int numNode2 = 0;
      posPrevH = -1;
      for (int j = 0; j < numNode; j++) {
        const auto node = haarNode[j];
        bool newPair = (posPrevH ^ node.pos) >> (level - 1) != 0;
        posPrevH = node.pos;

        if (newPair) {
          haarNode[numNode2].pos = node.pos;
          for (int k = 0; k < numAttrs; k++) {
            haarNode[numNode2].attr[k] = node.attr[k];
          }
          numNode2++;
        } else {
          auto& lastH = haarNode[numNode2 - 1];
          for (int k = 0; k < numAttrs; k++) {
            auto temp = node.attr[k] - lastH.attr[k];
            lastH.attr[k] += temp >> 1;
          }
        }
      }

      // third direction (at most 2 nodes).
      auto& lastH = haarNode[0];
      for (int k = 0; k < numAttrs; k++) {
        last.sumAttr[k] = lastH.attr[k];
      }

      if (numNode2 == 2) {
        lastH = haarNode[1];
        for (int k = 0; k < numAttrs; k++) {
          auto temp = lastH.attr[k] - last.sumAttr[k];
          last.sumAttr[k] += temp >> 1;
        }
      }
    }  // end Haar attributes

    last.lastChild = it;
    weightsOut->push_back(last);
    i = i2;
  }

  // number of nodes in next level
  return weightsOut->size();
}
//============================================================================
// Merge sum and difference values to form a tree.
template<bool haarFlag, int numAttrs>
void expandLevel(
  int level,
  int numNodes,
  std::vector<UrahtNode>* weightsIn,   // expand by numNodes before expand
  std::vector<UrahtNode>* weightsOut,  // shrink after expand
  std::vector<int64_t>* attrsIn,
  std::vector<int64_t>* attrsOut)
{
  if (numNodes == 0)
    return;

  // process a single level of the tree
  auto weightsInWrIt = weightsIn->rbegin();
  auto weightsInRdIt = std::next(weightsIn->crbegin(), numNodes);
  auto weightsOutRdIt = weightsOut->crbegin();
  auto attrsInWrIt = attrsIn->rbegin();
  auto attrsInRdIt = std::next(attrsIn->crbegin(), numNodes * numAttrs);
  auto attrsOutRdIt = attrsOut->crbegin();

  for (int i = 0; i < numNodes;) {
    bool isPair = (weightsOutRdIt->pos ^ weightsInRdIt->pos) >> level == 0;
    if (!isPair) {
      *weightsInWrIt++ = *weightsInRdIt++;
      for (int k = 0; k < numAttrs; k++)
        *attrsInWrIt++ = *attrsInRdIt++;

      continue;
    }

    // going to process a pair
    i++;

    // Out node is inserted before In node.
    const auto& nodeDelta = *weightsInWrIt++ = *weightsOutRdIt++;
    auto curAttrIt = attrsInWrIt;
    for (int k = 0; k < numAttrs; k++)
      *attrsInWrIt++ = *attrsOutRdIt++;

    // move In node to correct position, subtracting delta
    *weightsInWrIt = *weightsInRdIt++;
    (weightsInWrIt++)->weight -= nodeDelta.weight;
    for (int k = 0; k < numAttrs; k++) {
      *attrsInWrIt = *attrsInRdIt++;
      if (haarFlag) {
        *attrsInWrIt -= *curAttrIt >> 1;
        *curAttrIt++ += *attrsInWrIt++;
      } else {
        *attrsInWrIt++ -= *curAttrIt++;
      }
    }
  }
}

//============================================================================

} /* namespace RAHT */

} /* namespace pcc */
