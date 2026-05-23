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

#include "AttributeDecoder.h"

#include "AttributeCommon.h"
#include "DualLutCoder.h"
#include "attribute_raw.h"
#include "constants.h"
#include "entropy.h"
#include "hls.h"
#include "io_hls.h"
#include "RAHT.h"
#include "FixedPoint.h"

namespace pcc {

//============================================================================
// An encapsulation of the entropy decoding methods used in attribute coding

class PCCResidualsDecoder : protected AttributeContexts {
public:
  PCCResidualsDecoder(
    const AttributeBrickHeader& abh, const AttributeContexts& ctxtMem);

  EntropyDecoder arithmeticDecoder;

  const AttributeContexts& getCtx() const { return *this; }

  void start(const SequenceParameterSet& sps, const char* buf, int buf_len);
  void stop();

  int decodeRunLength();
  int decodeSymbol(int k1, int k2, int k3);
  void decode(int32_t values[3]);
  void decode420(int32_t values[3]);
  int32_t decode();
};

//----------------------------------------------------------------------------

PCCResidualsDecoder::PCCResidualsDecoder(
  const AttributeBrickHeader& abh, const AttributeContexts& ctxtMem)
  : AttributeContexts(ctxtMem)
{}

//----------------------------------------------------------------------------

void
PCCResidualsDecoder::start(
  const SequenceParameterSet& sps, const char* buf, int buf_len)
{
  arithmeticDecoder.setBuffer(buf_len, buf);
  arithmeticDecoder.enableBypassStream(sps.cabac_bypass_stream_enabled_flag);
  arithmeticDecoder.setBypassBinCodingWithoutProbUpdate(sps.bypass_bin_coding_without_prob_update);
  arithmeticDecoder.start();
}

//----------------------------------------------------------------------------

void
PCCResidualsDecoder::stop()
{
  arithmeticDecoder.stop();
}

//----------------------------------------------------------------------------

int
PCCResidualsDecoder::decodeRunLength()
{
  int runLength = 0;
  auto* ctx = ctxRunLen;
  for (; runLength < 3; runLength++, ctx++) {
    int bin = arithmeticDecoder.decode(*ctx);
    if (!bin)
      return runLength;
  }

  for (int i = 0; i < 4; i++) {
    int bin = arithmeticDecoder.decode(*ctx);
    if (!bin) {
      runLength += arithmeticDecoder.decode();
      return runLength;
    }
    runLength += 2;
  }

  runLength += arithmeticDecoder.decodeExpGolomb(2, *++ctx);
  return runLength;
}

//----------------------------------------------------------------------------

int
PCCResidualsDecoder::decodeSymbol(int k1, int k2, int k3)
{
  if (!arithmeticDecoder.decode(ctxCoeffGtN[0][k1]))
    return 0;

  if (!arithmeticDecoder.decode(ctxCoeffGtN[1][k2]))
    return 1;

  int coeff_abs_minus2 = arithmeticDecoder.decodeExpGolomb(
    1, ctxCoeffRemPrefix[k3], ctxCoeffRemSuffix[k3]);

  return coeff_abs_minus2 + 2;
}

//----------------------------------------------------------------------------

void
PCCResidualsDecoder::decode(int32_t value[3])
{
  value[1] = decodeSymbol(0, 0, 1);
  int b0 = value[1] == 0;
  int b1 = value[1] <= 1;
  value[2] = decodeSymbol(1 + b0, 1 + b1, 1);
  int b2 = value[2] == 0;
  int b3 = value[2] <= 1;
  value[0] = decodeSymbol(3 + (b0 << 1) + b2, 3 + (b1 << 1) + b3, 0);

  if (b0 && b2)
    value[0] += 1;

  if (value[0] && arithmeticDecoder.decode())
    value[0] = -value[0];
  if (value[1] && arithmeticDecoder.decode())
    value[1] = -value[1];
  if (value[2] && arithmeticDecoder.decode())
    value[2] = -value[2];
}

//----------------------------------------------------------------------------
 void
 PCCResidualsDecoder::decode420(int32_t value[3])
 {
   const int b0 = 1, b1 = 1, b2 = 1, b3 = 1;
   auto mag = decodeSymbol(3 + (b0 << 1) + b2, 3 + (b1 << 1) + b3, 0) + 1;
   bool sign = arithmeticDecoder.decode();
   value[0] = sign ? -mag : mag;
   value[1] = 0;
   value[2] = 0;
 }

//----------------------------------------------------------------------------

int32_t
PCCResidualsDecoder::decode()
{
  auto mag = decodeSymbol(0, 0, 0) + 1;
  bool sign = arithmeticDecoder.decode();
  return sign ? -mag : mag;
}

//============================================================================
// AttributeDecoderIntf

AttributeDecoderIntf::~AttributeDecoderIntf() = default;

//============================================================================
// AttributeDecoder factory

std::unique_ptr<AttributeDecoderIntf>
makeAttributeDecoder()
{
  return std::unique_ptr<AttributeDecoder>(new AttributeDecoder());
}

//============================================================================
// AttributeDecoder Members

void
AttributeDecoder::decode(
  const SequenceParameterSet& sps,
  const AttributeDescription& attr_desc,
  const AttributeParameterSet& attr_aps,
  const AttributeBrickHeader& abh,
  int geom_num_points_minus1,
  int minGeomNodeSizeLog2,
  const char* payload,
  size_t payloadLen,
  AttributeContexts& ctxtMem,
  PCCPointSet3& pointCloud,
  AttributeInterPredParams& attrInterPredParams,
  AttributeGranularitySlicingParam &slicingParam,
  ModeDecoder& predDecoder)
{
  if (attr_aps.attr_encoding == AttributeEncoding::kRaw) {
    AttrRawDecoder::decode(
      attr_desc, attr_aps, abh, payload, payloadLen, pointCloud);
    return;
  }
  
  
  QpSet qpSet;
  if(slicingParam.is_dependent_unit)
    qpSet = deriveQpSet(attr_desc, attr_aps, abh, *slicingParam.buf.abh_dep);
  else
    qpSet = deriveQpSet(attr_desc, attr_aps, abh);


  PCCResidualsDecoder decoder(abh, ctxtMem);
  decoder.start(sps, payload, payloadLen);

  bool isEnableCrossTypePred = (sps.attributeSets.size() == 1)
    ? 0
    : attr_aps.cross_attr_prediction_enabled_this_type;
  bool isColor;
  if (attr_aps.refAttrIdx != -1 && isEnableCrossTypePred)
   isColor = (attr_aps.aps_attr_parameter_set_id == 1);

  int64_t maxPos = 1;
  int64_t maxR = 1;
  int64_t maxC = 1;
  int64_t wAttr = 0;

  if (attr_aps.lodParametersPresent() && isEnableCrossTypePred) {
    auto bbox = pointCloud.computeBoundingBox();
    int64_t comMaxPos = bbox.max[0] + bbox.max[1] + bbox.max[2] - bbox.min[0]
      - bbox.min[1] - bbox.min[2];
    maxPos = comMaxPos;
    const int64_t clipMax = (1ll << attr_desc.bitdepth) - 1;
    maxR = clipMax;
    maxC = clipMax * 3;
   
    if (!isColor) {
      int QP = attr_aps.init_qp_minus4 + 4;
      int64_t lambda =  940 - 16 * QP;
      //overflow
      wAttr = round((lambda * (maxPos << 10) / maxC) >> 12);

    } else if (isColor) {
      int QP = attr_aps.init_qp_minus4 + 4;
      int64_t lambda = 4011 - 67 * QP;
      wAttr = round((lambda * (maxPos << 10) / maxR) >> 15);
    }
  }
  // generate LoDs if necessary
  if (attr_aps.lodParametersPresent() && _lods.empty()) {
    _lods.canonical_lod_sampling_enabled_flag = canonical_lod_sampling_enabled_flag();
    _lods.generate(
      attr_aps, abh, pointCloud.getPointCount() - 1, minGeomNodeSizeLog2, pointCloud, attrInterPredParams, slicingParam.maxLevel, slicingParam.is_dependent_unit);
    if(slicingParam.is_dependent_unit)
        _lods.predictFromParent(pointCloud,*slicingParam.buf.pointCloudParent);
  }

  if (isEnableCrossTypePred && !isColor && !_lods.empty()) {
    decideNeighborWithColor(
      attr_aps, _lods.predictors, _lods.indexes, pointCloud, wAttr);

    updatePredictorsForCross(_lods.predictors);
    for (auto& predictor : _lods.predictors) {
      predictor.computeWeights();
      if (attr_aps.attr_encoding == AttributeEncoding::kPredictingTransform)
        if (attr_aps.pred_weight_blending_enabled_flag)
          predictor.blendWeights(
            pointCloud, _lods.indexes, attrInterPredParams);
    }
  } else if (isEnableCrossTypePred && isColor && !_lods.empty()) {
    decideNeighborWithRefl(
      attr_aps, _lods.predictors, _lods.indexes, pointCloud, wAttr);

    updatePredictorsForCross(_lods.predictors);

    for (auto& predictor : _lods.predictors) {
      predictor.computeWeights();
      if (attr_aps.attr_encoding == AttributeEncoding::kPredictingTransform)
        if (attr_aps.pred_weight_blending_enabled_flag)
          predictor.blendWeights(
            pointCloud, _lods.indexes, attrInterPredParams);
    }
  }
 
  firstAttributeInSlice = abh.firstAttributeInSlice;

  if (attr_desc.attr_num_dimensions_minus1 == 0) {
    switch (attr_aps.attr_encoding) {
    case AttributeEncoding::kRAHTransform:
      decodeReflectancesRaht(attr_desc, attr_aps, abh, qpSet, decoder, pointCloud, attrInterPredParams, predDecoder);
      break;

    case AttributeEncoding::kPredictingTransform:
      decodeReflectancesPred(
        attr_desc, attr_aps, abh, qpSet, decoder, pointCloud, attrInterPredParams,slicingParam);
      break;

    case AttributeEncoding::kLiftingTransform:
      decodeReflectancesLift(
        attr_desc, attr_aps, abh, qpSet, geom_num_points_minus1,
        minGeomNodeSizeLog2, decoder, pointCloud, attrInterPredParams,slicingParam);
      break;

    case AttributeEncoding::kRaw:
      // Already handled
      break;
    }
  } else if (attr_desc.attr_num_dimensions_minus1 == 2) {
    switch (attr_aps.attr_encoding) {
    case AttributeEncoding::kRAHTransform:
      decodeColorsRaht(
        attr_desc, attr_aps, abh, qpSet, decoder, pointCloud, attrInterPredParams, predDecoder);
      break;

    case AttributeEncoding::kPredictingTransform:
      decodeColorsPred(
        attr_desc, attr_aps, abh, qpSet, decoder, pointCloud,
        attrInterPredParams, slicingParam);
      break;

    case AttributeEncoding::kLiftingTransform:
      decodeColorsLift(
        attr_desc, attr_aps, abh, qpSet, geom_num_points_minus1,
        minGeomNodeSizeLog2, decoder, pointCloud, attrInterPredParams,
        slicingParam);
      break;

    case AttributeEncoding::kRaw:
      // Already handled
      break;
    }
  } else {
    assert(
      attr_desc.attr_num_dimensions_minus1 == 0
      || attr_desc.attr_num_dimensions_minus1 == 2);
  }

  decoder.stop();

  // save the context state for re-use by a future slice if required
  ctxtMem = decoder.getCtx();
}

//----------------------------------------------------------------------------

bool
AttributeDecoder::isReusable(
  const AttributeParameterSet& aps, const AttributeBrickHeader& abh) const
{
  return _lods.isReusable(aps, abh);
}

//----------------------------------------------------------------------------

void
AttributeDecoder::setRefReusable( const Vec3<int> Attr_coord_scale,
  const bool Predgeom_enabled_flag,
  const int Geom_angular_azimuth_scale_log2_minus11,
  const bool EnableAttrInterPred,
  const bool EnableAttrInterPred2
  ) 
{
  attr_coord_scale = Attr_coord_scale;
  predgeom_enabled_flag = Predgeom_enabled_flag;
  geom_angular_azimuth_scale_log2_minus11 = Geom_angular_azimuth_scale_log2_minus11;
  enableAttrInterPred = EnableAttrInterPred;
  enableAttrInterPred2 = EnableAttrInterPred2;
}

//----------------------------------------------------------------------------

bool
AttributeDecoder::isRefReusable( const Vec3<int> Attr_coord_scale,
  const bool Predgeom_enabled_flag,
  const int Geom_angular_azimuth_scale_log2_minus11,
  const bool EnableAttrInterPred,
  const bool EnableAttrInterPred2
  ) const
{
  if(attr_coord_scale != Attr_coord_scale)
    return false;
  if(predgeom_enabled_flag != Predgeom_enabled_flag)
    return false;
  if(geom_angular_azimuth_scale_log2_minus11 != Geom_angular_azimuth_scale_log2_minus11)
    return false;
  if(enableAttrInterPred != EnableAttrInterPred)
    return false;
  if (enableAttrInterPred2 != EnableAttrInterPred2)
    return false;
  return true;
}


//----------------------------------------------------------------------------
AttributeLods&
AttributeDecoder::getLods(){
    return _lods;
}

//----------------------------------------------------------------------------
std::vector<uint32_t>&
AttributeDecoder::getIndexes(){
    return _lods.indexes;
}

//----------------------------------------------------------------------------
std::vector<uint32_t>&
AttributeDecoder::pointIndexToPredictorIndex(){
  _pointIndexToPredictorIndex.clear();
  _pointIndexToPredictorIndex.resize(_lods.indexes.size());
  for(int predictorIndex=0; predictorIndex<_lods.indexes.size(); predictorIndex++)
      _pointIndexToPredictorIndex[_lods.indexes[predictorIndex]] = predictorIndex;

  return _pointIndexToPredictorIndex;
}

bool& 
AttributeDecoder::canonical_lod_sampling_enabled_flag(){
  return _canonical_lod_sampling_enabled_flag;
};

//----------------------------------------------------------------------------

void
AttributeDecoder::decodePredModeRefl(
  const AttributeParameterSet& aps, int32_t& coeff, PCCPredictor& predictor)
{
  int coeffAbs = abs(coeff);
  int coeffSign = coeff < 0 ? -1 : 1;
  int mode;

  int maxcand =
    aps.max_num_direct_predictors + !aps.direct_avg_predictor_disabled_flag;
  switch (maxcand) {
  case 4:
    mode = coeffAbs & 3;
    coeff = coeffSign * (coeffAbs >> 2);
    break;

  case 3:
    mode = coeffAbs & 1;
    coeffAbs >>= 1;
    if (mode > 0) {
      mode += coeffAbs & 1;
      coeffAbs >>= 1;
    }
    coeff = coeffSign * coeffAbs;
    break;

  case 2:
    mode = coeffAbs & 1;
    coeff = coeffSign * (coeffAbs >> 1);
    break;

  default: mode = 0;
  }

  predictor.predMode = mode + aps.direct_avg_predictor_disabled_flag;
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeReflectancesPred(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const QpSet& qpSet,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud,
  const AttributeInterPredParams& attrInterPredParams,
  AttributeGranularitySlicingParam &slicingParam)
{

  const size_t pointCount = pointCloud.getPointCount();
  const int64_t maxReflectance = (1ll << desc.bitdepth) - 1;
  
  int zeroRunRem = 0;
  int quantLayer = 0;
  int quantLayerID = 0;
  int maxQuantLayerID = int(qpSet.layers.size()) - 1;
  
  std::vector<uint64_t> quantWeights;
  if (!aps.scalable_lifting_enabled_flag) {
    computeQuantizationWeights(
      _lods.predictors, quantWeights, aps.quant_neigh_weight,
      attrInterPredParams.enableAttrInterPred);
  } else {
    computeQuantizationWeightsScalable(
      _lods.predictors, _lods.numPointsInLod, pointCount, 0, quantWeights);
  }

  if(slicingParam.is_dependent_unit){
    if(slicingParam.buf.abh_dep->subgroup_weight_adjustment_enabled_flag) {
      updateQuantizationWeights(_lods.predictors, quantWeights, aps.quant_neigh_weight,slicingParam.buf.abh_dep->subgroup_weight_adj_coeff_a,slicingParam.buf.abh_dep->subgroup_weight_adj_coeff_b);
    }
  }else{
    if(abh.subgroup_weight_adjustment_enabled_flag) {
      updateQuantizationWeights(_lods.predictors, quantWeights, aps.quant_neigh_weight,abh.subgroup_weight_adj_coeff_a,abh.subgroup_weight_adj_coeff_b);
    }
  }

  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    if (predictorIndex == _lods.numPointsInLod[quantLayer]) {
      quantLayer = std::min(int(_lods.numPointsInLod.size()) - 1, quantLayer + 1);
      quantLayerID = std::min(maxQuantLayerID, quantLayerID + 1);
    }
    const uint32_t pointIndex = _lods.indexes[predictorIndex];
    auto quant = qpSet.quantizers(pointCloud[pointIndex], quantLayerID);
    auto& predictor = _lods.predictors[predictorIndex];
    predictor.predMode = 0;

    if (--zeroRunRem < 0)
      zeroRunRem = decoder.decodeRunLength();

    int32_t attValue0 = 0;
    if (!zeroRunRem)
      attValue0 = decoder.decode();

    
    if(slicingParam.is_dependent_unit && (predictorIndex < _lods.numPointsInLod[0])){} else{
        if (predModeEligibleRefl(desc, aps, pointCloud, _lods.indexes, predictor, attrInterPredParams))
            decodePredModeRefl(aps, attValue0, predictor);
    }
    
    attr_t& reflectance = pointCloud.getReflectance(pointIndex);
    int64_t quantPredAttValue;
    
    if(slicingParam.is_dependent_unit && (predictorIndex < _lods.numPointsInLod[0])){
        const auto parentIndex = predictor.neighbors[0].predictorIndex;
        quantPredAttValue = slicingParam.buf.pointCloudParent->getReflectance(parentIndex);
    }else{
        quantPredAttValue = predictor.predictReflectance(pointCloud, _lods.indexes,
      attrInterPredParams);
    }


    int64_t qStep = quant[0].stepSize();
    int64_t weight =
      std::min(static_cast<int64_t>(quantWeights[predictorIndex]), qStep)
      >> kFixedPointWeightShift;
    int64_t delta =
      divExp2RoundHalfUp(quant[0].scale(attValue0), kFixedPointAttributeShift);
    delta /= weight;

    const int64_t reconstructedQuantAttValue = quantPredAttValue + delta;
    reflectance =
      attr_t(PCCClip(reconstructedQuantAttValue, int64_t(0), maxReflectance));
  }
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodePredModeColor(
  const AttributeParameterSet& aps,
  Vec3<int32_t>& coeff,
  PCCPredictor& predictor)
{
  int signk1 = coeff[1] < 0 ? -1 : 1;
  int signk2 = coeff[2] < 0 ? -1 : 1;
  int coeffAbsk1 = abs(coeff[1]);
  int coeffAbsk2 = abs(coeff[2]);

  int mode;
  int maxcand =
    aps.max_num_direct_predictors + !aps.direct_avg_predictor_disabled_flag;
  switch (maxcand) {
    int parityk1, parityk2;
  case 4:
    parityk1 = coeffAbsk1 & 1;
    parityk2 = coeffAbsk2 & 1;
    coeff[1] = signk1 * (coeffAbsk1 >> 1);
    coeff[2] = signk2 * (coeffAbsk2 >> 1);

    mode = (parityk1 << 1) + parityk2;
    break;

  case 3:
    parityk1 = coeffAbsk1 & 1;
    coeff[1] = signk1 * (coeffAbsk1 >> 1);
    mode = parityk1;
    if (parityk1) {
      parityk2 = coeffAbsk2 & 1;
      coeff[2] = signk2 * (coeffAbsk2 >> 1);
      mode += parityk2;
    }
    break;

  case 2:
    parityk1 = coeffAbsk1 & 1;
    coeff[1] = signk1 * (coeffAbsk1 >> 1);
    mode = parityk1;
    break;

  default: assert(maxcand >= 2); mode = 0;
  }

  predictor.predMode = mode + aps.direct_avg_predictor_disabled_flag;
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeColorsPred(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const QpSet& qpSet,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud,
  const AttributeInterPredParams& attrInterPredParams,
  AttributeGranularitySlicingParam &slicingParam)
{
  const size_t pointCount = pointCloud.getPointCount();

  int64_t clipMax = (1 << desc.bitdepth) - 1;
  Vec3<int32_t> values;
  

  bool icpPresent = false;
  Vec3<int8_t> icpCoeff = {0,0,0};
  
  if(slicingParam.is_dependent_unit){
    icpPresent = slicingParam.buf.abh_dep->icpPresent(desc, aps);
    icpCoeff = icpPresent ? slicingParam.buf.abh_dep->icpCoeffs[0] : 0;
  }else{
    icpPresent = abh.icpPresent(desc, aps);
    icpCoeff = icpPresent ? abh.icpCoeffs[0] : 0;
  }

  int lod = 0;
  int zeroRunRem = 0;
  int quantLayer = 0;
  int quantLayerID = 0;
  int maxQuantLayerID = int(qpSet.layers.size()) - 1;
  

  std::vector<uint64_t> quantWeights;
  if (!aps.scalable_lifting_enabled_flag) {
    computeQuantizationWeights(
      _lods.predictors, quantWeights, aps.quant_neigh_weight,
      attrInterPredParams.enableAttrInterPred);
  } else {
    computeQuantizationWeightsScalable(
      _lods.predictors, _lods.numPointsInLod, pointCount, 0, quantWeights);
  }

  
  if(slicingParam.is_dependent_unit){
    if(slicingParam.buf.abh_dep->subgroup_weight_adjustment_enabled_flag) {
      updateQuantizationWeights(_lods.predictors, quantWeights, aps.quant_neigh_weight,slicingParam.buf.abh_dep->subgroup_weight_adj_coeff_a,slicingParam.buf.abh_dep->subgroup_weight_adj_coeff_b);
    }
  }else{
    if(abh.subgroup_weight_adjustment_enabled_flag) {
      updateQuantizationWeights(_lods.predictors, quantWeights, aps.quant_neigh_weight,abh.subgroup_weight_adj_coeff_a,abh.subgroup_weight_adj_coeff_b);
    }
  }

  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    if (predictorIndex == _lods.numPointsInLod[quantLayer]) {
      quantLayer = std::min(int(_lods.numPointsInLod.size()) - 1, quantLayer + 1);
      quantLayerID = std::min(maxQuantLayerID, quantLayerID + 1);
    }
    const uint32_t pointIndex = _lods.indexes[predictorIndex];
    auto quant = qpSet.quantizers(pointCloud[pointIndex], quantLayerID);
    auto& predictor = _lods.predictors[predictorIndex];
    predictor.predMode = 0;

    if (--zeroRunRem < 0)
      zeroRunRem = decoder.decodeRunLength();

    if (zeroRunRem)
      values[0] = values[1] = values[2] = 0;
    else
      decoder.decode(&values[0]);
    
    if(slicingParam.is_dependent_unit && (predictorIndex < _lods.numPointsInLod[0])){} else{
      if (predModeEligibleColor(
            desc, aps, pointCloud, _lods.indexes, predictor,
            attrInterPredParams))
          decodePredModeColor(aps, values, predictor);
    }
    

    Vec3<attr_t>& color = pointCloud.getColor(pointIndex);
    Vec3<attr_t> predictedColor;
    
    if(slicingParam.is_dependent_unit && (predictorIndex < _lods.numPointsInLod[0])){
        const auto parentIndex = predictor.neighbors[0].predictorIndex;
        predictedColor = slicingParam.buf.pointCloudParent->getColor(parentIndex);
    }else{
        predictedColor = predictor.predictColor(
          pointCloud, _lods.indexes, attrInterPredParams);
    }

    if (icpPresent && predictorIndex == _lods.numPointsInLod[lod]){
      if(slicingParam.is_dependent_unit)
        icpCoeff = slicingParam.buf.abh_dep->icpCoeffs[++lod];
      else
        icpCoeff = abh.icpCoeffs[++lod];
    }

    int64_t residual0 = 0;
    for (int k = 0; k < 3; ++k) {
      const auto& q = quant[std::min(k, 1)];

      int64_t qStep = q.stepSize();
      int64_t weight =
        std::min(static_cast<int64_t>(quantWeights[predictorIndex]), qStep)
        >> kFixedPointWeightShift;
      int64_t residual =
        divExp2RoundHalfUp(q.scale(values[k]), kFixedPointAttributeShift);
      residual /= weight;

      const int64_t recon =
        predictedColor[k] + residual + ((icpCoeff[k] * residual0 + 2) >> 2);
      color[k] = attr_t(PCCClip(recon, int64_t(0), clipMax));

      if (!k && aps.inter_component_prediction_enabled_flag)
        residual0 = residual;
    }
  }
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeReflectancesRaht(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const QpSet& qpSet,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud,
  AttributeInterPredParams& attrInterPredParams,
  ModeDecoder& predDecoder)
{
  const int voxelCount = int(pointCloud.getPointCount());
  if(firstAttributeInSlice){
    packedVoxel.resize(voxelCount);
    for (int n = 0; n < voxelCount; n++) {
      packedVoxel[n].mortonCode = mortonAddr(times(pointCloud[n], aps.lodNeighBias));
      packedVoxel[n].index = n;
    }
    sort(packedVoxel.begin(), packedVoxel.end());    
    mortonCode.resize(voxelCount);  
    pointQpOffsets.resize(voxelCount);
    for (int n = 0; n < voxelCount; n++) {
      mortonCode[n] = packedVoxel[n].mortonCode;
      pointQpOffsets[n] = qpSet.regionQpOffset(pointCloud[packedVoxel[n].index]);     
    }
  }
  // Entropy decode
  const int attribCount = 1;
  std::vector<int> coefficients(attribCount * voxelCount);
 
  int zeroRunRem = 0;
  for (int n = 0; n < voxelCount; ++n) {
    if (--zeroRunRem < 0)
      zeroRunRem = decoder.decodeRunLength();

    uint32_t value = 0;
    if (!zeroRunRem)
      value = decoder.decode();
    coefficients[n] = value;
  }

  std::vector<int> attributes(attribCount * voxelCount);
  bool enableACRDOInterLayer = aps.raht_enable_code_layer && attrInterPredParams.enableAttrInterPred;
  bool enableACRDOIntraLayer = aps.rahtPredParams.raht_enable_intraPred_nonPred_code_layer && aps.rahtPredParams.raht_prediction_enabled_flag;
  bool enableRDOCodinglayer = enableACRDOInterLayer || enableACRDOIntraLayer;

  if (attrInterPredParams.enableAttrInterPred) {
    if (enableRDOCodinglayer)
      predDecoder.set(&decoder.arithmeticDecoder);
    const int voxelCount_ref = int(attrInterPredParams.getPointCount());
    attrInterPredParams.paramsForInterRAHT.voxelCount = voxelCount_ref;
    if(firstAttributeInSlice){
      attrInterPredParams.paramsForInterRAHT.packedVoxel.resize(voxelCount_ref);
      for (int n = 0; n < voxelCount_ref; n++) {
        attrInterPredParams.paramsForInterRAHT.packedVoxel[n].mortonCode =
          mortonAddr(times(attrInterPredParams.getPoint(n), aps.lodNeighBias));
        attrInterPredParams.paramsForInterRAHT.packedVoxel[n].index = n;
      }
      sort(attrInterPredParams.paramsForInterRAHT.packedVoxel.begin(), attrInterPredParams.paramsForInterRAHT.packedVoxel.end());
      attrInterPredParams.paramsForInterRAHT.mortonCode.resize(voxelCount_ref);
      for (int n = 0; n < voxelCount_ref; n++) {      
        attrInterPredParams.paramsForInterRAHT.mortonCode[n] =
          attrInterPredParams.paramsForInterRAHT.packedVoxel[n].mortonCode;
      }       
    }

    attrInterPredParams.paramsForInterRAHT.attributes.resize(
      attribCount * voxelCount_ref);
    // Populate input arrays.
    for (int n = 0; n < voxelCount_ref; n++) {
      attrInterPredParams.paramsForInterRAHT.attributes[n] =
        attrInterPredParams.getReflectance(attrInterPredParams.paramsForInterRAHT.packedVoxel[n].index);
    }
    if (attrInterPredParams.hasDupPoints()) {
      attrInterPredParams.paramsForInterRAHT.dupPoints.resize(voxelCount_ref);
      attrInterPredParams.paramsForInterRAHT.dupPoints_available =1;
      for (int n = 0; n < voxelCount_ref; n++) {
        attrInterPredParams.paramsForInterRAHT.dupPoints[n] =
          attrInterPredParams.getDupPoints(attrInterPredParams.paramsForInterRAHT.packedVoxel[n].index);
      }
    }
  }
  else {
    if (enableRDOCodinglayer) {
      predDecoder.set(&decoder.arithmeticDecoder);
    }
  }
  regionAdaptiveHierarchicalInverseTransform(
    aps.rahtPredParams, abh, qpSet, pointQpOffsets.data(), mortonCode.data(),
    attributes.data(), attribCount, voxelCount, coefficients.data(),
    aps.raht_extension, attrInterPredParams,predDecoder);
  const int64_t maxReflectance = (1 << (int64_t)desc.bitdepth) - 1;
  const int64_t minReflectance = 0;
  for (int n = 0; n < voxelCount; n++) {
    int64_t val = attributes[attribCount * n];
    const attr_t reflectance =
      attr_t(PCCClip(val, minReflectance, maxReflectance));
    pointCloud.setReflectance(packedVoxel[n].index, reflectance);
    attributes[attribCount * n] = reflectance;
  }
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeColorsRaht(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const QpSet& qpSet,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud,
  AttributeInterPredParams& attrInterPredParams,
  ModeDecoder& predDecoder)
{
  const int voxelCount = int(pointCloud.getPointCount());

  if(firstAttributeInSlice){
    packedVoxel.resize(voxelCount);
    for (int n = 0; n < voxelCount; n++) {
      packedVoxel[n].mortonCode = mortonAddr(times(pointCloud[n], aps.lodNeighBias));
      packedVoxel[n].index = n;
    }
    sort(packedVoxel.begin(), packedVoxel.end());    
    mortonCode.resize(voxelCount);  
    pointQpOffsets.resize(voxelCount);
    for (int n = 0; n < voxelCount; n++) {
      mortonCode[n] = packedVoxel[n].mortonCode;
      pointQpOffsets[n] = qpSet.regionQpOffset(pointCloud[packedVoxel[n].index]);     
    }
  }

  // Entropy decode
  const int attribCount = 3;
  std::vector<int> coefficients(attribCount * voxelCount);
  const bool is420 = abh.is420;
  int first420 = is420 ? abh.first420 : -1;

  int zeroRunRem = 0;
  for (int n = 0; n < voxelCount; ++n) {
    if (--zeroRunRem < 0)
      zeroRunRem = decoder.decodeRunLength();

    int32_t values[3] = {};
    if (!zeroRunRem) {
      bool voxel420 = is420 && n >= first420;
      if (voxel420)
        decoder.decode420(values);
      else
        decoder.decode(values);
    }

    for (int d = 0; d < attribCount; ++d) {
      coefficients[voxelCount * d + n] = values[d];
    }
  }

  std::vector<int> attributes(attribCount * voxelCount);
  bool enableACRDOInterLayer = aps.raht_enable_code_layer && attrInterPredParams.enableAttrInterPred;
  bool enableACRDOIntraLayer = aps.rahtPredParams.raht_enable_intraPred_nonPred_code_layer && aps.rahtPredParams.raht_prediction_enabled_flag;
  bool enableRDOCodinglayer = enableACRDOInterLayer || enableACRDOIntraLayer;
  if (attrInterPredParams.enableAttrInterPred) {
    if (enableRDOCodinglayer)
      predDecoder.set(&decoder.arithmeticDecoder);

    const int voxelCount_ref = int(attrInterPredParams.getPointCount());
    attrInterPredParams.paramsForInterRAHT.voxelCount = voxelCount_ref;

    if(firstAttributeInSlice){
      attrInterPredParams.paramsForInterRAHT.packedVoxel.resize(voxelCount_ref);
      for (int n = 0; n < voxelCount_ref; n++) {
        attrInterPredParams.paramsForInterRAHT.packedVoxel[n].mortonCode =
          mortonAddr(times(attrInterPredParams.getPoint(n), aps.lodNeighBias));
        attrInterPredParams.paramsForInterRAHT.packedVoxel[n].index = n;
      }

      sort(
        attrInterPredParams.paramsForInterRAHT.packedVoxel.begin(),
        attrInterPredParams.paramsForInterRAHT.packedVoxel.end());

      attrInterPredParams.paramsForInterRAHT.mortonCode.resize(voxelCount_ref);

	  for (int n = 0; n < voxelCount_ref; n++) {
        attrInterPredParams.paramsForInterRAHT.mortonCode[n] =
          attrInterPredParams.paramsForInterRAHT.packedVoxel[n].mortonCode;
      }
	}

    attrInterPredParams.paramsForInterRAHT.attributes.resize(attribCount * voxelCount_ref);

    // Populate input arrays.
    for (int n = 0; n < voxelCount_ref; n++) {

      auto color = attrInterPredParams.getColor(attrInterPredParams.paramsForInterRAHT.packedVoxel[n].index);

      attrInterPredParams.paramsForInterRAHT.attributes[attribCount * n] = color[0];
      attrInterPredParams.paramsForInterRAHT.attributes[attribCount * n + 1] = color[1];
      attrInterPredParams.paramsForInterRAHT.attributes[attribCount * n + 2] = color[2];

    }

  }
  else {
    if (enableRDOCodinglayer) {
      predDecoder.set(&decoder.arithmeticDecoder);
    }
  }

  regionAdaptiveHierarchicalInverseTransform(
    aps.rahtPredParams, abh, qpSet, pointQpOffsets.data(), mortonCode.data(),
    attributes.data(), attribCount, voxelCount, coefficients.data(),
    aps.raht_extension, attrInterPredParams,predDecoder);

  int clipMax = (1 << desc.bitdepth) - 1;
  for (int n = 0; n < voxelCount; n++) {
    const int r = attributes[attribCount * n];
    const int g = attributes[attribCount * n + 1];
    const int b = attributes[attribCount * n + 2];
    Vec3<attr_t> color;
    color[0] = attr_t(PCCClip(r, 0, clipMax));
    color[1] = attr_t(PCCClip(g, 0, clipMax));
    color[2] = attr_t(PCCClip(b, 0, clipMax));
    pointCloud.setColor(packedVoxel[n].index, color);
  }
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeColorsLift(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const QpSet& qpSet,
  int geom_num_points_minus1,
  int minGeomNodeSizeLog2,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud,
  const AttributeInterPredParams& attrInterPredParams,
  AttributeGranularitySlicingParam &slicingParam)
{
  const size_t pointCount = pointCloud.getPointCount();
  std::vector<uint64_t> weights;
  
  if (!aps.scalable_lifting_enabled_flag) {
    PCCComputeQuantizationWeights(
      _lods.predictors, weights, attrInterPredParams.enableAttrInterPred);
  } else {
      computeQuantizationWeightsScalable(
        _lods.predictors, _lods.numPointsInLod, geom_num_points_minus1 + 1,
        minGeomNodeSizeLog2, weights);
  }

  const size_t lodCount = _lods.numPointsInLod.size();
  std::vector<Vec3<int64_t>> colors;
  colors.resize(pointCount);

  std::vector<Vec3<int64_t>> colorsRef;
  colorsRef.resize(attrInterPredParams.getPointCount());

  for (size_t index = 0; index < attrInterPredParams.getPointCount();
       ++index) {
      const auto& colorRef = attrInterPredParams.getColor(index);
      for (size_t d = 0; d < 3; ++d) {
      colorsRef[index][d] = int32_t(colorRef[d]) << kFixedPointAttributeShift;
      }
  }
  
  size_t pointCountParent = 0;
  std::vector<Vec3<int64_t>> colorsParent;
  if(slicingParam.is_dependent_unit){
      pointCountParent = slicingParam.buf.pointCloudParent->getPointCount();
      colorsParent.resize(pointCountParent);

      for (size_t index = 0; index < pointCountParent; ++index) {
        const auto& color = slicingParam.buf.pointCloudParent->getColor(index);
        for (size_t d = 0; d < 3; ++d) {
          colorsParent[index][d] = int32_t(color[d]) << kFixedPointAttributeShift;
        }
      }
  }

  // decompress
  // Per level-of-detail coefficients for last component prediction
  int lod = 0;
  int8_t lastCompPredCoeff = 0;
  if (aps.last_component_prediction_enabled_flag){
    if(slicingParam.is_dependent_unit){
      lastCompPredCoeff = slicingParam.buf.abh_dep->attrLcpCoeffs[0];    
    }else{
      lastCompPredCoeff = abh.attrLcpCoeffs[0];
    }
  }

  int zeroRunRem = 0;
  int quantLayer = 0;
  int quantLayerID = 0;
  int maxQuantLayerID = int(qpSet.layers.size()) - 1;
  

  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    if (predictorIndex == _lods.numPointsInLod[quantLayer]) {
      quantLayer = std::min(int(_lods.numPointsInLod.size()) - 1, quantLayer + 1);
      quantLayerID = std::min(maxQuantLayerID, quantLayerID + 1);
    }

    if (predictorIndex == _lods.numPointsInLod[lod]) {
      lod++;
      if (aps.last_component_prediction_enabled_flag){        
        if(slicingParam.is_dependent_unit){
          lastCompPredCoeff = slicingParam.buf.abh_dep->attrLcpCoeffs[lod];    
        }else{
          lastCompPredCoeff = abh.attrLcpCoeffs[lod];
        }      
      }
    }
    
    const uint32_t pointIndex = _lods.indexes[predictorIndex];
    auto quant = qpSet.quantizers(pointCloud[pointIndex], quantLayerID);

    if (--zeroRunRem < 0)
      zeroRunRem = decoder.decodeRunLength();

    int32_t values[3] = {};
    if (!zeroRunRem)
      decoder.decode(values);

    const int64_t iQuantWeight = irsqrt(weights[predictorIndex]);
    auto& color = colors[predictorIndex];

    int64_t scaled = quant[0].scale(values[0]);
    color[0] = divExp2RoundHalfInf(scaled * iQuantWeight, 36);

    scaled = quant[1].scale(values[1]);
    color[1] = divExp2RoundHalfInf(scaled * iQuantWeight, 36);

    scaled *= lastCompPredCoeff;
    scaled >>= 2;

    scaled += quant[1].scale(values[2]);
    color[2] = divExp2RoundHalfInf(scaled * iQuantWeight, 36);
  }

  // reconstruct
  
  if(slicingParam.is_dependent_unit){
    const size_t startIndex = 0;
    const size_t endIndex = _lods.numPointsInLod[0];
    PCCLiftPredict(_lods.predictors, startIndex, endIndex, false, colors, &colorsParent);  
  }

  for (size_t lodIndex = 1; lodIndex < lodCount; ++lodIndex) {
    const size_t startIndex = _lods.numPointsInLod[lodIndex - 1];
    const size_t endIndex = _lods.numPointsInLod[lodIndex];
    PCCLiftUpdate(
      _lods.predictors, weights, startIndex, endIndex, false, colors,
      attrInterPredParams.enableAttrInterPred);
    PCCLiftPredict(
      _lods.predictors, startIndex, endIndex, false, colors,
      attrInterPredParams.enableAttrInterPred, colorsRef);
  }

  int64_t clipMax = (1 << desc.bitdepth) - 1;
  for (size_t f = 0; f < pointCount; ++f) {
    const auto color0 =
      divExp2RoundHalfInf(colors[f], kFixedPointAttributeShift);
    Vec3<attr_t> color;
    for (size_t d = 0; d < 3; ++d) {
      color[d] = attr_t(PCCClip(color0[d], int64_t(0), clipMax));
    }
    pointCloud.setColor(_lods.indexes[f], color);
  }
}

//----------------------------------------------------------------------------

void
AttributeDecoder::decodeReflectancesLift(
  const AttributeDescription& desc,
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  const QpSet& qpSet,
  int geom_num_points_minus1,
  int minGeomNodeSizeLog2,
  PCCResidualsDecoder& decoder,
  PCCPointSet3& pointCloud,
  const AttributeInterPredParams& attrInterPredParams,
  AttributeGranularitySlicingParam &slicingParam)
{
  const size_t pointCount = pointCloud.getPointCount();
  std::vector<uint64_t> weights;
    
  if (!aps.scalable_lifting_enabled_flag) {
    PCCComputeQuantizationWeights(_lods.predictors, weights, attrInterPredParams.enableAttrInterPred);
  } else {
      computeQuantizationWeightsScalable(
        _lods.predictors, _lods.numPointsInLod, geom_num_points_minus1 + 1,
        minGeomNodeSizeLog2, weights);
  }

  const size_t lodCount = _lods.numPointsInLod.size();
  std::vector<int64_t> reflectances;
  reflectances.resize(pointCount);
  
  std::vector<int64_t> reflectancesRef;
  const int numPts = attrInterPredParams.getPointCount();
  reflectancesRef.resize(numPts);
  for (size_t index = 0; index < numPts; ++index) {
    reflectancesRef[index] = int32_t(attrInterPredParams.getReflectance(index))
      << kFixedPointAttributeShift;
  }

  size_t pointCountParent = 0;
  std::vector<int64_t> reflectancesParent;
  if(slicingParam.is_dependent_unit){
      pointCountParent = slicingParam.buf.pointCloudParent->getPointCount();
      reflectancesParent.resize(pointCountParent);

      for (size_t index = 0; index < pointCountParent; ++index) {
        const auto& reflectance = slicingParam.buf.pointCloudParent->getReflectance(index);
          reflectancesParent[index] = int32_t(reflectance) << kFixedPointAttributeShift;
      }
  }

  // decompress
  int zeroRunRem = 0;
  int quantLayer = 0;
  int quantLayerID = 0;
  int maxQuantLayerID = int(qpSet.layers.size()) - 1;
  

  for (size_t predictorIndex = 0; predictorIndex < pointCount;
       ++predictorIndex) {
    if (predictorIndex == _lods.numPointsInLod[quantLayer]) {
      quantLayer = std::min(int(_lods.numPointsInLod.size()) - 1, quantLayer + 1);
      quantLayerID = std::min(maxQuantLayerID, quantLayerID + 1);
    }
    const uint32_t pointIndex = _lods.indexes[predictorIndex];
    auto quant = qpSet.quantizers(pointCloud[pointIndex], quantLayerID);


    if (--zeroRunRem < 0)
      zeroRunRem = decoder.decodeRunLength();

    int64_t detail = 0;
    if (!zeroRunRem)
      detail = decoder.decode();

    const int64_t iQuantWeight = irsqrt(weights[predictorIndex]);
    auto& reflectance = reflectances[predictorIndex];
    const int64_t delta = detail;
    const int64_t reconstructedDelta = quant[0].scale(delta);
    reflectance = divExp2RoundHalfInf(reconstructedDelta * iQuantWeight, 36);
  }

  // reconstruct
  
  if(slicingParam.is_dependent_unit){
    const size_t startIndex = 0;
    const size_t endIndex = _lods.numPointsInLod[0];
    PCCLiftPredict(_lods.predictors, startIndex, endIndex, false, reflectances, &reflectancesParent);  
  }

  for (size_t lodIndex = 1; lodIndex < lodCount; ++lodIndex) {
    const size_t startIndex = _lods.numPointsInLod[lodIndex - 1];
    const size_t endIndex = _lods.numPointsInLod[lodIndex];
    PCCLiftUpdate(
      _lods.predictors, weights, startIndex, endIndex, false, reflectances
      , attrInterPredParams.enableAttrInterPred);
    PCCLiftPredict(
      _lods.predictors, startIndex, endIndex, false, reflectances
      , attrInterPredParams.enableAttrInterPred, reflectancesRef);
  }
  const int64_t maxReflectance = (1 << desc.bitdepth) - 1;
  for (size_t f = 0; f < pointCount; ++f) {
    const auto refl =
      divExp2RoundHalfInf(reflectances[f], kFixedPointAttributeShift);
    pointCloud.setReflectance(
      _lods.indexes[f], attr_t(PCCClip(refl, int64_t(0), maxReflectance)));
  }
}

//============================================================================

} /* namespace pcc */
