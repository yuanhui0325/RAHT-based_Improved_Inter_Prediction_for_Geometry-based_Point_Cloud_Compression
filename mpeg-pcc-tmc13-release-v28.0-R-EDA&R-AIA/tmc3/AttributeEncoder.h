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

#include <stdint.h>
#include <vector>

#include "Attribute.h"
#include "AttributeCommon.h"
#include "PayloadBuffer.h"
#include "PCCTMC3Common.h"
#include "hls.h"
#include "quantization.h"
#include "RAHT.h"

namespace pcc {

//============================================================================
// Opaque definitions (Internal detail)

class PCCResidualsEncoder;
struct PCCResidualsEntropyEstimator;

//============================================================================

class AttributeEncoder : public AttributeEncoderIntf {
public:
  void encode(
    const SequenceParameterSet& sps,
    const AttributeDescription& desc,
    const AttributeParameterSet& attr_aps,
	const GeometryBrickHeader& gbh,
    AttributeBrickHeader& abh,
    AttributeContexts& ctxtMem,
    PCCPointSet3& pointCloud,
    PayloadBuffer* payload,
    AttributeInterPredParams &attrInterPredParams,
    AttributeGranularitySlicingParam &slicingParam,
    ModeEncoder& predEncoder,
    const std::array<double, 4>* rdoqFactors = nullptr) override;

  bool isReusable(
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh) const override;

  void setRefReusable( const Vec3<int> attr_coord_scale,
  const bool predgeom_enabled_flag,
  const int geom_angular_azimuth_scale_log2_minus11,
  const bool enableAttrInterPred,
    const bool enableAttrInterPred2) override;

  bool isRefReusable( const Vec3<int> attr_coord_scale,
  const bool predgeom_enabled_flag,
  const int geom_angular_azimuth_scale_log2_minus11,
  const bool enableAttrInterPred,
    const bool enableAttrInterPred2) const override;

  AttributeLods& getLods() override;
  std::vector<uint32_t>& getIndexes() override;
  std::vector<uint32_t>& pointIndexToPredictorIndex() override;
  bool& canonical_lod_sampling_enabled_flag() override;

protected:
  // todo(df): consider alternative encapsulation

  void encodeReflectancesLift(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const QpSet& qpSet,
    PCCPointSet3& pointCloud,
    PCCResidualsEncoder& encoder,
    AttributeInterPredParams& attrInterPredParams,
    AttributeGranularitySlicingParam &slicingParam);

  void encodeColorsLift(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const QpSet& qpSet,
    PCCPointSet3& pointCloud,
    PCCResidualsEncoder& encoder,
    AttributeInterPredParams& attrInterPredParams,
    AttributeGranularitySlicingParam &slicingParam);

  void encodeReflectancesPred(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const QpSet& qpSet,
    PCCPointSet3& pointCloud,
    PCCResidualsEncoder& encoder,
    AttributeInterPredParams& attrInterPredParams,
    AttributeGranularitySlicingParam &slicingParam);

  void encodeColorsPred(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const QpSet& qpSet,
    PCCPointSet3& pointCloud,
    PCCResidualsEncoder& encoder,
    AttributeInterPredParams& attrInterPredParams,
    AttributeGranularitySlicingParam &slicingParam);

  void encodeReflectancesTransformRaht(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    AttributeBrickHeader& abh,
    const QpSet& qpSet,
    PCCPointSet3& pointCloud,
    PCCResidualsEncoder& encoder,
    AttributeInterPredParams& attrInterPredParams,
    ModeEncoder& predEncoder);

  void encodeColorsTransformRaht(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    AttributeBrickHeader& abh,
    const QpSet& qpSet,
    PCCPointSet3& pointCloud,
    PCCResidualsEncoder& encoder,
    AttributeInterPredParams& attrInterPredParams,
    ModeEncoder& predEncoder,
    const std::array<double, 4>* rdoqFactors = nullptr);

  static Vec3<int64_t> computeColorResiduals(
    const AttributeParameterSet& aps,
    const Vec3<attr_t> color,
    const Vec3<attr_t> predictedColor,
    const Vec3<int8_t> icpCoeff,
    const Quantizers& quant);

  static int computeColorDistortions(
    const AttributeDescription& desc,
    const Vec3<attr_t> color,
    const Vec3<attr_t> predictedColor,
    const Quantizers& quant);

  static void decidePredModeColor(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const PCCPointSet3& pointCloud,
    const std::vector<uint32_t>& indexesLOD,
    const uint32_t predictorIndex,
    PCCPredictor& predictor,
    PCCResidualsEncoder& encoder,
    PCCResidualsEntropyEstimator& context,
    const Vec3<int8_t>& icpCoeff,
    const Quantizers& quant,
    const AttributeInterPredParams& attrInterPredParams);

  static void encodePredModeColor(
    const AttributeParameterSet& aps, int predMode, Vec3<int32_t>& coeff);

  static int64_t computeReflectanceResidual(
    const uint64_t reflectance,
    const uint64_t predictedReflectance,
    const Quantizer& quant);

  static void decidePredModeRefl(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const PCCPointSet3& pointCloud,
    const std::vector<uint32_t>& indexesLOD,
    const uint32_t predictorIndex,
    PCCPredictor& predictor,
    PCCResidualsEncoder& encoder,
    PCCResidualsEntropyEstimator& context,
    const Quantizer& quant,
    const AttributeInterPredParams& attrInterPredParams);

  static void encodePredModeRefl(
    const AttributeParameterSet& aps, int predMode, int32_t& coeff);

private:
  std::vector<int8_t> computeLastComponentPredictionCoeff(
    const AttributeParameterSet& aps,
    const std::vector<Vec3<int64_t>>& coeffs,
    int maxSizeOfCoeff=0);

  std::vector<Vec3<int8_t>> computeInterComponentPredictionCoeffs(
    const AttributeParameterSet& aps,
    const PCCPointSet3& pointCloud,
    const AttributeInterPredParams& attrInterPredParams,
    PCCPointSet3* parentPointCloud = NULL,
     int maxSizeOfCoeff = 0);

private:
  // The current attribute slice header
  AttributeBrickHeader* _abh;
  // The current attribute slice header
  DependentAttributeDataUnitHeader* _abh_dep;

  AttributeLods _lods;
  std::vector<uint32_t> _pointIndexToPredictorIndex;
  bool _canonical_lod_sampling_enabled_flag;
  std::vector<MortonCodeWithIndex> packedVoxel;
  std::vector<int64_t> mortonCode;
  std::vector<Qps> pointQpOffsets;
  bool firstAttributeInSlice;

  Vec3<int> attr_coord_scale;
  bool predgeom_enabled_flag;
  int geom_angular_azimuth_scale_log2_minus11;
  bool enableAttrInterPred;
  bool enableAttrInterPred2;
};

//============================================================================

} /* namespace pcc */
