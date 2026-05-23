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

#include "Attribute.h"
#include "AttributeCommon.h"
#include "PayloadBuffer.h"
#include "PCCTMC3Common.h"
#include "quantization.h"
#include "RAHT.h"
namespace pcc {

//============================================================================
// Opaque definitions (Internal detail)

class PCCResidualsDecoder;

//============================================================================

class AttributeDecoder : public AttributeDecoderIntf {
public:
  void decode(
    const SequenceParameterSet& sps,
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh,
    int geom_num_points_minus1,
    int minGeomNodeSizeLog2,
    const char* payload,
    size_t payloadLen,
    AttributeContexts& ctxtMem,
    PCCPointSet3& pointCloud, 
    AttributeInterPredParams& attrInterPredParams,
    AttributeGranularitySlicingParam &slicingParam,
    ModeDecoder& decoder) override;

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

  void decodeReflectancesLift(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh,
    const QpSet& qpSet,
    int geom_num_points_minus1,
    int minGeomNodeSizeLog2,
    PCCResidualsDecoder& decoder,
    PCCPointSet3& pointCloud,
    const AttributeInterPredParams& attrInterPredParams,
    AttributeGranularitySlicingParam &slicingParam);

  void decodeColorsLift(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh,
    const QpSet& qpSet,
    int geom_num_points_minus1,
    int minGeomNodeSizeLog2,
    PCCResidualsDecoder& decoder,
    PCCPointSet3& pointCloud,
    const AttributeInterPredParams& attrInterPredParams,
    AttributeGranularitySlicingParam &slicingParam);

  void decodeReflectancesPred(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh,
    const QpSet& qpSet,
    PCCResidualsDecoder& decoder,
    PCCPointSet3& pointCloud,
    const AttributeInterPredParams& attrInterPredParams,
    AttributeGranularitySlicingParam &slicingParam);

  void decodeColorsPred(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh,
    const QpSet& qpSet,
    PCCResidualsDecoder& decoder,
    PCCPointSet3& pointCloud,
    const AttributeInterPredParams& attrInterPredParams,
    AttributeGranularitySlicingParam &slicingParam);

  void decodeReflectancesRaht(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh,
    const QpSet& qpSet,
    PCCResidualsDecoder& decoder,
    PCCPointSet3& pointCloud,
    AttributeInterPredParams& attrInterPredParams,
    ModeDecoder& predDecoder);

  void decodeColorsRaht(
    const AttributeDescription& desc,
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh,
    const QpSet& qpSet,
    PCCResidualsDecoder& decoder,
    PCCPointSet3& pointCloud,
    AttributeInterPredParams& attrInterPredParams,
    ModeDecoder& predDecoder);

  static void decodePredModeColor(
    const AttributeParameterSet& aps,
    Vec3<int32_t>& coeff,
    PCCPredictor& predictor);

  static void decodePredModeRefl(
    const AttributeParameterSet& aps, int32_t& coeff, PCCPredictor& predictor);

private:
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
