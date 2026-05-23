/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2017-2019, ISO/IEC
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

#include <memory>

#include "hls.h"
#include "PayloadBuffer.h"
#include "PCCPointSet.h"
#include "PCCTMC3Common.h"
#include "entropy.h"
#include "AttributeCommon.h"
#include "RAHT.h"

namespace pcc {

//============================================================================

class AttributeContexts;

//============================================================================

class AttributeDecoderIntf {
public:
  virtual ~AttributeDecoderIntf();

  virtual void decode(
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
    AttributeGranularitySlicingParam& slicingParam,
    ModeDecoder& decoder) = 0;

  // Indicates if the attribute decoder can decode the given aps
  virtual bool isReusable(
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh) const = 0;

  virtual void setRefReusable( const Vec3<int> attr_coord_scale,
  const bool predgeom_enabled_flag,
  const int geom_angular_azimuth_scale_log2_minus11,
  const bool enableAttrInterPred,
    const bool enableAttrInterPred2) = 0;

  virtual bool isRefReusable( const Vec3<int> attr_coord_scale,
  const bool predgeom_enabled_flag,
  const int geom_angular_azimuth_scale_log2_minus11,
  const bool enableAttrInterPred,
    const bool enableAttrInterPred2) const = 0;
  
  virtual AttributeLods& getLods() = 0;
  virtual std::vector<uint32_t>& getIndexes() = 0;
  virtual std::vector<uint32_t>& pointIndexToPredictorIndex() = 0;
  virtual bool& canonical_lod_sampling_enabled_flag() = 0;
};

//----------------------------------------------------------------------------

std::unique_ptr<AttributeDecoderIntf> makeAttributeDecoder();

//============================================================================

class AttributeEncoderIntf {
public:
  virtual ~AttributeEncoderIntf();

  virtual void encode(
    const SequenceParameterSet& sps,
    const AttributeDescription& desc,
    const AttributeParameterSet& attr_aps,
	const GeometryBrickHeader& gbh,
    AttributeBrickHeader& abh,
    AttributeContexts& ctxtMem,
    PCCPointSet3& pointCloud,
    PayloadBuffer* payload,
    AttributeInterPredParams &attrInterPredParams,
    AttributeGranularitySlicingParam& slicingParam,
    ModeEncoder& predEncoder,
    const std::array<double, 4>* rdoqFactors = nullptr) = 0;

  // Indicates if the attribute decoder can decode the given aps
  virtual bool isReusable(
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh) const = 0;

  virtual void setRefReusable( const Vec3<int> attr_coord_scale,
  const bool predgeom_enabled_flag,
  const int geom_angular_azimuth_scale_log2_minus11,
  const bool enableAttrInterPred,
    const bool enableAttrInterPred2) = 0;

  virtual bool isRefReusable( const Vec3<int> attr_coord_scale,
  const bool predgeom_enabled_flag,
  const int geom_angular_azimuth_scale_log2_minus11,
  const bool enableAttrInterPred,
    const bool enableAttrInterPred2) const = 0;

  virtual AttributeLods& getLods() = 0;
  virtual std::vector<uint32_t>& getIndexes() = 0;
  virtual std::vector<uint32_t>& pointIndexToPredictorIndex() = 0;
  virtual bool& canonical_lod_sampling_enabled_flag() = 0;
};

//----------------------------------------------------------------------------

std::unique_ptr<AttributeEncoderIntf> makeAttributeEncoder();

//============================================================================

int estimateDist2(
  const PCCPointSet3& cloud,
  int samplingPeriod,
  int searchRange,
  float percentileEstimate,
  const Vec3<int32_t> &lodNeighBias);

//============================================================================

}  // namespace pcc
