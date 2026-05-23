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

#include "BitReader.h"
#include "BitWriter.h"
#include "PCCMisc.h"
#include "hls.h"
#include "io_hls.h"

#include <iomanip>
#include <iterator>
#include <algorithm>
#include <sstream>

namespace pcc {

//============================================================================

Oid::operator std::string() const
{
  std::stringstream ss;
  int subidentifier = 0;
  int firstSubidentifier = 1;
  for (auto byte : this->contents) {
    if (byte & 0x80) {
      subidentifier = (subidentifier << 7) | (byte & 0x7f);
      continue;
    }

    // end of subidentifer.
    // NB: the first subidentifier encodes two oid components
    if (firstSubidentifier) {
      firstSubidentifier = 0;
      if (subidentifier < 40)
        ss << '0';
      else if (subidentifier < 80) {
        ss << '1';
        subidentifier -= 40;
      } else {
        ss << '2';
        subidentifier -= 80;
      }
    }

    ss << '.' << std::to_string(subidentifier);
    subidentifier = 0;
  }

  return ss.str();
}

//----------------------------------------------------------------------------

bool
operator==(const Oid& lhs, const Oid& rhs)
{
  // NB: there is a unique encoding for each OID.  Equality may be determined
  // comparing just the content octets
  return lhs.contents == rhs.contents;
}

//----------------------------------------------------------------------------

template<typename T>
void
writeOid(T& bs, const Oid& oid)
{
  // write out the length according to the BER definite shoft form.
  // NB: G-PCC limits the length to 127 octets.
  constexpr int oid_reserved_zero_bit = 0;
  int oid_len = oid.contents.size();
  bs.writeUn(1, oid_reserved_zero_bit);
  bs.writeUn(7, oid_len);

  const auto& oid_contents = oid.contents;
  for (int i = 0; i < oid_len; i++)
    bs.writeUn(8, oid_contents[i]);
}

//----------------------------------------------------------------------------

template<typename T>
void
readOid(T& bs, Oid* oid)
{
  int oid_reserved_zero_bit;
  int oid_len;
  bs.readUn(1, &oid_reserved_zero_bit);
  bs.readUn(7, &oid_len);
  oid->contents.resize(oid_len);
  for (int i = 0; i < oid_len; i++)
    bs.readUn(8, &oid->contents[i]);
}

//----------------------------------------------------------------------------

int
lengthOid(const Oid& oid)
{
  return 1 + oid.contents.size();
}

//============================================================================

std::ostream&
operator<<(std::ostream& os, const AttributeLabel& label)
{
  switch (label.known_attribute_label) {
  case KnownAttributeLabel::kColour: os << "color"; break;
  case KnownAttributeLabel::kReflectance: os << "reflectance"; break;
  case KnownAttributeLabel::kFrameIndex: os << "frame index"; break;
  case KnownAttributeLabel::kFrameNumber: os << "frame number"; break;
  case KnownAttributeLabel::kMaterialId: os << "material id"; break;
  case KnownAttributeLabel::kOpacity: os << "opacity"; break;
  case KnownAttributeLabel::kNormal: os << "normal"; break;
  case KnownAttributeLabel::kOid: os << std::string(label.oid);
  default:
    // An unknown known attribute
    auto iosFlags = os.flags(std::ios::hex);
    os << std::setw(8) << label.known_attribute_label;
    os.flags(iosFlags);
  }

  return os;
}

//============================================================================

template<typename Bs>
void
writeAttrParamCicp(Bs& bs, const AttributeParameters& param)
{
  bs.writeUe(param.cicp_colour_primaries_idx);
  bs.writeUe(param.cicp_transfer_characteristics_idx);
  bs.writeUe(param.cicp_matrix_coefficients_idx);
  bs.write(param.cicp_video_full_range_flag);
  bs.byteAlign();
}

//----------------------------------------------------------------------------

template<typename Bs>
void
parseAttrParamCicp(Bs& bs, AttributeParameters* param)
{
  bs.readUe(&param->cicp_colour_primaries_idx);
  bs.readUe(&param->cicp_transfer_characteristics_idx);
  bs.readUe(&param->cicp_matrix_coefficients_idx);
  bs.read(&param->cicp_video_full_range_flag);
  param->cicpParametersPresent = 1;
  bs.byteAlign();
}

//============================================================================

template<typename Bs>
void
writeAttrParamScaling(Bs& bs, const AttributeParameters& param)
{
  int attr_offset_bits = numBits(std::abs(param.attr_offset));
  int attr_scale_bits = numBits(param.attr_scale_minus1);
  bs.writeUe(attr_offset_bits);
  bs.writeSn(attr_offset_bits, param.attr_offset);
  bs.writeUe(attr_scale_bits);
  bs.writeUn(attr_scale_bits, param.attr_scale_minus1);
  bs.writeUe(param.attr_frac_bits);
  bs.byteAlign();
}

//----------------------------------------------------------------------------

template<typename Bs>
void
parseAttrParamScaling(Bs& bs, AttributeParameters* param)
{
  int attr_offset_bits, attr_scale_bits;
  bs.readUe(&attr_offset_bits);
  bs.readSn(attr_offset_bits, &param->attr_offset);
  bs.readUe(&attr_scale_bits);
  bs.readUn(attr_scale_bits, &param->attr_scale_minus1);
  bs.readUe(&param->attr_frac_bits);
  param->scalingParametersPresent = 1;
  bs.byteAlign();
}

//============================================================================

template<typename Bs>
void
writeAttrParamDefaultValue(
  const AttributeDescription& desc, Bs& bs, const AttributeParameters& param)
{
  bs.writeUn(desc.bitdepth, param.attr_default_value[0]);
  for (int k = 1; k <= desc.attr_num_dimensions_minus1; k++)
    bs.writeUn(desc.bitdepth, param.attr_default_value[k]);
  bs.byteAlign();
}

//----------------------------------------------------------------------------

template<typename Bs>
void
parseAttrParamDefaultValue(
  const AttributeDescription& desc, Bs& bs, AttributeParameters* param)
{
  param->attr_default_value.resize(desc.attr_num_dimensions_minus1 + 1);

  bs.readUn(desc.bitdepth, &param->attr_default_value[0]);
  for (int k = 1; k <= desc.attr_num_dimensions_minus1; k++)
    bs.readUn(desc.bitdepth, &param->attr_default_value[k]);
  bs.byteAlign();
}

//============================================================================

template<typename Bs>
void
writeAttrParamOpaque(Bs& bs, const OpaqueAttributeParameter& param)
{
  if (param.attr_param_type == AttributeParameterType::kItuT35) {
    bs.writeUn(8, param.attr_param_itu_t_t35_country_code);
    if (param.attr_param_itu_t_t35_country_code == 0xff)
      bs.writeUn(8, param.attr_param_itu_t_t35_country_code_extension);
  } else if (param.attr_param_type == AttributeParameterType::kOid)
    writeOid(bs, param.attr_param_oid);

  for (auto attr_param_byte : param.attr_param_byte)
    bs.writeUn(8, attr_param_byte);

  bs.byteAlign();
}

//----------------------------------------------------------------------------

template<typename Bs>
OpaqueAttributeParameter
parseAttrParamOpaque(
  Bs& bs, AttributeParameterType attr_param_type, int attrParamLen)
{
  bs.byteAlign();

  OpaqueAttributeParameter param;
  param.attr_param_type = attr_param_type;

  if (param.attr_param_type == AttributeParameterType::kItuT35) {
    bs.readUn(8, &param.attr_param_itu_t_t35_country_code);
    attrParamLen--;
    if (param.attr_param_itu_t_t35_country_code == 0xff) {
      bs.readUn(8, &param.attr_param_itu_t_t35_country_code_extension);
      attrParamLen--;
    }
  } else if (param.attr_param_type == AttributeParameterType::kOid) {
    readOid(bs, &param.attr_param_oid);
    attrParamLen -= lengthOid(param.attr_param_oid);
  }

  if (attrParamLen > 0) {
    param.attr_param_byte.resize(attrParamLen);
    for (int i = 0; i < attrParamLen; i++)
      bs.readUn(8, &param.attr_param_byte[i]);
  }

  return param;
}

//============================================================================
// NB: this writes all present parameters, whereas parse parses only one
// The encoder works in a fixed order.  However, this is non-normative.

template<typename T>
void
writeAttributeParameters(
  const AttributeDescription& attr, T& bs, const AttributeParameters& params)
{
  int num_attr_parameters = params.numParams();
  bs.writeUe(num_attr_parameters);
  bs.byteAlign();

  if (!params.attr_default_value.empty()) {
    int attr_param_len = 0;
    auto bsCounter = makeBitWriter(InsertionCounter(&attr_param_len));
    writeAttrParamDefaultValue(attr, bsCounter, params);

    auto attr_param_type = AttributeParameterType::kDefaultValue;
    bs.writeUn(8, attr_param_type);
    bs.writeUn(8, attr_param_len);
    writeAttrParamDefaultValue(attr, bs, params);
  }

  if (params.cicpParametersPresent) {
    int attr_param_len = 0;
    auto bsCounter = makeBitWriter(InsertionCounter(&attr_param_len));
    writeAttrParamCicp(bsCounter, params);

    auto attr_param_type = AttributeParameterType::kCicp;
    bs.writeUn(8, attr_param_type);
    bs.writeUn(8, attr_param_len);
    writeAttrParamCicp(bs, params);
  }

  if (params.scalingParametersPresent) {
    int attr_param_len = 0;
    auto bsCounter = makeBitWriter(InsertionCounter(&attr_param_len));
    writeAttrParamScaling(bsCounter, params);

    auto attr_param_type = AttributeParameterType::kScaling;
    bs.writeUn(8, attr_param_type);
    bs.writeUn(8, attr_param_len);
    writeAttrParamScaling(bs, params);
  }

  for (const auto& param : params.opaqueParameters) {
    int attr_param_len = 0;
    auto bsCounter = makeBitWriter(InsertionCounter(&attr_param_len));
    writeAttrParamOpaque(bsCounter, param);

    bs.writeUn(8, param.attr_param_type);
    bs.writeUn(8, attr_param_len);
    writeAttrParamOpaque(bs, param);
  }
}

//----------------------------------------------------------------------------

template<typename T>
void
parseAttributeParameter(
  const AttributeDescription& attr, T& bs, AttributeParameters& params)
{
  AttributeParameterType attr_param_type;
  int attr_param_len;
  bs.readUn(8, &attr_param_type);
  bs.readUn(8, &attr_param_len);
  // todo(df): check that all attr_param_len bytes are consumed
  switch (attr_param_type) {
    using Type = AttributeParameterType;
  case Type::kCicp: parseAttrParamCicp(bs, &params); break;
  case Type::kScaling: parseAttrParamScaling(bs, &params); break;
  case Type::kDefaultValue:
    parseAttrParamDefaultValue(attr, bs, &params);
    break;

  case Type::kItuT35:
  case Type::kOid:
  default:
    params.opaqueParameters.emplace_back(
      parseAttrParamOpaque(bs, attr_param_type, attr_param_len));
  }
}

//============================================================================

PayloadBuffer
write(const SequenceParameterSet& sps)
{
  PayloadBuffer buf(PayloadType::kSequenceParameterSet);
  auto bs = makeBitWriter(std::back_inserter(buf));

  // NB: if taking bits from reserved_profile_compatibility_XXbits, be sure
  // not to change the bitstream position of the other constraint flags.
  bs.writeUn(1, sps.profile.main_profile_compatibility_flag);
  bs.writeUn(21, sps.profile.reserved_profile_compatibility_21bits);
  bs.writeUn(1, sps.profile.slice_reordering_constraint_flag);
  bs.writeUn(1, sps.profile.unique_point_positions_constraint_flag);

  bs.writeUn(8, sps.level);
  bs.writeUn(4, sps.sps_seq_parameter_set_id);

  bs.writeUn(5, sps.frame_ctr_bits);
  bs.writeUn(5, sps.slice_tag_bits);

  bs.writeUe(sps.sps_bounding_box_offset_bits);
  if (auto bbOriginBits = sps.sps_bounding_box_offset_bits) {
    auto sps_bounding_box_offset_xyz =
      toXyz(sps.geometry_axis_order, sps.seqBoundingBoxOrigin);

    bs.writeSn(bbOriginBits, sps_bounding_box_offset_xyz.x());
    bs.writeSn(bbOriginBits, sps_bounding_box_offset_xyz.y());
    bs.writeSn(bbOriginBits, sps_bounding_box_offset_xyz.z());

    int seq_bounding_box_offset_log2_scale = 0;
    bs.writeUe(seq_bounding_box_offset_log2_scale);
  }

  bs.writeUe(sps.sps_bounding_box_size_bits);
  if (auto bbSizeBits = sps.sps_bounding_box_size_bits) {
    auto seq_bounding_box_size_minus1 =
      toXyz(sps.geometry_axis_order, sps.seqBoundingBoxSize - 1);

    bs.writeUn(bbSizeBits, seq_bounding_box_size_minus1.x());
    bs.writeUn(bbSizeBits, seq_bounding_box_size_minus1.y());
    bs.writeUn(bbSizeBits, seq_bounding_box_size_minus1.z());
  }

  int seq_unit_numerator_minus1 = sps.seqGeomScale.numerator - 1;
  int seq_unit_denominator_minus1 = sps.seqGeomScale.denominator - 1;
  bs.writeUe(seq_unit_numerator_minus1);
  bs.writeUe(seq_unit_denominator_minus1);
  bs.writeUn(1, sps.seq_geom_scale_unit_flag);

  bs.writeUe(sps.global_scale_mul_log2());
  bs.writeUe(sps.global_scale_fp_bits());
  bs.writeUn(sps.global_scale_fp_bits(), sps.global_scale_rem());

  int num_attribute_sets = int(sps.attributeSets.size());
  bs.writeUe(num_attribute_sets);

  for (const auto& attr : sps.attributeSets) {
    bs.writeUe(attr.attr_num_dimensions_minus1);
    bs.writeUe(attr.attr_instance_id);

    int attr_bitdepth_minus1 = attr.bitdepth - 1;
    bs.writeUe(attr_bitdepth_minus1);

    const auto& label = attr.attributeLabel;
    bs.write(label.known_attribute_label_flag());
    if (label.known_attribute_label_flag())
      bs.writeUe(label.known_attribute_label);
    else
      writeOid(bs, label.oid);

    writeAttributeParameters(attr, bs, attr.params);
  }

  bs.writeUn(3, sps.geometry_axis_order);
  bs.write(sps.cabac_bypass_stream_enabled_flag);
  bs.write(sps.entropy_continuation_enabled_flag);

  bool sps_extension_flag = true;
  bs.write(sps_extension_flag);

  if (sps_extension_flag) {
    bs.write(sps.inter_frame_prediction_enabled_flag);
    if (sps.inter_frame_prediction_enabled_flag)
      bs.write(sps.inter_entropy_continuation_enabled_flag);
    bs.write(sps.bypass_bin_coding_without_prob_update);

    if (num_attribute_sets > 1)
      bs.write(sps.cross_attr_prediction_enabled_flag);
      
    bs.write(sps.layer_group_enabled_flag);
  }

  bs.byteAlign();

  return buf;
}

//----------------------------------------------------------------------------

SequenceParameterSet
parseSps(const PayloadBuffer& buf)
{
  SequenceParameterSet sps;
  assert(buf.type == PayloadType::kSequenceParameterSet);
  auto bs = makeBitReader(buf.begin(), buf.end());

  bs.readUn(1, &sps.profile.main_profile_compatibility_flag);
  bs.readUn(21, &sps.profile.reserved_profile_compatibility_21bits);
  bs.readUn(1, &sps.profile.slice_reordering_constraint_flag);
  bs.readUn(1, &sps.profile.unique_point_positions_constraint_flag);

  bs.readUn(8, &sps.level);
  bs.readUn(4, &sps.sps_seq_parameter_set_id);

  bs.readUn(5, &sps.frame_ctr_bits);
  bs.readUn(5, &sps.slice_tag_bits);

  sps.seqBoundingBoxOrigin = 0;
  bs.readUe(&sps.sps_bounding_box_offset_bits);
  if (auto bbOriginBits = sps.sps_bounding_box_offset_bits) {
    Vec3<int> seq_bounding_box_offset;
    bs.readSn(bbOriginBits, &seq_bounding_box_offset.x());
    bs.readSn(bbOriginBits, &seq_bounding_box_offset.y());
    bs.readSn(bbOriginBits, &seq_bounding_box_offset.z());

    int seq_bounding_box_offset_log2_scale;
    bs.readUe(&seq_bounding_box_offset_log2_scale);
    seq_bounding_box_offset *= 1 << seq_bounding_box_offset_log2_scale;

    // NB: this is in XYZ axis order until the SPS is converted to STV
    sps.seqBoundingBoxOrigin = seq_bounding_box_offset;
  }

  sps.seqBoundingBoxSize = 0;
  bs.readUe(&sps.sps_bounding_box_size_bits);
  if (auto bbSizeBits = sps.sps_bounding_box_size_bits) {
    Vec3<int> seq_bounding_box_whd_minus1;
    bs.readUn(bbSizeBits, &seq_bounding_box_whd_minus1.x());
    bs.readUn(bbSizeBits, &seq_bounding_box_whd_minus1.y());
    bs.readUn(bbSizeBits, &seq_bounding_box_whd_minus1.z());

    // NB: this is in XYZ axis order until the SPS is converted to STV
    sps.seqBoundingBoxSize = seq_bounding_box_whd_minus1 + 1;
  }

  int seq_unit_numerator_minus1;
  int seq_unit_denominator_minus1;
  bs.readUe(&seq_unit_numerator_minus1);
  bs.readUe(&seq_unit_denominator_minus1);
  bs.readUn(1, &sps.seq_geom_scale_unit_flag);

  sps.seqGeomScale.numerator = seq_unit_numerator_minus1 + 1;
  sps.seqGeomScale.denominator = seq_unit_denominator_minus1 + 1;

  bs.readUe(&sps.global_scale_mul_log2());
  bs.readUe(&sps.global_scale_fp_bits());
  bs.readUn(sps.global_scale_fp_bits(), &sps.global_scale_rem());

  int num_attribute_sets = int(bs.readUe());

  for (int i = 0; i < num_attribute_sets; i++) {
    sps.attributeSets.emplace_back();
    auto& attr = sps.attributeSets.back();
    bs.readUe(&attr.attr_num_dimensions_minus1);
    bs.readUe(&attr.attr_instance_id);

    int attr_bitdepth_minus1;
    bs.readUe(&attr_bitdepth_minus1);
    attr.bitdepth = attr_bitdepth_minus1 + 1;

    auto& label = attr.attributeLabel;
    bool known_attribute_label_flag = bs.read();
    if (known_attribute_label_flag)
      bs.readUe(&label.known_attribute_label);
    else {
      label.known_attribute_label = KnownAttributeLabel::kOid;
      readOid(bs, &label.oid);
    }

    int num_attribute_parameters;
    bs.readUe(&num_attribute_parameters);
    bs.byteAlign();
    for (int i = 0; i < num_attribute_parameters; i++)
      parseAttributeParameter(attr, bs, attr.params);
  }

  bs.readUn(3, &sps.geometry_axis_order);
  bs.read(&sps.cabac_bypass_stream_enabled_flag);
  bs.read(&sps.entropy_continuation_enabled_flag);

  sps.inter_frame_prediction_enabled_flag = false;
  sps.bypass_bin_coding_without_prob_update = false;
  sps.inter_entropy_continuation_enabled_flag = false;
  sps.cross_attr_prediction_enabled_flag = false;
  sps.layer_group_enabled_flag = false;
  
  bool sps_extension_flag = bs.read();
  if (sps_extension_flag) {
    bs.read(&sps.inter_frame_prediction_enabled_flag);
    if (sps.inter_frame_prediction_enabled_flag)
      bs.read(&sps.inter_entropy_continuation_enabled_flag);
    bs.read(&sps.bypass_bin_coding_without_prob_update);

    if (num_attribute_sets > 1)
      bs.read(&sps.cross_attr_prediction_enabled_flag);
	
    bs.read(&sps.layer_group_enabled_flag);
  }

    // conformance check: reordering constraint must be set with continuation
  if (
    sps.entropy_continuation_enabled_flag
    || sps.inter_entropy_continuation_enabled_flag)
    assert(sps.profile.slice_reordering_constraint_flag);

  bs.byteAlign();

  return sps;
}

//----------------------------------------------------------------------------

void
convertXyzToStv(SequenceParameterSet* sps)
{
  // permute the bounding box from xyz to internal stv order
  sps->seqBoundingBoxOrigin =
    fromXyz(sps->geometry_axis_order, sps->seqBoundingBoxOrigin);

  sps->seqBoundingBoxSize =
    fromXyz(sps->geometry_axis_order, sps->seqBoundingBoxSize);
}

//============================================================================

PayloadBuffer
write(const SequenceParameterSet& sps, const GeometryParameterSet& gps)
{
  PayloadBuffer buf(PayloadType::kGeometryParameterSet);
  auto bs = makeBitWriter(std::back_inserter(buf));

  bs.writeUn(4, gps.gps_geom_parameter_set_id);
  bs.writeUn(4, gps.gps_seq_parameter_set_id);
  bs.write(gps.geom_box_log2_scale_present_flag);
  if (!gps.geom_box_log2_scale_present_flag)
    bs.writeUe(gps.gps_geom_box_log2_scale);

  bs.write(gps.geom_unique_points_flag);

  bs.write(gps.predgeom_enabled_flag);
  if (!gps.predgeom_enabled_flag) {
    bs.write(gps.octree_point_count_list_present_flag);

    bs.writeUn(2, gps.inferred_direct_coding_mode);
    if (gps.inferred_direct_coding_mode)
      bs.write(gps.joint_2pt_idcm_enabled_flag);

    bs.write(gps.qtbt_enabled_flag);
    bs.writeUn(3, gps.neighbour_avail_boundary_log2_minus1);
    if (gps.neighbour_avail_boundary_log2_minus1 > 0) {
      bs.write(gps.adjacent_child_contextualization_enabled_flag);
      bs.writeUe(gps.intra_pred_max_node_size_log2);
    }
    bs.write(gps.bitwise_occupancy_coding_flag);

    bs.write(gps.geom_planar_mode_enabled_flag);
    if (gps.geom_planar_mode_enabled_flag) {
      bs.writeUe(gps.geom_planar_threshold0);
      bs.writeUe(gps.geom_planar_threshold1);
      bs.writeUe(gps.geom_planar_threshold2);
      if (gps.inferred_direct_coding_mode == 1)
        bs.writeUn(5, gps.geom_idcm_rate_minus1);
    }
  }

  bs.write(gps.geom_angular_mode_enabled_flag);
  if (gps.geom_angular_mode_enabled_flag) {
    bs.write(gps.geom_slice_angular_origin_present_flag);
    if (!gps.geom_slice_angular_origin_present_flag) {
      auto gps_angular_origin =
        toXyz(sps.geometry_axis_order, gps.gpsAngularOrigin);

      int gps_angular_origin_bits_minus1 =
        numBits(gps_angular_origin.abs().max()) - 1;

      bs.writeUe(gps_angular_origin_bits_minus1);
      bs.writeSn(gps_angular_origin_bits_minus1 + 1, gps_angular_origin.x());
      bs.writeSn(gps_angular_origin_bits_minus1 + 1, gps_angular_origin.y());
      bs.writeSn(gps_angular_origin_bits_minus1 + 1, gps_angular_origin.z());
    }

    if (gps.predgeom_enabled_flag) {
      bs.writeUe(gps.geom_angular_azimuth_scale_log2_minus11);
      bs.writeUe(gps.geom_angular_azimuth_speed_minus1);
      bs.writeUe(gps.geom_angular_radius_inv_scale_log2);
    }

    int geom_angular_num_lidar_lasers_minus1 = gps.numLasers() - 1;
    bs.writeUe(geom_angular_num_lidar_lasers_minus1);
    int geom_angular_theta0 = gps.angularTheta[0];
    int geom_angular_z0 = gps.angularZ[0];
    bs.writeSe(geom_angular_theta0);
    bs.writeSe(geom_angular_z0);
    if (!gps.predgeom_enabled_flag) {
      int geom_angular_num_phi_per_turn0_minus1 =
        gps.angularNumPhiPerTurn[0] - 1;
      bs.writeUe(geom_angular_num_phi_per_turn0_minus1);
      
    }
  
    for (int i = 1; i <= geom_angular_num_lidar_lasers_minus1; i++) {
      int geom_angular_theta_laser_diff =
        gps.angularTheta[i] - gps.geomAngularThetaPred(i);

      int geom_angular_z_laser_diff = gps.angularZ[i] - gps.angularZ[i - 1];

      
      bs.writeSe(geom_angular_theta_laser_diff);
      bs.writeSe(geom_angular_z_laser_diff);

        int geom_angular_num_phi_laser_diff =
          gps.angularNumPhiPerTurn[i] - gps.angularNumPhiPerTurn[i - 1];

        if (!gps.predgeom_enabled_flag) {
          bs.writeSe(geom_angular_num_phi_laser_diff);
      }
    }

    if (gps.geom_planar_mode_enabled_flag)
      bs.write(gps.planar_buffer_disabled_flag);
  }

  bs.write(gps.geom_scaling_enabled_flag);
  if (gps.geom_scaling_enabled_flag) {
    bs.writeUe(gps.geom_base_qp);
    bs.writeUn(2, gps.geom_qp_multiplier_log2);
    if (gps.predgeom_enabled_flag)
      bs.writeUe(gps.geom_qp_offset_intvl_log2);
    else if (gps.inferred_direct_coding_mode)
      bs.writeSe(gps.geom_idcm_qp_offset);
  }

  // NB: bitstreams conforming to the first edition must set
  // gps_extension_flag equal to 0.
  bool gps_extension_flag = sps.profile.isDraftProfile();
  bs.write(gps_extension_flag);
  if (gps_extension_flag) {
    if(!gps.predgeom_enabled_flag)
      bs.write(gps.trisoup_enabled_flag);

    if (gps.trisoup_enabled_flag){
      bs.write(gps.non_cubic_node_start_edge);
      bs.write(gps.non_cubic_node_end_edge);
    }

    if (gps.geom_planar_mode_enabled_flag && gps.geom_angular_mode_enabled_flag && gps.inferred_direct_coding_mode)
      bs.write(gps.geom_planar_disabled_idcm_angular_flag);

    if (!gps.predgeom_enabled_flag || gps.geom_angular_mode_enabled_flag)
      bs.write(gps.interPredictionEnabledFlag);
    if (gps.interPredictionEnabledFlag) {
      bs.write(gps.globalMotionEnabled);
      if (gps.predgeom_enabled_flag) {
        bs.writeUe(gps.interAzimScaleLog2);
        bs.write(gps.resamplingEnabled);
        bs.writeUe(gps.maxPointsPerEntryMinus1);
        if (gps.maxPointsPerEntryMinus1 > 0)
          bs.writeUe(gps.dn_sampling_range + 1);
      }
      bs.writeUe(gps.biPredictionEnabledFlag);
      if (gps.biPredictionEnabledFlag)
        bs.write(gps.frameMergeEnabledFlag);
    }

    if (gps.predgeom_enabled_flag && gps.geom_angular_mode_enabled_flag) {
      bs.write(gps.residual2_disabled_flag);
      bs.write(gps.azimuth_scaling_enabled_flag);
      if (gps.azimuth_scaling_enabled_flag)
        bs.writeUe(gps.predgeom_max_pred_index);
        bs.writeUe(gps.predgeom_radius_threshold_for_pred_list);
        bs.write(gps.resR_context_qphi_threshold_present_flag);
        if (gps.resR_context_qphi_threshold_present_flag)
          bs.writeUe(gps.resR_context_qphi_threshold);
    }
    if (!gps.predgeom_enabled_flag && gps.geom_angular_mode_enabled_flag)
      bs.write(gps.octree_angular_extension_flag);

    if (gps.geom_planar_mode_enabled_flag)
      bs.write(gps.geom_octree_depth_planar_eligibiity_enabled_flag);

    if (gps.geom_planar_mode_enabled_flag && !gps.geom_angular_mode_enabled_flag)
      bs.write(gps.geom_octree_planar_dynamic_obuf_eligibiity_enabled_flag);

    if (!gps.predgeom_enabled_flag && gps.geom_planar_mode_enabled_flag)
      bs.write(gps.geom_multiple_planar_mode_enable_flag);

    if (gps.geom_angular_mode_enabled_flag) {
      if (!gps.predgeom_enabled_flag) {
        bs.write(gps.geom_z_compensation_enabled_flag);
        bs.write(gps.geom_xy_compensation_enabled_flag);
      }
      if (gps.geom_xy_compensation_enabled_flag)
      {
        bs.writeUe(gps.geom_xy_compensation_previous_points_num_minus1);
        bs.writeSe(gps.lasersHorizontalOffset[0]);

        for (int i = 1; i <= gps.numLasers() - 1; i++) {
          int lasers_horizontal_offset_abs_diff =
            abs(gps.lasersHorizontalOffset[i])
            - abs(gps.lasersHorizontalOffset[i - 1]);
          bs.writeSe(lasers_horizontal_offset_abs_diff);
          if (abs(gps.lasersHorizontalOffset[i]))
            bs.write(gps.lasersHorizontalOffset[i] > 0);
        }
      }
       
      bs.write(gps.geom_inter_idcm_enabled_flag);
      if (gps.geom_inter_idcm_enabled_flag)
        bs.write(gps.one_point_alone_laser_beam_flag);
    }
  }
  bs.byteAlign();

  return buf;
}

//----------------------------------------------------------------------------

GeometryParameterSet
parseGps(const PayloadBuffer& buf)
{
  GeometryParameterSet gps;
  assert(buf.type == PayloadType::kGeometryParameterSet);
  auto bs = makeBitReader(buf.begin(), buf.end());

  bs.readUn(4, &gps.gps_geom_parameter_set_id);
  bs.readUn(4, &gps.gps_seq_parameter_set_id);
  bs.read(&gps.geom_box_log2_scale_present_flag);
  if (!gps.geom_box_log2_scale_present_flag)
    bs.readUe(&gps.gps_geom_box_log2_scale);

  bs.read(&gps.geom_unique_points_flag);

  gps.geom_planar_mode_enabled_flag = false;
  gps.octree_point_count_list_present_flag = false;
  gps.geom_multiple_planar_mode_enable_flag = false;
  bs.read(&gps.predgeom_enabled_flag);
  if (!gps.predgeom_enabled_flag) {
    bs.read(&gps.octree_point_count_list_present_flag);

    bs.readUn(2, &gps.inferred_direct_coding_mode);
    if (gps.inferred_direct_coding_mode)
      bs.read(&gps.joint_2pt_idcm_enabled_flag);

    bs.read(&gps.qtbt_enabled_flag);
    bs.readUn(3, &gps.neighbour_avail_boundary_log2_minus1);

    gps.adjacent_child_contextualization_enabled_flag = 0;
    gps.intra_pred_max_node_size_log2 = 0;
    if (gps.neighbour_avail_boundary_log2_minus1 > 0) {
      bs.read(&gps.adjacent_child_contextualization_enabled_flag);
      bs.readUe(&gps.intra_pred_max_node_size_log2);
    }
    bs.read(&gps.bitwise_occupancy_coding_flag);

    bs.read(&gps.geom_planar_mode_enabled_flag);
    if (gps.geom_planar_mode_enabled_flag) {
      bs.readUe(&gps.geom_planar_threshold0);
      bs.readUe(&gps.geom_planar_threshold1);
      bs.readUe(&gps.geom_planar_threshold2);
      if (gps.inferred_direct_coding_mode == 1)
        bs.readUn(5, &gps.geom_idcm_rate_minus1);
    }
  }

  gps.planar_buffer_disabled_flag = false;
  gps.geom_slice_angular_origin_present_flag = false;
  gps.geom_z_compensation_enabled_flag = false;
  gps.geom_xy_compensation_enabled_flag = false;
  gps.geom_inter_idcm_enabled_flag = false;
  gps.one_point_alone_laser_beam_flag = false;
  bs.read(&gps.geom_angular_mode_enabled_flag);
  gps.geom_xy_compensation_previous_points_num_minus1 = 0;
  if (gps.geom_angular_mode_enabled_flag) {
    bs.read(&gps.geom_slice_angular_origin_present_flag);
    if (!gps.geom_slice_angular_origin_present_flag) {
      int gps_angular_origin_bits_minus1;
      bs.readUe(&gps_angular_origin_bits_minus1);

      // NB: this is in XYZ axis order until the GPS is converted to STV
      Vec3<int>& gps_angular_origin = gps.gpsAngularOrigin;
      bs.readSn(gps_angular_origin_bits_minus1 + 1, &gps_angular_origin.x());
      bs.readSn(gps_angular_origin_bits_minus1 + 1, &gps_angular_origin.y());
      bs.readSn(gps_angular_origin_bits_minus1 + 1, &gps_angular_origin.z());
    }

    if (gps.predgeom_enabled_flag) {
      bs.readUe(&gps.geom_angular_azimuth_scale_log2_minus11);
      bs.readUe(&gps.geom_angular_azimuth_speed_minus1);
      bs.readUe(&gps.geom_angular_radius_inv_scale_log2);
    }

    int geom_angular_num_lidar_lasers_minus1;
    bs.readUe(&geom_angular_num_lidar_lasers_minus1);
    gps.angularTheta.resize(geom_angular_num_lidar_lasers_minus1 + 1);
    gps.angularZ.resize(geom_angular_num_lidar_lasers_minus1 + 1);
    gps.angularNumPhiPerTurn.resize(geom_angular_num_lidar_lasers_minus1 + 1);
    gps.lasersHorizontalOffset.resize(
      geom_angular_num_lidar_lasers_minus1 + 1);

    int& geom_angular_theta0 = gps.angularTheta[0];
    int& geom_angular_z0 = gps.angularZ[0];
    bs.readSe(&geom_angular_theta0);
    bs.readSe(&geom_angular_z0);
    if (!gps.predgeom_enabled_flag) {
      int geom_angular_num_phi_per_turn0_minus1;
      bs.readUe(&geom_angular_num_phi_per_turn0_minus1);
      gps.angularNumPhiPerTurn[0] = geom_angular_num_phi_per_turn0_minus1 + 1;

     
    }

    for (int i = 1; i <= geom_angular_num_lidar_lasers_minus1; i++) {
      int geom_angular_theta_laser_diff;
      int geom_angular_z_laser_diff;
      bs.readSe(&geom_angular_theta_laser_diff);
      bs.readSe(&geom_angular_z_laser_diff);

      gps.angularZ[i] = gps.angularZ[i - 1] + geom_angular_z_laser_diff;
      gps.angularTheta[i] =
        gps.geomAngularThetaPred(i) + geom_angular_theta_laser_diff;

      if (!gps.predgeom_enabled_flag) {
        int geom_angular_num_phi_laser_diff;
        bs.readSe(&geom_angular_num_phi_laser_diff);

        gps.angularNumPhiPerTurn[i] =
          gps.angularNumPhiPerTurn[i - 1] + geom_angular_num_phi_laser_diff;

      }
    }

    if (gps.geom_planar_mode_enabled_flag)
      bs.read(&gps.planar_buffer_disabled_flag);
  }

  gps.geom_base_qp = 0;
  gps.geom_qp_multiplier_log2 = 0;
  gps.geom_idcm_qp_offset = 0;
  bs.read(&gps.geom_scaling_enabled_flag);
  if (gps.geom_scaling_enabled_flag) {
    bs.readUe(&gps.geom_base_qp);
    bs.readUn(2, &gps.geom_qp_multiplier_log2);
    if (gps.predgeom_enabled_flag)
      bs.readUe(&gps.geom_qp_offset_intvl_log2);
    else if (gps.inferred_direct_coding_mode)
      bs.readSe(&gps.geom_idcm_qp_offset);
  }

  gps.trisoup_enabled_flag = false;
  gps.azimuth_scaling_enabled_flag = false;
  gps.octree_angular_extension_flag = false;
  gps.interPredictionEnabledFlag = false;
  gps.globalMotionEnabled = false;
  gps.biPredictionEnabledFlag = false;
  gps.frameMergeEnabledFlag = false;
  gps.geom_planar_disabled_idcm_angular_flag = false;
  gps.residual2_disabled_flag = false;
  gps.geom_octree_depth_planar_eligibiity_enabled_flag = false;
  gps.geom_octree_planar_dynamic_obuf_eligibiity_enabled_flag = false;
  gps.dn_sampling_range = 0;
  bool gps_extension_flag = bs.read();
  if (gps_extension_flag) {
    if(!gps.predgeom_enabled_flag)
      bs.read(&gps.trisoup_enabled_flag);

    if (gps.trisoup_enabled_flag) {
      bs.read(&gps.non_cubic_node_start_edge);
      bs.read(&gps.non_cubic_node_end_edge);
    }

    if (
      gps.geom_planar_mode_enabled_flag && gps.geom_angular_mode_enabled_flag
      && gps.inferred_direct_coding_mode)
      bs.read(&gps.geom_planar_disabled_idcm_angular_flag);
    
    if (!gps.predgeom_enabled_flag || gps.geom_angular_mode_enabled_flag)
      bs.read(&gps.interPredictionEnabledFlag);
    if (gps.interPredictionEnabledFlag) {
      bs.read(&gps.globalMotionEnabled);
      if (gps.predgeom_enabled_flag) {
        bs.readUe(&gps.interAzimScaleLog2);
        bs.read(&gps.resamplingEnabled);
        bs.readUe(&gps.maxPointsPerEntryMinus1);
        if (gps.maxPointsPerEntryMinus1 > 0)
          bs.readUe(&gps.dn_sampling_range);
        gps.dn_sampling_range--;
      }
      bs.readUe(&gps.biPredictionEnabledFlag);
      if (gps.biPredictionEnabledFlag)
        bs.read(&gps.frameMergeEnabledFlag);
    }

    if (gps.predgeom_enabled_flag && gps.geom_angular_mode_enabled_flag) {
      bs.read(&gps.residual2_disabled_flag);
      bs.read(&gps.azimuth_scaling_enabled_flag);
      if (gps.azimuth_scaling_enabled_flag)
        bs.readUe(&gps.predgeom_max_pred_index);
        bs.readUe(&gps.predgeom_radius_threshold_for_pred_list);
        bs.read(&gps.resR_context_qphi_threshold_present_flag);
        if (gps.resR_context_qphi_threshold_present_flag){
          bs.readUe(&gps.resR_context_qphi_threshold);
        }else{
          gps.resR_context_qphi_threshold = 0;
        }
    }
    if (!gps.predgeom_enabled_flag && gps.geom_angular_mode_enabled_flag)
      bs.read(&gps.octree_angular_extension_flag);

    if (gps.geom_planar_mode_enabled_flag)
      bs.read(&gps.geom_octree_depth_planar_eligibiity_enabled_flag);

    if (gps.geom_planar_mode_enabled_flag && !gps.geom_angular_mode_enabled_flag)
      bs.read(&gps.geom_octree_planar_dynamic_obuf_eligibiity_enabled_flag);

    if (!gps.predgeom_enabled_flag && gps.geom_planar_mode_enabled_flag)
      bs.read(&gps.geom_multiple_planar_mode_enable_flag);

    if (gps.geom_angular_mode_enabled_flag) {
      if (!gps.predgeom_enabled_flag) {
        bs.read(&gps.geom_z_compensation_enabled_flag);
        bs.read(&gps.geom_xy_compensation_enabled_flag);
      }
      if (gps.geom_xy_compensation_enabled_flag) {
        bs.readUe(&gps.geom_xy_compensation_previous_points_num_minus1);
        bs.readSe(&gps.lasersHorizontalOffset[0]);

        for (int i = 1; i <= gps.lasersHorizontalOffset.size()-1; i++) {
          int lasers_horizontal_offset_diff;
          bs.readSe(&lasers_horizontal_offset_diff);
          gps.lasersHorizontalOffset[i] = lasers_horizontal_offset_diff
            + abs(gps.lasersHorizontalOffset[i - 1]);
          if (gps.lasersHorizontalOffset[i]) {
            int lasersHorizontalOffsetSign;
            bs.read(&lasersHorizontalOffsetSign);
            if (!lasersHorizontalOffsetSign)
              gps.lasersHorizontalOffset[i] = -gps.lasersHorizontalOffset[i];
          }
        }
      
      }
        
      bs.read(&gps.geom_inter_idcm_enabled_flag);
      if (gps.geom_inter_idcm_enabled_flag)
        bs.read(&gps.one_point_alone_laser_beam_flag);
    }
  }
  bs.byteAlign();

  return gps;
}

//----------------------------------------------------------------------------

void
convertXyzToStv(const SequenceParameterSet& sps, GeometryParameterSet* gps)
{
  gps->gpsAngularOrigin =
    fromXyz(sps.geometry_axis_order, gps->gpsAngularOrigin);
}

//============================================================================

PayloadBuffer
write(const SequenceParameterSet& sps, const AttributeParameterSet& aps)
{
  PayloadBuffer buf(PayloadType::kAttributeParameterSet);
  auto bs = makeBitWriter(std::back_inserter(buf));

  bs.writeUn(4, aps.aps_attr_parameter_set_id);
  bs.writeUn(4, aps.aps_seq_parameter_set_id);
  bs.writeUe(aps.attr_encoding);

  bs.writeUe(aps.init_qp_minus4);
  bs.writeSe(aps.aps_chroma_qp_offset);
  bs.write(aps.aps_slice_qp_deltas_present_flag);

  auto lod_neigh_bias_minus1 =
    toXyz(sps.geometry_axis_order, aps.lodNeighBias) - 1;
  bs.writeUe(lod_neigh_bias_minus1.x());
  bs.writeUe(lod_neigh_bias_minus1.y());
  bs.writeUe(lod_neigh_bias_minus1.z());

  if (aps.lodParametersPresent()) {
    bs.writeUe(aps.num_pred_nearest_neighbours_minus1);
    bs.writeUe(aps.inter_lod_search_range);

    if (aps.attr_encoding == AttributeEncoding::kLiftingTransform)
      bs.write(aps.last_component_prediction_enabled_flag);

    bs.write(aps.scalable_lifting_enabled_flag);
    if (aps.scalable_lifting_enabled_flag)
      bs.writeUe(aps.max_neigh_range_minus1);

    if (!aps.scalable_lifting_enabled_flag) {
      bs.writeUe(aps.num_detail_levels_minus1);
      if (!aps.num_detail_levels_minus1)
        bs.write(aps.canonical_point_order_flag);
      else {
        bs.writeUe(aps.lod_decimation_type);

        if (aps.lod_decimation_type != LodDecimationMethod::kNone) {
          for (int idx = 0; idx < aps.num_detail_levels_minus1; idx++) {
            auto lod_sampling_period_minus2 = aps.lodSamplingPeriod[idx] - 2;
            bs.writeUe(lod_sampling_period_minus2);
          }
        }

        bs.writeUe(aps.dist2);
        bs.write(aps.aps_slice_dist2_deltas_present_flag);
      }
    }
  }

  if (aps.attr_encoding == AttributeEncoding::kPredictingTransform) {
    bs.writeUe(aps.max_num_direct_predictors);
    if (aps.max_num_direct_predictors) {
      bs.writeUn(8, aps.adaptive_prediction_threshold);
      bs.write(aps.direct_avg_predictor_disabled_flag);
    }
    bs.writeUe(aps.intra_lod_search_range);
    if (aps.intra_lod_search_range) 
      bs.writeUe(aps.intra_lod_prediction_skip_layers);
    bs.write(aps.inter_component_prediction_enabled_flag);
    bs.write(aps.pred_weight_blending_enabled_flag);
  }

  if (aps.attr_encoding == AttributeEncoding::kRAHTransform) {
    bs.write(aps.rahtPredParams.raht_prediction_enabled_flag);
    if (aps.rahtPredParams.raht_prediction_enabled_flag) {
      bs.writeUe(aps.rahtPredParams.raht_prediction_threshold0);
      bs.writeUe(aps.rahtPredParams.raht_prediction_threshold1);
    }
  }

  if (aps.attr_encoding == AttributeEncoding::kRaw)
    bs.write(aps.raw_attr_variable_len_flag);

  if (!aps.scalable_lifting_enabled_flag)
    bs.write(aps.spherical_coord_flag);
  if (aps.spherical_coord_flag) {
    assert(!aps.scalable_lifting_enabled_flag);
    for (int k = 0; k < 3; k++) {
      int attr_coord_scale_bits_minus1 = numBits(aps.attr_coord_scale[k]) - 1;
      bs.writeUn(5, attr_coord_scale_bits_minus1);
      bs.writeUn(attr_coord_scale_bits_minus1 + 1, aps.attr_coord_scale[k]);
    }
  }
 

  
  // NB: bitstreams conforming to the first edition must set
  // aps_extension_flag equal to 0.
  bool aps_extension_flag = sps.profile.isDraftProfile();
  bs.write(aps_extension_flag);
  if (aps_extension_flag) {
    if (aps.attr_encoding == AttributeEncoding::kRAHTransform) {
      bs.write(aps.rahtPredParams.integer_haar_enable_flag);
    }
    if (aps.attr_encoding == AttributeEncoding::kPredictingTransform) {
      for (int i = 0; i <= aps.num_pred_nearest_neighbours_minus1; i++)
        bs.writeUe(aps.quant_neigh_weight[i]);
    }

    bs.write(aps.attrInterPredictionEnabled);
    if (aps.attrInterPredictionEnabled){
      if (aps.attr_encoding == AttributeEncoding::kRAHTransform) {
        bs.writeUe(aps.raht_inter_prediction_depth_minus1);
        if(!aps.rahtPredParams.integer_haar_enable_flag){
          bs.write(aps.raht_send_inter_filters);
          bs.writeUe(aps.raht_inter_skip_layers);          
        }
        bs.write(aps.raht_enable_code_layer);
        bs.write(aps.rahtPredParams.raht_hybrid_prediction_enabled_flag);
        if(aps.rahtPredParams.raht_hybrid_prediction_enabled_flag){
          bs.writeUe(aps.rahtPredParams.raht_hybrid_prediction_num_enabled_layers);
          if(aps.rahtPredParams.raht_hybrid_prediction_num_enabled_layers)
            bs.writeUe(aps.rahtPredParams.raht_hybrid_prediction_lower_depth_minus1);
        }
      } else
        bs.writeUe(aps.attrInterPredSearchRange);       
    }


    if (
      aps.lodParametersPresent() && !aps.scalable_lifting_enabled_flag
      && !aps.num_detail_levels_minus1) {
      bs.writeUe(aps.max_points_per_sort_log2_plus1);
    }

    if (
      aps.lodParametersPresent()
      && aps.num_pred_nearest_neighbours_minus1 >= 2)
      bs.write(aps.predictionWithDistributionEnabled);

    if(
      aps.lodParametersPresent() 
      && aps.lod_decimation_type == LodDecimationMethod::kPeriodic
      && !aps.attrInterPredictionEnabled)
      bs.write(aps.enableMortonCodeScaling);


    if (aps.attr_encoding == AttributeEncoding::kRAHTransform) {
      bs.write(aps.raht_extension);
    }

    if (
      aps.attr_encoding == AttributeEncoding::kRAHTransform
      && aps.rahtPredParams.raht_prediction_enabled_flag) {
      bs.write(aps.rahtPredParams.raht_subnode_prediction_enabled_flag);
      if (aps.rahtPredParams.raht_subnode_prediction_enabled_flag) 
        for (int i = 0; i < 5; i++) 
          bs.writeUe(aps.rahtPredParams.raht_prediction_weights[i]);
      bs.writeUe(aps.rahtPredParams.raht_prediction_search_range);
      bs.write(aps.rahtPredParams.raht_enable_intraPred_nonPred_code_layer);
    }

    if (aps.attr_encoding == AttributeEncoding::kRAHTransform)
      bs.write(aps.last_component_prediction_enabled_flag);

    
      bs.write(aps.cross_attr_prediction_enabled_this_type);
      if (aps.cross_attr_prediction_enabled_this_type)
        bs.writeUe(aps.refAttrIdx);
    
    
	  if (sps.layer_group_enabled_flag)
		  bs.write(aps.attr_ref_id_present_flag);

  }

  bs.byteAlign();

  return buf;
}

//----------------------------------------------------------------------------

AttributeParameterSet
parseAps(const PayloadBuffer& buf, const SequenceParameterSet& sps)
{
  AttributeParameterSet aps;
  assert(buf.type == PayloadType::kAttributeParameterSet);
  auto bs = makeBitReader(buf.begin(), buf.end());

  bs.readUn(4, &aps.aps_attr_parameter_set_id);
  bs.readUn(4, &aps.aps_seq_parameter_set_id);
  bs.readUe(&aps.attr_encoding);

  bs.readUe(&aps.init_qp_minus4);
  bs.readSe(&aps.aps_chroma_qp_offset);
  bs.read(&aps.aps_slice_qp_deltas_present_flag);

  aps.scalable_lifting_enabled_flag = false;
  aps.aps_slice_dist2_deltas_present_flag = false;
  aps.dist2 = 0;
  aps.last_component_prediction_enabled_flag = false;

  Vec3<int> lod_neigh_bias_minus1;
  bs.readUe(&lod_neigh_bias_minus1.x());
  bs.readUe(&lod_neigh_bias_minus1.y());
  bs.readUe(&lod_neigh_bias_minus1.z());
  // NB: this is in XYZ axis order until the GPS is converted to STV
  aps.lodNeighBias = lod_neigh_bias_minus1 + 1;

  if (aps.lodParametersPresent()) {
    bs.readUe(&aps.num_pred_nearest_neighbours_minus1);
    bs.readUe(&aps.inter_lod_search_range);

    if (aps.attr_encoding == AttributeEncoding::kLiftingTransform)
      bs.read(&aps.last_component_prediction_enabled_flag);

    bs.read(&aps.scalable_lifting_enabled_flag);
    if (aps.scalable_lifting_enabled_flag)
      bs.readUe(&aps.max_neigh_range_minus1);

    aps.canonical_point_order_flag = false;
    if (!aps.scalable_lifting_enabled_flag) {
      bs.readUe(&aps.num_detail_levels_minus1);
      if (!aps.num_detail_levels_minus1)
        bs.read(&aps.canonical_point_order_flag);
      else {
        bs.readUe(&aps.lod_decimation_type);

        if (aps.lod_decimation_type != LodDecimationMethod::kNone) {
          aps.lodSamplingPeriod.resize(aps.num_detail_levels_minus1);
          for (int idx = 0; idx < aps.num_detail_levels_minus1; idx++) {
            int lod_sampling_period_minus2;
            bs.readUe(&lod_sampling_period_minus2);
            aps.lodSamplingPeriod[idx] = lod_sampling_period_minus2 + 2;
          }
        }

        aps.dist2 = 0;
        bs.readUe(&aps.dist2);
        bs.read(&aps.aps_slice_dist2_deltas_present_flag);
      }
    }
  }

  aps.pred_weight_blending_enabled_flag = false;
  aps.intra_lod_prediction_skip_layers = aps.kSkipAllLayers;
  aps.quant_neigh_weight = 0;

  if (aps.attr_encoding == AttributeEncoding::kPredictingTransform) {
    bs.readUe(&aps.max_num_direct_predictors);
    aps.adaptive_prediction_threshold = 0;
    aps.direct_avg_predictor_disabled_flag = false;
    if (aps.max_num_direct_predictors) {
      bs.readUn(8, &aps.adaptive_prediction_threshold);
      bs.read(&aps.direct_avg_predictor_disabled_flag);
    }
    bs.readUe(&aps.intra_lod_search_range);
    if (aps.intra_lod_search_range)
      bs.readUe(&aps.intra_lod_prediction_skip_layers);
    bs.read(&aps.inter_component_prediction_enabled_flag);
    bs.read(&aps.pred_weight_blending_enabled_flag);
  }

  if (aps.attr_encoding == AttributeEncoding::kRAHTransform) {
    bs.read(&aps.rahtPredParams.raht_prediction_enabled_flag);
    if (aps.rahtPredParams.raht_prediction_enabled_flag) {
      bs.readUe(&aps.rahtPredParams.raht_prediction_threshold0);
      bs.readUe(&aps.rahtPredParams.raht_prediction_threshold1);
    }
  }

  if (aps.attr_encoding == AttributeEncoding::kRaw)
    bs.read(&aps.raw_attr_variable_len_flag);

  aps.spherical_coord_flag = false;
  if (!aps.scalable_lifting_enabled_flag)
    bs.read(&aps.spherical_coord_flag);
  if (aps.spherical_coord_flag) {
    for (int k = 0; k < 3; k++) {
      int attr_coord_scale_bits_minus1;
      bs.readUn(5, &attr_coord_scale_bits_minus1);
      bs.readUn(attr_coord_scale_bits_minus1 + 1, &aps.attr_coord_scale[k]);
    }
  }

  bool aps_extension_flag = bs.read();
  aps.max_points_per_sort_log2_plus1 = 0;
  aps.raht_extension = false;
  aps.rahtPredParams.raht_subnode_prediction_enabled_flag = false;
  aps.attrInterPredictionEnabled = false;
  aps.raht_inter_prediction_depth_minus1 = 0;
  aps.attrInterPredSearchRange = 0;
  aps.raht_send_inter_filters = false;
  aps.raht_inter_skip_layers = 0;
  aps.predictionWithDistributionEnabled = false;
  aps.enableMortonCodeScaling = false;
  aps.rahtPredParams.raht_enable_intraPred_nonPred_code_layer = false;
  aps.rahtPredParams.raht_hybrid_prediction_enabled_flag = false;
  aps.rahtPredParams.raht_hybrid_prediction_lower_depth_minus1 = 0;
  aps.rahtPredParams.raht_hybrid_prediction_num_enabled_layers = 0;

  if (aps_extension_flag) {
    if (aps.attr_encoding == AttributeEncoding::kRAHTransform) {
      bs.read(&aps.rahtPredParams.integer_haar_enable_flag);
    }
    if (aps.attr_encoding == AttributeEncoding::kPredictingTransform) {
      for (int i = 0; i <= aps.num_pred_nearest_neighbours_minus1; i++)
        bs.readUe(&aps.quant_neigh_weight[i]);
    }

    bs.read(&aps.attrInterPredictionEnabled);
    if (aps.attrInterPredictionEnabled) {
      if (aps.attr_encoding == AttributeEncoding::kRAHTransform) {
        bs.readUe(&aps.raht_inter_prediction_depth_minus1);
        if(!aps.rahtPredParams.integer_haar_enable_flag){
          bs.read(&aps.raht_send_inter_filters);
          bs.readUe(&aps.raht_inter_skip_layers);          
        }
        bs.read(&aps.raht_enable_code_layer);
        bs.read(&aps.rahtPredParams.raht_hybrid_prediction_enabled_flag);
        if(aps.rahtPredParams.raht_hybrid_prediction_enabled_flag){
          bs.readUe(&aps.rahtPredParams.raht_hybrid_prediction_num_enabled_layers);
          if(aps.rahtPredParams.raht_hybrid_prediction_num_enabled_layers)
            bs.readUe(&aps.rahtPredParams.raht_hybrid_prediction_lower_depth_minus1);
        }
      }
      else
      bs.readUe(&aps.attrInterPredSearchRange);
    }

    if (
      aps.lodParametersPresent() && !aps.scalable_lifting_enabled_flag
      && !aps.num_detail_levels_minus1) {
      bs.readUe(&aps.max_points_per_sort_log2_plus1);
    }

    if (
      aps.lodParametersPresent()
      && aps.num_pred_nearest_neighbours_minus1 >= 2)
      bs.read(&aps.predictionWithDistributionEnabled);

    if(
      aps.lodParametersPresent() 
      && aps.lod_decimation_type == LodDecimationMethod::kPeriodic
      && !aps.attrInterPredictionEnabled)
      bs.read(&aps.enableMortonCodeScaling);

    if (aps.attr_encoding == AttributeEncoding::kRAHTransform) {
      bs.read(&aps.raht_extension);
    }

    if (
      aps.attr_encoding == AttributeEncoding::kRAHTransform
      && aps.rahtPredParams.raht_prediction_enabled_flag) {
      bs.read(&aps.rahtPredParams.raht_subnode_prediction_enabled_flag);
      if (aps.rahtPredParams.raht_subnode_prediction_enabled_flag) {
        aps.rahtPredParams.raht_prediction_weights.resize(5);
        for (int i = 0; i < 5; i++)
          bs.readUe(&aps.rahtPredParams.raht_prediction_weights[i]);
        aps.rahtPredParams.setPredictionWeights();
      }
      bs.readUe(&aps.rahtPredParams.raht_prediction_search_range);
      bs.read(&aps.rahtPredParams.raht_enable_intraPred_nonPred_code_layer);
    }

    if (aps.attr_encoding == AttributeEncoding::kRAHTransform)
      bs.read(&aps.last_component_prediction_enabled_flag);

    
      bs.read(&aps.cross_attr_prediction_enabled_this_type);
      if (aps.cross_attr_prediction_enabled_this_type)
        bs.readUe(&aps.refAttrIdx);
    
    
	  if (sps.layer_group_enabled_flag){
      aps.layer_group_enabled_flag = true;
		  bs.read(&aps.attr_ref_id_present_flag);
    }else{
      aps.layer_group_enabled_flag = false;    
    }
  }

  bs.byteAlign();

  return aps;
}

//----------------------------------------------------------------------------

void
convertXyzToStv(const SequenceParameterSet& sps, AttributeParameterSet* aps)
{
  aps->lodNeighBias = fromXyz(sps.geometry_axis_order, aps->lodNeighBias);
}

//============================================================================

void
write(
  const SequenceParameterSet& sps,
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  PayloadBuffer* buf)
{
  assert(buf->type == PayloadType::kGeometryBrick);
  auto bs = makeBitWriter(std::back_inserter(*buf));

  int gbh_temporal_id = 0;
  bs.writeUn(4, gbh.geom_geom_parameter_set_id);
  bs.writeUn(3, gbh_temporal_id);
  bs.writeUe(gbh.geom_slice_id);
  bs.writeUn(sps.slice_tag_bits, gbh.slice_tag);
  bs.writeUn(sps.frame_ctr_bits, gbh.frame_ctr_lsb);

  if (sps.entropy_continuation_enabled_flag) {
    bs.write(gbh.entropy_continuation_flag);
    if (gbh.entropy_continuation_flag)
      bs.writeUe(gbh.prev_slice_id);
  }

  int trisoupNodeSizeLog2 = 0;
  if (gps.trisoup_enabled_flag) {
    bs.writeUe(gbh.trisoup_node_size_log2_minus2);
    trisoupNodeSizeLog2 = gbh.trisoup_node_size_log2_minus2 + 2;
  }

  int geomBoxLog2Scale =
    std::max(gbh.geomBoxLog2Scale(gps), trisoupNodeSizeLog2);
  auto geom_box_origin = toXyz(sps.geometry_axis_order, gbh.geomBoxOrigin);
  geom_box_origin.x() >>= geomBoxLog2Scale;
  geom_box_origin.y() >>= geomBoxLog2Scale;
  geom_box_origin.z() >>= geomBoxLog2Scale;

  if (gps.geom_box_log2_scale_present_flag)
    bs.writeUe(gbh.geom_box_log2_scale);

  bs.writeUe(gbh.geom_box_origin_bits_minus1);
  if (auto originBits = gbh.geom_box_origin_bits_minus1 + 1) {
    bs.writeUn(originBits, geom_box_origin.x());
    bs.writeUn(originBits, geom_box_origin.y());
    bs.writeUn(originBits, geom_box_origin.z());
  }

  if (gps.geom_slice_angular_origin_present_flag) {
    auto gbh_angular_origin =
      toXyz(sps.geometry_axis_order, gbh.gbhAngularOrigin);

    int gbh_angular_origin_bits_minus1 =
      numBits(gbh_angular_origin.abs().max()) - 1;

    bs.writeUe(gbh_angular_origin_bits_minus1);
    bs.writeSn(gbh_angular_origin_bits_minus1 + 1, gbh_angular_origin.x());
    bs.writeSn(gbh_angular_origin_bits_minus1 + 1, gbh_angular_origin.y());
    bs.writeSn(gbh_angular_origin_bits_minus1 + 1, gbh_angular_origin.z());
  }

  if (!gps.predgeom_enabled_flag) {
    int tree_depth_minus1 = gbh.tree_depth_minus1();

    if (!gps.trisoup_enabled_flag)
      bs.writeUe(tree_depth_minus1);
    else {
      int tree_depth = tree_depth_minus1 + 1;
      bs.writeUe(tree_depth);
    }
    if (gps.qtbt_enabled_flag)
      for (int i = 0; i <= tree_depth_minus1; i++)
        bs.writeUn(3, gbh.tree_lvl_coded_axis_list[i]);

    bs.writeUe(gbh.geom_stream_cnt_minus1);
  }

  if (gps.geom_scaling_enabled_flag) {
    bs.writeSe(gbh.geom_slice_qp_offset);
    if (gps.predgeom_enabled_flag)
      bs.writeUe(gbh.geom_qp_offset_intvl_log2_delta);
  }

  if (gps.trisoup_enabled_flag) {
    bs.writeUe(gbh.trisoup_sampling_value_minus1);
    bs.writeUe(gbh.num_unique_segments_bits_minus1);
    auto segmentBits = gbh.num_unique_segments_bits_minus1 + 1;
    bs.writeUn(segmentBits, gbh.num_unique_segments_minus1);
    bs.writeUe(gbh.trisoup_vertex_quantization_bits);
    bs.write(gbh.trisoup_centroid_vertex_residual_flag);
    if (gbh.trisoup_centroid_vertex_residual_flag) {
      bs.write(gbh.trisoup_face_vertex_flag);
    }
    bs.write(gbh.trisoup_halo_flag);
    if (gbh.trisoup_halo_flag)
      bs.write(gbh.trisoup_adaptive_halo_flag);
    bs.write(gbh.trisoup_fine_ray_tracing_flag);

    bs.write(gbh.trisoup_vertex_merge);
    bs.write(gbh.trisoup_vertex_fix);
    int _write_bits = 0;
    if (gps.non_cubic_node_start_edge) {
      bs.writeUe(gbh.slice_bb_pos_bits);
      if (gbh.slice_bb_pos_bits > 0) {
        bs.writeUe(gbh.slice_bb_pos_log2_scale);
        _write_bits = gbh.slice_bb_pos_bits;
        bs.writeUn(_write_bits, gbh.slice_bb_pos[0]);
        bs.writeUn(_write_bits, gbh.slice_bb_pos[1]);
        bs.writeUn(_write_bits, gbh.slice_bb_pos[2]);
      }
    }
    if (gps.non_cubic_node_end_edge) {
      bs.writeUe(gbh.slice_bb_width_bits);
      if (gbh.slice_bb_width_bits > 0) {
        bs.writeUe(gbh.slice_bb_width_log2_scale);
        _write_bits = gbh.slice_bb_width_bits;
        bs.writeUn(_write_bits, gbh.slice_bb_width[0]);
        bs.writeUn(_write_bits, gbh.slice_bb_width[1]);
        bs.writeUn(_write_bits, gbh.slice_bb_width[2]);
      }
    }
  }

  if (gps.predgeom_enabled_flag) {
    for (int k = 0; k < 3; k++)
      bs.writeUn(3, gbh.pgeom_resid_abs_log2_bits[k]);

    if (gps.geom_angular_mode_enabled_flag)
      bs.writeUe(gbh.pgeom_min_radius);
  }
  if (gps.interPredictionEnabledFlag)
    bs.write(gbh.interPredictionEnabledFlag);
  else
    assert(!gbh.interPredictionEnabledFlag);

  if (gbh.interPredictionEnabledFlag) {
    if (gps.globalMotionEnabled) {
      if (gps.predgeom_enabled_flag)
        bs.write(gbh.interFrameRefGmcFlag);
      if (!gps.predgeom_enabled_flag || gbh.interFrameRefGmcFlag) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if (i == j)
              bs.writeSe(gbh.gm_param[0].gm_matrix[3 * i + j] - 65536);
            else
              bs.writeSe(gbh.gm_param[0].gm_matrix[3 * i + j]);
          }
        }
        for (int j = 0; j < 3; j++)
          bs.writeSe(gbh.gm_param[0].gm_trans[j]);
      }
      if (!gps.predgeom_enabled_flag) {
        bs.write(gbh.lpu_type);
        bs.write(gbh.min_zero_origin_flag);
        if (gbh.lpu_type != 0)
          for (int i = 0; i < 3; i++)
            bs.writeUe(gbh.motion_block_size[i]);
      }
      const bool signalThres0 = (gps.predgeom_enabled_flag || !gbh.lpu_type)
        && (!gps.predgeom_enabled_flag || gbh.interFrameRefGmcFlag);
      if (signalThres0) {
        bs.writeSe(gbh.gm_param[0].gm_thres.first);
        bs.writeSe(gbh.gm_param[0].gm_thres.second);
      }
    }
    if (gps.biPredictionEnabledFlag)
      bs.write(gbh.biPredictionEnabledFlag);
    else
      assert(!gbh.biPredictionEnabledFlag);
    if (gbh.biPredictionEnabledFlag && gps.globalMotionEnabled) {
      if (gps.predgeom_enabled_flag)
        bs.write(gbh.interFrameRefGmcFlag2);
      if (!gps.predgeom_enabled_flag || gbh.interFrameRefGmcFlag2) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if (i == j)
              bs.writeSe(gbh.gm_param[1].gm_matrix[3 * i + j] - 65536);
            else
              bs.writeSe(gbh.gm_param[1].gm_matrix[3 * i + j]);
          }
        }
        for (int j = 0; j < 3; j++)
          bs.writeSe(gbh.gm_param[1].gm_trans[j]);
      }
      const bool signalThres1 = (gps.predgeom_enabled_flag || !gbh.lpu_type)
        && (!gps.predgeom_enabled_flag || gbh.interFrameRefGmcFlag2);
      if (signalThres1) {
        bs.writeSe(gbh.gm_param[1].gm_thres.first);
        bs.writeSe(gbh.gm_param[1].gm_thres.second);
      }
    }
  }
  if (
    gbh.interPredictionEnabledFlag
    && sps.inter_entropy_continuation_enabled_flag) {
    bs.write(&gbh.slice_inter_entropy_continuation_flag);
    if (gbh.slice_inter_entropy_continuation_flag) {
      bs.writeUn(sps.frame_ctr_bits, gbh.inter_entropy_prev_frame_lsb);
      bs.writeUe(gbh.inter_entropy_prev_slice_id);
    }
  }

  // layer-group slicing
  if (sps.layer_group_enabled_flag) {
	  if (!gps.predgeom_enabled_flag) {
		  bs.writeUe(gbh.num_layer_groups_minus1);
		  bool acc_subgroup_enabled_flag = 0;
		  for (int i = 0; i <= gbh.num_layer_groups_minus1; i++) {
			  //bs.writeUe(gbh.layer_group_id[i]);
			  bs.writeUe(gbh.num_layers_minus1[i]);
			  bs.write(gbh.subgroup_enabled_flag[i]);

			  acc_subgroup_enabled_flag |= gbh.subgroup_enabled_flag[i];
		  }
		  if (acc_subgroup_enabled_flag) {
			  bs.writeUe(gbh.subgroupBboxOrigin_bits_minus1);
			  bs.writeUe(gbh.subgroupBboxSize_bits_minus1);
		  }

		  for (int i = 0; i < 3; i++)
			  bs.writeUe(gbh.root_node_size_log2[i]);

		  bs.writeUn(16, gbh.numSubsequentSubgroups);

		  const bool checkPlanarEligibilityBasedOnOctreeDepth =
			  gps.geom_planar_mode_enabled_flag
			  && gps.geom_octree_depth_planar_eligibiity_enabled_flag
			  && !gps.geom_angular_mode_enabled_flag;

		  if (checkPlanarEligibilityBasedOnOctreeDepth) {
			  for (int i = 0; i <= gbh.num_layers_minus1[0]; i++)
				  bs.write(gbh.planarEligibleKOctreeDepth[i]);
		  }

		  bs.writeUn(8, gbh.fgs_occtree_depth_minus1);
	  }
  }

  bs.byteAlign();
}

//----------------------------------------------------------------------------

GeometryBrickHeader
parseGbh(
  const SequenceParameterSet& sps,
  const GeometryParameterSet& gps,
  const PayloadBuffer& buf,
  int* bytesReadHead,
  int* bytesReadFoot)
{
  GeometryBrickHeader gbh;
  assert(buf.type == PayloadType::kGeometryBrick);
  auto bs = makeBitReader(buf.begin(), buf.end());

  int gbh_temporal_id;
  bs.readUn(4, &gbh.geom_geom_parameter_set_id);
  bs.readUn(3, &gbh_temporal_id);
  bs.readUe(&gbh.geom_slice_id);
  bs.readUn(sps.slice_tag_bits, &gbh.slice_tag);
  bs.readUn(sps.frame_ctr_bits, &gbh.frame_ctr_lsb);

  gbh.entropy_continuation_flag = false;
  if (sps.entropy_continuation_enabled_flag) {
    bs.read(&gbh.entropy_continuation_flag);
    if (gbh.entropy_continuation_flag)
      bs.readUe(&gbh.prev_slice_id);
  }

  int trisoupNodeSizeLog2 = 0;
  if (gps.trisoup_enabled_flag) {
    bs.readUe(&gbh.trisoup_node_size_log2_minus2);
    trisoupNodeSizeLog2 = gbh.trisoup_node_size_log2_minus2 + 2;
  }

  if (gps.geom_box_log2_scale_present_flag)
    bs.readUe(&gbh.geom_box_log2_scale);

  Vec3<int> geom_box_origin;
  bs.readUe(&gbh.geom_box_origin_bits_minus1);
  if (auto originBits = gbh.geom_box_origin_bits_minus1 + 1) {
    bs.readUn(originBits, &geom_box_origin.x());
    bs.readUn(originBits, &geom_box_origin.y());
    bs.readUn(originBits, &geom_box_origin.z());
  }
  gbh.geomBoxOrigin = fromXyz(sps.geometry_axis_order, geom_box_origin);
  gbh.geomBoxOrigin *=
    1 << std::max(gbh.geomBoxLog2Scale(gps), trisoupNodeSizeLog2);

  if (gps.geom_slice_angular_origin_present_flag) {
    int gbh_angular_origin_bits_minus1;
    bs.readUe(&gbh_angular_origin_bits_minus1);

    Vec3<int> gbh_angular_origin;
    bs.readSn(gbh_angular_origin_bits_minus1 + 1, &gbh_angular_origin.x());
    bs.readSn(gbh_angular_origin_bits_minus1 + 1, &gbh_angular_origin.y());
    bs.readSn(gbh_angular_origin_bits_minus1 + 1, &gbh_angular_origin.z());

    gbh.gbhAngularOrigin =
      fromXyz(sps.geometry_axis_order, gbh_angular_origin);
  }

  gbh.geom_stream_cnt_minus1 = 0;
  if (!gps.predgeom_enabled_flag) {
    int tree_depth_minus1;
    if (!gps.trisoup_enabled_flag)
      bs.readUe(&tree_depth_minus1);
    else {
      int tree_depth;
      bs.readUe(&tree_depth);
      tree_depth_minus1 = tree_depth - 1;
    }

    gbh.tree_lvl_coded_axis_list.resize(tree_depth_minus1 + 1, 7);
    if (gps.qtbt_enabled_flag)
      for (int i = 0; i <= tree_depth_minus1; i++)
        bs.readUn(3, &gbh.tree_lvl_coded_axis_list[i]);

    bs.readUe(&gbh.geom_stream_cnt_minus1);
  }

  gbh.geom_slice_qp_offset = 0;
  if (gps.geom_scaling_enabled_flag) {
    bs.readSe(&gbh.geom_slice_qp_offset);
    if (gps.predgeom_enabled_flag)
      bs.readUe(&gbh.geom_qp_offset_intvl_log2_delta);
  }

  gbh.trisoup_vertex_merge = false;
  gbh.trisoup_vertex_fix = false;
  if (gps.trisoup_enabled_flag) {
    bs.readUe(&gbh.trisoup_sampling_value_minus1);
    bs.readUe(&gbh.num_unique_segments_bits_minus1);
    auto segmentBits = gbh.num_unique_segments_bits_minus1 + 1;
    bs.readUn(segmentBits, &gbh.num_unique_segments_minus1);
    bs.readUe(&gbh.trisoup_vertex_quantization_bits);
    bs.read(&gbh.trisoup_centroid_vertex_residual_flag);
    gbh.trisoup_face_vertex_flag = false;
    if( gbh.trisoup_centroid_vertex_residual_flag ){
      bs.read(&gbh.trisoup_face_vertex_flag);
    }
    bs.read(&gbh.trisoup_halo_flag);
    gbh.trisoup_adaptive_halo_flag = false;
    if (gbh.trisoup_halo_flag)
      bs.read(&gbh.trisoup_adaptive_halo_flag);
    bs.read(&gbh.trisoup_fine_ray_tracing_flag);

    bs.read(&gbh.trisoup_vertex_merge);
    bs.read(&gbh.trisoup_vertex_fix);
    gbh.slice_bb_pos_bits = 0;
    gbh.slice_bb_pos_log2_scale = 0;
    gbh.slice_bb_pos = 0;
    gbh.slice_bb_width_bits = 0;
    gbh.slice_bb_width_log2_scale = 0;
    gbh.slice_bb_width = 0;
    int _read_bits = 0;
    if( gps.non_cubic_node_start_edge ){
      bs.readUe( &gbh.slice_bb_pos_bits );
      if( gbh.slice_bb_pos_bits > 0 ) {
        bs.readUe( &gbh.slice_bb_pos_log2_scale );
        _read_bits = gbh.slice_bb_pos_bits;
        bs.readUn( _read_bits, &gbh.slice_bb_pos[0] );
        bs.readUn( _read_bits, &gbh.slice_bb_pos[1] );
        bs.readUn( _read_bits, &gbh.slice_bb_pos[2] );
      }
    }
    if( gps.non_cubic_node_end_edge ){
      bs.readUe( &gbh.slice_bb_width_bits );
      if( gbh.slice_bb_width_bits > 0 ) {
        bs.readUe( &gbh.slice_bb_width_log2_scale );
        _read_bits = gbh.slice_bb_width_bits;
        bs.readUn( _read_bits, &gbh.slice_bb_width[0] );
        bs.readUn( _read_bits, &gbh.slice_bb_width[1] );
        bs.readUn( _read_bits, &gbh.slice_bb_width[2] );
      }
    }
  }

  gbh.pgeom_min_radius = 0;
  if (gps.predgeom_enabled_flag) {
    for (int k = 0; k < 3; k++)
      bs.readUn(3, &gbh.pgeom_resid_abs_log2_bits[k]);

    if (gps.geom_angular_mode_enabled_flag)
      bs.readUe(&gbh.pgeom_min_radius);
  }
  if (gps.interPredictionEnabledFlag)
    bs.read(&gbh.interPredictionEnabledFlag);
  else
    gbh.interPredictionEnabledFlag = false;


  gbh.gm_param[0].init();
  gbh.gm_param[1].init();
  gbh.motion_block_size = {0, 0, 0};
  gbh.interFrameRefGmcFlag = false;
  gbh.interFrameRefGmcFlag2 = false;
  gbh.biPredictionEnabledFlag = false;
  if (gbh.interPredictionEnabledFlag) {
    if (gps.globalMotionEnabled) {
      if (gps.predgeom_enabled_flag)
        bs.read(&gbh.interFrameRefGmcFlag);
      if (!gps.predgeom_enabled_flag || gbh.interFrameRefGmcFlag) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++)
            bs.readSe(&gbh.gm_param[0].gm_matrix[3 * i + j]);
          gbh.gm_param[0].gm_matrix[3 * i + i] += 65536;
        }
        for (int j = 0; j < 3; j++)
          bs.readSe(&gbh.gm_param[0].gm_trans[j]);
      }

      if (!gps.predgeom_enabled_flag) {
        bs.read(&gbh.lpu_type);
        bs.read(&gbh.min_zero_origin_flag);
        if (gbh.lpu_type != 0)
          for (int i = 0; i < 3; i++)
            bs.readUe(&gbh.motion_block_size[i]);
      }
      const bool signalThres0 = (gps.predgeom_enabled_flag || !gbh.lpu_type)
        && (!gps.predgeom_enabled_flag
            || gbh.interFrameRefGmcFlag);  // to be simplified
      if (signalThres0) {
        bs.readSe(&gbh.gm_param[0].gm_thres.first);
        bs.readSe(&gbh.gm_param[0].gm_thres.second);
      }
    }
    if (gps.biPredictionEnabledFlag)
      bs.read(&gbh.biPredictionEnabledFlag);
    if (gbh.biPredictionEnabledFlag && gps.globalMotionEnabled) {
      if (gps.predgeom_enabled_flag)
        bs.read(&gbh.interFrameRefGmcFlag2);
      if (!gps.predgeom_enabled_flag || gbh.interFrameRefGmcFlag2) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++)
            bs.readSe(&gbh.gm_param[1].gm_matrix[3 * i + j]);
          gbh.gm_param[1].gm_matrix[3 * i + i] += 65536;
        }
        for (int j = 0; j < 3; j++)
          bs.readSe(&gbh.gm_param[1].gm_trans[j]);
      }
      const bool signalThres1 = (gps.predgeom_enabled_flag || !gbh.lpu_type)
        && (!gps.predgeom_enabled_flag
            || gbh.interFrameRefGmcFlag2);  // to be simplified
      if (signalThres1) {
        bs.readSe(&gbh.gm_param[1].gm_thres.first);
        bs.readSe(&gbh.gm_param[1].gm_thres.second);
      }
    }
  }
  gbh.slice_inter_entropy_continuation_flag = false;
  if (
    gbh.interPredictionEnabledFlag
    && sps.inter_entropy_continuation_enabled_flag) {
    bs.read(&gbh.slice_inter_entropy_continuation_flag);
    if (gbh.slice_inter_entropy_continuation_flag) {
      bs.readUn(sps.frame_ctr_bits, &gbh.inter_entropy_prev_frame_lsb);
      bs.readUe(&gbh.inter_entropy_prev_slice_id);
    }
  }
  // layer-group slicing
  if (sps.layer_group_enabled_flag) {
	  if (sps.layer_group_enabled_flag) {
		  bs.readUe(&gbh.num_layer_groups_minus1);

		  gbh.layer_group_id.resize(gbh.num_layer_groups_minus1 + 1);
		  gbh.num_layers_minus1.resize(gbh.num_layer_groups_minus1 + 1);
		  gbh.subgroup_enabled_flag.resize(gbh.num_layer_groups_minus1 + 1);
		  bool acc_subgroup_enabled_flag = 0;

		  for (int i = 0; i <= gbh.num_layer_groups_minus1; i++) {
			  bs.readUe(&gbh.num_layers_minus1[i]);
			  gbh.subgroup_enabled_flag[i] = bs.read();

			  acc_subgroup_enabled_flag |= gbh.subgroup_enabled_flag[i];
		  }
		  if (acc_subgroup_enabled_flag) {
			  bs.readUe(&gbh.subgroupBboxOrigin_bits_minus1);
			  bs.readUe(&gbh.subgroupBboxSize_bits_minus1);
		  }
		  for (int i = 0; i < 3; i++)
			  bs.readUe(&gbh.root_node_size_log2[i]);

		  gbh.tree_lvl_coded_axis_list.resize(gbh.root_node_size_log2.max(), 7);

		  bs.readUn(16, &gbh.numSubsequentSubgroups);

		  const bool checkPlanarEligibilityBasedOnOctreeDepth =
			  gps.geom_planar_mode_enabled_flag
			  && gps.geom_octree_depth_planar_eligibiity_enabled_flag
			  && !gps.geom_angular_mode_enabled_flag;

		  if (checkPlanarEligibilityBasedOnOctreeDepth) {
			  gbh.planarEligibleKOctreeDepth.resize(gbh.num_layers_minus1[0] + 1);
			  for (int i = 0; i <= gbh.num_layers_minus1[0]; i++) {
				  gbh.planarEligibleKOctreeDepth[i] = bs.read();
			  }
		  }
		  bs.readUn(8, &gbh.fgs_occtree_depth_minus1);
	  }
  }

  bs.byteAlign();

  if (bytesReadHead)
    *bytesReadHead = int(std::distance(buf.begin(), bs.pos()));

  // To avoid having to make separate calls, the footer is parsed here
  gbh.footer = parseGbf(gps, gbh, buf, bytesReadFoot);

  return gbh;
}

//----------------------------------------------------------------------------

GeometryBrickHeader
parseGbhIds(const PayloadBuffer& buf)
{
  GeometryBrickHeader gbh;
  assert(buf.type == PayloadType::kGeometryBrick);
  auto bs = makeBitReader(buf.begin(), buf.end());

  int gbh_temporal_id;
  bs.readUn(4, &gbh.geom_geom_parameter_set_id);
  bs.readUn(3, &gbh_temporal_id);
  bs.readUe(&gbh.geom_slice_id);
  // NB: to decode slice_tag requires sps activation

  /* NB: this function only decodes ids at the start of the header. */
  /* NB: do not attempt to parse any further */

  return gbh;
}

//============================================================================

void
write(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& /* gbh */,
  const GeometryBrickFooter& gbf,
  PayloadBuffer* buf)
{
  assert(buf->type == PayloadType::kGeometryBrick);
  auto bs = makeBitWriter(std::back_inserter(*buf));

  // NB: if modifying this footer, it is essential that the decoder can
  // either decode backwards, or seek to the start.

  if (gps.octree_point_count_list_present_flag) 
	  for (int i = 0; i < gbf.octree_lvl_num_points_minus1.size(); i++) 
		  bs.writeUn(24, gbf.octree_lvl_num_points_minus1[i]);

  bs.writeUn(24, gbf.geom_num_points_minus1);
}

//----------------------------------------------------------------------------

GeometryBrickFooter
parseGbf(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  const PayloadBuffer& buf,
  int* bytesRead)
{
  GeometryBrickFooter gbf;
  assert(buf.type == PayloadType::kGeometryBrick);

  // todo(df): this would be simpler to parse if it were written in reverse
  auto bufStart = buf.end() - 3; /* geom_num_points_minus1 */

  auto tree_depth_minus1 = gbh.tree_depth_minus1();
  if (gbh.fgs_occtree_depth_minus1 >= 0)
	  tree_depth_minus1 = gbh.fgs_occtree_depth_minus1;

  if (gps.octree_point_count_list_present_flag)
    bufStart -= tree_depth_minus1 * 3;

  if (bytesRead)
    *bytesRead = int(std::distance(bufStart, buf.end()));

  auto bs = makeBitReader(bufStart, buf.end());

  if (gps.octree_point_count_list_present_flag) {
    gbf.octree_lvl_num_points_minus1.resize(tree_depth_minus1);
	for (int i = 0; i < tree_depth_minus1; i++) 
		bs.readUn(24, &gbf.octree_lvl_num_points_minus1[i]);
  }

  bs.readUn(24, &gbf.geom_num_points_minus1);

  return gbf;
}


//============================================================================

void
write(
	const GeometryParameterSet& gps,
	const GeometryBrickHeader& /* gbh */,
	const DependentGeometryDataUnitFooter& dgduf,
	PayloadBuffer* buf)
{
	assert(buf->type == PayloadType::kDependentGeometryDataUnit);
	auto bs = makeBitWriter(std::back_inserter(*buf));

	// NB: if modifying this footer, it is essential that the decoder can
	// either decode backwards, or seek to the start.

	if (gps.octree_point_count_list_present_flag) 
		for (int i = 0; i < dgduf.octree_lvl_node_cnt_minus1.size(); i++) 
			bs.writeUn(24, dgduf.octree_lvl_node_cnt_minus1[i]);

	bs.writeUn(24, dgduf.num_nodes_minus1);
}

//----------------------------------------------------------------------------

DependentGeometryDataUnitFooter
parseDgbf(
	const GeometryParameterSet& gps,
	//const GeometryBrickHeader& gbh,
	int numDepthMinus1,
	const PayloadBuffer& buf,
	int* bytesRead)
{
	DependentGeometryDataUnitFooter dgduf;
	assert(buf.type == PayloadType::kDependentGeometryDataUnit);
	
	// todo(df): this would be simpler to parse if it were written in reverse
	auto bufStart = buf.end() - 3; /* num_nodes_minus1 */
	if (gps.octree_point_count_list_present_flag)
		bufStart -= numDepthMinus1 * 3;

	if (bytesRead)
		*bytesRead = int(std::distance(bufStart, buf.end()));

	auto bs = makeBitReader(bufStart, buf.end());
	
	if (gps.octree_point_count_list_present_flag) {
		dgduf.octree_lvl_node_cnt_minus1.resize(numDepthMinus1);
		for (int i = 0; i < numDepthMinus1; i++) 
			bs.readUn(24, &dgduf.octree_lvl_node_cnt_minus1[i]);
	}

	bs.readUn(24, &dgduf.num_nodes_minus1);

	return dgduf;
}

//============================================================================

void
write(
	const GeometryParameterSet& gps,
	const GeometryBrickHeader& gbh,
	const DependentGeometryDataUnitHeader& dep_gbh,
	PayloadBuffer* buf)
{
	assert(buf->type == PayloadType::kDependentGeometryDataUnit);
	auto bs = makeBitWriter(std::back_inserter(*buf));

	bs.writeUn(4, dep_gbh.geom_parameter_set_id);
	bs.writeUe(dep_gbh.geom_slice_id);

	bs.writeUn(8, dep_gbh.layer_group_id);
	if (gbh.subgroup_enabled_flag[dep_gbh.layer_group_id]) {
		bs.writeUn(16, dep_gbh.subgroup_id);

		if (auto originBits = gbh.subgroupBboxOrigin_bits_minus1 + 1) {
			bs.writeUn(originBits, dep_gbh.subgroupBboxOrigin.x());
			bs.writeUn(originBits, dep_gbh.subgroupBboxOrigin.y());
			bs.writeUn(originBits, dep_gbh.subgroupBboxOrigin.z());
		}
		if (auto sizeBits = gbh.subgroupBboxSize_bits_minus1 + 1) {
			bs.writeUn(sizeBits, dep_gbh.subgroupBboxSize.x());
			bs.writeUn(sizeBits, dep_gbh.subgroupBboxSize.y());
			bs.writeUn(sizeBits, dep_gbh.subgroupBboxSize.z());
		}
	}

	bs.writeUn(8, dep_gbh.ref_layer_group_id);
	if (gbh.subgroup_enabled_flag[dep_gbh.ref_layer_group_id])
		bs.writeUn(16, dep_gbh.ref_subgroup_id);

	bs.write(dep_gbh.context_reference_indication_flag);

  if (dep_gbh.context_reference_indication_flag)
	  bs.writeUn(8, dep_gbh.numSubsequentSubgroups);


	const bool checkPlanarEligibilityBasedOnOctreeDepth =
		gps.geom_planar_mode_enabled_flag
		&& gps.geom_octree_depth_planar_eligibiity_enabled_flag
		&& !gps.geom_angular_mode_enabled_flag;
	if (checkPlanarEligibilityBasedOnOctreeDepth) {
		for (int i = 0; i <= gbh.num_layers_minus1[dep_gbh.layer_group_id]; i++)
			bs.write(dep_gbh.planarEligibleKOctreeDepth[i]);
	}

	bs.byteAlign();
}

//----------------------------------------------------------------------------

DependentGeometryDataUnitHeader
parseDepGbh(
	const SequenceParameterSet& sps,
	const GeometryParameterSet& gps,
	GeometryBrickHeader gbh,
	const PayloadBuffer& buf,
	int* bytesReadHead,
	int* bytesReadFoot)
{
	DependentGeometryDataUnitHeader dep_gbh;
	assert(buf.type == PayloadType::kDependentGeometryDataUnit);

	auto bs = makeBitReader(buf.begin(), buf.end());

	bs.readUn(4, &dep_gbh.geom_parameter_set_id);
	bs.readUe(&dep_gbh.geom_slice_id);

	bs.readUn(8, &dep_gbh.layer_group_id);
	if (gbh.subgroup_enabled_flag[dep_gbh.layer_group_id]) {
		bs.readUn(16, &dep_gbh.subgroup_id);

		if (auto originBits = gbh.subgroupBboxOrigin_bits_minus1 + 1) {
			bs.readUn(originBits, &dep_gbh.subgroupBboxOrigin.x());
			bs.readUn(originBits, &dep_gbh.subgroupBboxOrigin.y());
			bs.readUn(originBits, &dep_gbh.subgroupBboxOrigin.z());
		}
		if (auto sizeBits = gbh.subgroupBboxSize_bits_minus1 + 1) {
			bs.readUn(sizeBits, &dep_gbh.subgroupBboxSize.x());
			bs.readUn(sizeBits, &dep_gbh.subgroupBboxSize.y());
			bs.readUn(sizeBits, &dep_gbh.subgroupBboxSize.z());
		}
	}
	else
		dep_gbh.subgroup_id = 0;


	bs.readUn(8, &dep_gbh.ref_layer_group_id);
	if (gbh.subgroup_enabled_flag[dep_gbh.ref_layer_group_id])
		bs.readUn(16, &dep_gbh.ref_subgroup_id);
	else
		dep_gbh.ref_subgroup_id = 0;

	bs.read(&dep_gbh.context_reference_indication_flag);	
  
  if (dep_gbh.context_reference_indication_flag)
		bs.readUn(8, &dep_gbh.numSubsequentSubgroups);
	else
		dep_gbh.numSubsequentSubgroups = 0;


	const bool checkPlanarEligibilityBasedOnOctreeDepth =
		gps.geom_planar_mode_enabled_flag
		&& gps.geom_octree_depth_planar_eligibiity_enabled_flag
		&& !gps.geom_angular_mode_enabled_flag;
	if (checkPlanarEligibilityBasedOnOctreeDepth) {
		dep_gbh.planarEligibleKOctreeDepth.resize(gbh.num_layers_minus1[dep_gbh.layer_group_id] + 1);
		for (int i = 0; i <= gbh.num_layers_minus1[dep_gbh.layer_group_id]; i++) {
			dep_gbh.planarEligibleKOctreeDepth[i] = bs.read();
		}
	}

	bs.byteAlign();

	if (bytesReadHead)
		*bytesReadHead = int(std::distance(buf.begin(), bs.pos()));

	// To avoid having to make separate calls, the footer is parsed here
	dep_gbh.footer = parseDgbf(gps, gbh.num_layers_minus1[dep_gbh.layer_group_id], buf, bytesReadFoot);

	return dep_gbh;
}

DependentGeometryDataUnitHeader
parseDepGbhIds(const PayloadBuffer& buf)
{
	DependentGeometryDataUnitHeader dep_gbh;
	assert(buf.type == PayloadType::kDependentGeometryDataUnit);
	auto bs = makeBitReader(buf.begin(), buf.end());

	bs.readUn(4, &dep_gbh.geom_parameter_set_id);
	bs.readUe(&dep_gbh.geom_slice_id);
	// NB: to decode slice_tag requires sps activation

	/* NB: this function only decodes ids at the start of the header. */
	/* NB: do not attempt to parse any further */

	return dep_gbh;
}

//============================================================================

PayloadBuffer
write(
	const SequenceParameterSet& sps, const LayerGroupStructureInventory& inventory)
{
	PayloadBuffer buf(PayloadType::kLayerGroupStructureInventory);
	auto bs = makeBitWriter(std::back_inserter(buf));

	bs.writeUn(4, inventory.lgsi_seq_parameter_set_id);
	bs.writeUn(5, inventory.lgsi_frame_idx_bits);
	bs.writeUn(inventory.lgsi_frame_idx_bits, inventory.lgsi_frame_idx);

	int num_slice_ids_minus1 = inventory.slice_ids.size() - 1;
	bs.writeUn(16, num_slice_ids_minus1);
	if (num_slice_ids_minus1 < 0) {
		bs.byteAlign();
		return buf;
	}

	for (const auto& slice : inventory.slice_ids) {
		bs.writeUe(slice.lgsi_slice_id);
		bs.writeUn(8, slice.lgsi_num_layer_groups_minus1);
		bs.writeUe(slice.lgsi_subgroupBboxOrigin_bits_minus1);
		bs.writeUe(slice.lgsi_subgroupBboxSize_bits_minus1);

		for (const auto& entry : slice.layerGroups) {
			bs.writeUn(8, entry.lgsi_layer_group_id);
			bs.writeUn(8, entry.lgsi_num_layers_minus1);
			bs.writeUn(16, entry.lgsi_num_subgroups_minus1);

			for (const auto& subentry : entry.subgroups) {
				bs.writeUn(16, subentry.lgsi_subgroup_id);
				bs.writeUn(16, subentry.lgsi_parent_subgroup_id);

				if (auto originBits = slice.lgsi_subgroupBboxOrigin_bits_minus1 + 1) {
					bs.writeUn(originBits, subentry.lgsi_subgroupBboxOrigin.x());
					bs.writeUn(originBits, subentry.lgsi_subgroupBboxOrigin.y());
					bs.writeUn(originBits, subentry.lgsi_subgroupBboxOrigin.z());
				}
				if (auto sizeBits = slice.lgsi_subgroupBboxSize_bits_minus1 + 1) {
					bs.writeUn(sizeBits, subentry.lgsi_subgroupBboxSize.x());
					bs.writeUn(sizeBits, subentry.lgsi_subgroupBboxSize.y());
					bs.writeUn(sizeBits, subentry.lgsi_subgroupBboxSize.z());
				}
			}
		}
	}

	// NB: this is at the end of the inventory to aid fixed-width parsing
	bs.writeUe(inventory.lgsi_origin_bits_minus1);
	if (auto originBits = inventory.lgsi_origin_bits_minus1 + 1) {
		auto origin_xyz = toXyz(sps.geometry_axis_order, inventory.lgsi_origin);
		bs.writeSn(originBits, origin_xyz.x());
		bs.writeSn(originBits, origin_xyz.y());
		bs.writeSn(originBits, origin_xyz.z());
	}

	int origin_log2_scale = 0;
	bs.writeUe(origin_log2_scale);

	bs.byteAlign();

	return buf;
}

//============================================================================

LayerGroupStructureInventory
parseLayerGroupStructureInventory(const PayloadBuffer& buf)
{
	LayerGroupStructureInventory inventory;
	assert(buf.type == PayloadType::kLayerGroupStructureInventory);
	auto bs = makeBitReader(buf.begin(), buf.end());

	bs.readUn(4, &inventory.lgsi_seq_parameter_set_id);
	bs.readUn(5, &inventory.lgsi_frame_idx_bits);
	bs.readUn(inventory.lgsi_frame_idx_bits, &inventory.lgsi_frame_idx);
	bs.readUn(16, &inventory.lgsi_num_slice_ids_minus1);
	if (inventory.lgsi_num_slice_ids_minus1 < 0) {
		bs.byteAlign();
		return inventory;
	}

	for (int slice_id = 0; slice_id < inventory.lgsi_num_slice_ids_minus1 + 1; slice_id++) {
		LayerGroupStructureInventory::SliceEntry slice;

		bs.readUe(&slice.lgsi_slice_id);
		bs.readUn(8, &slice.lgsi_num_layer_groups_minus1);

		bs.readUe(&slice.lgsi_subgroupBboxOrigin_bits_minus1);
		bs.readUe(&slice.lgsi_subgroupBboxSize_bits_minus1);

		for (int layer_group_id = 0; layer_group_id < slice.lgsi_num_layer_groups_minus1 + 1; layer_group_id++) {
			LayerGroupStructureInventory::GroupEntry entry;

			bs.readUn(8, &entry.lgsi_layer_group_id);
			bs.readUn(8, &entry.lgsi_num_layers_minus1);
			bs.readUn(16, &entry.lgsi_num_subgroups_minus1);

			for (int subgroup_id = 0; subgroup_id < entry.lgsi_num_subgroups_minus1 + 1; subgroup_id++) {
				LayerGroupStructureInventory::SubGroupEntry subentry;

				bs.readUn(16, &subentry.lgsi_subgroup_id);
				bs.readUn(16, &subentry.lgsi_parent_subgroup_id);

				if (auto originBits = slice.lgsi_subgroupBboxOrigin_bits_minus1 + 1) {
					bs.readUn(originBits, &subentry.lgsi_subgroupBboxOrigin.x());
					bs.readUn(originBits, &subentry.lgsi_subgroupBboxOrigin.y());
					bs.readUn(originBits, &subentry.lgsi_subgroupBboxOrigin.z());
				}
				if (auto sizeBits = slice.lgsi_subgroupBboxSize_bits_minus1 + 1) {
					bs.readUn(sizeBits, &subentry.lgsi_subgroupBboxSize.x());
					bs.readUn(sizeBits, &subentry.lgsi_subgroupBboxSize.y());
					bs.readUn(sizeBits, &subentry.lgsi_subgroupBboxSize.z());
				}
				entry.subgroups.push_back(subentry);
			}
			slice.layerGroups.push_back(entry);
		}
		inventory.slice_ids.push_back(slice);
	}

	Vec3<int> lgsi_origin_xyz;
	bs.readUe(&inventory.lgsi_origin_bits_minus1);
	if (auto originBits = inventory.lgsi_origin_bits_minus1 + 1) {
		bs.readSn(originBits, &lgsi_origin_xyz.x());
		bs.readSn(originBits, &lgsi_origin_xyz.y());
		bs.readSn(originBits, &lgsi_origin_xyz.z());
	}

	int lgsi_origin_log2_scale;
	bs.readUe(&lgsi_origin_log2_scale);
	lgsi_origin_xyz *= 1 << lgsi_origin_log2_scale;

	// NB: this is in XYZ axis order until converted to STV
	inventory.lgsi_origin = lgsi_origin_xyz;

	bs.byteAlign();

	return inventory;
}

//----------------------------------------------------------------------------

void
write(
  const SequenceParameterSet& sps,
  const AttributeParameterSet& aps,
  const AttributeBrickHeader& abh,
  PayloadBuffer* buf)
{
  assert(buf->type == PayloadType::kAttributeBrick);
  auto bs = makeBitWriter(std::back_inserter(*buf));

  int abh_temporal_id = 0;
  bs.writeUn(4, abh.attr_attr_parameter_set_id);
  bs.writeUn(3, abh_temporal_id);
  bs.writeUe(abh.attr_sps_attr_idx);
  bs.writeUe(abh.attr_geom_slice_id);

  if (aps.aps_slice_dist2_deltas_present_flag || aps.attrInterPredictionEnabled)
    bs.writeSe(abh.attr_dist2_delta);
  if(aps.enableMortonCodeScaling)
    bs.writeUe(abh.mortonScaleFactorLog2);

  assert(abh.attr_sps_attr_idx < sps.attributeSets.size());
  if (abh.lcpPresent(sps.attributeSets[abh.attr_sps_attr_idx], aps)) {
    assert(abh.attrLcpCoeffs.size() == aps.maxNumDetailLevels());
    int pred = 4;

    for (int i = 0; i < abh.attrLcpCoeffs.size(); i++) {
      int lcp_coeff_diff = abh.attrLcpCoeffs[i] - pred;
      pred = abh.attrLcpCoeffs[i];
      bs.writeSe(lcp_coeff_diff);
    }
  }

  if (abh.icpPresent(sps.attributeSets[abh.attr_sps_attr_idx], aps)) {
    assert(abh.icpCoeffs.size() == aps.maxNumDetailLevels());
    Vec3<int8_t> pred = {0, 4, 4};

    for (int i = 0; i < abh.icpCoeffs.size(); i++) {
      auto icp_coeff_diff = abh.icpCoeffs[i] - pred;
      pred = abh.icpCoeffs[i];
      // NB: only k > 1 is coded
      for (int k = 1; k < 3; k++)
        bs.writeSe(icp_coeff_diff[k]);
    }
  }

  if (aps.aps_slice_qp_deltas_present_flag) {
    bs.writeSe(abh.attr_qp_delta_luma);
    bs.writeSe(abh.attr_qp_delta_chroma);
  }

  bool attr_layer_qp_present_flag = !abh.attr_layer_qp_delta_luma.empty();
  bs.write(attr_layer_qp_present_flag);
  if (attr_layer_qp_present_flag) {
    int attr_num_qp_layers_minus1 = abh.attr_num_qp_layers_minus1();
    bs.writeUe(attr_num_qp_layers_minus1);
    for (int i = 0; i <= attr_num_qp_layers_minus1; i++) {
      bs.writeSe(abh.attr_layer_qp_delta_luma[i]);
      bs.writeSe(abh.attr_layer_qp_delta_chroma[i]);
    }
  }

  int attr_num_regions = abh.qpRegions.size();
  bs.writeUe(attr_num_regions);

  if (attr_num_regions)
    bs.writeUe(abh.attr_region_bits_minus1);

  for (int i = 0; i < attr_num_regions; i++) {
    // NB: only one region is currently permitted.
    auto& region = abh.qpRegions[i];

    auto attr_region_origin =
      toXyz(sps.geometry_axis_order, region.regionOrigin);

    auto attr_region_whd_minus1 =
      toXyz(sps.geometry_axis_order, region.regionSize - 1);

    auto regionBits = abh.attr_region_bits_minus1 + 1;
    bs.writeUn(regionBits, attr_region_origin.x());
    bs.writeUn(regionBits, attr_region_origin.y());
    bs.writeUn(regionBits, attr_region_origin.z());
    bs.writeUn(regionBits, attr_region_whd_minus1.x());
    bs.writeUn(regionBits, attr_region_whd_minus1.y());
    bs.writeUn(regionBits, attr_region_whd_minus1.z());
    bs.writeSe(region.attr_region_qp_offset[0]);
    if (sps.attributeSets[abh.attr_sps_attr_idx].attr_num_dimensions_minus1)
      bs.writeSe(region.attr_region_qp_offset[1]);
  }
 
  if (aps.attr_encoding == AttributeEncoding::kRAHTransform && !aps.rahtPredParams.integer_haar_enable_flag) {
    bs.write(abh.attr_raht_ac_coeff_qp_offset_preset());
    if (abh.attr_raht_ac_coeff_qp_offset_preset()) {
      const auto attr_num_raht_ac_coeff_qp_layers_minus1 =
        abh.attr_num_raht_ac_coeff_qp_layers_minus1();
      bs.writeUe(attr_num_raht_ac_coeff_qp_layers_minus1);
      for (auto i = 0; i <= attr_num_raht_ac_coeff_qp_layers_minus1; i++)
        for (auto coeffIdx = 0; coeffIdx < 7; coeffIdx++) {
          bs.writeSe(abh.attr_raht_ac_coeff_qp_delta_luma[i][coeffIdx]);
          bs.writeSe(abh.attr_raht_ac_coeff_qp_delta_chroma[i][coeffIdx]);
        }
    }
    bs.write(abh.is420);
    if (abh.is420) {
      int first420 = abh.first420 +1;
      bs.writeUe(first420);
    }
  }

  if (aps.attrInterPredictionEnabled) {
    bs.write(abh.enableAttrInterPred);
    if (abh.geomEnableBiInterPred) {
      bs.write(abh.enableAttrInterPred2);
    }
      
    if (abh.enableAttrInterPred && aps.raht_send_inter_filters
      && aps.attr_encoding == AttributeEncoding::kRAHTransform) {
      int numFilters = abh.RAHTFilterTaps.size();
      bs.writeUe(numFilters);
      for (int filteridx = 0; filteridx < numFilters; filteridx++) {
        int currenttap = abh.RAHTFilterTaps[filteridx];
        bs.writeSe(currenttap);
      }
    }
  }

  if (sps.layer_group_enabled_flag) {
	  bs.write(abh.subgroup_weight_adjustment_enabled_flag);
	  if (abh.subgroup_weight_adjustment_enabled_flag) {
		  bs.writeSe(abh.subgroup_weight_adj_coeff_a);
		  bs.writeSe(abh.subgroup_weight_adj_coeff_b);
	  }
  }
  bs.byteAlign();
}

//----------------------------------------------------------------------------

AttributeBrickHeader
parseAbhIds(const PayloadBuffer& buf)
{
  AttributeBrickHeader abh;
  assert(buf.type == PayloadType::kAttributeBrick);
  auto bs = makeBitReader(buf.begin(), buf.end());

  int abh_temporal_id;
  bs.readUn(4, &abh.attr_attr_parameter_set_id);
  bs.readUn(3, &abh_temporal_id);
  bs.readUe(&abh.attr_sps_attr_idx);
  bs.readUe(&abh.attr_geom_slice_id);

  /* NB: this function only decodes ids at the start of the header. */
  /* NB: do not attempt to parse any further */

  return abh;
}

//----------------------------------------------------------------------------

AttributeBrickHeader
parseAbh(
  const SequenceParameterSet& sps,
  const AttributeParameterSet& aps,
  const PayloadBuffer& buf,
  int* bytesRead)
{
  AttributeBrickHeader abh;
  assert(buf.type == PayloadType::kAttributeBrick);
  auto bs = makeBitReader(buf.begin(), buf.end());

  int abh_temporal_id;
  bs.readUn(4, &abh.attr_attr_parameter_set_id);
  bs.readUn(3, &abh_temporal_id);
  bs.readUe(&abh.attr_sps_attr_idx);
  bs.readUe(&abh.attr_geom_slice_id);

  abh.attr_dist2_delta = 0;
  abh.mortonScaleFactorLog2 = 0;
  if (aps.aps_slice_dist2_deltas_present_flag || aps.attrInterPredictionEnabled)
    bs.readSe(&abh.attr_dist2_delta);
  if(aps.enableMortonCodeScaling)
    bs.readUe(&abh.mortonScaleFactorLog2);

  assert(abh.attr_sps_attr_idx < sps.attributeSets.size());
  if (abh.lcpPresent(sps.attributeSets[abh.attr_sps_attr_idx], aps)) {
    abh.attrLcpCoeffs.resize(aps.maxNumDetailLevels(), 0);
    int pred = 4;

    for (int i = 0; i < abh.attrLcpCoeffs.size(); i++) {
      auto& lcp_coeff_diff = abh.attrLcpCoeffs[i];
      bs.readSe(&lcp_coeff_diff);
      abh.attrLcpCoeffs[i] += pred;
      pred = abh.attrLcpCoeffs[i];
    }
  }

  if (abh.icpPresent(sps.attributeSets[abh.attr_sps_attr_idx], aps)) {
    abh.icpCoeffs.resize(aps.maxNumDetailLevels(), 0);
    Vec3<int8_t> pred{0, 4, 4};

    for (int i = 0; i < abh.icpCoeffs.size(); i++) {
      auto& icp_coeff_diff = abh.icpCoeffs[i];
      for (int k = 1; k < 3; k++)
        bs.readSe(&icp_coeff_diff[k]);

      // NB: pred[0] is always 0.
      abh.icpCoeffs[i] += pred;
      pred = abh.icpCoeffs[i];
    }
  }

  if (aps.aps_slice_qp_deltas_present_flag) {
    bs.readSe(&abh.attr_qp_delta_luma);
    bs.readSe(&abh.attr_qp_delta_chroma);
  }

  bool attr_layer_qp_present_flag;
  bs.read(&attr_layer_qp_present_flag);
  if (attr_layer_qp_present_flag) {
    int attr_num_qp_layers_minus1;
    bs.readUe(&attr_num_qp_layers_minus1);
    abh.attr_layer_qp_delta_luma.resize(attr_num_qp_layers_minus1 + 1);
    abh.attr_layer_qp_delta_chroma.resize(attr_num_qp_layers_minus1 + 1);
    for (int i = 0; i <= attr_num_qp_layers_minus1; i++) {
      bs.readSe(&abh.attr_layer_qp_delta_luma[i]);
      bs.readSe(&abh.attr_layer_qp_delta_chroma[i]);
    }
  }

  // NB: Number of regions is restricted in this version of specification.
  int attr_num_regions;
  bs.readUe(&attr_num_regions);
  assert(attr_num_regions <= 1);

  if (attr_num_regions)
    bs.readUe(&abh.attr_region_bits_minus1);

  abh.qpRegions.resize(attr_num_regions);
  for (int i = 0; i < attr_num_regions; i++) {
    auto regionBits = abh.attr_region_bits_minus1 + 1;
    auto& region = abh.qpRegions[i];

    Vec3<int> attr_region_origin;
    bs.readUn(regionBits, &attr_region_origin.x());
    bs.readUn(regionBits, &attr_region_origin.y());
    bs.readUn(regionBits, &attr_region_origin.z());
    region.regionOrigin = fromXyz(sps.geometry_axis_order, attr_region_origin);

    Vec3<int> attr_region_whd_minus1;
    bs.readUn(regionBits, &attr_region_whd_minus1.x());
    bs.readUn(regionBits, &attr_region_whd_minus1.y());
    bs.readUn(regionBits, &attr_region_whd_minus1.z());
    region.regionSize =
      fromXyz(sps.geometry_axis_order, attr_region_whd_minus1 + 1);

    bs.readSe(&region.attr_region_qp_offset[0]);
    if (sps.attributeSets[abh.attr_sps_attr_idx].attr_num_dimensions_minus1)
      bs.readSe(&region.attr_region_qp_offset[1]);
  }
  abh.is420 = false;
  abh.first420 = -1;
  
  if (aps.attr_encoding == AttributeEncoding::kRAHTransform && !aps.rahtPredParams.integer_haar_enable_flag) {
    bool raht_ac_coeff_qp_offset_present;
    bs.read(&raht_ac_coeff_qp_offset_present);
    if (raht_ac_coeff_qp_offset_present) {
      int attr_num_raht_ac_coeff_qp_layers_minus1;
      bs.readUe(&attr_num_raht_ac_coeff_qp_layers_minus1);
      abh.resizeRahtAcCoeffQpOffset(attr_num_raht_ac_coeff_qp_layers_minus1 + 1);
      for (auto i = 0; i <= attr_num_raht_ac_coeff_qp_layers_minus1; i++)
        for (auto coeffIdx = 0; coeffIdx < 7; coeffIdx++) {
          bs.readSe(&abh.attr_raht_ac_coeff_qp_delta_luma[i][coeffIdx]);
          bs.readSe(&abh.attr_raht_ac_coeff_qp_delta_chroma[i][coeffIdx]);
        }
    }
    bs.read(&abh.is420);
    if (abh.is420) {
      bs.readUe(&abh.first420);
      abh.first420--;
    }
  }
  abh.enableAttrInterPred2 = false;
  abh.enableAttrInterPred = false;
  if (aps.attrInterPredictionEnabled) {
    bs.read(&abh.enableAttrInterPred);
    if (abh.geomEnableBiInterPred)
      bs.read(&abh.enableAttrInterPred2);
    
    if (aps.raht_send_inter_filters && abh.enableAttrInterPred
      && aps.attr_encoding == AttributeEncoding::kRAHTransform) {
      int numFilters = 0;
      bs.readUe(&numFilters);
      abh.RAHTFilterTaps.resize(numFilters);
      for (int i = 0; i < numFilters; i++) {
        bs.readSe(&abh.RAHTFilterTaps[i]);
      }
    }
  }

 
  if (sps.layer_group_enabled_flag) {
	  bs.read(&abh.subgroup_weight_adjustment_enabled_flag);
	  if (abh.subgroup_weight_adjustment_enabled_flag) {
		  bs.readSe(&abh.subgroup_weight_adj_coeff_a);
		  bs.readSe(&abh.subgroup_weight_adj_coeff_b);
	  }
	  else {
		  abh.subgroup_weight_adj_coeff_a = 0;
		  abh.subgroup_weight_adj_coeff_b = 0;
	  }
  }

  bs.byteAlign();

  if (bytesRead)
    *bytesRead = int(std::distance(buf.begin(), bs.pos()));

  return abh;
}

//----------------------------------------------------------------------------

void
write(
  const SequenceParameterSet& sps,
  const AttributeParameterSet& aps,
  const GeometryBrickHeader& gbh,
  const DependentAttributeDataUnitHeader& dep_abh,
  PayloadBuffer* buf)
{
  assert(buf->type == PayloadType::kDependentAttributeDataUnit);
  auto bs = makeBitWriter(std::back_inserter(*buf));

  int abh_reserved_zero_3bits = 0;
  bs.writeUn(4, dep_abh.attr_attr_parameter_set_id);
  bs.writeUe(dep_abh.attr_sps_attr_idx);
  bs.writeUe(dep_abh.attr_geom_slice_id);

	
	bs.writeUn(8, dep_abh.layer_group_id);
	if (gbh.subgroup_enabled_flag[dep_abh.layer_group_id]) {
		bs.writeUn(16, dep_abh.subgroup_id);
		
	}
	
	if (aps.attr_ref_id_present_flag)
		bs.writeUn(8, dep_abh.ref_layer_group_id);

	if (gbh.subgroup_enabled_flag[dep_abh.layer_group_id]) {

		if (aps.attr_ref_id_present_flag) {
			bs.writeUn(16, dep_abh.ref_subgroup_id);
			bs.write(dep_abh.context_reference_indication_flag);
		}
	}
  
  assert(dep_abh.attr_sps_attr_idx < sps.attributeSets.size());
  if (dep_abh.lcpPresent(sps.attributeSets[dep_abh.attr_sps_attr_idx], aps)) {
    assert(dep_abh.attrLcpCoeffs.size() >= gbh.num_layers_minus1[dep_abh.layer_group_id]+2);
    int pred = 4;

    for (int i = 0; i < gbh.num_layers_minus1[dep_abh.layer_group_id]+2; i++) {
      int lcp_coeff_diff = dep_abh.attrLcpCoeffs[i] - pred;
      pred = dep_abh.attrLcpCoeffs[i];
      bs.writeSe(lcp_coeff_diff);
    }
  }

  
  if (dep_abh.icpPresent(sps.attributeSets[dep_abh.attr_sps_attr_idx], aps)) {
    assert(dep_abh.icpCoeffs.size() >= gbh.num_layers_minus1[dep_abh.layer_group_id]+2);
    Vec3<int8_t> pred = {0, 4, 4};

    for (int i = 0; i < gbh.num_layers_minus1[dep_abh.layer_group_id]+2; i++) {
      auto icp_coeff_diff = dep_abh.icpCoeffs[i] - pred;
      pred = dep_abh.icpCoeffs[i];
      // NB: only k > 1 is coded
      for (int k = 1; k < 3; k++)
        bs.writeSe(icp_coeff_diff[k]);
    }
  }
  
  if (aps.aps_slice_qp_deltas_present_flag){
    bs.writeSe(dep_abh.attr_qp_delta_luma);
    bs.writeSe(dep_abh.attr_qp_delta_chroma);
  }

  bool attr_layer_qp_present_flag = !dep_abh.attr_layer_qp_delta_luma.empty();
  bs.write(attr_layer_qp_present_flag);
  if (attr_layer_qp_present_flag) {
    int attr_num_qp_layers_minus1 = dep_abh.attr_num_qp_layers_minus1();
    bs.writeUe(attr_num_qp_layers_minus1);
    for (int i = 0; i <= attr_num_qp_layers_minus1; i++) {
      bs.writeSe(dep_abh.attr_layer_qp_delta_luma[i]);
      bs.writeSe(dep_abh.attr_layer_qp_delta_chroma[i]);
    }
  }
  
  
	bs.write(dep_abh.subgroup_weight_adjustment_enabled_flag);
	if (dep_abh.subgroup_weight_adjustment_enabled_flag) {
		bs.writeSe(dep_abh.subgroup_weight_adj_coeff_a);
		bs.writeSe(dep_abh.subgroup_weight_adj_coeff_b);
	}

  

	bs.byteAlign();
}

//----------------------------------------------------------------------------

DependentAttributeDataUnitHeader
parseDepAbhIds(const PayloadBuffer& buf)
{
  DependentAttributeDataUnitHeader abh;
  assert(buf.type == PayloadType::kDependentAttributeDataUnit);
  auto bs = makeBitReader(buf.begin(), buf.end());

  int abh_reserved_zero_3bits;
  bs.readUn(4, &abh.attr_attr_parameter_set_id);
  bs.readUe(&abh.attr_sps_attr_idx);
  bs.readUe(&abh.attr_geom_slice_id);

  /* NB: this function only decodes ids at the start of the header. */
  /* NB: do not attempt to parse any further */

  return abh;
}

//============================================================================

DependentAttributeDataUnitHeader
parseDepAbh(
  const SequenceParameterSet& sps,
  const AttributeParameterSet& aps,
  //const AttributeBrickHeader& abh,
  const GeometryBrickHeader& gbh,
  const PayloadBuffer& buf,
  int* bytesRead)
{
  DependentAttributeDataUnitHeader dep_abh;
  assert(buf.type == PayloadType::kDependentAttributeDataUnit);
  auto bs = makeBitReader(buf.begin(), buf.end());

  bs.readUn(4, &dep_abh.attr_attr_parameter_set_id);
  bs.readUe(&dep_abh.attr_sps_attr_idx);
  bs.readUe(&dep_abh.attr_geom_slice_id);
  
  bs.readUn(8, &dep_abh.layer_group_id);
    if (gbh.subgroup_enabled_flag[dep_abh.layer_group_id]) {
	    bs.readUn(16, &dep_abh.subgroup_id);
    }
	  else
		  dep_abh.subgroup_id = 0;

    
	if (aps.attr_ref_id_present_flag)
		bs.readUn(8, &dep_abh.ref_layer_group_id);

	if (gbh.subgroup_enabled_flag[dep_abh.layer_group_id]) {
		if (aps.attr_ref_id_present_flag) {
			bs.readUn(16, &dep_abh.ref_subgroup_id);
			bs.read(&dep_abh.context_reference_indication_flag);
		}
		else
			dep_abh.context_reference_indication_flag = false;
	}
  
  assert(dep_abh.attr_sps_attr_idx < sps.attributeSets.size());
  if (dep_abh.lcpPresent(sps.attributeSets[dep_abh.attr_sps_attr_idx], aps)) {
    dep_abh.attrLcpCoeffs.resize(gbh.num_layers_minus1[dep_abh.layer_group_id]+2, 0);
    int pred = 4;

    for (int i = 0; i < dep_abh.attrLcpCoeffs.size(); i++) {
      auto& lcp_coeff_diff = dep_abh.attrLcpCoeffs[i];
      bs.readSe(&lcp_coeff_diff);
      dep_abh.attrLcpCoeffs[i] += pred;
      pred = dep_abh.attrLcpCoeffs[i];
    }
  }
  
  if (dep_abh.icpPresent(sps.attributeSets[dep_abh.attr_sps_attr_idx], aps)) {
    dep_abh.icpCoeffs.resize(gbh.num_layers_minus1[dep_abh.layer_group_id]+2, 0);
    Vec3<int8_t> pred{0, 4, 4};

    for (int i = 0; i < dep_abh.icpCoeffs.size(); i++) {
      auto& icp_coeff_diff = dep_abh.icpCoeffs[i];
      for (int k = 1; k < 3; k++)
        bs.readSe(&icp_coeff_diff[k]);

      // NB: pred[0] is always 0.
      dep_abh.icpCoeffs[i] += pred;
      pred = dep_abh.icpCoeffs[i];
    }
  }
  
  if (aps.aps_slice_qp_deltas_present_flag) {
    bs.readSe(&dep_abh.attr_qp_delta_luma);
    bs.readSe(&dep_abh.attr_qp_delta_chroma);
  }

  bool attr_layer_qp_present_flag;
  bs.read(&attr_layer_qp_present_flag);
  if (attr_layer_qp_present_flag) {
    int attr_num_qp_layers_minus1;
    bs.readUe(&attr_num_qp_layers_minus1);
    dep_abh.attr_layer_qp_delta_luma.resize(attr_num_qp_layers_minus1 + 1);
    dep_abh.attr_layer_qp_delta_chroma.resize(attr_num_qp_layers_minus1 + 1);
    for (int i = 0; i <= attr_num_qp_layers_minus1; i++) {
      bs.readSe(&dep_abh.attr_layer_qp_delta_luma[i]);
      bs.readSe(&dep_abh.attr_layer_qp_delta_chroma[i]);
    }
  }

	bs.read(&dep_abh.subgroup_weight_adjustment_enabled_flag);
	if (dep_abh.subgroup_weight_adjustment_enabled_flag) {
		bs.readSe(&dep_abh.subgroup_weight_adj_coeff_a);
		bs.readSe(&dep_abh.subgroup_weight_adj_coeff_b);
	}
	else {
		dep_abh.subgroup_weight_adj_coeff_a = 0;
		dep_abh.subgroup_weight_adj_coeff_b = 0;
	}


  bs.byteAlign();

  if (bytesRead)
    *bytesRead = int(std::distance(buf.begin(), bs.pos()));

  return dep_abh;
}
//============================================================================

ConstantAttributeDataUnit
parseConstantAttribute(
  const SequenceParameterSet& sps, const PayloadBuffer& buf)
{
  ConstantAttributeDataUnit cadu;
  assert(buf.type == PayloadType::kConstantAttribute);
  auto bs = makeBitReader(buf.begin(), buf.end());

  bs.readUn(4, &cadu.constattr_attr_parameter_set_id);
  bs.readUe(&cadu.constattr_sps_attr_idx);
  bs.readUe(&cadu.constattr_geom_slice_id);

  // todo(df): check bounds
  const auto& attrDesc = sps.attributeSets[cadu.constattr_sps_attr_idx];

  cadu.constattr_default_value.resize(attrDesc.attr_num_dimensions_minus1 + 1);
  bs.readUn(attrDesc.bitdepth, &cadu.constattr_default_value[0]);
  for (int k = 1; k <= attrDesc.attr_num_dimensions_minus1; k++)
    bs.readUn(attrDesc.bitdepth, &cadu.constattr_default_value[k]);

  return cadu;
}

//============================================================================

void
write(
  const SequenceParameterSet& sps,
  const FrameBoundaryMarker& fbm,
  PayloadBuffer* buf)
{
  assert(buf->type == PayloadType::kFrameBoundaryMarker);
  auto bs = makeBitWriter(std::back_inserter(*buf));

  int fbdu_frame_ctr_lsb_bits = sps.frame_ctr_bits;
  bs.writeUn(5, fbdu_frame_ctr_lsb_bits);
  bs.writeUn(fbdu_frame_ctr_lsb_bits, fbm.fbdu_frame_ctr_lsb);
  bs.byteAlign();
}

//----------------------------------------------------------------------------

FrameBoundaryMarker
parseFrameBoundaryMarker(const PayloadBuffer& buf)
{
  FrameBoundaryMarker fbm;
  assert(buf.type == PayloadType::kFrameBoundaryMarker);
  auto bs = makeBitReader(buf.begin(), buf.end());

  int fbdu_frame_ctr_lsb_bits;
  bs.readUn(5, &fbdu_frame_ctr_lsb_bits);
  bs.readUn(fbdu_frame_ctr_lsb_bits, &fbm.fbdu_frame_ctr_lsb);

  return fbm;
}

//============================================================================

PayloadBuffer
write(const SequenceParameterSet& sps, const TileInventory& inventory)
{
  PayloadBuffer buf(PayloadType::kTileInventory);
  auto bs = makeBitWriter(std::back_inserter(buf));

  bs.writeUn(4, inventory.ti_seq_parameter_set_id);

  bs.writeUn(5, inventory.ti_frame_ctr_bits);
  bs.writeUn(inventory.ti_frame_ctr_bits, inventory.ti_frame_ctr);

  int num_tiles = inventory.tiles.size();
  bs.writeUn(16, num_tiles);
  if (!num_tiles) {
    bs.byteAlign();
    return buf;
  }

  bs.writeUn(5, inventory.tile_id_bits);
  bs.writeUn(8, inventory.tile_origin_bits_minus1);
  bs.writeUn(8, inventory.tile_size_bits_minus1);

  for (const auto& entry : inventory.tiles) {
    bs.writeUn(inventory.tile_id_bits, entry.tile_id);

    auto tile_origin = toXyz(sps.geometry_axis_order, entry.tileOrigin);
    if (auto tileOriginBits = inventory.tile_origin_bits_minus1 + 1) {
      bs.writeSn(tileOriginBits, tile_origin.x());
      bs.writeSn(tileOriginBits, tile_origin.y());
      bs.writeSn(tileOriginBits, tile_origin.z());
    }

    auto tile_size_minus1 = toXyz(sps.geometry_axis_order, entry.tileSize) - 1;
    if (auto tileSizeBits = inventory.tile_size_bits_minus1 + 1) {
      bs.writeUn(tileSizeBits, tile_size_minus1.x());
      bs.writeUn(tileSizeBits, tile_size_minus1.y());
      bs.writeUn(tileSizeBits, tile_size_minus1.z());
    }
  }

  // NB: this is at the end of the inventory to aid fixed-width parsing
  bs.writeUe(inventory.ti_origin_bits_minus1);
  if (auto tiOriginBits = inventory.ti_origin_bits_minus1 + 1) {
    auto ti_origin_xyz = toXyz(sps.geometry_axis_order, inventory.origin);
    bs.writeSn(tiOriginBits, ti_origin_xyz.x());
    bs.writeSn(tiOriginBits, ti_origin_xyz.y());
    bs.writeSn(tiOriginBits, ti_origin_xyz.z());
  }

  int ti_origin_log2_scale = 0;
  bs.writeUe(ti_origin_log2_scale);

  bs.byteAlign();

  return buf;
}

//----------------------------------------------------------------------------

TileInventory
parseTileInventory(const PayloadBuffer& buf)
{
  TileInventory inventory;
  assert(buf.type == PayloadType::kTileInventory);
  auto bs = makeBitReader(buf.begin(), buf.end());

  bs.readUn(4, &inventory.ti_seq_parameter_set_id);

  bs.readUn(5, &inventory.ti_frame_ctr_bits);
  bs.readUn(inventory.ti_frame_ctr_bits, &inventory.ti_frame_ctr);

  int num_tiles;
  bs.readUn(16, &num_tiles);
  if (!num_tiles) {
    bs.byteAlign();
    return inventory;
  }

  bs.readUn(5, &inventory.tile_id_bits);
  bs.readUn(8, &inventory.tile_origin_bits_minus1);
  bs.readUn(8, &inventory.tile_size_bits_minus1);

  for (int i = 0; i < num_tiles; i++) {
    int tile_id = i;
    bs.readUn(inventory.tile_id_bits, &tile_id);

    Vec3<int> tile_origin;
    if (auto tileOriginBits = inventory.tile_origin_bits_minus1 + 1) {
      bs.readSn(tileOriginBits, &tile_origin.x());
      bs.readSn(tileOriginBits, &tile_origin.y());
      bs.readSn(tileOriginBits, &tile_origin.z());
    }

    Vec3<int> tile_size_minus1;
    if (auto tileSizeBits = inventory.tile_size_bits_minus1 + 1) {
      bs.readUn(tileSizeBits, &tile_size_minus1.x());
      bs.readUn(tileSizeBits, &tile_size_minus1.y());
      bs.readUn(tileSizeBits, &tile_size_minus1.z());
    }
    // NB: this is in XYZ axis order until the inventory is converted to STV
    TileInventory::Entry entry;
    entry.tile_id = tile_id;
    entry.tileOrigin = tile_origin;
    entry.tileSize = tile_size_minus1 + 1;
    inventory.tiles.push_back(entry);
  }

  Vec3<int> ti_origin_xyz;
  bs.readUe(&inventory.ti_origin_bits_minus1);
  if (auto tiOriginBits = inventory.ti_origin_bits_minus1 + 1) {
    bs.readSn(tiOriginBits, &ti_origin_xyz.x());
    bs.readSn(tiOriginBits, &ti_origin_xyz.y());
    bs.readSn(tiOriginBits, &ti_origin_xyz.z());
  }

  int ti_origin_log2_scale;
  bs.readUe(&ti_origin_log2_scale);
  ti_origin_xyz *= 1 << ti_origin_log2_scale;

  // NB: this is in XYZ axis order until converted to STV
  inventory.origin = ti_origin_xyz;

  bs.byteAlign();

  return inventory;
}

//----------------------------------------------------------------------------

void
convertXyzToStv(const SequenceParameterSet& sps, TileInventory* inventory)
{
  for (auto& tile : inventory->tiles) {
    tile.tileOrigin = fromXyz(sps.geometry_axis_order, tile.tileOrigin);
    tile.tileSize = fromXyz(sps.geometry_axis_order, tile.tileSize);
  }
}

//============================================================================

PayloadBuffer
write(
  const SequenceParameterSet& sps,
  const AttributeParamInventoryHdr& inv,
  const AttributeParameters& params)
{
  PayloadBuffer buf(PayloadType::kGeneralizedAttrParamInventory);
  auto bs = makeBitWriter(std::back_inserter(buf));

  assert(inv.attr_param_seq_parameter_set_id == sps.sps_seq_parameter_set_id);
  bs.writeUn(4, inv.attr_param_seq_parameter_set_id);
  int attr_param_frame_ctr_lsb_bits = sps.frame_ctr_bits;
  bs.writeUn(5, attr_param_frame_ctr_lsb_bits);
  bs.writeUn(attr_param_frame_ctr_lsb_bits, inv.attr_param_frame_ctr_lsb);
  bs.writeUe(inv.attr_param_sps_attr_idx);

  assert(inv.attr_param_sps_attr_idx < sps.attributeSets.size());
  auto& desc = sps.attributeSets[inv.attr_param_sps_attr_idx];
  writeAttributeParameters(desc, bs, params);

  return buf;
}

//----------------------------------------------------------------------------

AttributeParamInventoryHdr
parseAttrParamInventoryHdr(const PayloadBuffer& buf)
{
  AttributeParamInventoryHdr inv;
  assert(buf.type == PayloadType::kGeneralizedAttrParamInventory);
  auto bs = makeBitReader(buf.begin(), buf.end());

  bs.readUn(4, &inv.attr_param_seq_parameter_set_id);
  int attr_param_frame_ctr_lsb_bits;
  bs.readUn(5, &attr_param_frame_ctr_lsb_bits);
  bs.readUn(attr_param_frame_ctr_lsb_bits, &inv.attr_param_frame_ctr_lsb);
  bs.readUe(&inv.attr_param_sps_attr_idx);

  return inv;
}

//----------------------------------------------------------------------------

AttributeParameters&
parseAttrParamInventory(
  const AttributeDescription& attr,
  const PayloadBuffer& buf,
  AttributeParameters& params)
{
  AttributeParamInventoryHdr inv;
  assert(buf.type == PayloadType::kGeneralizedAttrParamInventory);
  auto bs = makeBitReader(buf.begin(), buf.end());

  bs.readUn(4, &inv.attr_param_seq_parameter_set_id);
  int attr_param_frame_ctr_lsb_bits;
  bs.readUn(5, &attr_param_frame_ctr_lsb_bits);
  bs.readUn(attr_param_frame_ctr_lsb_bits, &inv.attr_param_frame_ctr_lsb);
  bs.readUe(&inv.attr_param_sps_attr_idx);

  int num_attr_parameters;
  bs.readUe(&num_attr_parameters);
  bs.byteAlign();
  for (auto i = 0; i < num_attr_parameters; i++)
    parseAttributeParameter(attr, bs, params);

  return params;
}

//============================================================================

PayloadBuffer
write(const UserData& ud)
{
  PayloadBuffer buf(PayloadType::kUserData);
  auto bs = makeBitWriter(std::back_inserter(buf));

  writeOid(bs, ud.user_data_oid);
  // todo(df): write a blob of userdata

  return buf;
}

//----------------------------------------------------------------------------

UserData
parseUserData(const PayloadBuffer& buf)
{
  UserData ud;
  assert(buf.type == PayloadType::kUserData);
  auto bs = makeBitReader(buf.begin(), buf.end());

  readOid(bs, &ud.user_data_oid);
  // todo(df): read the blob of userdata

  return ud;
}

//============================================================================
// Helpers for Global scaling

SequenceParameterSet::GlobalScale::GlobalScale(Rational x)
{
  // x must be representable using fixed point
  if (popcntGt1(uint32_t(x.denominator)))
    throw std::runtime_error("cannot convert Rational to GlobalScale");

  // Direct conversion
  denominatorLog2 = ilog2(uint32_t(x.numerator));
  numeratorModDenominator = x.numerator - (1 << denominatorLog2);
  numeratorMulLog2 = denominatorLog2 - ilog2(uint32_t(x.denominator));

  // Simplify: remove powers of two from numeratorModDenominator
  while (!(numeratorModDenominator & 1) && denominatorLog2) {
    numeratorModDenominator >>= 1;
    denominatorLog2--;
  }
}

//----------------------------------------------------------------------------

SequenceParameterSet::GlobalScale::operator Rational() const
{
  int numeratorPreMul = ((1 << denominatorLog2) + numeratorModDenominator);

  // Simplify 2^numeratorMulLog2 / 2^denominatorLog2
  int numeratorS = std::max(0, numeratorMulLog2 - denominatorLog2);
  int denominatorS = denominatorLog2 - (numeratorMulLog2 - numeratorS);

  // Simplify numeratorPreMul / 2^denominatorLog2
  while (!(numeratorPreMul & 1) && denominatorS) {
    numeratorPreMul >>= 1;
    denominatorS--;
  }

  return Rational(numeratorPreMul << numeratorS, 1 << denominatorS);
}

//============================================================================

}  // namespace pcc
