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

#include "PCCTMC3Decoder.h"

#include <algorithm>
#include <cassert>
#include <string>
#include <sstream>

#include "AttributeCommon.h"
#include "PayloadBuffer.h"
#include "PCCPointSet.h"
#include "coordinate_conversion.h"
#include "geometry.h"
#include "geometry_octree.h"
#include "geometry_predictive.h"
#include "hls.h"
#include "io_hls.h"
#include "io_tlv.h"
#include "pcc_chrono.h"
#include "osspecific.h"

namespace pcc {

//============================================================================

PCCTMC3Decoder3::PCCTMC3Decoder3(const DecoderParams& params) : _params(params)
{
  init();
}

//----------------------------------------------------------------------------

void
PCCTMC3Decoder3::init()
{
  _firstSliceInFrame = true;
  _outputInitialized = false;
  _suppressOutput = 1;
  _sps = nullptr;
  _gps = nullptr;
  _spss.clear();
  _gpss.clear();
  _apss.clear();
  _refFrameSeq.clear();
  _refFrameAltSeq.clear();
  biPredDecodeParams.init();

  _ctxtMemOctreeGeom.reset(new GeometryOctreeContexts);
  _ctxtMemPredGeom.reset(new PredGeomContexts);
}

//----------------------------------------------------------------------------

PCCTMC3Decoder3::~PCCTMC3Decoder3() = default;

//============================================================================

static bool
payloadStartsNewSlice(PayloadType type)
{
  return type == PayloadType::kGeometryBrick
    || type == PayloadType::kFrameBoundaryMarker;
}

//============================================================================

bool
PCCTMC3Decoder3::dectectFrameBoundary(const PayloadBuffer* buf)
{
  // This may be from either geometry brick or attr param inventory
  int frameCtrLsb;

  switch (buf->type) {
  case PayloadType::kFrameBoundaryMarker: {
    // the frame boundary data marker explcitly indicates a boundary
    // However, this implementation doesn't flush the output, rather
    // this happens naturally when the frame boundary is detected by
    // a change in frameCtr.
    auto fbm = parseFrameBoundaryMarker(*buf);
    frameCtrLsb = fbm.fbdu_frame_ctr_lsb;
    break;
  }

  case PayloadType::kGeometryBrick:
  case PayloadType::kGeometryBrickUnusedForRef: {
    activateParameterSets(parseGbhIds(*buf));
    auto gbh = parseGbh(*_sps, *_gps, *buf, nullptr, nullptr);
    frameCtrLsb = gbh.frame_ctr_lsb;
    break;
  }

  case PayloadType::kGeneralizedAttrParamInventory: {
    auto apih = parseAttrParamInventoryHdr(*buf);
    activateParameterSets(apih);
    // todo(conf): check lsb_bits is same as sps
    frameCtrLsb = apih.attr_param_frame_ctr_lsb;
    break;
  }

  // other data units don't indicate a boundary
  default: return false;
  }

  auto bdry = _frameCtr.isDifferentFrame(frameCtrLsb, _sps->frame_ctr_bits);
  if (bdry)
    _prevFrameLsb = _frameCtr & (1 << _sps->frame_ctr_bits - 1);
  _frameCtr.update(frameCtrLsb, _sps->frame_ctr_bits);

  return bdry;
}

//============================================================================

void
PCCTMC3Decoder3::outputCurrentCloud(PCCTMC3Decoder3::Callbacks* callback)
{
  if (_suppressOutput)
    return;

  std::swap(_outCloud.cloud, _accumCloud);


  // Apply global scaling to output for integer conformance
  // todo: add other output scaling modes
  // NB: if accumCloud is reused for future inter-prediction, global scaling
  //     must be applied to a copy.
  scaleGeometry(_outCloud.cloud, _sps->globalScale, _outCloud.outputFpBits);

  callback->onOutputCloud(_outCloud);

  std::swap(_outCloud.cloud, _accumCloud);
  _accumCloud.clear();
}

void
PCCTMC3Decoder3::storeCurrentCloudAsRef()
{
  if (_sps->inter_frame_prediction_enabled_flag && !_suppressOutput) {
    _refFrameSeq[_sps->sps_seq_parameter_set_id].cloud = _accumCloud;
    auto& altCloud = _refFrameAltSeq[_sps->sps_seq_parameter_set_id].cloud;
    altCloud.clear();
    altCloud.swap(_accumCloudAltPositions);
  }
}

void
PCCTMC3Decoder3::storeCurrentCloudAsBRef()
{
  if (_suppressOutput)
    return;

  if (_frameCtr == 0)
    _gbh.biPredictionEnabledFlag = false;

  if (!_gbh.biPredictionEnabledFlag) {
    biPredDecodeParams.refPointCloud2 = _accumCloud;
  } else {
    _refFrameSeq[_sps->sps_seq_parameter_set_id].cloud = _accumCloud;
  }

  if (_gps->biPredictionEnabledFlag == 2 && _gbh.biPredictionEnabledFlag)
    processHierarchicalGOFPost();
}

//============================================================================

void
PCCTMC3Decoder3::outputGOFCurrentCloud(PCCTMC3Decoder3::Callbacks* callback)
{
  if (!_gbh.biPredictionEnabledFlag) {
    if ((_outCloud.frameNum == 0) || ((_outCloud.frameNum - biPredDecodeParams.preOutFrameNum) == 1)) {
      compensateXYZ();
      outputCurrentCloud(callback);
      biPredDecodeParams.preOutFrameNum = _outCloud.frameNum;
    } else {
      _accumCloud.clear();
    }
    return;
  }

  if (_gps->biPredictionEnabledFlag == 1) {
    auto outputOrigin = _outCloud.outputOrigin;
    auto outputUnitLength = _outCloud.outputUnitLength;
    compensateXYZ();
    outputCurrentCloud(callback);
    biPredDecodeParams.preOutFrameNum = _outCloud.frameNum;
    if (_outCloud.frameNum + 1 == biPredDecodeParams.preIPFrame) {
      _accumCloud = biPredDecodeParams.refPointCloud2;
      _outCloud.frameNum = biPredDecodeParams.preIPFrame;
      _outCloud.outputOrigin = outputOrigin;
      _outCloud.outputUnitLength = outputUnitLength;
      compensateXYZ();
      outputCurrentCloud(callback);
      biPredDecodeParams.preOutFrameNum = _outCloud.frameNum;
    }
  } else {
    int outputFrameNum = biPredDecodeParams.preOutFrameNum + 1;
    int outputFrameIndexInGOF =
      outputFrameNum - biPredDecodeParams.prePreIPFrame;
    int currFrameNum = _outCloud.frameNum;

    const PCCPointSet3 _accumCloudTmp = _accumCloud;
    auto outputOrigin = _outCloud.outputOrigin;
    auto outputUnitLength = _outCloud.outputUnitLength;
    while (biPredDecodeParams.frameReadyForOutput(outputFrameIndexInGOF)) {
      _accumCloud =
        (outputFrameNum != currFrameNum
           ? hGOFDecodeParams.gof[outputFrameIndexInGOF]
           : _accumCloudTmp);

      if (outputFrameIndexInGOF != biPredDecodeParams.refTimesList.size() - 1)
        hGOFDecodeParams.clearFrame(outputFrameIndexInGOF);

      _outCloud.frameNum = outputFrameNum;
      _outCloud.outputOrigin = outputOrigin;
      _outCloud.outputUnitLength = outputUnitLength;
      compensateXYZ();
      outputCurrentCloud(callback);
      biPredDecodeParams.preOutFrameNum = _outCloud.frameNum;

      outputFrameNum++;
      outputFrameIndexInGOF++;
    }
  }
  _accumCloud.clear();
  return;
}

//============================================================================

void
PCCTMC3Decoder3::compensateXYZ()
{
  if (
    !_gps->geom_z_compensation_enabled_flag
    && !_gps->geom_xy_compensation_enabled_flag)
    return;

  auto plyScale = reciprocal(_sps->seqGeomScale);
  plyScale.numerator *= 1000;
  auto laserOrigin = _gps->gpsAngularOrigin;
  compensateXYZCoordinate(
    _accumCloud, _gps, plyScale, laserOrigin, _outCloud.outputUnitLength,
    _outCloud.outputOrigin);
}
//============================================================================

void
PCCTMC3Decoder3::startFrame()
{
  _outputInitialized = true;
  _firstSliceInFrame = true;
  _outCloud.frameNum = _frameCtr;

  // the following could be set once when the SPS is discovered
  _outCloud.setParametersFrom(*_sps, _params.outputFpBits);
  // create a new reference frame for the sps, if needed
  emplaceRefFrame(*_sps);
}

//============================================================================

void
PCCTMC3Decoder3::emplaceRefFrame(const SequenceParameterSet& sps)
{
  if (sps.inter_frame_prediction_enabled_flag) {
    _refFrameSeq.emplace(std::make_pair(sps.sps_seq_parameter_set_id, CloudFrame(_outCloud)));
    _refFrameAltSeq.emplace(std::make_pair(sps.sps_seq_parameter_set_id, CloudFrame(_outCloud)));
  }
}

//============================================================================

int
PCCTMC3Decoder3::decompress(
  const PayloadBuffer* buf, PCCTMC3Decoder3::Callbacks* callback)
{
  // Starting a new geometry brick/slice/tile, transfer any
  // finished points to the output accumulator
  if (!buf || payloadStartsNewSlice(buf->type)) {
    if (size_t numPoints = _currentPointCloud.getPointCount()) {
      for (size_t i = 0; i < numPoints; i++)
        for (int k = 0; k < 3; k++)
          _currentPointCloud[i][k] += _sliceOrigin[k];
      _accumCloud.append(_currentPointCloud);

      bool hasAttr = _accumCloud.hasReflectances() || _accumCloud.hasColors()
        || _accumCloud.hasFrameIndex() || _accumCloud.hasLaserAngles();

      if (!hasAttr)
      {
        _storeAsSphe = _gps->geom_angular_mode_enabled_flag && _gps->predgeom_enabled_flag;
        _currentPointCloudAttr = _currentPointCloud;
      }
      
      if (_storeAsSphe)
        _accumCloudAltPositions.append(_currentPointCloudAttr, _posSph);
      else
        _accumCloudAltPositions.append(_currentPointCloudAttr);
    }
  }

  if (!buf) {
    // flush decoder, output pending cloud if any
    if (_gps->biPredictionEnabledFlag)
      outputGOFCurrentCloud(callback);
    else{ 
      if(_sps->layer_group_enabled_flag){  
        _gHandler.releaseNodes();
        _gHandler.releaseCtxForGeometry();
        _gHandler.releaseCtxForAttribute();
        _gHandler.releaseIndexes();

        _subgroupPointCloud.clear();      
      }
      compensateXYZ();
      outputCurrentCloud(callback);
    }
    return 0;
  }

  // process a frame boundary
  //  - this may update FrameCtr
  //  - this will activate the sps for GeometryBrick and AttrParamInventory
  //  - after outputing the current frame, the output must be reinitialized
  if (dectectFrameBoundary(buf)) {
    if (_gps->biPredictionEnabledFlag) {
      storeCurrentCloudAsBRef();
      outputGOFCurrentCloud(callback);
    } else {

      if(_sps->layer_group_enabled_flag){  
        _gHandler.releaseNodes();
        _gHandler.releaseCtxForGeometry();
        _gHandler.releaseCtxForAttribute();
        _gHandler.releaseIndexes();

        _subgroupPointCloud.clear();      
      }

      storeCurrentCloudAsRef();
      compensateXYZ();
      outputCurrentCloud(callback);
    }
    _outputInitialized = false;
  }

  // process the buffer
  switch (buf->type) {
  case PayloadType::kSequenceParameterSet: {
    auto sps = parseSps(*buf);
    convertXyzToStv(&sps);
    storeSps(std::move(sps));
    return 0;
  }

  case PayloadType::kGeometryParameterSet: {
    auto gps = parseGps(*buf);
    // HACK: assume that an SPS has been received prior to the GPS.
    // This is not required, and parsing of the GPS is independent of the SPS.
    // todo(df): move GPS fixup to activation process
    _sps = &_spss.cbegin()->second;
    convertXyzToStv(*_sps, &gps);
    storeGps(std::move(gps));
    return 0;
  }

  case PayloadType::kAttributeParameterSet: {
    auto aps = parseAps(*buf,*_sps);
    aps.rahtPredParams.raht_last_component_prediction_enabled_flag =
      aps.last_component_prediction_enabled_flag;
    // HACK: assume that an SPS has been received prior to the APS.
    // This is not required, and parsing of the APS is independent of the SPS.
    // todo(df): move APS fixup to activation process
    _sps = &_spss.cbegin()->second;
    convertXyzToStv(*_sps, &aps);
    storeAps(std::move(aps));
    return 0;
  }

  case PayloadType::kFrameBoundaryMarker:
    if (!_outputInitialized)
      startFrame();
    return 0;

  case PayloadType::kGeometryBrick:
  case PayloadType::kGeometryBrickUnusedForRef:
    if (!_outputInitialized)
      startFrame();

    // avoid accidents with stale attribute decoder on next slice
    _attrDecoder.reset();
    // Avoid dropping an actual frame
    _suppressOutput = false;

    return decodeGeometryBrick(*buf);

  case PayloadType::kAttributeBrick:
  case PayloadType::kAttributeBrickUnusedForRef:
    decodeAttributeBrick(*buf);
    return 0;

  case PayloadType::kConstantAttribute:
    decodeConstantAttribute(*buf);
    return 0;

  case PayloadType::kTileInventory:
    // NB: the tile inventory is decoded in xyz order.  It may need
    //     conversion if it is used (it currently isn't).
    storeTileInventory(parseTileInventory(*buf));
    return 0;

  case PayloadType::kGeneralizedAttrParamInventory: {
    if (!_outputInitialized)
      startFrame();

    auto hdr = parseAttrParamInventoryHdr(*buf);
    assert(hdr.attr_param_sps_attr_idx < int(_sps->attributeSets.size()));
    auto& attrDesc = _outCloud.attrDesc[hdr.attr_param_sps_attr_idx];
    parseAttrParamInventory(attrDesc, *buf, attrDesc.params);
    return 0;
  }

  case PayloadType::kUserData: parseUserData(*buf); return 0;

  case PayloadType::kDependentGeometryDataUnit:
    return decodeDependentGeometryBrick(*buf);

  case PayloadType::kDependentAttributeDataUnit:
    decodeDependentAttributeBrick(*buf);
    return 0;

  case PayloadType::kLayerGroupStructureInventory:
	  storeLayerGroupStructureInventory(parseLayerGroupStructureInventory(*buf));
	  return 0;
  }

  // todo(df): error, unhandled payload type
  return 1;
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::storeSps(SequenceParameterSet&& sps)
{
  // todo(df): handle replacement semantics
  _spss.emplace(std::make_pair(sps.sps_seq_parameter_set_id, sps));
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::storeGps(GeometryParameterSet&& gps)
{
  // todo(df): handle replacement semantics
  _gpss.emplace(std::make_pair(gps.gps_geom_parameter_set_id, gps));
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::storeAps(AttributeParameterSet&& aps)
{
  // todo(df): handle replacement semantics
  _apss.emplace(std::make_pair(aps.aps_attr_parameter_set_id, aps));
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::storeTileInventory(TileInventory&& inventory)
{
  // todo(df): handle replacement semantics
  _tileInventory = inventory;
}

//==========================================================================

void
PCCTMC3Decoder3::storeLayerGroupStructureInventory(
	LayerGroupStructureInventory&& inventory)
{
	_layerGroupStructureInventory = inventory;
}

//==========================================================================

void
PCCTMC3Decoder3::activateParameterSets(const GeometryBrickHeader& gbh)
{
  // HACK: assume activation of the first SPS and GPS
  // todo(df): parse brick header here for propper sps & gps activation
  //  -- this is currently inconsistent between trisoup and octree
  assert(!_spss.empty());
  assert(!_gpss.empty());
  _sps = &_spss.cbegin()->second;
  _gps = &_gpss.cbegin()->second;

  _refFrame = _sps->inter_frame_prediction_enabled_flag
    ? &_refFrameSeq[_sps->sps_seq_parameter_set_id]
    : nullptr;
  _refFrameAlt = _sps->inter_frame_prediction_enabled_flag
    ? &_refFrameAltSeq[_sps->sps_seq_parameter_set_id]
    : nullptr;
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::activateParameterSets(const AttributeParamInventoryHdr& hdr)
{
  // HACK: assume activation of the first SPS and GPS
  // todo(df): parse brick header here for propper sps & gps activation
  //  -- this is currently inconsistent between trisoup and octree
  assert(!_spss.empty());
  assert(!_gpss.empty());
  _sps = &_spss.cbegin()->second;
  _gps = &_gpss.cbegin()->second;

  _refFrame = _sps->inter_frame_prediction_enabled_flag
    ? &_refFrameSeq[_sps->sps_seq_parameter_set_id]
    : nullptr;
  _refFrameAlt = _sps->inter_frame_prediction_enabled_flag
    ? &_refFrameAltSeq[_sps->sps_seq_parameter_set_id]
    : nullptr;
}
//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::processHierarchicalGOF()
{
  auto& refCloud1 = attrInterPredParams.referencePointCloud;
  auto& refCloud2 =
    biPredDecodeParams.attrInterPredParams2.referencePointCloud;

  std::vector<point_t>& refPosSph1 = _refPosSph;
  std::vector<point_t>& refPosSph2 = biPredDecodeParams._refPosSph2; 

  PredGeomPredictor& refFrameSph1 = _refFrameSph; 
  PredGeomPredictor& refFrameSph2 = biPredDecodeParams._refFrameSph2;

  if (_gbh.biPredictionEnabledFlag) {
    const auto deltaIPFrame =
      biPredDecodeParams.preIPFrame - biPredDecodeParams.prePreIPFrame;
    if ((deltaIPFrame - 1) != hGOFDecodeParams.codeOrderList.size()) {
      hGOFDecodeParams.reInitializeLists(deltaIPFrame);
    }
    if (!biPredDecodeParams.preFrameAsBframe) {
      biPredDecodeParams.refTimesList = hGOFDecodeParams.refTimesList;
    }
  }

  if (!_gbh.biPredictionEnabledFlag) {
    if (hGOFDecodeParams.gof.size()) {
      biPredDecodeParams.refPointCloud2 = hGOFDecodeParams.gof.back();
      refCloud2 = hGOFDecodeParams.gof_spherical.back();
      refPosSph2 = hGOFDecodeParams.gof_posSph.back();
      hGOFDecodeParams.clearGofs();
    }
  } else {
    if (!hGOFDecodeParams.gof.size()) {
      hGOFDecodeParams.initializeGof(
        biPredDecodeParams.refTimesList.size(),
        _refFrameSeq[_sps->sps_seq_parameter_set_id].cloud,
        biPredDecodeParams.refPointCloud2, refCloud1, refCloud2, refPosSph1, refPosSph2);

      biPredDecodeParams.currFrameInGOF = 0;
    }

    biPredDecodeParams.currFrameIndexInGOF =
      hGOFDecodeParams.codeOrderList[biPredDecodeParams.currFrameInGOF];
    int _preRefFrame =
      hGOFDecodeParams.refFrameList[biPredDecodeParams.currFrameInGOF * 2];
    int _backRefFrame =
      hGOFDecodeParams.refFrameList[biPredDecodeParams.currFrameInGOF * 2 + 1];

    hGOFDecodeParams.updateReferenceFrames(
      _refFrameSeq[_sps->sps_seq_parameter_set_id].cloud,
      biPredDecodeParams.refPointCloud2, refCloud1, refCloud2, refPosSph1, refPosSph2,
      refFrameSph1, refFrameSph2, _preRefFrame,_backRefFrame);

    biPredDecodeParams.decrementRefTimesList(
      _preRefFrame, _backRefFrame, biPredDecodeParams.currFrameIndexInGOF);

    biPredDecodeParams.currFrameInGOF++;
  }
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::processHierarchicalGOFPost()
{
  hGOFDecodeParams.storeReferenceFrame(
    biPredDecodeParams.currFrameIndexInGOF,
    _refFrameSeq[_sps->sps_seq_parameter_set_id].cloud,
    attrInterPredParams.referencePointCloud, _refPosSph);
}
//==========================================================================
// Initialise the point cloud storage and decode a single geometry slice.

int
PCCTMC3Decoder3::decodeGeometryBrick(const PayloadBuffer& buf)
{
  assert(buf.type == PayloadType::kGeometryBrick);
  std::cout << "positions bitstream size " << buf.size() << " B\n";

  // todo(df): replace with attribute mapping
  bool hasColour = std::any_of(
    _sps->attributeSets.begin(), _sps->attributeSets.end(),
    [](const AttributeDescription& desc) {
      return desc.attributeLabel == KnownAttributeLabel::kColour;
    });

  bool hasReflectance = std::any_of(
    _sps->attributeSets.begin(), _sps->attributeSets.end(),
    [](const AttributeDescription& desc) {
      return desc.attributeLabel == KnownAttributeLabel::kReflectance;
    });

  _currentPointCloud.clear();
  _currentPointCloud.addRemoveAttributes(hasColour, hasReflectance);

  pcc::chrono::Stopwatch<pcc::chrono::utime_inc_children_clock> clock_user;
  clock_user.start();

  int gbhSize, gbfSize;
  _gbh = parseGbh(*_sps, *_gps, buf, &gbhSize, &gbfSize);
  _prevSliceId = _sliceId;
  _sliceId = _gbh.geom_slice_id;
  _sliceOrigin = _gbh.geomBoxOrigin;

  if (_frameCtr == 0) {
    _refFrameSph.setGlobalMotionEnabled(_gps->globalMotionEnabled);
    biPredDecodeParams._refFrameSph2.setGlobalMotionEnabled(_gps->globalMotionEnabled);
  } else if (_firstSliceInFrame) {
    if (_gps->biPredictionEnabledFlag) {
      if (_gps->biPredictionEnabledFlag == 2) {
        processHierarchicalGOF();
      }
      if (!_gbh.biPredictionEnabledFlag) {
        attrInterPredParams.referencePointCloud =
          biPredDecodeParams.attrInterPredParams2.referencePointCloud;
        _refFrameSeq[_sps->sps_seq_parameter_set_id].cloud =
          biPredDecodeParams.refPointCloud2;
      }
      if (_gps->predgeom_enabled_flag){
        if (!_gbh.biPredictionEnabledFlag){
          _refFrameSph.clearRefFrameCur();
          _refFrameSph.insert(biPredDecodeParams._refPosSph2);
          _refPosSph = biPredDecodeParams._refPosSph2;
        }
        if (_gps->globalMotionEnabled){
          _refFrameSph.setFrameMovingState(_gbh.interFrameRefGmcFlag);
          _refFrameSph.setMotionParams(
            _gbh.gm_param[0].gm_thres, _gbh.gm_param[0].gm_matrix,
            _gbh.gm_param[0].gm_trans);
          _refFrameSph.setRefFrameCtr(_refFrameSph.getFrameCtr() + 1);
          if (_gbh.biPredictionEnabledFlag){
            biPredDecodeParams._refFrameSph2.setFrameMovingState(_gbh.interFrameRefGmcFlag2);
            biPredDecodeParams._refFrameSph2.setMotionParams(
              _gbh.gm_param[1].gm_thres, _gbh.gm_param[1].gm_matrix,
              _gbh.gm_param[1].gm_trans);    
            biPredDecodeParams._refFrameSph2.setRefFrameCtr(biPredDecodeParams._refFrameSph2.getFrameCtr() + 1);        
          }
        }
        _refFrameSph.updateFrame(*_gps, *_refFrameAlt);
        if(_gbh.biPredictionEnabledFlag){
          biPredDecodeParams._refFrameSph2.updateFrame(*_gps, *_refFrameAlt);
        }
      }
    }     
    else if(_gps->interPredictionEnabledFlag) {
      if (_gps->predgeom_enabled_flag) {
        if (_gps->globalMotionEnabled) {
          _refFrameSph.setFrameMovingState(_gbh.interFrameRefGmcFlag);
          _refFrameSph.setMotionParams(
            _gbh.gm_param[0].gm_thres, _gbh.gm_param[0].gm_matrix,
            _gbh.gm_param[0].gm_trans);
        }

        if (_currentPointCloud.hasReflectances())
          _refFrameSph.updateFrame(
            *_gps, *_refFrameAlt, minPos_ref, _apss[0].attr_coord_scale);
        else
          _refFrameSph.updateFrame(*_gps, *_refFrameAlt);

      }
    }
  }

  // sanity check for loss detection
  if (_gbh.entropy_continuation_flag) {
    assert(!_firstSliceInFrame);
    assert(_gbh.prev_slice_id == _prevSliceId);
  } else if (_gbh.slice_inter_entropy_continuation_flag) {
    assert(_firstSliceInFrame);
    assert(_gbh.inter_entropy_prev_slice_id == _prevSliceId);
    assert(_gbh.inter_entropy_prev_frame_lsb == _prevFrameLsb);
  } else {
    // forget (reset) all saved context state at boundary
    _ctxtMemOctreeGeom->reset();
    _ctxtMemPredGeom->reset();
    for (auto& ctxtMem : _ctxtMemAttrs)
      ctxtMem.reset();
  }

  // set default attribute values (in case an attribute data unit is lost)
  // NB: it is a requirement that geom_num_points_minus1 is correct
  _currentPointCloud.resize(_gbh.footer.geom_num_points_minus1 + 1);
  if (hasColour) {
    auto it = std::find_if(
      _outCloud.attrDesc.cbegin(), _outCloud.attrDesc.cend(),
      [](const AttributeDescription& desc) {
        return desc.attributeLabel == KnownAttributeLabel::kColour;
      });

    Vec3<attr_t> defAttrVal = 1 << (it->bitdepth - 1);
    if (!it->params.attr_default_value.empty())
      for (int k = 0; k < 3; k++)
        defAttrVal[k] = it->params.attr_default_value[k];
    for (int i = 0; i < _currentPointCloud.getPointCount(); i++)
      _currentPointCloud.setColor(i, defAttrVal);
  }

  if (hasReflectance) {
    auto it = std::find_if(
      _outCloud.attrDesc.cbegin(), _outCloud.attrDesc.cend(),
      [](const AttributeDescription& desc) {
        return desc.attributeLabel == KnownAttributeLabel::kReflectance;
      });
    attr_t defAttrVal = 1 << (it->bitdepth - 1);
    if (!it->params.attr_default_value.empty())
      defAttrVal = it->params.attr_default_value[0];
    for (int i = 0; i < _currentPointCloud.getPointCount(); i++)
      _currentPointCloud.setReflectance(i, defAttrVal);
  }

  // Calculate a tree level at which to stop
  // It should result in at most max points being decoded
  if (_params.decodeMaxPoints && _gps->octree_point_count_list_present_flag) {
    if (_params.decodeMaxPoints > _gbh.footer.geom_num_points_minus1)
      _params.minGeomNodeSizeLog2 = 0;
    else {
      auto it = std::lower_bound(
        std::next(_gbh.footer.octree_lvl_num_points_minus1.begin()),
        _gbh.footer.octree_lvl_num_points_minus1.end(),
        _params.decodeMaxPoints);

      _params.minGeomNodeSizeLog2 =
        std::distance(it, _gbh.footer.octree_lvl_num_points_minus1.end()) + 1;
    }
  }

  EntropyDecoder aec;
  aec.setBuffer(buf.size() - gbhSize - gbfSize, buf.data() + gbhSize);
  aec.enableBypassStream(_sps->cabac_bypass_stream_enabled_flag);
  aec.setBypassBinCodingWithoutProbUpdate(_sps->bypass_bin_coding_without_prob_update);
  aec.start();

  if (_gps->predgeom_enabled_flag) {
    _refFrameSph.setInterEnabled(_gbh.interPredictionEnabledFlag);
    biPredDecodeParams._refFrameSph2.setInterEnabled(_gbh.interPredictionEnabledFlag && _gbh.biPredictionEnabledFlag);
    if (!_gbh.interPredictionEnabledFlag){
      _refFrameSph.clearRefFrame();
      biPredDecodeParams._refFrameSph2.clearRefFrame();
    }
    decodePredictiveGeometry(
      *_gps, _gbh, _currentPointCloud, &_posSph, _refFrameSph, biPredDecodeParams._refFrameSph2,
      *_ctxtMemPredGeom, aec);
  } else if (!_gps->trisoup_enabled_flag) {
    if (!_params.minGeomNodeSizeLog2) {
		if (_sps->layer_group_enabled_flag) {
      
      _gHandler = LayerGroupHandler(*_sps, _gbh);     

      _gHandler.initializeForGeometry(_layerGroupStructureInventory,_sliceId,_params.numSkipLayerGroups, false);

      _subgroupPointCloud.resize(1);
      _subgroupPointCloud[0].addRemoveAttributes(_currentPointCloud.hasColors(), _currentPointCloud.hasReflectances());    
      

      int curArrayIdx = 0;
			_gHandler.resetStoredCtxForGeometry(curArrayIdx);

      
      GeometryGranularitySlicingParam slicingParam;
      _gHandler.setSlicingParamGeom(curArrayIdx,_gbh,slicingParam);


			decodeGeometryOctreeGranularitySlicing(
				*_gps, _gbh, _currentPointCloud, *_gHandler._ctxtMemSaved[curArrayIdx].get(), aec, _gHandler._nodesSaved[curArrayIdx].get(),slicingParam);
      
      std::swap(_currentPointCloud,_subgroupPointCloud[0]);

      
      _gHandler.storeSlicingParamGeom(curArrayIdx,slicingParam);      
      _gHandler.setStartIdxForEachOutputPoints(0, 0);
      _currentPointCloud.append(_gHandler.getOutputPoints(_currentPointCloud.hasColors(), _currentPointCloud.hasReflectances(), curArrayIdx, _subgroupPointCloud[0]));

			auto rootBboxSize = slicingParam.bboxMax - slicingParam.bboxMin;
      _gHandler.setBox(curArrayIdx, Vec3<int>{0,0,0}, rootBboxSize);      
      _gHandler.setRoi(_params.roiEnabledFlag, _params.roiPointScale,_params.roiSize);

      _gHandler.releaseGeometryDecoderResource(curArrayIdx, true, _gbh.numSubsequentSubgroups);

		}
		else
      decodeGeometryOctree(
        *_gps, _gbh, _currentPointCloud, *_ctxtMemOctreeGeom, aec, _refFrame,
        biPredDecodeParams.refPointCloud2, attrInterPredParams.compensatedPointCloud, biPredDecodeParams.compensatedPointCloud2, _sps->seqBoundingBoxOrigin);
    } else {
      decodeGeometryOctreeScalable(
        *_gps, _gbh, _params.minGeomNodeSizeLog2, _currentPointCloud,
        *_ctxtMemOctreeGeom, aec, _refFrame,
        biPredDecodeParams.refPointCloud2, attrInterPredParams.compensatedPointCloud, biPredDecodeParams.compensatedPointCloud2);
    }
  } else {
    decodeGeometryTrisoup(
      *_gps, _gbh, _currentPointCloud, *_ctxtMemOctreeGeom, aec,
      _refFrame, _sps->seqBoundingBoxOrigin, attrInterPredParams.compensatedPointCloud, biPredDecodeParams.compensatedPointCloud2);
  }

  bool currFrameNotCodedAsB = (_gps->biPredictionEnabledFlag) && !_gbh.biPredictionEnabledFlag;
  auto& refFrameSph = currFrameNotCodedAsB ? biPredDecodeParams._refFrameSph2 : _refFrameSph;
  auto& refPosSph = currFrameNotCodedAsB ? biPredDecodeParams._refPosSph2 : _refPosSph;

  if (_gps->interPredictionEnabledFlag && _gps->predgeom_enabled_flag) {
    if (_gbh.biPredictionEnabledFlag)
      refPosSph = _posSph;
  }


   biPredDecodeParams.preFrameAsBframe = _gbh.biPredictionEnabledFlag;

  if (_gps->biPredictionEnabledFlag) {
    if (!_gbh.biPredictionEnabledFlag) {
      biPredDecodeParams.prePreIPFrame = biPredDecodeParams.preIPFrame;
      biPredDecodeParams.preIPFrame = _outCloud.frameNum;
    }
  }

  // At least the first slice's geometry has been decoded
  _firstSliceInFrame = false;

  clock_user.stop();

  auto total_user =
    std::chrono::duration_cast<std::chrono::microseconds>(clock_user.count());
  auto old_precision = std::cout.precision(6);
  std::cout << std::fixed;
  std::cout << "positions processing time (user): "
    << total_user.count() / 1000000.0 << " s\n";
  std::cout << std::endl;
  std::cout << std::defaultfloat << std::setprecision(old_precision);

  firstAttributeInSlice = true;

  return 0;
}

//--------------------------------------------------------------------------

int
PCCTMC3Decoder3::decodeDependentGeometryBrick(const PayloadBuffer& buf)
{
	assert(buf.type == PayloadType::kDependentGeometryDataUnit);
	std::cout << "positions bitstream size " << buf.size() << " B\n";

	// todo(df): replace with attribute mapping
	bool hasColour = std::any_of(
		_sps->attributeSets.begin(), _sps->attributeSets.end(),
		[](const AttributeDescription& desc) {
		return desc.attributeLabel == KnownAttributeLabel::kColour;
	});

	bool hasReflectance = std::any_of(
		_sps->attributeSets.begin(), _sps->attributeSets.end(),
		[](const AttributeDescription& desc) {
		return desc.attributeLabel == KnownAttributeLabel::kReflectance;
	});

	PCCPointSet3 tmpPoints;
	tmpPoints.clear();
	tmpPoints.addRemoveAttributes(hasColour, hasReflectance);

	pcc::chrono::Stopwatch<pcc::chrono::utime_inc_children_clock> clock_user;
	clock_user.start();

	int gbhSize, gbfSize;
	_dep_gbh = parseDepGbh(*_sps, *_gps, _gbh, buf, &gbhSize, &gbfSize);

	if (_dep_gbh.layer_group_id > _gbh.num_layer_groups_minus1 - _params.numSkipLayerGroups) {
		return 0;
	}

	bool isOutputLayerGroup = false;
	if (_dep_gbh.layer_group_id == _gbh.num_layer_groups_minus1 - _params.numSkipLayerGroups)
		isOutputLayerGroup = true;

  
	

  bool isRequiredLayer = _gHandler.isRequiredLayer(_dep_gbh.layer_group_id) ;
  bool isRequiredDataunit = ( !_gbh.subgroup_enabled_flag[_dep_gbh.layer_group_id] ||
      _gHandler.checkRoi(_dep_gbh.subgroupBboxOrigin, _dep_gbh.subgroupBboxSize) );

  if(!(isRequiredLayer && isRequiredDataunit)){
    return 0;
  }


	// set default attribute values (in case an attribute data unit is lost)
	// NB: it is a requirement that fgs_num_nodes_minus1 is correct
	tmpPoints.resize(_dep_gbh.footer.num_nodes_minus1 + 1);

	if (hasColour) {
		auto it = std::find_if(
			_outCloud.attrDesc.cbegin(), _outCloud.attrDesc.cend(),
			[](const AttributeDescription& desc) {
			return desc.attributeLabel == KnownAttributeLabel::kColour;
		});

		Vec3<attr_t> defAttrVal = 1 << (it->bitdepth - 1);
		if (!it->params.attr_default_value.empty())
			for (int k = 0; k < 3; k++)
				defAttrVal[k] = it->params.attr_default_value[k];
		for (int i = 0; i < tmpPoints.getPointCount(); i++)
			tmpPoints.setColor(i, defAttrVal);
	}

	if (hasReflectance) {
		auto it = std::find_if(
			_outCloud.attrDesc.cbegin(), _outCloud.attrDesc.cend(),
			[](const AttributeDescription& desc) {
			return desc.attributeLabel == KnownAttributeLabel::kReflectance;
		});
		attr_t defAttrVal = 1 << (it->bitdepth - 1);
		if (!it->params.attr_default_value.empty())
			defAttrVal = it->params.attr_default_value[0];
		for (int i = 0; i < tmpPoints.getPointCount(); i++)
			tmpPoints.setReflectance(i, defAttrVal);
	}

  
  
	Vec3<int> bboxMin = { 0, 0, 0 };
	Vec3<int> bboxMax = _gHandler.getRootBoxSize();
  if (_gbh.subgroup_enabled_flag[_dep_gbh.layer_group_id]) {
		bboxMin = _dep_gbh.subgroupBboxOrigin;
		bboxMax = bboxMin + _dep_gbh.subgroupBboxSize;  
  }
  _gHandler.setNewDependentUnit(_dep_gbh.layer_group_id, _dep_gbh.subgroup_id, _dep_gbh.ref_layer_group_id, _dep_gbh.ref_subgroup_id,bboxMin,bboxMax);
  
  int curArrayIdx = _gHandler.getArrayId(_dep_gbh.layer_group_id, _dep_gbh.subgroup_id);
	int refArrayIdx4Context = _gHandler.getReferenceIdx(curArrayIdx);
	int refArrayIdx4Parent = _gHandler.getParentIdx(curArrayIdx);
  

  if(_subgroupPointCloud.size()<curArrayIdx+1)
    _subgroupPointCloud.resize(curArrayIdx+1);

	_gHandler.resetStoredCtxForGeometry(curArrayIdx);
	_gHandler.loadCtxForGeometry(curArrayIdx, refArrayIdx4Context);
  
  
  GeometryGranularitySlicingParam slicingParam;  
  _gHandler.setSlicingParamGeom(curArrayIdx,refArrayIdx4Parent, _dep_gbh,slicingParam);
  
	EntropyDecoder aec;
	aec.setBuffer(buf.size() - gbhSize - gbfSize, buf.data() + gbhSize);
	aec.enableBypassStream(_sps->cabac_bypass_stream_enabled_flag);
	aec.setBypassBinCodingWithoutProbUpdate(_sps->bypass_bin_coding_without_prob_update);
	aec.start();

	decodeGeometryOctreeGranularitySlicing(
		*_gps, _gbh, tmpPoints, *_gHandler._ctxtMemSaved[curArrayIdx].get(), aec, _gHandler._nodesSaved[curArrayIdx].get(),slicingParam);
  
  
  std::swap(_subgroupPointCloud[curArrayIdx],tmpPoints);
  
  _gHandler.storeSlicingParamGeom(curArrayIdx,slicingParam);
  
  _gHandler.setStartIdxForEachOutputPoints(curArrayIdx, _currentPointCloud.getPointCount());
  _currentPointCloud.append(_gHandler.getOutputPoints(_currentPointCloud.hasColors(), _currentPointCloud.hasReflectances(), curArrayIdx, _subgroupPointCloud[curArrayIdx]));

  
  _gHandler.releaseGeometryDecoderResource(curArrayIdx, true, _dep_gbh.numSubsequentSubgroups);



	clock_user.stop();

	auto total_user =
		std::chrono::duration_cast<std::chrono::microseconds>(clock_user.count());
	auto old_precision = std::cout.precision(6);
	std::cout << std::fixed;
	std::cout << "positions processing time (user): "
		<< total_user.count() / 1000000.0 << " s\n";
	std::cout << std::endl;
	std::cout << std::defaultfloat << std::setprecision(old_precision);
	return 0;
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::decodeAttributeBrick(const PayloadBuffer& buf)
{
  assert(buf.type == PayloadType::kAttributeBrick);
  // todo(df): replace assertions with error handling
  assert(_sps);
  assert(_gps);

  // verify that this corresponds to the correct geometry slice
  AttributeBrickHeader abh = parseAbhIds(buf);
  assert(abh.attr_geom_slice_id == _sliceId);

  // todo(df): validate that sps activation is not changed via the APS
  const auto it_attr_aps = _apss.find(abh.attr_attr_parameter_set_id);

  assert(it_attr_aps != _apss.cend());
  const auto& attr_aps = it_attr_aps->second;

  assert(abh.attr_sps_attr_idx < _sps->attributeSets.size());
  const auto& attr_sps = _sps->attributeSets[abh.attr_sps_attr_idx];
  const auto& label = attr_sps.attributeLabel;

  // sanity check for loss detection
  if (_gbh.entropy_continuation_flag)
    assert(_gbh.prev_slice_id == _ctxtMemAttrSliceIds[abh.attr_sps_attr_idx]);
  if (_gbh.slice_inter_entropy_continuation_flag) {
    assert(
      _gbh.inter_entropy_prev_slice_id
      == _ctxtMemAttrSliceIds[abh.attr_sps_attr_idx]);
    assert(
      _gbh.inter_entropy_prev_frame_lsb
      == _ctxtMemAttrFrameLsbs[abh.attr_sps_attr_idx]);
  }

  // Ensure context arrays are allocated context arrays
  // todo(df): move this to sps activation
  _ctxtMemAttrSliceIds.resize(_sps->attributeSets.size());
  _ctxtMemAttrFrameLsbs.resize(_sps->attributeSets.size());
  _ctxtMemAttrs.resize(_sps->attributeSets.size());

  // In order to determinet hat the attribute decoder is reusable, the abh
  // must be inspected.
  abh.geomEnableBiInterPred = _gbh.biPredictionEnabledFlag;
  int abhSize;
  abh.RAHTFilterTaps.clear();
  abh = parseAbh(*_sps, attr_aps, buf, &abhSize);

  attrInterPredParams.frameDistance = 1;
  attrInterPredParams.enableAttrInterPred =
    attr_aps.attrInterPredictionEnabled && abh.enableAttrInterPred;
  
  abh.attrInterPredSearchRange = attr_aps.attrInterPredSearchRange; 

    biPredDecodeParams.attrInterPredParams2.enableAttrInterPred =
      _gps->biPredictionEnabledFlag && attr_aps.attrInterPredictionEnabled
      && abh.enableAttrInterPred2;

    if (!attr_aps.spherical_coord_flag) {
      if (attrInterPredParams.enableAttrInterPred)
        attrInterPredParams.referencePointCloud = attrInterPredParams.compensatedPointCloud;
      if (_gps->biPredictionEnabledFlag)
        if (_gbh.biPredictionEnabledFlag
          && biPredDecodeParams.attrInterPredParams2.enableAttrInterPred)
          biPredDecodeParams.attrInterPredParams2.referencePointCloud = biPredDecodeParams.compensatedPointCloud2;
    }



  if (_gps->biPredictionEnabledFlag) {
      //FrameMerge for attribute inter prediction
    if (
      _gbh.biPredictionEnabledFlag
      && biPredDecodeParams.attrInterPredParams2.enableAttrInterPred) {
      if (attrInterPredParams.enableAttrInterPred) {
        attrInterPredParams.referencePointCloud.append(
          biPredDecodeParams.attrInterPredParams2.referencePointCloud);

        //The times of neighbor search process is halfed
        abh.attrInterPredSearchRange /= 2;
      } else {
        attrInterPredParams = biPredDecodeParams.attrInterPredParams2;
        attrInterPredParams.frameDistance = 1;
      }
    }
  }


  attrInterPredParams.paramsForInterRAHT.raht_inter_prediction_depth_minus1 =
    attr_aps.raht_inter_prediction_depth_minus1;
  attrInterPredParams.paramsForInterRAHT.enableFilterEstimation = attr_aps.raht_send_inter_filters;
  attrInterPredParams.paramsForInterRAHT.skipInitLayersForFiltering = attr_aps.raht_inter_skip_layers;
  attrInterPredParams.paramsForInterRAHT.FilterTaps.clear();
  if (attr_aps.raht_send_inter_filters && abh.enableAttrInterPred) {
    attrInterPredParams.paramsForInterRAHT.FilterTaps = abh.RAHTFilterTaps;
  }

  attrInterPredParams.paramsForInterRAHT.raht_enable_inter_intra_layer_RDO =
    attr_aps.raht_enable_code_layer;
  
  if (attr_aps.spherical_coord_flag && _gps->predgeom_enabled_flag && !_gps->biPredictionEnabledFlag)
    attrInterPredParams.useRefCloudIndex = true;
  
  pcc::chrono::Stopwatch<pcc::chrono::utime_inc_children_clock> clock_user;

  int curArrayIdx = 0;
  if (_sps->layer_group_enabled_flag && attr_aps.layer_group_enabled_flag) {

    _gHandler.releaseCtxForGeometry();
    _gHandler.releaseNodes();

    curArrayIdx = _gHandler.getArrayId(0,0);

    
    if( !_attrDecoder ){
      
      int numAttr = _sps->attributeSets.size();
      _gHandler.initializeForAttribute(numAttr, false);

      _abh.clear();
      _abh.resize(numAttr);

      _codingtime.clear();
      _codingtime.resize(numAttr);

    }

    _gHandler.buildLoDSettings(attr_aps, abh.attr_sps_attr_idx);


  } 


  // replace the attribute decoder if not compatible
  bool differentArray = _sps->layer_group_enabled_flag && attr_aps.layer_group_enabled_flag 
                          && _prevArrayIdx != curArrayIdx;
  if (!_attrDecoder || !_attrDecoder->isReusable(attr_aps, abh) || differentArray){
    _attrDecoder = makeAttributeDecoder();
  }

  clock_user.start();


  if(_sps->layer_group_enabled_flag && attr_aps.layer_group_enabled_flag){
    
    _gHandler.resetStoredCtxForAttribute(abh.attr_sps_attr_idx,curArrayIdx);
    

    AttributeGranularitySlicingParam slicingParam;
    _gHandler.setSlicingParamAttr(abh.attr_sps_attr_idx, 0, nullptr, nullptr, abh.subgroup_weight_adjustment_enabled_flag, 0, nullptr, nullptr, slicingParam);

    _attrDecoder->decode(
      *_sps, attr_sps, attr_aps, abh, 
      _gbh.footer.geom_num_points_minus1,
      _gHandler.getMinNodeSize(abh.attr_sps_attr_idx, 0), buf.data() + abhSize, buf.size() - abhSize,
      *_gHandler._ctxtMemAttrsSaved[abh.attr_sps_attr_idx][curArrayIdx], 
      _subgroupPointCloud[curArrayIdx],
      attrInterPredParams,slicingParam, predDecoder);
    
    _abh[abh.attr_sps_attr_idx] = abh;
    _prevArrayIdx = curArrayIdx;
    

  }
  else{
      // Convert cartesian positions to spherical for use in attribute coding.
    // NB: this retains the original cartesian positions to restore afterwards

    if (attr_aps.spherical_coord_flag) {
      // If predgeom was used, re-use the internal positions rather than
      // calculating afresh.
      if(firstAttributeInSlice || !_attrDecoder->isRefReusable(attr_aps.attr_coord_scale, _gps->predgeom_enabled_flag, _gps->geom_angular_azimuth_scale_log2_minus11,
           abh.enableAttrInterPred, abh.enableAttrInterPred2)) {
        Box3<int> bboxRpl;

        pcc::point_t minPos = 0;

        if (_gps->predgeom_enabled_flag) {
          altPositions = _posSph;
          bboxRpl = Box3<int>(altPositions.begin(), altPositions.end());
          minPos = bboxRpl.min;
          if (
            attrInterPredParams.enableAttrInterPred
            && attr_aps.attr_encoding == AttributeEncoding::kRAHTransform) {

            attrInterPredParams.copyReferenceCloud(_refFrameAlt);

          }
          if (
            attrInterPredParams.enableAttrInterPred
            || biPredDecodeParams.attrInterPredParams2.enableAttrInterPred) {
            for (auto i = 0; i < 3; i++)
              minPos[i] = minPos[i] < minPos_ref[i] ? minPos[i] : minPos_ref[i];
            auto minPos_shift = minPos_ref - minPos;

            attrInterPredParams.setMinPosAndScale(
              minPos, attr_aps.attr_coord_scale);

            if (
              (minPos_shift[0] || minPos_shift[1] || minPos_shift[2])
              && attr_aps.attr_encoding == AttributeEncoding::kRAHTransform
              && !attrInterPredParams.useRefCloudIndex){
                offsetAndScaleShift(
                  minPos_shift, attr_aps.attr_coord_scale,
                  &attrInterPredParams.referencePointCloud[0],
                  &attrInterPredParams.referencePointCloud[0]
                    + attrInterPredParams.getPointCount());
              }
          }
          minPos_ref = minPos;
        } else {
          altPositions.resize(_currentPointCloud.getPointCount());

          auto laserOrigin = _gbh.geomAngularOrigin(*_gps);
          bboxRpl = convertXyzToRpl(
            laserOrigin, _gps->angularTheta.data(), _gps->angularTheta.size(),
            &_currentPointCloud[0],
            &_currentPointCloud[0] + _currentPointCloud.getPointCount(),
            altPositions.data());

          if(!attr_aps.attrInterPredictionEnabled){
            minPos = bboxRpl.min;
          }
        }

        offsetAndScale(
          minPos, attr_aps.attr_coord_scale, altPositions.data(),
          altPositions.data() + altPositions.size());   
          
        _attrDecoder->setRefReusable(attr_aps.attr_coord_scale, _gps->predgeom_enabled_flag, _gps->geom_angular_azimuth_scale_log2_minus11,
          abh.enableAttrInterPred, abh.enableAttrInterPred2);     
      }
      _currentPointCloud.swapPoints(altPositions);
    }

    if (!attr_aps.spherical_coord_flag && attr_aps.attrInterPredictionEnabled)
      for (auto i = 0; i < _currentPointCloud.getPointCount(); i++)
        _currentPointCloud[i] += _sliceOrigin;

    if (attr_aps.attrInterPredictionEnabled && firstAttributeInSlice) {
      if (_refFrameAlt) {
        attrInterPredParams.updateReferencePointCloud(
          _currentPointCloud.computeBoundingBox(), _refFrameAlt,
          attr_aps.spherical_coord_flag && !_gps->biPredictionEnabledFlag,
          minPos_ref, attr_aps.attr_coord_scale);
      }
    }

    if (_gps->predgeom_enabled_flag && !attr_aps.canonical_point_order_flag) {
      _attrDecoder->canonical_lod_sampling_enabled_flag() = true;
    }
    else {
      _attrDecoder->canonical_lod_sampling_enabled_flag() = false;
    }

    auto& ctxtMemAttr = _ctxtMemAttrs.at(abh.attr_sps_attr_idx);
    abh.firstAttributeInSlice = firstAttributeInSlice;
    AttributeGranularitySlicingParam slicingParam;
    _attrDecoder->decode(
      *_sps, attr_sps, attr_aps, abh, _gbh.footer.geom_num_points_minus1,
      _params.minGeomNodeSizeLog2, buf.data() + abhSize, buf.size() - abhSize,
      ctxtMemAttr, _currentPointCloud, attrInterPredParams, slicingParam, predDecoder);

    _currentPointCloudAttr = _currentPointCloud;
    _storeAsSphe = attr_aps.spherical_coord_flag && _gps->predgeom_enabled_flag;

    bool currFrameNotCodedAsB =
      (_gps->biPredictionEnabledFlag && !_gbh.biPredictionEnabledFlag);
    auto& refCloud = currFrameNotCodedAsB
      ? biPredDecodeParams.attrInterPredParams2.referencePointCloud
      : attrInterPredParams.referencePointCloud;

    if (attr_aps.spherical_coord_flag) {
      if (!_gps->predgeom_enabled_flag)
        refCloud = _currentPointCloud;
      _currentPointCloud.swapPoints(altPositions);
    } 
    else {
      refCloud = _currentPointCloud;
    }

    if (!attr_aps.spherical_coord_flag && attr_aps.attrInterPredictionEnabled)
      for (auto i = 0; i < _currentPointCloud.getPointCount(); i++)
        _currentPointCloud[i] -= _sliceOrigin;

  }

  // Note the current sliceID for loss detection
  _ctxtMemAttrSliceIds[abh.attr_sps_attr_idx] = _sliceId;
  _ctxtMemAttrFrameLsbs[abh.attr_sps_attr_idx] =
    _frameCtr & (1 << _sps->frame_ctr_bits - 1);

  if (_sps->layer_group_enabled_flag && attr_aps.layer_group_enabled_flag) {

    assert(_gHandler.isRequiredLayer(0));

    _gHandler.setDecodedAttribute(curArrayIdx,_currentPointCloud,_subgroupPointCloud[curArrayIdx],attr_sps.attr_num_dimensions_minus1 == 2, attr_sps.attr_num_dimensions_minus1 == 0);

    
    _gHandler.releaseAttributeDecoderResource(
      abh.attr_sps_attr_idx,
      curArrayIdx,
      true,
      attr_aps.attr_ref_id_present_flag,
      _subgroupPointCloud);
  }

  clock_user.stop();

  std::cout << label << "s bitstream size " << buf.size() << " B\n";

  auto total_user =
    std::chrono::duration_cast<std::chrono::microseconds>(clock_user.count());
  auto old_precision = std::cout.precision(6);
  std::cout << std::fixed;
  std::cout << label
    << "s processing time (user): " << total_user.count() / 1000000.0
    << " s\n";
  std::cout << std::defaultfloat << std::setprecision(old_precision);

  firstAttributeInSlice = false;
  
  if (_sps->layer_group_enabled_flag && attr_aps.layer_group_enabled_flag)
    _codingtime[abh.attr_sps_attr_idx] = total_user.count()/1000.0;
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::decodeDependentAttributeBrick(const PayloadBuffer& buf)
{
  assert(buf.type == PayloadType::kDependentAttributeDataUnit);
  // todo(df): replace assertions with error handling
  assert(_sps);
  assert(_gps);

  // verify that this corresponds to the correct geometry slice
  DependentAttributeDataUnitHeader dep_abh = parseDepAbhIds(buf);
  assert(dep_abh.attr_geom_slice_id == _sliceId);

  // todo(df): validate that sps activation is not changed via the APS
  const auto it_attr_aps = _apss.find(dep_abh.attr_attr_parameter_set_id);

  assert(it_attr_aps != _apss.cend());
  const auto& attr_aps = it_attr_aps->second;

  assert(dep_abh.attr_sps_attr_idx < _sps->attributeSets.size());
  const auto& attr_sps = _sps->attributeSets[dep_abh.attr_sps_attr_idx];
  const auto& label = attr_sps.attributeLabel;
  
  assert(_sps->layer_group_enabled_flag);
  assert(attr_aps.layer_group_enabled_flag);


  // In order to determinet hat the attribute decoder is reusable, the abh
  // must be inspected.
  int abhSize;
  dep_abh = parseDepAbh(*_sps, attr_aps, _gbh, buf, &abhSize);

  PCCPointSet3 tmpPoints;
  
  //Check corresponding geometry brick exsisting
  if(!_gHandler.hasCorrespondingGeometry(dep_abh.layer_group_id, dep_abh.subgroup_id))
    return;
  if(!_gHandler.isRequiredLayer(dep_abh.layer_group_id))
    return;
  
  
  //todo: inherit required&output flag of subgroups from geometry

  auto abh = _abh[dep_abh.attr_sps_attr_idx];
  
  int curArrayIdx = _gHandler.getArrayId(dep_abh.layer_group_id, dep_abh.subgroup_id);
  std::swap(tmpPoints, _subgroupPointCloud[curArrayIdx]);
  
  //Get parent information and make parameters for LOD generation
  int parentArrayIdx = _gHandler.getParentIdx(curArrayIdx);
  
  int refArrayIdx;
  if(attr_aps.attr_ref_id_present_flag){
    _gHandler.setNewDependentUnitForAttribute(dep_abh.layer_group_id, dep_abh.subgroup_id, dep_abh.ref_layer_group_id, dep_abh.ref_subgroup_id);
  }
  refArrayIdx = _gHandler.getReferenceIdxAttribute(curArrayIdx);

  pcc::chrono::Stopwatch<pcc::chrono::utime_inc_children_clock> clock_user;


  // replace the attribute decoder if not compatible
  if (!_attrDecoder || !_attrDecoder->isReusable(attr_aps, abh) || _prevArrayIdx != curArrayIdx){
    _attrDecoder = makeAttributeDecoder();
  }

  clock_user.start();

  _gHandler.loadCtxForAttribute(dep_abh.attr_sps_attr_idx,curArrayIdx,refArrayIdx);
  
  auto parentPointCloud = _gHandler.getParentPoints(
    curArrayIdx, parentArrayIdx, _subgroupPointCloud[parentArrayIdx]);
      
  AttributeGranularitySlicingParam slicingParam;
  _gHandler.setSlicingParamAttr(dep_abh.attr_sps_attr_idx, dep_abh.layer_group_id, &dep_abh, &parentPointCloud, dep_abh.subgroup_weight_adjustment_enabled_flag, 0, nullptr, nullptr, slicingParam);
              

  _attrDecoder->decode(
    *_sps, attr_sps, attr_aps, abh, tmpPoints.getPointCount()-1,
    _gHandler.getMinNodeSize(abh.attr_sps_attr_idx,dep_abh.layer_group_id), 
    buf.data() + abhSize, buf.size() - abhSize,
    *_gHandler._ctxtMemAttrsSaved[abh.attr_sps_attr_idx][curArrayIdx], tmpPoints,
    attrInterPredParams,slicingParam, predDecoder);
    

  
  _prevArrayIdx = curArrayIdx;
  std::swap(_subgroupPointCloud[curArrayIdx], tmpPoints);

  
  assert(_gHandler.isRequiredLayer(dep_abh.layer_group_id));

  _gHandler.setDecodedAttribute(curArrayIdx,_currentPointCloud,_subgroupPointCloud[curArrayIdx],attr_sps.attr_num_dimensions_minus1 == 2, attr_sps.attr_num_dimensions_minus1 == 0);


  _gHandler.releaseAttributeDecoderResource(
    abh.attr_sps_attr_idx,
    curArrayIdx,
    dep_abh.context_reference_indication_flag,
    attr_aps.attr_ref_id_present_flag,
    _subgroupPointCloud);

  
  clock_user.stop();

  std::cout << label << "s bitstream size " << buf.size() << " B\n";

  auto total_user =
    std::chrono::duration_cast<std::chrono::microseconds>(clock_user.count());
  auto old_precision = std::cout.precision(6);
  std::cout << std::fixed;
  std::cout << label
    << "s processing time (user): " << total_user.count() / 1000000.0
    << " s\n";
  std::cout << std::endl;
  std::cout << std::defaultfloat << std::setprecision(old_precision);
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::decodeConstantAttribute(const PayloadBuffer& buf)
{
  assert(buf.type == PayloadType::kConstantAttribute);
  // todo(df): replace assertions with error handling
  assert(_sps);
  assert(_gps);

  ConstantAttributeDataUnit cadu = parseConstantAttribute(*_sps, buf);

  // verify that this corresponds to the correct geometry slice
  assert(cadu.constattr_geom_slice_id == _sliceId);

  assert(cadu.constattr_sps_attr_idx < _sps->attributeSets.size());
  const auto& attrDesc = _sps->attributeSets[cadu.constattr_sps_attr_idx];
  const auto& label = attrDesc.attributeLabel;

  // todo(df): replace with proper attribute mapping
  if (label == KnownAttributeLabel::kColour) {
    Vec3<attr_t> defAttrVal;
    for (int k = 0; k < 3; k++)
      defAttrVal[k] = attrDesc.params.attr_default_value[k];
    for (int i = 0; i < _currentPointCloud.getPointCount(); i++)
      _currentPointCloud.setColor(i, defAttrVal);
  }

  if (label == KnownAttributeLabel::kReflectance) {
    attr_t defAttrVal = attrDesc.params.attr_default_value[0];
    for (int i = 0; i < _currentPointCloud.getPointCount(); i++)
      _currentPointCloud.setReflectance(i, defAttrVal);
  }
}

//============================================================================

}  // namespace pcc
