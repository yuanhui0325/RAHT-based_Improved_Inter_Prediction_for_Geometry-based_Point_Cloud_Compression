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

#include <cstdio>

#include "geometry_trisoup.h"

#include "pointset_processing.h"
#include "geometry.h"
#include "geometry_octree.h"

namespace pcc {

//============================================================================

bool
operator<(const TrisoupSegment& s1, const TrisoupSegment& s2)
{
  // assert all quantities are at most 21 bits
  uint64_t s1startpos = (uint64_t(s1.startpos[0]) << 42)
    | (uint64_t(s1.startpos[1]) << 21) | s1.startpos[2];

  uint64_t s2startpos = (uint64_t(s2.startpos[0]) << 42)
    | (uint64_t(s2.startpos[1]) << 21) | s2.startpos[2];

  if (s1startpos < s2startpos)
    return true;

  if (s1startpos != s2startpos)
    return false;

  uint64_t s1endpos = (uint64_t(s1.endpos[0]) << 42)
    | (uint64_t(s1.endpos[1]) << 21) | s1.endpos[2];

  uint64_t s2endpos = (uint64_t(s2.endpos[0]) << 42)
    | (uint64_t(s2.endpos[1]) << 21) | s2.endpos[2];

  if (s1endpos < s2endpos)
    return true;

  if (s1endpos == s2endpos)
    if (s1.index < s2.index)  // stable sort
      return true;

  return false;
}

//============================================================================
bool
operator<(const TrisoupSegmentNeighbours& s1, const TrisoupSegmentNeighbours& s2)
{
  // assert all quantities are at most 21 bits
  uint64_t s1startpos = (uint64_t(s1.startpos[0]) << 42)
    | (uint64_t(s1.startpos[1]) << 21) | s1.startpos[2];

  uint64_t s2startpos = (uint64_t(s2.startpos[0]) << 42)
    | (uint64_t(s2.startpos[1]) << 21) | s2.startpos[2];

  if (s1startpos < s2startpos)
    return true;

  if (s1startpos != s2startpos)
    return false;

  uint64_t s1endpos = (uint64_t(s1.endpos[0]) << 42)
    | (uint64_t(s1.endpos[1]) << 21) | s1.endpos[2];

  uint64_t s2endpos = (uint64_t(s2.endpos[0]) << 42)
    | (uint64_t(s2.endpos[1]) << 21) | s2.endpos[2];

  if (s1endpos < s2endpos)
    return true;

  if (s1endpos == s2endpos)
    if (s1.index < s2.index)  // stable sort
      return true;

  return false;
}

bool
operator==(const TrisoupSegment& s1, const TrisoupSegment& s2)
{
  if (s1.startpos == s2.startpos && s1.endpos == s2.endpos)
    return true;
  else
    return false;
}

//============================================================================

void
decodeGeometryTrisoup(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  PCCPointSet3& pointCloud,
  GeometryOctreeContexts& ctxtMemOctree,
  EntropyDecoder& arithmeticDecoder,
  const CloudFrame* refFrame,
  const Vec3<int> minimum_position,
  PCCPointSet3& compensatedPointCloud,
  PCCPointSet3& compensatedPointCloud2)
{
  // trisoup uses octree coding until reaching the triangulation level.
  // todo(df): pass trisoup node size rather than 0?
  pcc::ringbuf<PCCOctree3Node> nodes;
  PCCPointSet3 predPointCloud2;
  
  GeometryGranularitySlicingParam slicingParam;

  decodeGeometryOctree(
    gps, gbh, 0, pointCloud, ctxtMemOctree, arithmeticDecoder, &nodes,
	  refFrame, predPointCloud2, compensatedPointCloud,compensatedPointCloud2, minimum_position, slicingParam);

  int blockWidth = 1 << gbh.trisoupNodeSizeLog2(gps);
  const int maxVertexPrecisionLog2 = gbh.trisoup_vertex_quantization_bits ? gbh.trisoup_vertex_quantization_bits : gbh.trisoupNodeSizeLog2(gps);
  const int bitDropped =  std::max(0, gbh.trisoupNodeSizeLog2(gps) - maxVertexPrecisionLog2);
  const bool isCentroidDriftActivated = gbh.trisoup_centroid_vertex_residual_flag;
  const bool isFaceVertexActivated = gbh.trisoup_face_vertex_flag;

  std::vector<node6nei> nodes6nei;
  if( isFaceVertexActivated ){
    determineTrisoupNodeNeighbours(nodes, nodes6nei, blockWidth);
  }

  // Determine neighbours
  std::vector<uint16_t> neighbNodes;
  std::vector<std::array<int, 18>> edgePattern;
  determineTrisoupNeighbours(nodes, neighbNodes, edgePattern, blockWidth);


  // Decode vertex presence and position into bitstream  
  std::vector<bool> segind;
  std::vector<uint8_t> vertices;
  std::vector<TrisoupNodeEdgeVertex> eVerts;
  const bool mergeFlag = gbh.trisoup_vertex_merge;
  const bool fixFlag = gbh.trisoup_vertex_fix;
  decodeTrisoupVertices(nodes, blockWidth, segind, vertices, neighbNodes, edgePattern, bitDropped,
    gps, gbh, eVerts, arithmeticDecoder, mergeFlag, fixFlag);

  AdaptiveBitModel ctxDrift0[9];
  AdaptiveBitModel ctxDriftSign[3][8][8];
  AdaptiveBitModel ctxDriftMag[4];

  std::vector<TrisoupCentroidVertex> cVerts;
  std::vector<Vec3<int32_t>> normVs;
  std::vector<Vec3<int32_t>> gravityCenter;
  decodeTrisoupCentroids(nodes, nodes6nei, gps, gbh, blockWidth, bitDropped, eVerts,
    gravityCenter, cVerts, normVs, &arithmeticDecoder , isFaceVertexActivated, ctxDrift0, ctxDriftSign, ctxDriftMag);


  decodeTrisoupCentroids(nodes, nodes6nei, gps, gbh, blockWidth, bitDropped, eVerts,
    gravityCenter, cVerts, normVs, &arithmeticDecoder , isFaceVertexActivated, ctxDrift0, ctxDriftSign, ctxDriftMag, true);

  std::vector<TrisoupFace> faces;
  std::vector<TrisoupNodeFaceVertex> fVerts;
  fVerts.resize( nodes.size() );
  if( isFaceVertexActivated ){
    decodeTrisoupFaceList(nodes, gps, gbh, blockWidth,
      /*distanceSearchEncoder*/0, nodes6nei, eVerts, gravityCenter, cVerts,
      fVerts, normVs, faces, arithmeticDecoder);
  }

  PCCPointSet3 recPointCloud;
  recPointCloud.addRemoveAttributes(pointCloud);

  // Compute refinedVertices.
  std::vector<CentroidDrift> drifts;
  int32_t maxval = (1 << gbh.maxRootNodeDimLog2) - 1;
  bool haloFlag = gbh.trisoup_halo_flag;
  bool adaptiveHaloFlag = gbh.trisoup_adaptive_halo_flag;
  bool fineRayFlag = gbh.trisoup_fine_ray_tracing_flag;
  decodeTrisoupCommon(nodes, nodes6nei, eVerts, cVerts, gravityCenter, normVs,
    faces, fVerts, recPointCloud, gps, gbh, blockWidth, maxval,
    gbh.trisoup_sampling_value_minus1 + 1, bitDropped,
    isCentroidDriftActivated, isFaceVertexActivated, haloFlag,
    adaptiveHaloFlag, fineRayFlag);

  pointCloud.resize(0);
  pointCloud = std::move(recPointCloud);
}


//============================================================================


void
determineTrisoupNodeNeighbours
(
  const ringbuf<PCCOctree3Node>& leaves,
  std::vector<node6nei> &nodes6nei,
  int defaultBlockWidth
 )
{
  // 1. duplicate leaves into 6nei nodes and sort them by their coordinate.
  // 2. Scan the same coordinate node in the same hierarchy after
  //    the coordinate sort to find the original index of the
  //    6nei node from the central node.
  int bw = defaultBlockWidth;
  std::vector<duplicateNodes> dupNodes;
  Vec3<int32_t> offset[7] = {
    {0,0,-bw},{0,0,bw}, {0,-bw,0},{0,bw,0},
    {-bw,0,0},{bw,0,0}, {0,0,0} };
  for( int i=0; i<leaves.size(); i++ ) {
    for( int j=0; j<7; j++ ) {
      duplicateNodes dnode;
      dnode.pos = leaves[i].pos + offset[j];
      dnode.idx = (i<<3)+j;
      dupNodes.push_back(dnode);
    }
  }
  std::sort( dupNodes.begin(), dupNodes.end() );
  duplicateNodes localDupNode = dupNodes[0];
  node6nei local6nei;
  local6nei.pos = localDupNode.pos;
  int ofstIdx = 7 & localDupNode.idx;
  int nIdx = (6==ofstIdx) ? 6 : (ofstIdx^1);
  local6nei.idx[nIdx] = localDupNode.idx>>3;
  for( auto it=dupNodes.begin()+1; it!=dupNodes.end(); it++ ) {
    if( localDupNode.pos != it->pos ) {
      if( local6nei.idx[6] != -1 ) {
        nodes6nei.push_back(local6nei);
      }
      local6nei.clear();
      local6nei.pos = it->pos;
    }
    localDupNode = *it;
    // flip neighbour direction by LSB flip
    ofstIdx = 7 & localDupNode.idx; // 0-5,6(center)
    nIdx = (6==ofstIdx) ? 6 : (ofstIdx^1);
    local6nei.idx[nIdx] = localDupNode.idx>>3;
  }
  if( local6nei.idx[6] != -1 ) {
    nodes6nei.push_back(local6nei);
  }
  std::sort( nodes6nei.begin(), nodes6nei.end() );
  return;
}


void determineTrisoupNeighbours(
  const ringbuf<PCCOctree3Node>& leaves,
  std::vector<uint16_t>& neighbNodes,
  std::vector<std::array<int, 18>>& edgePattern,
  const int defaultBlockWidth) {

  // Width of block.
  // in future, may override with leaf blockWidth
  const int32_t blockWidth = defaultBlockWidth;

  // Eight corners of block.
  const Vec3<int32_t> pos000({ 0, 0, 0 });
  const Vec3<int32_t> posW00({ blockWidth, 0, 0 });
  const Vec3<int32_t> pos0W0({ 0, blockWidth, 0 });
  const Vec3<int32_t> posWW0({ blockWidth, blockWidth, 0 });
  const Vec3<int32_t> pos00W({ 0, 0, blockWidth });
  const Vec3<int32_t> posW0W({ blockWidth, 0, blockWidth });
  const Vec3<int32_t> pos0WW({ 0, blockWidth, blockWidth });
  const Vec3<int32_t> posWWW({ blockWidth, blockWidth, blockWidth });

  // Put all leaves' edges into a list.
  std::vector<TrisoupSegmentNeighbours> segments;
  segments.reserve(36 * leaves.size());
  for (int i = 0; i < leaves.size(); i++) {

    const auto& leaf = leaves[i];
    int ii = 36 * i;
    int ii2 = ii + 12;
    int ii3 = ii + 24;
    // x: left to right; y: bottom to top; z: far to near
    auto posNode = leaf.pos + blockWidth;

    // ------------ edges along x
    // in node
    segments.push_back(TrisoupSegmentNeighbours({ posNode + pos000, posNode + posW00, ii + 0, 1 })); // far bottom edge
    segments.push_back(TrisoupSegmentNeighbours({ posNode + pos0W0, posNode + posWW0, ii + 2,  2 })); // far top edge
    segments.push_back(TrisoupSegmentNeighbours({ posNode + pos00W, posNode + posW0W, ii + 8,  4 })); // near bottom edge
    segments.push_back(TrisoupSegmentNeighbours({ posNode + pos0WW, posNode + posWWW, ii + 10,  8 })); // near top edge
    // left 
    auto posLeft = posNode - posW00;
    segments.push_back(TrisoupSegmentNeighbours({ posLeft + pos000, posLeft + posW00, ii2 + 0, 16 })); // far bottom edge
    segments.push_back(TrisoupSegmentNeighbours({ posLeft + pos0W0, posLeft + posWW0, ii2 + 2,  32 })); // far top edge
    segments.push_back(TrisoupSegmentNeighbours({ posLeft + pos00W, posLeft + posW0W, ii2 + 8,  64 })); // near bottom edge
    segments.push_back(TrisoupSegmentNeighbours({ posLeft + pos0WW, posLeft + posWWW, ii2 + 10,  128 })); // near top edge
    //right
    auto posRight = posNode + posW00;
    segments.push_back(TrisoupSegmentNeighbours({ posRight + pos000, posRight + posW00, ii3 + 0, 256 })); // far bottom edge
    segments.push_back(TrisoupSegmentNeighbours({ posRight + pos0W0, posRight + posWW0, ii3 + 2,  512 })); // far top edge
    segments.push_back(TrisoupSegmentNeighbours({ posRight + pos00W, posRight + posW0W, ii3 + 8,  1024 })); // near bottom edge
    segments.push_back(TrisoupSegmentNeighbours({ posRight + pos0WW, posRight + posWWW, ii3 + 10,  2048 })); // near top edge                                                                                                       

    // ------------ edges along y 
    // in node
    segments.push_back(TrisoupSegmentNeighbours({ posNode + pos000, posNode + pos0W0, ii + 1, 1 + (1 << 13) })); // far left edge
    segments.push_back(TrisoupSegmentNeighbours({ posNode + posW00, posNode + posWW0, ii + 3,  2 + (1 << 13) })); // far right edge
    segments.push_back(TrisoupSegmentNeighbours({ posNode + pos00W, posNode + pos0WW, ii + 9,  4 + (1 << 13) })); // near left edge
    segments.push_back(TrisoupSegmentNeighbours({ posNode + posW0W, posNode + posWWW, ii + 11,  8 + (1 << 13) })); // near right edge   
    // bottom
    auto posBottom = posNode - pos0W0;
    segments.push_back(TrisoupSegmentNeighbours({ posBottom + pos000, posBottom + pos0W0, ii2 + 1, 16 })); // far left edge
    segments.push_back(TrisoupSegmentNeighbours({ posBottom + posW00, posBottom + posWW0, ii2 + 3,  32 })); // far right edge
    segments.push_back(TrisoupSegmentNeighbours({ posBottom + pos00W, posBottom + pos0WW, ii2 + 9,  64 })); // near left edge
    segments.push_back(TrisoupSegmentNeighbours({ posBottom + posW0W, posBottom + posWWW, ii2 + 11,  128 })); // near right edge  
    // top
    auto posTop = posNode + pos0W0;
    segments.push_back(TrisoupSegmentNeighbours({ posTop + pos000, posTop + pos0W0, ii3 + 1, 256 })); // far left edge
    segments.push_back(TrisoupSegmentNeighbours({ posTop + posW00, posTop + posWW0, ii3 + 3,  512 })); // far right edge
    segments.push_back(TrisoupSegmentNeighbours({ posTop + pos00W, posTop + pos0WW, ii3 + 9,  1024 })); // near left edge
    segments.push_back(TrisoupSegmentNeighbours({ posTop + posW0W, posTop + posWWW, ii3 + 11,  2048 })); // near right edge 

    // ------------ edges along z 
    // in node
    segments.push_back(TrisoupSegmentNeighbours({ posNode + pos000, posNode + pos00W, ii + 4, 1 + (1 << 14) })); // bottom left edge
    segments.push_back(TrisoupSegmentNeighbours({ posNode + pos0W0, posNode + pos0WW, ii + 5,  2 + (1 << 14) })); // top left edge
    segments.push_back(TrisoupSegmentNeighbours({ posNode + posWW0, posNode + posWWW, ii + 6,  4 + (1 << 14) })); // top right edge
    segments.push_back(TrisoupSegmentNeighbours({ posNode + posW00, posNode + posW0W, ii + 7,  8 + (1 << 14) })); // bottom right edge    
    // near 
    auto posNear = posNode - pos00W;
    segments.push_back(TrisoupSegmentNeighbours({ posNear + pos000, posNear + pos00W, ii2 + 4, 16 })); // bottom left edge
    segments.push_back(TrisoupSegmentNeighbours({ posNear + pos0W0, posNear + pos0WW, ii2 + 5,  32 })); // top left edge
    segments.push_back(TrisoupSegmentNeighbours({ posNear + posWW0, posNear + posWWW, ii2 + 6,  64 })); // top right edge
    segments.push_back(TrisoupSegmentNeighbours({ posNear + posW00, posNear + posW0W, ii2 + 7,  128 })); // bottom right edge
    // far 
    auto posFar = posNode + pos00W;
    segments.push_back(TrisoupSegmentNeighbours({ posFar + pos000, posFar + pos00W, ii3 + 4, 256 })); // bottom left edge
    segments.push_back(TrisoupSegmentNeighbours({ posFar + pos0W0, posFar + pos0WW, ii3 + 5,  512 })); // top left edge
    segments.push_back(TrisoupSegmentNeighbours({ posFar + posWW0, posFar + posWWW, ii3 + 6,  1024 })); // top right edge
    segments.push_back(TrisoupSegmentNeighbours({ posFar + posW00, posFar + posW0W, ii3 + 7,  2048 })); // bottom right edge
  }

  // Sort the list and find unique segments.
  std::sort(segments.begin(), segments.end());

  // find neighbourgs for unique segments
  TrisoupSegmentNeighbours localSegment = segments[0];

  auto it = segments.begin() + 1;
  neighbNodes.clear();
  std::vector<int> correspondanceUnique(segments.size(), -1);

  int uniqueIndex = 0;
  std::array<int, 18> pattern = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

  for (; it != segments.end(); it++) {
    if (localSegment.startpos != it->startpos || localSegment.endpos != it->endpos) {

      if (localSegment.neighboursMask & 15) {
        // the segment is a true segment and not only accumulation of copies; then push and jump to next segment
        neighbNodes.push_back(localSegment.neighboursMask);
        edgePattern.push_back(pattern);

        uniqueIndex++;
        for (int v = 0; v < 18; v++)
          pattern[v] = -1;
      }
      localSegment = *it;
    }
    else {
      // segment[i] is the same as localSegment
      // Accumulate into localSegment       
      localSegment.neighboursMask |= it->neighboursMask;
    }
    correspondanceUnique[it->index] = uniqueIndex;

    // ---------- neighbouring vertex parallel before
    if (it->neighboursMask >= 256 && it->neighboursMask <= 2048) { // lookinbg for vertex before (x left or y front or z below)
      int indexBefore = it->index - 24;

      if (correspondanceUnique[indexBefore] != -1) {
        pattern[0] = correspondanceUnique[indexBefore];
      }
    }

    // ---------    8-bit pattern = 0 before, 1-4 perp, 5-12 others
    static const int localEdgeindex[12][11] = {
      { 4,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // vertex 0
      { 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // vertex 1
      { 1,  5,  4,  9,  0,  8, -1, -1, -1, -1, -1}, // vertex 2
      { 0,  7,  4,  8,  2, 10,  1,  9, -1, -1, -1}, // vertex 3
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // vertex 4
      { 1,  0,  9,  4, -1, -1, -1, -1, -1, -1, -1}, // vertex 5
      { 3,  2,  0, 10, 11,  9,  8,  7,  5,  4, -1}, // vertex 6
      { 0,  1,  2,  8, 10,  4,  5, -1, -1, -1, -1}, // vertex 7
      { 4,  9,  1,  0, -1, -1, -1, -1, -1, -1, -1}, // vertex 8
      { 4,  0,  1, -1, -1, -1, -1, -1, -1, -1, -1}, // vertex 9
      { 5,  9,  1,  2,  8,  0, -1, -1, -1, -1, -1}, // vertex 10
      { 7,  8,  0, 10,  5,  2,  3,  9,  1, -1, -1}  // vertex 11
    };
    static const int patternIndex[12][11] = {
      { 3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // vertex 0
      { 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // vertex 1
      { 2,  3,  5,  8, 15, 17, -1, -1, -1, -1, -1}, // vertex 2
      { 2,  3,  5,  8,  9, 12, 15, 17, -1, -1, -1}, // vertex 3
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, // vertex 4
      { 1,  7, 10, 14, -1, -1, -1, -1, -1, -1, -1}, // vertex 5
      { 1,  2,  6,  9, 10, 11, 13, 14, 15, 16, -1}, // vertex 6
      { 2,  5,  8,  9, 12, 15, 17, -1, -1, -1, -1}, // vertex 7
      { 1,  4,  7, 14, -1, -1, -1, -1, -1, -1, -1}, // vertex 8
      { 1,  7, 14, -1, -1, -1, -1, -1, -1, -1, -1}, // vertex 9
      { 1,  2,  6, 14, 15, 16, -1, -1, -1, -1, -1}, // vertex 10
      { 1,  2,  6,  9, 11, 13, 14, 15, 16, -1, -1}  // vertex 11
    };

    if ((it->neighboursMask & 4095) <= 8) { // true edge, not a copy; so done as many times a nodes for the  true edge
      int indexLow = it->index % 12;   // true edge index within node
      for (int v = 0; v < 11; v++) {
        if (localEdgeindex[indexLow][v] == -1)
          break;

        int indexV = it->index - indexLow + localEdgeindex[indexLow][v]; // index of segment
        int Vidx = correspondanceUnique[indexV];
        if (Vidx != -1)  // check if already coded
          pattern[patternIndex[indexLow][v]] = Vidx;
        else
          std::cout << "#" << indexLow << "/" << v << " ";

      }
    }

  }
  if (localSegment.neighboursMask & 15) {
    neighbNodes.push_back(localSegment.neighboursMask);
    edgePattern.push_back(pattern);
  }
}
//============================================================================

Vec3<int32_t>
truncate(const Vec3<int32_t> in, const int32_t offset)
{
  Vec3<int32_t> out = in + offset;  
  if (out[0] < 0)
    out[0] = 0;
  if (out[1] < 0)
    out[1] = 0;
  if (out[2] < 0)
    out[2] = 0;

  return out;
}



//---------------------------------------------------------------------------

int32_t
trisoupVertexArc(int32_t x, int32_t y, int32_t Width_x, int32_t Width_y)
{
  int32_t score;

  if (x >= Width_x) {
    score = y;
  } else if (y >= Width_y) {
    score = Width_y + Width_x - x;
  } else if (x <= 0) {
    score = Width_y*2 + Width_x - y;
  } else {
    score = Width_y*2 + Width_x + x;
  }

  return score;
}

//---------------------------------------------------------------------------
bool
boundaryinsidecheck(const Vec3<int32_t> a, const int bbsize)
{
  return a[0] >= 0 && a[0] <= bbsize && a[1] >= 0 && a[1] <= bbsize && a[2] >= 0 && a[2] <= bbsize;
}

//---------------------------------------------------------------------------

void nonCubicNode
(
 const GeometryParameterSet& gps,
 const GeometryBrickHeader& gbh,
 const Vec3<int32_t>& leafpos,
 const int32_t blockWidth,
 const Box3<int32_t>& bbox,
 Vec3<int32_t>& newp,
 Vec3<int32_t>& neww,
 Vec3<int32_t>* corner )
{
  bool flag_n = gps.non_cubic_node_start_edge && ( gbh.slice_bb_pos_bits   > 0 );
  bool flag_f = gps.non_cubic_node_end_edge   && ( gbh.slice_bb_width_bits > 0 );
  for( int k=0; k<3; k++ ) {
    newp[k] = ( ( flag_n ) && ( leafpos[k] < bbox.min[k] ) ) ? bbox.min[k] : leafpos[k];
    neww[k] = ( ( flag_n ) && ( leafpos[k] < bbox.min[k] ) ) ?
      (blockWidth-(bbox.min[k]-leafpos[k])) :
      ( flag_f ) ? std::min(bbox.max[k]-leafpos[k]+1, blockWidth) : blockWidth;

    if (neww[k] == 0) {
      newp[k] = ((flag_n) && (leafpos[k] < bbox.min[k])) ? bbox.min[k] - 1
                                                         : leafpos[k];
      neww[k] = ((flag_n) && (leafpos[k] < bbox.min[k]))
        ? (blockWidth - (bbox.min[k] - 1 - leafpos[k]))
        : (flag_f) ? std::min(bbox.max[k] - leafpos[k] + 2, blockWidth)
                   : blockWidth;
    }
  }
  corner[POS_000] = {       0,       0,       0 };
  corner[POS_W00] = { neww[0],       0,       0 };
  corner[POS_0W0] = {       0, neww[1],       0 };
  corner[POS_WW0] = { neww[0], neww[1],       0 };
  corner[POS_00W] = {       0,       0, neww[2] };
  corner[POS_W0W] = { neww[0],       0, neww[2] };
  corner[POS_0WW] = {       0, neww[1], neww[2] };
  corner[POS_WWW] = { neww[0], neww[1], neww[2] };
  return;
}

Vec3<int64_t> findDirectionOfNormalCentroid(Vec3<int32_t> reference, std::vector<Vertex> list, TrisoupCentroidVertex cVert){
  Vec3<int> avg = 0;
  for (auto p1: list){
    avg += p1.pos;
  }
  avg /= list.size();
  return cVert.valid ? reference - cVert.pos : reference - avg;
}

// Determines if two trisoup vertecies are in the same plane
int areVerticiesOnSameFace(Vec3<int32_t> v1, Vec3<int32_t> v2, Vec3<int32_t>& nodew){
  for (int i = 0; i < 3; ++i){
    if (v1[i] == v2[i] && (v1[i]<0 || v1[i]>=nodew[i]-kTrisoupFpHalf)) return true;
  }
  return false;
}

// Determines if two trisoup vertecies are in the same segment (edge)
bool areVerticiesOnSameEdge(Vec3<int32_t> v1, Vec3<int32_t> v2, Vec3<int32_t> nodew){
  int counter = 0;
  for (int i = 0; i < 3; ++i){
    if (v1[i] == v2[i] && (v1[i] < 0 || v1[i] >= nodew[i] - kTrisoupFpHalf)) ++counter;
  }
  if (counter >= 2)
    return true;
  return false;
}

int findIndexOfPlane(Vec3<int32_t> v1, Vec3<int32_t> v2, Vec3<int>& offset){
  for (int i = 0; i < 3; ++i){
    if (v1[i] == v2[i]){
      if (v1[i] < 0) {
        offset[i] = -1;
        return 4 - 2*i;
      } else {
        offset[i] = 1;
        return 5 - 2*i;
      }
    }
  }
  return -1;
}

bool
determineNormVandCentroidContexts(
  const Vec3<int32_t>& nodeWidth,
  const std::vector<TrisoupNodeEdgeVertex>& eVerts,
  const TrisoupNodeEdgeVertex& eVert,
  node6nei& neiInds,
  const int bitDropped,
  Vec3<int32_t>& gravityCenter,
  Vec3<int32_t>& normalV,
  TrisoupCentroidContext& cctx,
  const std::vector<TrisoupCentroidVertex>& cVerts)
{
  // compute centroid
  int triCount = (int)eVert.vertices.size();
  std::vector<int> Weigths(triCount, 0);
  int Wtotal = 0;
  for (int k = 0; k < triCount; k++) {
    int k2 = k + 1;
    if (k2 >= triCount)
      k2 -= triCount;
    Vec3<int32_t> segment =
      (eVert.vertices[k].pos - eVert.vertices[k2].pos).abs();
    int weight = segment[0] + segment[1] + segment[2];

    Weigths[k] += weight;
    Weigths[k2] += weight;
    Wtotal += 2 * weight;
  }

  Vec3<int64_t> blockCentroid2 = 0;
  for (int j = 0; j < triCount; j++) {
    blockCentroid2 += int64_t(Weigths[j]) * eVert.vertices[j].pos;
  }
  blockCentroid2 /= int64_t(Wtotal);
  gravityCenter = {
    int(blockCentroid2[0]),int(blockCentroid2[1]), int(blockCentroid2[2]) };


  // judgment for refinement of the centroid along the domiannt axis
  if(triCount == 3){
    normalV = { 0,0,0 };
    cctx = { 0,0,0,0,0 };
    return false;
  }

  int dominantAxis = eVert.dominantAxis;
  int bitDropped2 = bitDropped;
  int halfDropped2 = bitDropped2 == 0 ? 0 : 1 << bitDropped2 - 1;

  // contextual information  for drift coding
  int minPos = eVert.vertices[0].pos[dominantAxis];
  int maxPos = eVert.vertices[0].pos[dominantAxis];
  for (int k = 1; k < triCount; k++) {
    if (eVert.vertices[k].pos[dominantAxis] < minPos)
      minPos = eVert.vertices[k].pos[dominantAxis];
    if (eVert.vertices[k].pos[dominantAxis] > maxPos)
      maxPos = eVert.vertices[k].pos[dominantAxis];
  }

  // find normal vector
  Vec3<int64_t> accuNormal = 0;
  if (triCount != 2){
    for (int k = 0; k < triCount; k++) {
      int k2 = k + 1;
      if (k2 >= triCount)
        k2 -= triCount;
      accuNormal += crossProduct(
        eVert.vertices[k].pos - gravityCenter,
        eVert.vertices[k2].pos - gravityCenter);
    }
  } else { // 2 vertices
    Vec3<int> offset = 0;
    int ind = findIndexOfPlane(eVert.vertices[0].pos, eVert.vertices[1].pos, offset);

	if (ind == -1)
      return false;

    if (neiInds.idx[ind] == -1) {
      normalV = { 0,0,0 };
      cctx = { 0,0,0,0,0 };
      return false;
    }
    Vec3<int> v = {offset[0] * (nodeWidth[0] << kTrisoupFpBits), offset[1] * (nodeWidth[1] << kTrisoupFpBits), offset[2] * (nodeWidth[2] << kTrisoupFpBits)};
    if (eVerts[neiInds.idx[ind]].vertices.size() != 0) {
      accuNormal = findDirectionOfNormalCentroid((eVert.vertices[0].pos + eVert.vertices[1].pos)/2 - v, eVerts[neiInds.idx[ind]].vertices, cVerts[neiInds.idx[ind]]);
    } else if (accuNormal == 0) {
      normalV = { 0,0,0 };
      cctx = { 0,0,0,0,0 };
      return false;
    }
  }
  int64_t normN = isqrt(
      accuNormal[0] * accuNormal[0]
    + accuNormal[1] * accuNormal[1]
    + accuNormal[2] * accuNormal[2]);
  Vec3<int32_t> _normalV = (accuNormal<< kTrisoupFpBits) / normN;
  normalV = _normalV;

  //  drift bounds
  cctx.ctxMinMax =
    std::min(8, (maxPos - minPos) >> (kTrisoupFpBits + bitDropped));
  int bound = (int(nodeWidth[dominantAxis]) - 1) << kTrisoupFpBits;
  int m = 1;
  int bw = nodeWidth[dominantAxis];
  for (; m < bw; m++) {
    Vec3<int32_t> temp = gravityCenter + m * normalV;
    if (temp[0] < 0 || temp[1] < 0 || temp[2] < 0
        || temp[0] > bound || temp[1] > bound || temp[2]> bound)
      break;
  }
  cctx.highBound = (m - 1) + halfDropped2 >> bitDropped2;

  m = 1;
  for (; m < bw; m++) {
    Vec3<int32_t> temp = gravityCenter - m * normalV;
    if (temp[0] < 0 || temp[1] < 0 || temp[2] < 0
        || temp[0] > bound || temp[1] > bound || temp[2] > bound)
      break;
  }
  cctx.lowBound = (m - 1) + halfDropped2 >> bitDropped2;
  cctx.lowBoundSurface =
    (gravityCenter[dominantAxis] - minPos + kTrisoupFpHalf >> kTrisoupFpBits)
    + halfDropped2 >> bitDropped2;
  cctx.highBoundSurface =
    (maxPos - gravityCenter[dominantAxis] + kTrisoupFpHalf >> kTrisoupFpBits)
    + halfDropped2 >> bitDropped2;

  return true;
}

void
decodeFlags(
  pcc::EntropyDecoder& decoder,
  const TrisoupOriPC& oriPC,
  int i,
  int idx,
  uint8_t& axiFlag,
  Vec3<bool>& posFlag,
  AdaptiveBitModel (&axiFlagctx)[3][2],
  AdaptiveBitModel (&posFlagctx)[3][2])
{
  if (decoder.decode(axiFlagctx[idx][oriPC.axiFlagctx[i][idx]])) {
    axiFlag |= 1 << idx;
    posFlag[idx] = decoder.decode(posFlagctx[idx][oriPC.posFlagctx[i][idx]]);
  }
}

void
decodeTrisoupOriPC(
  const TrisoupOriPC& oriPC,
  pcc::EntropyDecoder& arithmeticDecoder,
  bool fixFlag,
  int i,
  uint8_t& peraxiFlagMask,
  Vec3<bool>& perposFlag,
  AdaptiveBitModel& multiFlagctx,
  AdaptiveBitModel (&axiFlagctx)[3][2],
  AdaptiveBitModel (&posFlagctx)[3][2])
{
  uint8_t axisFlagMask = 0;
  Vec3<bool> axiFlag = false;
  Vec3<bool> posFlag = false;
  bool perMultiFlag = arithmeticDecoder.decode(multiFlagctx);
  if (perMultiFlag) {
    if (oriPC.axesEligible[i] > 4) {
      decodeFlags(arithmeticDecoder, oriPC, i , 2,axisFlagMask, posFlag, axiFlagctx, posFlagctx);
    }
    if ((oriPC.axesEligible[i] & 3) > 2) {
      decodeFlags(arithmeticDecoder, oriPC, i , 1,axisFlagMask, posFlag, axiFlagctx, posFlagctx);
    }
    int b = !(oriPC.axesEligible[i] & 1) + (oriPC.axesEligible[i] == 4);
    axisFlagMask |=(!axisFlagMask|| arithmeticDecoder.decode(axiFlagctx[b][oriPC.axiFlagctx[i][b]]))<< b;
    if (axisFlagMask & 1 << b)
      posFlag[b] = arithmeticDecoder.decode(posFlagctx[b][oriPC.posFlagctx[i][b]]);
  }
  peraxiFlagMask=axisFlagMask;
  perposFlag = posFlag;
}

void
isEligible(
  const int bitDropped,
  bool eligibleFlag,
  int cnt,
  int* vertexCount,
  uint8_t& eligible,
  std::vector<bool>& nodeEligible)
{
  static const int eligible_th2[7] = {1, 1, 2, 2, 3, 3, 4};
  static const int eligible_th1[7] = {3, 3, 4, 4, 5, 5, 6};
  int vNum = (bitDropped <= 1) ? 5 : (bitDropped <= 3) ? 6 : 7;
  const int* eligible_th = eligibleFlag ? eligible_th1 : eligible_th2;
  if (cnt > vNum) {
    for (int k = 0; k < 3; ++k) {
      if ((vertexCount[k] >= cnt - eligible_th[cnt - 6]|| vertexCount[k] <= eligible_th[cnt - 6])&& vertexCount[k] != cnt && vertexCount[k] != 0) {
        eligible|=1<<k;
      }
    }
    nodeEligible.push_back(eligible);
  } else {
    nodeEligible.push_back(false);
  }
}

void
generateCtx(int cnt, int* vertexCount, TrisoupOriPC& oriPC)
{
  Vec3<uint8_t> axiFlagctx;
  Vec3<uint8_t> posFlagctx;
  for (int k = 0; k <3; k++) {
    axiFlagctx[k] = (vertexCount[k] >=(cnt-1) || vertexCount[k] <= 1) ? 1 : 0;
    posFlagctx[k] = (vertexCount[k] >= (cnt - 1)) ? 1 : 0;
  }
  oriPC.axiFlagctx.push_back(axiFlagctx);
  oriPC.posFlagctx.push_back(posFlagctx);
}

int
detectVertex(uint8_t per_oriPointCloud, int j)
{
  int isDelete = 0;
  if (!(per_oriPointCloud & faceIdxToSubBlockMask[LUTSegmentExtToFaceIdx[j][0]])
      || !(per_oriPointCloud & faceIdxToSubBlockMask[LUTSegmentExtToFaceIdx[j][1]]))
    isDelete = 1;
  return isDelete;
}
// Trisoup geometry decoding, at both encoder and decoder.
// Compute from leaves, 6nei-nodes, and edge,centroid,face-vertices
// a set of triangles, and output their vertices.

void
decodeTrisoupCommon(
  const ringbuf<PCCOctree3Node>& leaves,
  const std::vector<node6nei> nodes6nei,
  const std::vector<TrisoupNodeEdgeVertex>& eVerts,
  const std::vector<TrisoupCentroidVertex>& cVerts,
  const std::vector<Vec3<int32_t>>& gravityCenter,
  std::vector<Vec3<int32_t>> normVs,
  std::vector<TrisoupFace>& faces,
  std::vector<TrisoupNodeFaceVertex>& fVerts,
  PCCPointSet3& recPointCloud,
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  int defaultBlockWidth,
  int positionClipValue,
  uint32_t samplingValue,
  const int bitDropped,
  const bool isCentroidDriftActivated,
  const bool isFaceVertexActivated,
  bool haloFlag,
  bool adaptiveHaloFlag,
  bool fineRayflag)
{
  Box3<int32_t> sliceBB;
  sliceBB.min = gbh.slice_bb_pos << gbh.slice_bb_pos_log2_scale;
  sliceBB.max = sliceBB.min + ( gbh.slice_bb_width << gbh.slice_bb_width_log2_scale );

  // Width of block.
  // in future, may override with leaf blockWidth
  const int32_t blockWidth = defaultBlockWidth;

  // ray tracing parameters
  int haloTriangle = 0;
  int haloBit =
    (((1 << bitDropped) - 1) << kTrisoupFpBits) / blockWidth;  // 28
  haloBit = (haloBit * 24) / 32;
  haloBit = haloBit > 40 ? 40 : haloBit;

  if (haloFlag) {
    if (samplingValue > 1) {
      haloTriangle =
        haloFlag ? (adaptiveHaloFlag ? 50 * samplingValue : 50) : 0;
      haloTriangle = haloTriangle > 100 ? 100 : haloTriangle;
    } else {
      haloTriangle = haloBit;
    }
  }

  int thickness = samplingValue > 1 ? 16 : 32;

  // Create list of refined pointss, one leaf at a time.
  std::vector<Vec3<int32_t>> refinedVertices;

  // ----------- loop on leaf nodes ----------------------
  for (int i = 0; i < leaves.size(); i++) {
    Vec3<int32_t> nodepos, nodew, corner[8];
    nonCubicNode(
      gps, gbh, leaves[i].pos, defaultBlockWidth, sliceBB, nodepos, nodew,
      corner);

    std::vector<Vec3<int32_t>> refinedVerticesBlock;

    for (int j = 0; j < eVerts[i].vertices.size(); j++) {
      // Get distance along edge of vertex.
      // Vertex code is the index of the voxel along the edge of the block
      // of surface intersection./ Put decoded vertex at center of voxel,
      // unless voxel is first or last along the edge, in which case put the
      // decoded vertex at the start or endpoint of the segment.
      Vec3<int32_t> point =
        eVerts[i].vertices[j].pos + kTrisoupFpHalf >> kTrisoupFpBits;
      // Add vertex to list of points.
      if (bitDropped || samplingValue > 1) {
        if (boundaryinsidecheck(point, blockWidth - 1))
          refinedVerticesBlock.push_back(nodepos + point);
      }
    }

    // Skip leaves that have fewer than 3 vertices.
    if (eVerts[i].vertices.size() < 2 || (eVerts[i].vertices.size() == 2 && !cVerts[i].valid)) {
      std::sort(refinedVerticesBlock.begin(), refinedVerticesBlock.end());
      refinedVerticesBlock.erase(
        std::unique(refinedVerticesBlock.begin(), refinedVerticesBlock.end()),
        refinedVerticesBlock.end());
      refinedVertices.insert(
        refinedVertices.end(),
        refinedVerticesBlock.begin(), refinedVerticesBlock.end());
      continue;
    }

    if (eVerts[i].vertices.size() > 3) {
      Vec3<int32_t> foundvoxel =
        cVerts[i].pos + truncateValue >> kTrisoupFpBits;
      if (boundaryinsidecheck(foundvoxel, blockWidth - 1))
        refinedVerticesBlock.push_back( foundvoxel + nodepos );
    }

    std::vector<Vertex> nodeVertices;
    if (eVerts[i].vertices.size() != 2 || fVerts[i].vertices.size() == 0) {
      for (int j = 0; j < eVerts[i].vertices.size(); j++) {
        nodeVertices.push_back(eVerts[i].vertices[j]);
        for(int k = 0; k < fVerts[i].vertices.size(); k++) {
          if (j == fVerts[i].formerEdgeVertexIdx[k])
            nodeVertices.push_back(fVerts[i].vertices[k]);
        }
      }
    } else {
      for (int j = 0; j < eVerts[i].vertices.size(); ++j) {
        if (j != 0)
          nodeVertices.push_back(eVerts[i].vertices[j]);
        for (int k = 0; k < fVerts[i].vertices.size(); ++k) {
          if (areVerticiesOnSameFace(fVerts[i].vertices[k].pos, eVerts[i].vertices[j].pos, nodew))
            nodeVertices.push_back(fVerts[i].vertices[k]);
        }
        if (j == 0)
          nodeVertices.push_back(eVerts[i].vertices[j]);
      }
    }

    // Divide vertices into triangles around centroid
    // and upsample each triangle by an upsamplingFactor.
    int vtxCount = nodeVertices.size();
    Vec3<int32_t> blockCentroid = cVerts[i].pos;
    Vec3<int32_t> v2 =
      vtxCount == 3 ? nodeVertices[2].pos : blockCentroid;
    Vec3<int32_t> v1 = nodeVertices[0].pos;
    Vec3<int32_t> posNode = nodepos << kTrisoupFpBits;

    for (int vtxIndex = 0;
        vtxIndex < (vtxCount == 3 ? 1 : vtxCount);
        vtxIndex++) {
      int j0 = vtxIndex;
      int j1 = vtxIndex + 1;
      if (eVerts[i].vertices.size() == 2 && j1 >= vtxCount) {
        break;
      } else if (j1 >= vtxCount){
        j1 -= vtxCount;
      }

      Vec3<int32_t> v0 = v1;
      v1 = nodeVertices[j1].pos;

      // range
      int minRange[3];
      int maxRange[3];
      for (int k = 0; k < 3; k++) {
        minRange[k] =
          std::max(0,
            std::min(std::min(v0[k], v1[k]), v2[k])
            + truncateValue >> kTrisoupFpBits);
        maxRange[k] =
          std::min(blockWidth,
            std::max(std::max(v0[k], v1[k]), v2[k])
            + truncateValue >> kTrisoupFpBits);
      }

      // precompute for rays
      Vec3<int64_t> edge1 = v1 - v0;
      Vec3<int64_t> edge2 = v2 - v0;
      int minDir = 1 << 28;
      int directionExcluded = 0;
      for (int k = 0; k <= 2; k++) {
        Vec3<int64_t> rayVector = 0;
        rayVector[k] = 1 << kTrisoupFpBits;
        Vec3<int64_t> h = crossProduct(edge1, edge2) >> kTrisoupFpBits;
        int64_t a = (rayVector * h) >> kTrisoupFpBits;
        if (std::abs(a) < minDir) {
          minDir = std::abs(a);
          directionExcluded = k;
        }
      }

      int64_t preC[4];
      Vec3<int64_t> v0_64 = v0;
      preC[0] = edge1[1] * edge2[2] - edge1[2] * edge2[1];
      preC[1] = edge1[2] * edge2[0] - edge1[0] * edge2[2];
      preC[2] = edge1[0] * edge2[1] - edge1[1] * edge2[0];
      Vec3<int64_t> tmp = crossProduct(v0_64, edge1) >> kTrisoupFpBits;
      preC[3] = int64_t(tmp[0]) * edge2[0] + int64_t(tmp[1]) * edge2[1]
        + int64_t(tmp[2]) * edge2[2];

      // applying ray tracing along direction
      for (int direction = 0; direction < 3; direction++) {
        if (directionExcluded == direction)
          // exclude most parallel direction
          continue;
        rayTracingAlongdirection(
          refinedVerticesBlock, direction, samplingValue, blockWidth, nodepos,
          minRange, maxRange, edge1, edge2, v0, v1, v2, haloTriangle,
          thickness, fineRayflag, preC);
      }
    }  // end loop on triangles

    std::sort(refinedVerticesBlock.begin(), refinedVerticesBlock.end());
    refinedVerticesBlock.erase(
      std::unique(refinedVerticesBlock.begin(), refinedVerticesBlock.end()),
      refinedVerticesBlock.end());
    refinedVertices.insert(
      refinedVertices.end(),
      refinedVerticesBlock.begin(), refinedVerticesBlock.end());

  }// end loop on leaves

  // Move list of points to pointCloud.
  recPointCloud.resize(refinedVertices.size());
  for (int i = 0; i < refinedVertices.size(); i++) {
    recPointCloud[i] = refinedVertices[i];
  }
}

// ---------------------------------------------------------------------------

void
decodeTrisoupFaceList(
  const ringbuf<PCCOctree3Node>& leaves,
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  int blockWidth,
  const int distanceSearchEncoder,
  std::vector<node6nei> nodes6nei,
  const std::vector<TrisoupNodeEdgeVertex> eVerts,
  const std::vector<Vec3<int32_t>> gravityCenter,
  const std::vector<TrisoupCentroidVertex> cVerts,
  std::vector<TrisoupNodeFaceVertex>& fVerts,
  std::vector<Vec3<int32_t>> normVs,
  std::vector<TrisoupFace>& faces,
  pcc::EntropyDecoder& arithmeticDecoder)
{
  AdaptiveBitModel ctxFaces;
  Box3<int32_t> sliceBB;
  sliceBB.min = gbh.slice_bb_pos << gbh.slice_bb_pos_log2_scale;
  sliceBB.max = sliceBB.min + (gbh.slice_bb_width << gbh.slice_bb_width_log2_scale);

  for (int i = 0; i < leaves.size(); i++) {
    Vec3<int32_t> nodepos, nodew, corner[8];
    nonCubicNode(
      gps, gbh, leaves[i].pos, blockWidth, sliceBB, nodepos, nodew,
      corner);

    for (int j = 1, nei = 0; j < 6; j += 2, nei++) {
      TrisoupFace _face(false);
      if (!cVerts[i].valid || !cVerts[i].boundaryInside)
        continue;
      int ii = nodes6nei[i].idx[j];
      if (-1 == ii || !cVerts[ii].valid || !cVerts[ii].boundaryInside)
        continue;
      // neighbour-node exists on this direction
      // centroid of the neighbour-node exists and inside of the boundary
      int eIdx[2][2] = {-1};
      int axis = 2 - nei;  // z,y,x-axis order 2,1,0
      Vec3<int32_t> nodeW = nodew << kTrisoupFpBits;
      Vec3<int32_t> zeroW = 0;
      int neVtxBoundaryFace = countTrisoupEdgeVerticesOnFace(eVerts[i], nodeW, axis);      
      if (2 == neVtxBoundaryFace && eVerts[i].vertices.size() == 2) {
        continue;
      } else if (2 == neVtxBoundaryFace || 3 == neVtxBoundaryFace && eVerts[i].vertices.size() != 2 && eVerts[ii].vertices.size() != 2) {
        Vertex fVert[2];
        findTrisoupFaceVertex(i, nei, nodes6nei[i], cVerts, nodew, fVert);
        determineTrisoupEdgeBoundaryLine(eVerts[i], nodeW, axis, fVert[0], eIdx[0]);
        determineTrisoupEdgeBoundaryLine(eVerts[ii], zeroW, axis, fVert[1], eIdx[1]);
        if ((-1 != eIdx[0][0]) && (-1 != eIdx[0][1])) {
          bool judge = determineTrisoupDirectionOfCentroidsAndFvert(
            eVerts[i], cVerts[i], cVerts[ii], gravityCenter[i],
            gravityCenter[ii], eIdx[0][0], eIdx[0][1], fVert);
          if (judge) {
            _face.connect = !!(arithmeticDecoder.decode(ctxFaces));
            if (_face.connect) {
              fVerts[i].formerEdgeVertexIdx.push_back(eIdx[0][0]);
              fVerts[i].vertices.push_back(fVert[0]);
              fVerts[ii].formerEdgeVertexIdx.push_back(eIdx[1][0]);
              fVerts[ii].vertices.push_back(fVert[1]);
            }
          }
        }
      } else if (1 == neVtxBoundaryFace && eVerts[i].vertices.size() == 2 && eVerts[ii].vertices.size() == 2) {
        Vertex fVert[2];
        findTrisoupFaceVertex(i, nei, nodes6nei[i], cVerts, nodew, fVert);
        fVerts[i].formerEdgeVertexIdx.push_back(1);
        fVerts[i].vertices.push_back(fVert[0]);
        fVerts[ii].formerEdgeVertexIdx.push_back(1);
        fVerts[ii].vertices.push_back(fVert[1]);
      }      
      // limited-faces to full-faces
      faces.push_back(_face);
    }
  }
}

// ---------------------------------------------------------------------------

void decodeTrisoupCentroids(
  const ringbuf<PCCOctree3Node>& leaves,
  std::vector<node6nei>& nodes6nei,
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  int defaultBlockWidth,
  const int bitDropped,
  const std::vector<TrisoupNodeEdgeVertex>& eVerts,
  std::vector<Vec3<int32_t>>& gravityCenter,
  std::vector<TrisoupCentroidVertex>& cVerts,
  std::vector<Vec3<int32_t>>& normVs,
  pcc::EntropyDecoder* arithmeticDecoder,
  const bool isFaceVertexActivated,
  AdaptiveBitModel ctxDrift0[9],
  AdaptiveBitModel ctxDriftSign[3][8][8],
  AdaptiveBitModel ctxDriftMag[4],
  bool twoEdgeVertices
  )
{
  const bool isCentroidDriftActivated =
    gbh.trisoup_centroid_vertex_residual_flag;
  int bitDropped2 = bitDropped;
  Box3<int32_t> sliceBB;
  sliceBB.min = gbh.slice_bb_pos << gbh.slice_bb_pos_log2_scale;
  sliceBB.max =
  sliceBB.min + (gbh.slice_bb_width << gbh.slice_bb_width_log2_scale);
  for (int i = 0; i < leaves.size(); i++) {

    Vec3<int32_t> nodepos, nodew, corner[8];
    nonCubicNode(
      gps, gbh, leaves[i].pos, defaultBlockWidth, sliceBB, nodepos, nodew,
      corner);

    // Skip leaves that have fewer than 3 vertices.
    if (!twoEdgeVertices) {
      if (eVerts[i].vertices.size() <= 2) {
        cVerts.push_back({ false, { 0, 0, 0 }, 0, true });
        normVs.push_back({ 0, 0, 0 });
        gravityCenter.push_back({ 0, 0, 0 });
        continue;
      }
    } else {
      if (eVerts[i].vertices.size() != 2) {
        continue;
      } else if (!areVerticiesOnSameFace(eVerts[i].vertices[0].pos, eVerts[i].vertices[1].pos, nodew)) {
        continue;
      } else if (areVerticiesOnSameEdge(eVerts[i].vertices[0].pos, eVerts[i].vertices[1].pos, nodew << kTrisoupFpBits)){
        continue;
      } else if (!isFaceVertexActivated) {
        continue;
      } 
    }


    Vec3<int32_t> gCenter = { 0 }, normalV = { 0 };
    TrisoupCentroidContext cctx = { 0 };

	auto neighbour = isFaceVertexActivated ? nodes6nei[i] : node6nei();
    bool driftCondition = determineNormVandCentroidContexts(
      nodew, eVerts, eVerts[i], neighbour, bitDropped, gCenter, normalV, cctx, cVerts);

    if (!(driftCondition && isCentroidDriftActivated)) {
      if (!twoEdgeVertices){
        cVerts.push_back({ false, gCenter, 0, true });
        normVs.push_back(normalV);
        gravityCenter.push_back(gCenter);
        continue;
      } else {
        continue;
      }
    }

    Vec3<int32_t> blockCentroid = gCenter;
    int driftQ = 0;
    int lowBound = cctx.lowBound;
    int highBound = cctx.highBound;
    int ctxMinMax = cctx.ctxMinMax;
    int lowBoundSurface = cctx.lowBoundSurface;
    int highBoundSurface = cctx.highBoundSurface;

      driftQ = arithmeticDecoder->decode(ctxDrift0[ctxMinMax]) ? 0 : 1;

      // if not 0, drift in [-lowBound; highBound]
      if (driftQ) {
        int sign = 1;
        // code sign
        if (!twoEdgeVertices) {
          int lowS = std::min(7, lowBoundSurface);
          int highS = std::min(7, highBoundSurface);
          if (highBound && lowBound)  // otherwise sign is knwow
            sign = arithmeticDecoder->decode(ctxDriftSign[lowBound == highBound ? 0 : 1 + (lowBound < highBound)][lowS][highS]);
          else if (!highBound)  // highbound is 0, so sign is negative;
            sign = 0;
          // otherwise sign is already set to positive
        }
        // code remaining bits 1 to 7 at most
        int magBound = (sign ? highBound : lowBound) - 1;

        int ctx = 0;
        while (magBound > 0) {
          int bit;
          if (ctx < 4)
            bit = arithmeticDecoder->decode(ctxDriftMag[ctx]);
          else
            bit = arithmeticDecoder->decode();

          if (bit) // magDrift==0 and magnitude coding is finished
            break;

          driftQ++;
          magBound--;
          ctx++;
        }

        if (!sign)
          driftQ = -driftQ;
      }

      // dequantize and apply drift
      int driftDQ = 0;
        if (driftQ) {
          driftDQ = std::abs(driftQ) << bitDropped2 + 6;
          int half = 1 << 5 + bitDropped2;
          int DZ = 2 * half / 3;
          driftDQ += DZ - half; 
          if (driftQ < 0)
            driftDQ = -driftDQ;
        }

      blockCentroid += (driftDQ * normalV) >> 6;
      blockCentroid[0] = std::max(-kTrisoupFpHalf, blockCentroid[0]);
      blockCentroid[1] = std::max(-kTrisoupFpHalf, blockCentroid[1]);
      blockCentroid[2] = std::max(-kTrisoupFpHalf, blockCentroid[2]);
    int blockWidth = defaultBlockWidth;
      blockCentroid[0] =
        std::min(((blockWidth - 1) << kTrisoupFpBits) + kTrisoupFpHalf - 1,
          blockCentroid[0]);
      blockCentroid[1] =
        std::min(((blockWidth - 1) << kTrisoupFpBits) + kTrisoupFpHalf - 1,
          blockCentroid[1]);
      blockCentroid[2] =
        std::min(((blockWidth - 1) << kTrisoupFpBits) + kTrisoupFpHalf - 1,
          blockCentroid[2]);
         
    if (twoEdgeVertices) {
      cVerts[i].valid = true;
      cVerts[i].pos = blockCentroid;
      cVerts[i].drift = driftDQ;
      cVerts[i].boundaryInside = true;
      normVs[i] = normalV;
      gravityCenter[i] = gCenter;
    } else {
      bool boundaryInside = true;
      if (!nodeBoundaryInsideCheck(nodew << kTrisoupFpBits, blockCentroid)) {
        boundaryInside = false;
      }
      cVerts.push_back({ true, blockCentroid, driftDQ, boundaryInside});
      normVs.push_back(normalV);
      gravityCenter.push_back(gCenter);
    }
  }
}

// ---------------------------------------------------------------------------

void decodeTrisoupVerticesSub(
  std::vector<bool>& segind,
  std::vector<uint8_t>& vertices,
  std::vector<uint16_t>& neighbNodes,
  std::vector<std::array<int, 18>>& edgePattern,
  int bitDropped,
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  pcc::EntropyDecoder& arithmeticDecoder) {
  const int nbitsVertices = gbh.trisoupNodeSizeLog2(gps) - bitDropped;
  const int max2bits = nbitsVertices > 1 ? 3 : 1;
  const int mid2bits = nbitsVertices > 1 ? 2 : 1;

  int iV = 0;
  std::vector<int> correspondanceSegment2V;

  AdaptiveBitModel ctxTempV2[120];

  CtxModelDynamicOBUF ctxTriSoup;
  CtxMapDynamicOBUF MapOBUFTriSoup[3];
  MapOBUFTriSoup[0].reset(14+1, 7); // flag
  MapOBUFTriSoup[1].reset(10+1, 6);  // first bit position
  MapOBUFTriSoup[2].reset(10+1+3+1, 6 + 1);  // second bit position

  const uint8_t initValue0[128] = {
     15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  42,  96,  71,  37,  15,
     15,  22,  51,  15,  15,  30,  27,  15,  15,  64,  15,  48,  15, 224, 171,
    127,  24, 127,  34,  80,  46, 141,  44,  66,  49, 127, 116, 140, 116, 105,
     39, 127, 116, 114,  46, 172, 109,  60,  73, 181, 161, 112,  65, 240, 159,
    127, 127, 127,  87, 183, 127, 116, 116, 195,  88, 152, 141, 228, 141, 127,
     80, 127, 127, 160,  92, 224, 167, 129, 135, 240, 183, 240, 184, 240, 240,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    127, 127, 127, 127, 127, 127, 127, 127
  };
  const uint8_t initValue1[64] = {
    116, 127, 118,  15, 104,  56,  97,  15,  96,  15,  29,  15,  95,  15,  46,
     15, 196, 116, 182,  53, 210, 104, 163,  69, 169,  15, 114,  15, 121,  15,
    167,  63, 240, 127, 184,  92, 240, 163, 197,  77, 239,  73, 179,  59, 213,
     48, 185, 108, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    127, 127, 127, 127
  };
  const uint8_t initValue2[128] = {
    141, 127, 127, 127, 189,  81,  36, 127, 143, 105, 103, 116, 201,  60,  38,
    116, 116, 127,  15, 127, 153,  59,  15, 116,  69, 105,  15, 127, 158,  93,
     36,  79, 141, 161, 116, 127, 197, 102,  53, 127, 177, 125,  88,  79, 209,
     75, 102,  28,  95,  74,  72,  56, 189,  62,  78,  18,  88, 116,  28,  45,
    237, 100, 152,  35, 141, 240, 127, 127, 208, 133, 101, 141, 186, 210, 168,
     98, 201, 124, 138,  15, 195, 194, 103,  94, 229,  82, 167,  23,  92, 197,
    112,  59, 185,  87, 156,  79, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    127, 127, 127, 127, 127, 127, 127, 127
  };
  MapOBUFTriSoup[0].init(initValue0);
  MapOBUFTriSoup[1].init(initValue1);
  MapOBUFTriSoup[2].init(initValue2);

  uint8_t _BufferOBUFleaves[CtxMapDynamicOBUF::kLeafBufferSize * (1 << CtxMapDynamicOBUF::kLeafDepth)];
  memset(_BufferOBUFleaves, 0, sizeof(uint8_t) * CtxMapDynamicOBUF::kLeafBufferSize * (1 << CtxMapDynamicOBUF::kLeafDepth));
  int _OBUFleafNumber = 0;

  for (int i = 0; i <= gbh.num_unique_segments_minus1; i++) {
    // reduced neighbour contexts
    int ctxE = (!!(neighbNodes[i] & 1)) + (!!(neighbNodes[i] & 2)) + (!!(neighbNodes[i] & 4)) + (!!(neighbNodes[i] & 8)) - 1; // at least one node is occupied 
    int ctx0 = (!!(neighbNodes[i] & 16)) + (!!(neighbNodes[i] & 32)) + (!!(neighbNodes[i] & 64)) + (!!(neighbNodes[i] & 128));
    int ctx1 = (!!(neighbNodes[i] & 256)) + (!!(neighbNodes[i] & 512)) + (!!(neighbNodes[i] & 1024)) + (!!(neighbNodes[i] & 2048));
    int direction = neighbNodes[i] >> 13; // 0=x, 1=y, 2=z

    // construct pattern
    auto patternIdx = edgePattern[i];
    int pattern = 0;
    int patternClose  = 0;
    int patternClosest  = 0;
    int nclosestPattern = 0;

    int towardOrAway[18] = { // 0 = toward; 1 = away
      0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    int mapping18to9[3][9] = {
      { 0, 1, 2, 3,  4, 15, 14, 5,  7},
      { 0, 1, 2, 3,  9, 15, 14, 7, 12},
      { 0, 1, 2, 9, 10, 15, 14, 7, 12}
    };

    for (int v = 0; v < 9; v++) {
      int v18 = mapping18to9[direction][v];

      if (patternIdx[v18] != -1) {
        int idxEdge = patternIdx[v18];
        if (segind[idxEdge]) {
          pattern |= 1 << v;
          int vertexPos2bits = vertices[correspondanceSegment2V[idxEdge]] >> std::max(0, nbitsVertices - 2);
          if (towardOrAway[v18])
            vertexPos2bits = max2bits - vertexPos2bits; // reverses for away
          if (vertexPos2bits >= mid2bits)
            patternClose |= 1 << v;
          if (vertexPos2bits >= max2bits)
            patternClosest |= 1 << v;
          nclosestPattern += vertexPos2bits >= max2bits && v <= 4;
        }
      }
    }

    int missedCloseStart = /*!(pattern & 1)*/ + !(pattern & 2) + !(pattern & 4);
    int nclosestStart = !!(patternClosest & 1) + !!(patternClosest & 2) + !!(patternClosest & 4);
    if (direction == 0) {
      missedCloseStart +=  !(pattern & 8) + !(pattern & 16);
      nclosestStart +=  !!(patternClosest & 8) + !!(patternClosest & 16);
    }
    if (direction == 1) {
      missedCloseStart +=  !(pattern & 8);
      nclosestStart +=  !!(patternClosest & 8) - !!(patternClosest & 16) ;
    }
    if (direction == 2) {
      nclosestStart +=  - !!(patternClosest & 8) - !!(patternClosest & 16) ;
    }

    // reorganize neighbours of vertex /edge (endpoint) independently on xyz
    int neighbEdge = (neighbNodes[i] >> 0) & 15;
    int neighbEnd = (neighbNodes[i] >> 4) & 15;
    int neighbStart = (neighbNodes[i] >> 8) & 15;
    if (direction == 2) {
      neighbEdge = ((neighbNodes[i] >> 0 + 0) & 1);
      neighbEdge += ((neighbNodes[i] >> 0 + 3) & 1) << 1;
      neighbEdge += ((neighbNodes[i] >> 0 + 1) & 1) << 2;
      neighbEdge += ((neighbNodes[i] >> 0 + 2) & 1) << 3;

      neighbEnd = ((neighbNodes[i] >> 4 + 0) & 1);
      neighbEnd += ((neighbNodes[i] >> 4 + 3) & 1) << 1;
      neighbEnd += ((neighbNodes[i] >> 4 + 1) & 1) << 2;
      neighbEnd += ((neighbNodes[i] >> 4 + 2) & 1) << 3;

      neighbStart = ((neighbNodes[i] >> 8 + 0) & 1);
      neighbStart += ((neighbNodes[i] >> 8 + 3) & 1) << 1;
      neighbStart += ((neighbNodes[i] >> 8 + 1) & 1) << 2;
      neighbStart += ((neighbNodes[i] >> 8 + 2) & 1) << 3;
    }

    // encode flag vertex

    int ctxMap1 = std::min(nclosestPattern, 2) * 15 * 2 +  (neighbEdge-1) * 2 + ((ctx1 == 4));    // 2* 15 *3 = 90 -> 7 bits
    int ctxMap2 = neighbEnd << 11;
    ctxMap2 |= (patternClose & (0b00000110)) << 9 - 1 ; // perp that do not depend on direction = to start
    ctxMap2 |= direction << 7;
    ctxMap2 |= (patternClose & (0b00011000))<< 5-3; // perp that  depend on direction = to start or to end
    ctxMap2 |= (patternClose & (0b00000001))<< 4;  // before
    int orderedPclosePar = (((pattern >> 5) & 3) << 2) + (!!(pattern & 128) << 1) + !!(pattern & 256);
    ctxMap2 |= orderedPclosePar;
    bool c = MapOBUFTriSoup[0].decodeEvolve(&arithmeticDecoder, ctxTriSoup, ctxMap2, ctxMap1, &_OBUFleafNumber, _BufferOBUFleaves);

    segind.push_back(c);
    correspondanceSegment2V.push_back(-1);

    // encode position vertex 
    if (c) {
      correspondanceSegment2V.back() = iV;

      uint8_t v = 0;
      int ctxFullNbounds = (4 * (ctx0 <= 1 ? 0 : (ctx0 >= 3 ? 2 : 1)) + (std::max(1, ctx1) - 1)) * 2 + (ctxE == 3);
      int b = nbitsVertices - 1;

      // first bit
      ctxMap1 = ctxFullNbounds * 2 + (nclosestStart > 0);
      ctxMap2 = missedCloseStart << 8;
      ctxMap2 |= (patternClosest & 1) << 7;
      ctxMap2 |= direction << 5;
      ctxMap2 |= patternClose & (0b00011111);
      int orderedPclosePar = (((patternClose >> 5) & 3) << 2) + (!!(patternClose & 128) << 1) + !!(patternClose & 256);

      int bit = MapOBUFTriSoup[1].decodeEvolve(&arithmeticDecoder, ctxTriSoup, ctxMap2, ctxMap1, &_OBUFleafNumber, _BufferOBUFleaves);
      v = (v << 1) | bit;
      b--;

      // second bit
      if (b >= 0) {
        ctxMap1 = ctxFullNbounds * 2 + (nclosestStart > 0);
        ctxMap2 = missedCloseStart << 8;
        ctxMap2 |= (patternClose & 1) << 7;
        ctxMap2 |= (patternClosest & 1) << 6;
        ctxMap2 |= direction << 4;
        ctxMap2 |= (patternClose & (0b00011111)) >> 1;
        ctxMap2 = (ctxMap2 << 4) + orderedPclosePar;

        bit = MapOBUFTriSoup[2].decodeEvolve(&arithmeticDecoder, ctxTriSoup, ctxMap2, (ctxMap1 << 1) + v, &_OBUFleafNumber, _BufferOBUFleaves);
        v = (v << 1) | bit;
        b--;
      }


      // third bit
      if (b >= 0) {
        int ctxFullNboundsReduced1 = (5 * (ctx0 >> 1) + missedCloseStart) * 2 + (ctxE == 3);
        v = (v << 1) | arithmeticDecoder.decode(ctxTempV2[4 * ctxFullNboundsReduced1 + v]);
        b--;
      }

      // remaining bits are bypassed
      for (; b >= 0; b--)
        v = (v << 1) | arithmeticDecoder.decode();
      vertices.push_back(v);
      iV++;
    }

  }

}

// ---------------------------------------------------------------------------

void decodeTrisoupVertices(
  const ringbuf<PCCOctree3Node>& leaves,
  const int defaultBlockWidth,
  std::vector<bool>& segind,
  std::vector<uint8_t>& vertices,
  std::vector<uint16_t>& neighbNodes,
  std::vector<std::array<int, 18>>& edgePattern,
  int bitDropped,
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  std::vector<TrisoupNodeEdgeVertex>& eVerts,
  pcc::EntropyDecoder& arithmeticDecoder,
  bool mergeFlag,
  bool fixFlag)
{
  // not use
  PCCPointSet3 dummyPointCloud;
  int dummyDistanceSearchEncoder = 0;
  std::vector<PCCOctree3Node> nodesPadded;
  PCCPointSet3 pointCloudPadding;
  std::vector<int> indices;
  TrisoupOriPC oriPC;
  processTrisoupVertices(
    gps, gbh, leaves, defaultBlockWidth, bitDropped,mergeFlag,fixFlag, true, dummyPointCloud,
    dummyDistanceSearchEncoder, neighbNodes, edgePattern, arithmeticDecoder,eVerts, oriPC, segind, vertices, nodesPadded, pointCloudPadding,
    indices);
}

// ---------------------------------------------------------------------------
// Project vertices along dominant axis (i.e., into YZ, XZ, or XY plane).
// Sort projected vertices by decreasing angle in [-pi,+pi] around center
// of block (i.e., clockwise) breaking ties in angle by
// increasing distance along the dominant axis.   

int findDominantAxis(
  std::vector<Vertex>& leafVertices,
  Vec3<uint32_t> blockWidth,
  Vec3<int32_t> blockCentroid ) {

  int dominantAxis = 0;
  int triCount = leafVertices.size();
  if (triCount > 3) {
    Vertex vertex;
    Vec3<int32_t> Width = blockWidth << kTrisoupFpBits;

    const int sIdx1[3] = { 2,2,1 };
    const int sIdx2[3] = { 1,0,0 };

    int maxNormTri = 0;
    for (int axis = 0; axis <= 2; axis++) {
      // order along axis
      for (int j = 0; j < triCount; j++) {
        Vec3<int32_t> s = leafVertices[j].pos + kTrisoupFpHalf; // back to [0,B]^3 for ordering
        leafVertices[j].theta = trisoupVertexArc(s[sIdx1[axis]], s[sIdx2[axis]],
                                                 Width[sIdx1[axis]], Width[sIdx2[axis]]);
        leafVertices[j].tiebreaker = s[axis];
      }
      std::sort(leafVertices.begin(), leafVertices.end(), vertex);

      // compute sum normal
      int32_t accuN = 0;
      for (int k = 0; k < triCount; k++) {
        int k2 = k + 1;
        if (k2 >= triCount)
          k2 -= triCount;
        Vec3<int32_t> h = crossProduct(leafVertices[k].pos - blockCentroid, leafVertices[k2].pos - blockCentroid);
        accuN += std::abs(h[axis]);
      }

      // if sumnormal is bigger , this is dominantAxis
      if (accuN > maxNormTri) {
        maxNormTri = accuN;
        dominantAxis = axis;
      }
    }

    for (int j = 0; j < leafVertices.size(); j++) {
      Vec3<int32_t> s = leafVertices[j].pos + kTrisoupFpHalf; // back to [0,B]^3 for ordering
      leafVertices[j].theta = trisoupVertexArc(s[sIdx1[dominantAxis]], s[sIdx2[dominantAxis]],
                                               Width[sIdx1[dominantAxis]], Width[sIdx2[dominantAxis]]);
      leafVertices[j].tiebreaker = s[dominantAxis];
    }
    std::sort(leafVertices.begin(), leafVertices.end(), vertex);
  } // end find dominant axis 

  return dominantAxis;
}

// ---------------------------------------------------------------------------

inline bool
IsPointInTriangle(
  const std::array<int64_t, 2>& P,
  const std::array<int64_t, 2>& V0,
  const std::array<int64_t, 2>& V1,
  const std::array<int64_t, 2>& V2,
  const int32_t& halo)
{
  return ((P[0] - V0[0]) * (V1[1] - V0[1]) - (P[1] - V0[1]) * (V1[0] - V0[0])
          >= halo)
    && ((P[0] - V1[0]) * (V2[1] - V1[1]) - (P[1] - V1[1]) * (V2[0] - V1[0])
        >= halo)
    && ((P[0] - V2[0]) * (V0[1] - V2[1]) - (P[1] - V2[1]) * (V0[0] - V2[0])
        >= halo);
}

// ---------------------------------------------------------------------------

void
rayTracingAlongdirection(
  std::vector<Vec3<int32_t>>& refinedVerticesBlock,
  int direction,
  uint32_t samplingValue,
  int blockWidth,
  Vec3<int32_t> nodepos,
  int minRange[3],
  int maxRange[3],
  Vec3<int64_t> edge1,
  Vec3<int64_t> edge2,
  Vec3<int64_t> v0,
  Vec3<int64_t> v1,
  Vec3<int64_t> v2,
  int haloTriangle,
  int thickness,
  bool fineRayflag,
  int64_t preC[4])
{
  // check if ray tracing is valid; if not skip the direction
  Vec3<int64_t> rayVector = 0;
  rayVector[direction] = 1 << kTrisoupFpBits;
  Vec3<int64_t> h = crossProduct(rayVector, edge2) >> kTrisoupFpBits;
  int64_t a = (edge1 * h) >> kTrisoupFpBits;
  if (std::abs(a) <= kTrisoupFpOne)
    return;

  //bounds
  const int g1pos[3] = {1, 0, 0};
  const int g2pos[3] = {2, 2, 1};
  const int32_t startposG1 = minRange[g1pos[direction]];
  const int32_t startposG2 = minRange[g2pos[direction]];
  const int32_t endposG1 = maxRange[g1pos[direction]];
  const int32_t endposG2 = maxRange[g2pos[direction]];
  const int32_t rayStart = minRange[direction] << kTrisoupFpBits;
  Vec3<int32_t> rayOrigin = rayStart;

  int64_t c[4];
  c[0] = preC[0] / a;
  c[1] = preC[1] / a;
  c[2] = preC[2] / a;
  c[3] = preC[3] / a;
  // c[3] = ((crossProduct(v0, edge1) >> kTrisoupFpBits) * edge2) / a;

  int64_t a1 = c[g1pos[direction]];
  int64_t a2 = c[g2pos[direction]];
  int64_t a3 =
    ((c[direction] * rayOrigin[direction]) >> kTrisoupFpBits) - c[3];

  int32_t sv = samplingValue;

  int64_t deltaT1 = a1 * sv;
  int64_t deltaT2 = a2 * sv;

  std::array<int64_t, 2> V0 = {v0[g1pos[direction]], v0[g2pos[direction]]};
  std::array<int64_t, 2> V1 = {v1[g1pos[direction]], v1[g2pos[direction]]};
  std::array<int64_t, 2> V2 = {v2[g1pos[direction]], v2[g2pos[direction]]};

  int64_t S =
    (V0[0] - V1[0]) * (V2[1] - V1[1]) - (V0[1] - V1[1]) * (V2[0] - V1[0]);

  if (S < 0) {
    V0.swap(V2);
  }
  //directly convert halo parameters from 2D to 3D
  int32_t haloTriangle2D =
    -(((std::abs(S) + kTrisoupFpHalf) >> kTrisoupFpBits) * haloTriangle);

  bool isVisible[129][129];
  int32_t tBuffer[129][129];
  for (int32_t g1 = startposG1; g1 <= endposG1; g1 += sv)
    for (int32_t g2 = startposG2; g2 <= endposG2; g2 += sv)
      isVisible[g1][g2] = false;

  for (int32_t g1 = startposG1; g1 <= endposG1; g1 += sv) {
    rayOrigin[g1pos[direction]] = g1 << kTrisoupFpBits;
    bool preGrid = false;
    for (int32_t g2 = startposG2; g2 <= endposG2; g2 += sv) {
      rayOrigin[g2pos[direction]] = g2 << kTrisoupFpBits;
      std::array<int64_t, 2> P = {g1 << kTrisoupFpBits, g2 << kTrisoupFpBits};

      isVisible[g1][g2] =
        IsPointInTriangle(P, V0, V1, V2, haloTriangle2D) ? true : false;

      if (preGrid && !isVisible[g1][g2])
        break;
      preGrid = isVisible[g1][g2];

      if (!isVisible[g1][g2])
        continue;

      if ((g2 - sv) >= startposG2 && isVisible[g1][g2 - sv]) {
        tBuffer[g1][g2] = tBuffer[g1][g2 - sv] + deltaT2;
      } else if ((g1 - sv) >= startposG1 && isVisible[g1 - sv][g2]) {
        tBuffer[g1][g2] = tBuffer[g1 - sv][g2] + deltaT1;
      } else {
        tBuffer[g1][g2] = ((P[0] * a1) >> kTrisoupFpBits)
          + ((P[1] * a2) >> kTrisoupFpBits) + a3;
      }

      // middle ray at integer position
      Vec3<int32_t> intersection = rayOrigin;
      Vec3<int32_t> intersectionUp = rayOrigin;
      Vec3<int32_t> intersectionDown = rayOrigin;
      Vec3<int32_t> foundvoxel;

      intersection[direction] += tBuffer[g1][g2];

      intersectionUp = intersection;
      intersectionUp[direction] += thickness;

      intersectionDown = intersection;
      intersectionDown[direction] -= thickness;

      if (true || samplingValue == 1) {
        foundvoxel = (intersectionUp + truncateValue) >> kTrisoupFpBits;
        if (boundaryinsidecheck(foundvoxel, blockWidth - 1)) {
          refinedVerticesBlock.push_back(nodepos + foundvoxel);
        }
        foundvoxel = (intersectionDown + truncateValue) >> kTrisoupFpBits;
        if (boundaryinsidecheck(foundvoxel, blockWidth - 1)) {
          refinedVerticesBlock.push_back(nodepos + foundvoxel);
        }
      }

      foundvoxel = (intersection + truncateValue) >> kTrisoupFpBits;
      if (boundaryinsidecheck(foundvoxel, blockWidth - 1)) {
        refinedVerticesBlock.push_back(nodepos + foundvoxel);
        continue;  // ray interected , no need to launch other rays
      }

      // if ray not interected then  augment +- offset
      if (samplingValue == 1 && fineRayflag) {
        const int Offset1[8] = {0, 0, -1, +1, -1, -1, +1, +1};
        const int Offset2[8] = {-1, +1, 0, 0, -1, +1, -1, +1};
        const int offset = kTrisoupFpHalf >> 2;

        for (int pos = 0; pos < 8; pos++) {
          Vec3<int32_t> rayOrigin2 = rayOrigin;
          rayOrigin2[g1pos[direction]] += Offset1[pos] * offset;
          rayOrigin2[g2pos[direction]] += Offset2[pos] * offset;
          std::array<int64_t, 2> P = {
            rayOrigin2[g1pos[direction]], rayOrigin2[g2pos[direction]]};

          bool isVisible =
            IsPointInTriangle(P, V0, V1, V2, haloTriangle2D) ? true : false;

          if (!isVisible)
            continue;
          int32_t t = ((P[0] * a1) >> kTrisoupFpBits)
            + ((P[1] * a2) >> kTrisoupFpBits) + a3;

          Vec3<int32_t> intersection = rayOrigin2;
          intersection[direction] += t;
          foundvoxel = (intersection + truncateValue) >> kTrisoupFpBits;
          if (boundaryinsidecheck(foundvoxel, blockWidth - 1)) {
            refinedVerticesBlock.push_back(nodepos + foundvoxel);
            continue;  // ray interected , no need to launch other rays
          }
        }  //pos

      }  // augment

    }  // loop g2
  }    //loop g1
}

// ---------------------------------------------------------------------------

bool nodeBoundaryInsideCheck( Vec3<int32_t> bw, Vec3<int32_t> pt )
{
  if( ( 0 <= pt[0] ) && ( pt[0] <= bw[0] ) &&
      ( 0 <= pt[1] ) && ( pt[1] <= bw[1] ) &&
      ( 0 <= pt[2] ) && ( pt[2] <= bw[2] ) ) {
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// find face vertex position from connection of centroid vertices of
// current and neighbour nodes one by one.
// (totally this function is called up to three times per node.)

void findTrisoupFaceVertex(
  const int nodeIdx,
  const int neiOrderIdx, // z,y,x-order : nei=0,1,2
  const node6nei& nodes6nei,
  const std::vector<TrisoupCentroidVertex>& cVerts,
  const Vec3<int32_t>& nodew,
  Vertex *fVert)
{
  int axis = 2 - neiOrderIdx; // z,y,x-order axis=2,1,0
  int neiNodeIdx = nodes6nei.idx[neiOrderIdx*2+1];
  int32_t c0facePos = (nodew[axis] << kTrisoupFpBits) - kTrisoupFpHalf;
  Vec3<int32_t> c0 = cVerts[nodeIdx].pos;
  Vec3<int32_t> c1 = cVerts[neiNodeIdx].pos;
  c1[axis] += nodew[axis] << kTrisoupFpBits;
  int32_t denom = c1[axis] - c0[axis];
  // denom=0 means that c0 and c1 are on the same grid(t=0).
  int32_t t =
    denom ? (((c0facePos - c0[axis]) << kTrisoupFpBits) / denom) : 0;
  Vertex faceVertex(
    { c0 + ((t * (c1 - c0) + kTrisoupFpHalf) >> kTrisoupFpBits), 0, 0 });
  fVert[0] = faceVertex;
  fVert[0].pos[axis] = c0facePos;
  fVert[1] = faceVertex;
  fVert[1].pos[axis] = - kTrisoupFpHalf;
}

// ---------------------------------------------------------------------------

int countTrisoupEdgeVerticesOnFace(
  const TrisoupNodeEdgeVertex& eVerts, Vec3<int32_t>& nodeW, int axis)
{
  int neVtxBoundaryFace = 0;
  // count edge vertices included in the face
  for (int k = 0; k < eVerts.vertices.size(); k++) {
    Vec3<int32_t> vtxC = eVerts.vertices[k].pos + kTrisoupFpHalf;
    neVtxBoundaryFace += nodeW[axis] == vtxC[axis];
  }
  return neVtxBoundaryFace;
}

// ---------------------------------------------------------------------------

void determineTrisoupEdgeBoundaryLine(
  const TrisoupNodeEdgeVertex& eVerts,
  Vec3<int32_t>& nodeW,
  int axis,
  Vertex& fvert,
  int* eIdx)
{
  // if there were two or three edge vertices on the face,
  // then to select the nearest bridge segment
  // between two edge vertices from tentative face vertex of current node,
  // edge vertices within current node are already sorted,
  // and two vertices are selected which make the nearest segment from temtative face vertex.
  // if the surface has a hole within the current node,
  // the sequential number of edge vertex couldn't be found,
  // false is returned without creating a face vertex.
  int evCnt = eVerts.vertices.size();
  int distMin = 1 << 30; // initial value must be larger than any case.
  for (int evIdx=0; evIdx < ((evCnt==3) ? 1 : evCnt); evIdx++) {
    int ev0 = evIdx;
    int ev1 = evIdx + 1;
    if (ev1 >= evCnt) { ev1 -= evCnt; }
    Vec3<int32_t> evCoord0 = eVerts.vertices[ev0].pos + kTrisoupFpHalf;
    Vec3<int32_t> evCoord1 = eVerts.vertices[ev1].pos + kTrisoupFpHalf;
    if (nodeW[axis] != evCoord0[axis] || nodeW[axis] != evCoord1[axis]) {
      continue;
    }
    Vec3<int32_t> middlePoint = (evCoord0 + evCoord1) >>1;
    Vec3<int32_t> distVec = (middlePoint - fvert.pos) >> kTrisoupFpBits;
    int dist = distVec[0] * distVec[0] + distVec[1] * distVec[1] + distVec[2] * distVec[2];

    if (distMin > dist) {
      eIdx[0] = ev0;
      eIdx[1] = ev1;
      distMin = dist;
    }
  }
}

// ---------------------------------------------------------------------------
// finding vector:
//   1. gravityCenter to Centroid of current node
//   2. gravityCenter to Centroid of current neighbour node
//   3. face vertex vector on face of current node
// and confirm directions of these three vectors are
// not invert with inner product.

bool determineTrisoupDirectionOfCentroidsAndFvert(
  const TrisoupNodeEdgeVertex& eVerts,
  const TrisoupCentroidVertex& cVerts0,
  const TrisoupCentroidVertex& cVerts1,
  const Vec3<int32_t>& gravityCenter0,
  const Vec3<int32_t>& gravityCenter1,
  int e0, int e1, Vertex *fVert)
{
  // unit vector between two edge vertices on boundary face
  Vec3<int64_t> euv = eVerts.vertices[e1].pos - eVerts.vertices[e0].pos;
  Vec3<int64_t> ef = fVert[0].pos - eVerts.vertices[e0].pos;
  int64_t en = (ef * euv) >> kTrisoupFpBits;
  int64_t euvNorm2 = (euv[0] * euv[0] + euv[1] * euv[1] + euv[2] * euv[2])>> kTrisoupFpBits;
  int64_t dp0 = (cVerts0.pos - gravityCenter0)* (ef * euvNorm2 - euv * en);
  int64_t dp1 = (cVerts1.pos - gravityCenter1)* (ef * euvNorm2 - euv * en);
  bool judge = (dp0 > 0) && (dp1 > 0);
  return judge;
}

//============================================================================

}  // namespace pcc
