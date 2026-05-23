#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <unordered_map>
#include <map>
#include <functional>

#include "Attribute.h"
#include "PayloadBuffer.h"
#include "PCCMath.h"
#include "PCCPointSet.h"
#include "geometry.h"
#include "hls.h"
#include "geometry_octree.h"
#include "ringbuf.h"
#include "quantization.h"

namespace pcc {
  

struct LayerGroupSlicingParams {
	bool layerGroupEnabledFlag = false;

  int subgroupBboxSize_Cubic;


	int numLayerGroupsMinus1 = 0;

	std::vector<int> numLayersPerLayerGroup;
	std::vector<int> numSubgroupsMinus1;

	std::vector<std::vector<Vec3<int>>> subgrpBboxOrigin;
	std::vector<std::vector<Vec3<int>>> subgrpBboxSize;

	std::vector<std::vector<int>> refLayerGroupId;
	std::vector<std::vector<int>> refSubgroupId;

	std::vector<std::vector<int>> parentLayerGroupId;
	std::vector<std::vector<int>> parentSubgroupId;

	std::vector<std::vector<int>> numNodes;
	std::vector<int> numPoints;

	std::vector<std::vector<size_t>> numPointsInSubgroups;
	std::vector<std::vector<size_t>> numExcludedPoints;
  

  bool attr_ctxt_ref_id_present_flag = false;
	std::vector<std::vector<int>> refLayerGroupIdAttribute;
	std::vector<std::vector<int>> refSubgroupIdAttribute;

  int subgroupBboxOrigin_bits_minus1;
  int subgroupBboxSize_bits_minus1;


	//++++++++++++++++++++++++++++++++++++++++++++++
	Vec3<int> rootNodeSizeLog2;
	Vec3<int> rootNodeSizeLog2_coded = 0;

	std::vector<int> numberOfPointsPerLodPerSubgroups;
	std::vector<int> numPointsPerAttrSubgroups;

	std::vector<std::vector<std::vector<uint32_t>>> dcmNodesIdx;

	std::vector<std::vector<int>> sliceSelectionIndicationFlag;
	//++++++++++++++++++++++++++++++++++++++++++++++
};


//============================================================================
struct LayerGroupKey{
	int layerGroupID;
	int subgroupID;

	LayerGroupKey(): layerGroupID(0), subgroupID(0){}
	
	LayerGroupKey(const int layerGroupID_, const int subgroupID_): layerGroupID(layerGroupID_), subgroupID(subgroupID_){}

	bool operator<(const LayerGroupKey& rhs) const{
        if (layerGroupID < rhs.layerGroupID) {
          return true;
        }
        if (layerGroupID > rhs.layerGroupID) {
          return false;
        }
        if (subgroupID < rhs.subgroupID) {
          return true;
        }
        if (subgroupID > rhs.subgroupID) {
          return false;
        }
        return false;
    }

};

class LayerGroupHandler {

  private:

    //general
    int _num_layer_groups;
    int _num_subgroups;
    std::vector<int> _num_layers;

    int _max_depth;
    std::vector<int> _end_depth;
    std::vector<int> _end_depth_of_parent;
    std::vector<int> _num_remained_layers;
    std::vector<int> _min_geom_node_size_log2;
    std::vector<int> _bitshift;

    std::vector<std::vector<int>> _lod_max_level;
    std::vector<std::vector<int>> _lod_min_geom_node_size_log2;
    std::vector<std::vector<int>> _lod_num_layers;
    std::vector<std::vector<int>> _lod_acum_layer_id;


    //for both of encoder and decode
    std::map<LayerGroupKey, int> _layerGroupIdxToSavedArrayIdx;
    std::map<int, LayerGroupKey> _savedArrayIdxToLayerGroupIdx;
    int _arrayCounter;
    std::vector<int> _refIdxSubgroup;
    std::vector<int> _refIdxSubgroupAttribute;
    std::vector<int> _parentIdxSubgroup;
    int _prevArrayIdx;
    
    
    
    Vec3<int> _rootBboxSize;
    

    int _id_output_layer_group;

    bool _roi_enable_flag;
    Vec3<int> _roi_origin, _roi_size;
    Vec3<int> _roi_min, _roi_max;

    std::vector<bool> _isOutputSubgroup;
    std::vector<bool> _isRequiredSubgroup;

  public:
    
    Vec3<int> _bboxMin, _bboxMax;

    std::vector<int> _codingOrder;
    
            
    
    std::vector<std::vector<std::unique_ptr<AttributeContexts>>> _ctxtMemAttrsSaved;
    std::vector<std::vector<std::unique_ptr<std::vector<uint32_t>>>> _subGroupPointIndexToOrgPredictorIndex;    
    std::vector<int64_t> _numChildPoints;
    

	  std::vector<std::vector<std::map<int, int>>> SQWA_coeff;
    int _numPrevDCMPoints;


    //for decoder
    std::vector<std::unique_ptr<pcc::ringbuf<PCCOctree3Node>>> _nodesSaved;
    std::vector<std::unique_ptr<GeometryOctreeContexts>> _ctxtMemSaved;
    std::vector<std::unique_ptr<std::vector<int>>> _phiBufferSaved;
    std::vector<std::unique_ptr<OctreePlanarState>> _planarSaved;	
    std::vector<std::unique_ptr<int>> _nodesBeforePlanarUpdateSaved;	
    std::vector<uint32_t> _numDCMPointsSubgroup;
    std::vector<std::vector<uint32_t>> _indexExtractedDCMPointsSubgroup;
    std::vector<int> _startIdxForEachOutputPoints;
    std::vector<Vec3<int>> _bboxMinVector;
    std::vector<Vec3<int>> _bboxMaxVector;

	std::vector<std::vector<int>> _SubgroupOccNeighPatEq0P;
	std::vector<std::vector<int>> _SubgroupOccNodeChildCntP;
	std::vector<std::vector<int>> _SubgroupOccNodeChildCntGP;
        
    //for geometry    
	std::vector<bool> _available_geom;
    std::vector<std::map<int, int>> _idx_uncoded_referenced_geom;
    std::vector<std::map<int, int>> _idx_uncoded_children_geom;
    std::vector<std::vector<std::map<int, int>>> _idx_uncoded_referenced_attr;
    std::vector<std::vector<std::map<int, int>>> _idx_uncoded_children_attr;
    
    std::vector<int> _numSubsequentSubgroups;

    //for encoder
    std::vector<std::unique_ptr<bool>> _planarEligibleKOctreeDepthSaved;	
    std::vector<std::vector<uint32_t>> _subGroupPointIndexToOrgPointIndex;
    std::vector<uint32_t> _encodedPointIndexToOrgPointIndex;
    std::vector<uint32_t> _orgPointIndexToEncodedPointIndex;
    std::vector<std::vector<uint32_t>> _nodeStart;
    std::vector<std::vector<uint32_t>> _nodeEnd;

    std::vector<std::vector<uint32_t>> _subGroupPointIndexToEncodedPointIndex;

    
    std::vector<int> _visited;

    bool _use_nonnormative_memory_handling;

  public:
	  LayerGroupHandler();
	  LayerGroupHandler(const SequenceParameterSet& sps, const GeometryBrickHeader& gbh);

      void initializeForGeometry(
        const int numSubgroups, 
        const bool isEncoder,
        const int numSkipLayerGroups=0,
        const bool use_nonnormative_memory_handling=false);
      
      int initializeForGeometry(
        LayerGroupSlicingParams& layerGroupParams, int numSkipLayerGroups);

      int initializeForGeometry(
        LayerGroupStructureInventory& inventory, 
        int slice_id,
        const int numSkipLayerGroups=0, bool use_use_nonnormative_memory_handling = false);

      void initializeForAttribute(const int numAttr, const bool isEncoder);

      void buildLoDSettings(const AttributeParameterSet& aps, const int attrIdx);


      void setReferenceTree(LayerGroupSlicingParams& layerGroupParams);
      void setReferenceTree(LayerGroupStructureInventory& inventory, int slice_id);

      void initializeSubgroupPointIndexToOrgPredictorIndex(const int numAttr,
    const int groupIndex, const int subgroupIndex,
    const int numberOfPoint,
    std::vector<uint32_t>& pointIndexToPredictorIndex);



      void releaseGeometryEncoderResource(int curArrayIdx);
      void releaseAttributeEncoderResource(int attrIdx, int curArrayIdx, bool isReusable);
      void releaseAttributeEncoderResource(int curArrayIdx,
        std::vector<std::vector<PCCPointSet3>>& subgroupPointCloud);
      void releaseGeometryDecoderResource(int curArrayIdx, bool context_reference_indication_flag, int numSubsequentSubgroups);
      void releaseAttributeDecoderResource(
        int attrIdx, 
        int curArrayIdx, 
        bool context_reference_indication_flag, 
        bool attr_ref_id_present_flag,
        std::vector<PCCPointSet3>& subgroupPointCloud);
      
      void releaseNodes();
      void releaseNodes(int arrayIdx);
      void releaseCtxForGeometry();
      void releaseCtxForGeometry(int arrayIdx);
      void releaseCtxForAttribute();
      void releaseCtxForAttribute(int attrIdx);
      void releaseCtxForAttribute(int attrIdx, int arrayIdx);
      void releaseIndexes();
      void releaseIndexes(int attrIdx);
      void releaseIndexes(int attrIdx, int arrayIdx);

      int setNewDependentUnit(const int layer_group_id, const int subgroup_id, const int ref_layer_group_id, const int ref_subgroup_id, Vec3<int> bboxMin, Vec3<int> bboxMax);
      int setNewDependentUnitForAttribute(const int layer_group_id, const int subgroup_id, const int ref_layer_group_id, const int ref_subgroup_id);

      void resetStoredCtxForGeometry(int arrayIdx);
      void loadCtxForGeometry(int arrayIdx,int refArrayIdx);
      void resetStoredCtxForAttribute(int attrIdx, int arrayIdx);
      void loadCtxForAttribute(int attrIdx, int arrayIdx,int refArrayIdx);
      
      void setSlicingParamGeom(const int group_id, const int subgroup_id, const int curArrayIdx, const int parentArrayIdx, const LayerGroupSlicingParams& layerGroupParams, GeometryGranularitySlicingParam& slicingParam);
      void setSlicingParamGeom(const int curArrayIdx, const GeometryBrickHeader& gbh, GeometryGranularitySlicingParam& slicingParam);
      void setSlicingParamGeom(const int curArrayIdx, const int parentArrayIdx, const DependentGeometryDataUnitHeader& gbh, GeometryGranularitySlicingParam& slicingParam);

      void storeSlicingParamGeom(const int curArrayIdx, const GeometryGranularitySlicingParam& slicingParam);

      
      void setSlicingParamAttr(
        const int attrIdx,
        const int group_id,
        DependentAttributeDataUnitHeader* abh_dep,
        PCCPointSet3* parentPointCloud, 
        bool subgroup_weight_adjustment_enabled_flag,
        int subgroup_weight_adjustment_method,
        std::vector<uint32_t>* subGroupPointIndexToOrgPredictorIndex,
        std::vector<uint64_t>* orgWeight,
        AttributeGranularitySlicingParam& slicingParam);

      void setSlicingParamAttrWithWeightAdjustment(
        const int attrIdx,
        const int curArrayIdx,
        const bool isReusable,
        const std::map<std::string, int>& attributeIdxMap,
        DependentAttributeDataUnitHeader* abh_dep,
        PCCPointSet3* parentPointCloud,
        const int subgroup_weight_adjustment_method,
        std::vector<std::vector<uint64_t>>& quantWeights,
        AttributeGranularitySlicingParam& slicingParam);
      
      //used only for weight_adjustment_method==1
      void initializeDcmNodesIdx(LayerGroupSlicingParams& layerGroupParams);
      void setDcmNodesIdx(int curArrayIdx, LayerGroupSlicingParams& layerGroupParams, const GeometryGranularitySlicingParam& slicingParam);


    bool computeWeightAdjustmentParametersInEncoder(
      const std::vector<const AttributeParameterSet*> aps, 
      std::vector<AttributeBrickHeader>& attr_abh, 
      const std::map<std::string, int> attributeIdxMap, 
      LayerGroupSlicingParams& lgsp, 
      const int weight_adjustment_method, 
      std::vector<std::vector<uint64_t>>& quantWeights,
      const PCCPointSet3& pointCloud, 
      const std::vector<std::vector<PCCPointSet3>>& subgroupPointCloud,
      AttributeInterPredParams& attrInterPredParams);

    void setGeometryHeader(const LayerGroupSlicingParams& layerGroupParams, GeometryBrickHeader& gbh);
    void setDependentGeometryHeader(int curArrayIdx, const LayerGroupSlicingParams& layerGroupParams, const GeometryBrickHeader& gbh, DependentGeometryDataUnitHeader& dep_gbh);
    void setAttributeHeader(
      const int curArrayIdx, 
      const bool aps_slice_qp_deltas_present_flag,
      const int qpCoefDependentUnits, 
      const int pointCount, 
      const AttributeGranularitySlicingParam slicingParam,
      AttributeBrickHeader& abh);

    void setDependentAttributeHeader(
      const int curArrayIdx, 
      const bool aps_slice_qp_deltas_present_flag,
      const int qpCoefDependentUnits, 
      const int qpLayerCoefDependentUnits,
      const int pointCount, 
      const AttributeGranularitySlicingParam slicingParam,
      const AttributeBrickHeader& abh,
      DependentAttributeDataUnitHeader& abh_dep);

      void setRoi(bool roiEnabledFlag, Vec3<double> roiOriginScaled, Vec3<int> roiSize);

      bool checkRoi(int curArrayIdx);
      bool checkRoi(Vec3<int> bboxOrigin, Vec3<int> bboxSize);

      PCCPointSet3 getDCMPoints(bool hasColour, bool hasReflectance, 
        int curArrayIdx,
        PCCPointSet3& subgroupPointCloud);

      PCCPointSet3 getPoints(bool hasColour, bool hasReflectance, 
        PCCPointSet3& subgroupPointCloud);

      PCCPointSet3 getParentPoints(int curArrayIdx, int parentArrayIdx, 
    PCCPointSet3& subgroupPointCloud);

      PCCPointSet3 getOutputPoints(bool hasColour, bool hasReflectance, 
        int curArrayIdx,
        PCCPointSet3& subgroupPointCloud);

      void countExcludedPoints(const int curArrayIdx, LayerGroupSlicingParams& layerGroupParams, PCCPointSet3& subgroupPointCloud);

      int setEncodedAttribute(
        int curArrayIdx, 
        PCCPointSet3& pointCloud, 
        PCCPointSet3& subgroupPointCloud);

      void setDecodedAttribute(
        int curArrayIdx, 
        PCCPointSet3& pointCloud, 
        PCCPointSet3& subgroupPointCloud,
        bool setColor,
        bool setReflectance);
      
      int getNumGroups();
      int getNumSubGroups();
  
      int getMaxDepth();
      int getEndDepth(int groupIndex);
      int getNumRemainedLayers(int groupIndex);
      int getBitshift(int groupIndex);
  
      int getMinNodeSize(int attrIdx,int groupIndex);
      int getMaxLevel(int attrIdx,int groupIndex);
      int getLodNumLayers(int attrIdx,int groupIndex);
      int getLodAcumLayerId(int attrIdx,int groupIndex);
      bool isOutputLayer(int groupIndex);
      bool isRequiredLayer(int groupIndex);

      
      int getTrueEndOfLayers(int maxDepthInGBH);

      int getArrayId(int layerGroupId, int subgroupId);
      LayerGroupKey getLayerGroupIds(int currArayId);
      int getParentIdx(int curArrayIdx);
      int getReferenceIdx(int curArrayIdx);
      int getReferenceIdxAttribute(int curArrayIdx);

      Vec3<int> getRootBoxSize();
      void setBox(int curArrayIdx, Vec3<int> min, Vec3<int> max);
      pcc::LayerGroupKey checkBox(int curArrayIdx, int refGroupIdx);
      bool checkBox(int curArrayIdx, point_t pos);

      void setStartIdxForEachOutputPoints(int curArrayIdx, int idx);
      int getStartIdxForEachOutputPoints(int curArrayIdx);


      int getNumDecodedDCMPoints(int curArrayIdx);
      
      int getNumExtractedDCMPoints(int curArrayIdx);

      int getIdxDCMPoints(int curArrayIdx, int index);

      bool hasCorrespondingGeometry(int layer_group_id, int subgroup_id);

      std::vector<std::unordered_multimap<int64_t,uint32_t>> makeHashForResampling(
        const std::vector<uint32_t>& numInLod, 
        const std::vector<uint32_t>& lodIndexes, 
        PCCPointSet3& pointCloud);

      void resampling(
        const std::vector<AttributeDescription>& descs,
        const int attrIdx,
        std::vector<std::unordered_multimap<int64_t,uint32_t>>& hashTable,
        const std::vector<int> numSubgroupsMinus1, 
        const PCCPointSet3& pointCloud, 
        const bool weight_adjustment_enabled_flag,
        const std::vector<std::vector<uint32_t>>& orgIndexToPredictorIndex, 
        std::vector<std::vector<PCCPointSet3>>& subgroupPointCloud);

      void resampling(
        int groupIndex, int subgroupIndex,
        const int attrIdx,
        const std::vector<AttributeDescription>& descs,
        std::vector<std::unordered_multimap<int64_t,uint32_t>>& hashTable,
        const PCCPointSet3& pointCloud, 
        const std::vector<size_t>& subGroupPointIndexToOrgIndex, 
        const bool weight_adjustment_enabled_flag,
        const std::vector<std::vector<uint32_t>>& orgIndexToPredictorIndex, 
        PCCPointSet3& subgroupPointCloud);

      void countChildPoints(
        const std::vector<int> numSubgroupsMinus1     
      );

      void setLods(int attrIdx, int curArrayIdx, AttributeLods& lods);
      
      void setEncodedPointIndexToOrgPointIndex(int curArrayIdx, const std::vector<uint32_t>& encIndexToOrgIndex, const PCCPointSet3& subgroupPointCloud);
      void setSubGroupPointIndexToEncodedPointIndex(const PCCPointSet3& pointCloud);
      void setSubGroupAttribute(int curArrayIdx, PCCPointSet3& subgroupPointCloud, const PCCPointSet3& pointCloud);
};

void splitSubgroup(std::vector<Vec3<int>>& splitBboxOrigin, std::vector<Vec3<int>>& splitBboxSize, std::vector<int>& splitNumPoints, const std::vector<Vec3<int>> subgrpPointCloud,
	const Vec3<int> curBboxOrigin, const Vec3<int> initSubgroupBboxSize, const int maxNumPoint);


int splitOneDirection(std::vector<Vec3<int>>& splitBboxOrigin, std::vector<Vec3<int>>& splitBboxSize, std::vector<int>& splitNumPoints, const std::vector<int> numPointsInSplitedSubgroups, std::vector<std::vector<Vec3<int>>> splitSubgroupPointCloud,
	const Vec3<int> subOrigin_in, const Vec3<int> subSize_in, const Vec3<int> BestDirection, const int numDivMinus1, Vec3<int> posHigh, const int maxNumPoint);

void splitSubgroup(std::vector<Vec3<int>>& splitBboxOrigin, std::vector<Vec3<int>>& splitBboxSize, std::vector<int>& splitNumPoints, const std::vector<Vec3<int>> subgrpPointCloud,
	const Vec3<int> curBboxOrigin, const Vec3<int> initSubgroupBboxSize, const int maxNumPoint) ;

bool setLayerGroupParams(const PCCPointSet3& cloud, const GeometryParameterSet gps, const OctreeEncOpts geom, LayerGroupSlicingParams& params);
}
