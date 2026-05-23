#include "layerGroupSlicing.h"

namespace pcc {
	
  

	LayerGroupHandler::LayerGroupHandler(){
    _num_layer_groups=0;
    _num_subgroups = 0;
    _max_depth = 0;
  }

	LayerGroupHandler::LayerGroupHandler(const SequenceParameterSet& sps, const GeometryBrickHeader& gbh){

    _num_layer_groups = gbh.num_layer_groups_minus1+1;
    _num_subgroups = 0;

    _num_layers.resize(_num_layer_groups);
    _end_depth.resize(_num_layer_groups);
    _end_depth_of_parent.resize(_num_layer_groups);
    _num_remained_layers.resize(_num_layer_groups);
    _min_geom_node_size_log2.resize(_num_layer_groups);
    _bitshift.resize(_num_layer_groups);
    
    _lod_max_level.resize(sps.attributeSets.size());
    _lod_min_geom_node_size_log2.resize(sps.attributeSets.size());
    _lod_num_layers.resize(sps.attributeSets.size());
    _lod_acum_layer_id.resize(sps.attributeSets.size());
  
    int endDepth = 0;
    for (int groupIndex = 0; groupIndex < _num_layer_groups; groupIndex++){

      _num_layers[groupIndex] = gbh.num_layers_minus1[groupIndex] + 1;
            
      _end_depth_of_parent[groupIndex] = endDepth;

      endDepth += gbh.num_layers_minus1[groupIndex]+1;
      _end_depth[groupIndex] = endDepth;
      
    }

	_max_depth = gbh.root_node_size_log2.max();

    
    _bboxMax = { 1 << _max_depth, 1 << _max_depth, 1 << _max_depth };
    _bboxMin = {0,0,0};
    _rootBboxSize = _bboxMax - _bboxMin;
    
    for (int groupIndex = 0; groupIndex < _num_layer_groups; groupIndex++){

      _num_remained_layers[groupIndex] = _max_depth - _end_depth[groupIndex];
      _bitshift[groupIndex] = _num_remained_layers[groupIndex]*3;
      
      _min_geom_node_size_log2[groupIndex] = _max_depth - _end_depth[groupIndex];      
    }


    _arrayCounter = 0;
    _layerGroupIdxToSavedArrayIdx.clear();
    _savedArrayIdxToLayerGroupIdx.clear();
    _refIdxSubgroup.clear();
    _refIdxSubgroupAttribute.clear();
    _parentIdxSubgroup.clear();
    
    _ctxtMemAttrsSaved.clear();

    _nodesSaved.clear();
    _ctxtMemSaved.clear();
    _phiBufferSaved.clear();
    _planarSaved.clear();
    _planarEligibleKOctreeDepthSaved.clear();
    _nodesBeforePlanarUpdateSaved.clear();
    _numDCMPointsSubgroup.clear();
    _indexExtractedDCMPointsSubgroup.clear();
    _startIdxForEachOutputPoints.clear();
    _bboxMinVector.clear();
    _bboxMaxVector.clear();


	_SubgroupOccNeighPatEq0P.clear();
	_SubgroupOccNodeChildCntP.clear();
	_SubgroupOccNodeChildCntGP.clear();
    
    _id_output_layer_group = _num_layer_groups-1;
    _roi_enable_flag = false;

    _idx_uncoded_referenced_geom.clear();
    _idx_uncoded_children_geom.clear();
    _idx_uncoded_referenced_attr.clear();
    _idx_uncoded_children_attr.clear();

    _use_nonnormative_memory_handling = false;

    _subGroupPointIndexToEncodedPointIndex.clear();

    _visited.clear();
  }

  void LayerGroupHandler::initializeForGeometry(
    const int numSubgroups, 
    const bool isEncoder,
    const int numSkipLayerGroups,
    const bool use_nonnormative_memory_handling)
  {

    if(!isEncoder){
      _use_nonnormative_memory_handling = use_nonnormative_memory_handling;
    }else{
      _use_nonnormative_memory_handling = true;
    }

    if(_use_nonnormative_memory_handling)
      _num_subgroups = numSubgroups;
    else
      _num_subgroups = 1;


    _arrayCounter=0;
    LayerGroupKey key(0, 0);
    _layerGroupIdxToSavedArrayIdx[key] = 0;
    _savedArrayIdxToLayerGroupIdx[_arrayCounter] = key;
    _arrayCounter++;

    _num_subgroups = numSubgroups;
    _refIdxSubgroup.resize(_num_subgroups);
    _refIdxSubgroupAttribute.resize(_num_subgroups);
    _parentIdxSubgroup.resize(_num_subgroups);
    _startIdxForEachOutputPoints.resize(_num_subgroups);


    {
      _nodesSaved.resize(_num_subgroups);
      _numDCMPointsSubgroup.resize(_num_subgroups);
      _indexExtractedDCMPointsSubgroup.resize(_num_subgroups);
        
      _ctxtMemSaved.resize(_num_subgroups);
      _phiBufferSaved.resize(_num_subgroups);
      _planarSaved.resize(_num_subgroups);
      _planarEligibleKOctreeDepthSaved.resize(_num_subgroups);
      _nodesBeforePlanarUpdateSaved.resize(_num_subgroups);
      
      _bboxMinVector.resize(_num_subgroups);
      _bboxMaxVector.resize(_num_subgroups);

      _id_output_layer_group = std::max(_num_layer_groups - 1 - numSkipLayerGroups,0);

	  _available_geom.resize(_num_subgroups, false);
      _idx_uncoded_referenced_geom.resize(_num_subgroups);
      _idx_uncoded_children_geom.resize(_num_subgroups);

      
      _visited.resize(_num_subgroups,false);
    }

	_SubgroupOccNeighPatEq0P.resize(_num_subgroups);
	_SubgroupOccNodeChildCntP.resize(_num_subgroups);
	_SubgroupOccNodeChildCntGP.resize(_num_subgroups);

    if(isEncoder){
      
      _subGroupPointIndexToEncodedPointIndex.resize(_num_subgroups);
      _subGroupPointIndexToOrgPointIndex.resize(_num_subgroups);
      _encodedPointIndexToOrgPointIndex.clear();
      _orgPointIndexToEncodedPointIndex.clear();
      
      _nodeStart.resize(_num_subgroups);
      _nodeEnd.resize(_num_subgroups);
    }
  }
  
  int LayerGroupHandler::initializeForGeometry(
        LayerGroupSlicingParams& layerGroupParams, int numSkipLayerGroups)
  {
    int totalNumUnits = 0;

	  for (int i = 0; i <= layerGroupParams.numLayerGroupsMinus1; i++)
		  totalNumUnits += layerGroupParams.subgrpBboxOrigin[i].size();
    
		if (numSkipLayerGroups > layerGroupParams.numLayerGroupsMinus1)
			numSkipLayerGroups = layerGroupParams.numLayerGroupsMinus1;

    initializeForGeometry(totalNumUnits, true, numSkipLayerGroups,true);
    
    setReferenceTree(layerGroupParams);

    return totalNumUnits;
  }

  int LayerGroupHandler::initializeForGeometry(
      LayerGroupStructureInventory& inventory, 
      int slice_id,
      const int numSkipLayerGroups, bool use_nonnormative_memory_handling)
  {
    
    int totalNumUnits = 0;
    if(use_nonnormative_memory_handling){
      for (auto sl : inventory.slice_ids) 
        if (slice_id == sl.lgsi_slice_id) 
          for (int group_id = 0; group_id < sl.lgsi_num_layer_groups_minus1 + 1; group_id++) 
            totalNumUnits +=sl.layerGroups[group_id].lgsi_num_subgroups_minus1+1;    
    }else{
      totalNumUnits = 1;
    }
    _numSubsequentSubgroups.resize(totalNumUnits);

    initializeForGeometry(totalNumUnits, false, numSkipLayerGroups,use_nonnormative_memory_handling);

    
    //make context reference tree using group structure inventory
    if(use_nonnormative_memory_handling)
      setReferenceTree(inventory,slice_id);

    return totalNumUnits;
  }



  void LayerGroupHandler::setReferenceTree(LayerGroupSlicingParams& layerGroupParams){
  

	  for (int group_id = 0; group_id <= layerGroupParams.numLayerGroupsMinus1; group_id++){
      for(int subgroup_id=0; subgroup_id<=layerGroupParams.numSubgroupsMinus1[group_id]; subgroup_id++){

                    
        int curArrayIdx = 0;
        int refArrayIdx4Context = 0;
        int refArrayIdx4Parent = 0;
        

        Vec3<int> bboxMin = { 0, 0, 0 };
        Vec3<int> bboxMax = getRootBoxSize();

        if(group_id==0){

            setBox(curArrayIdx, bboxMin, bboxMax);  
        }else{
          
		      bboxMin = layerGroupParams.subgrpBboxOrigin[group_id][subgroup_id];
		      bboxMax = bboxMin + layerGroupParams.subgrpBboxSize[group_id][subgroup_id];

          curArrayIdx = setNewDependentUnit(group_id, subgroup_id, layerGroupParams.refLayerGroupId[group_id][subgroup_id], layerGroupParams.refSubgroupId[group_id][subgroup_id],bboxMin,bboxMax);

          if(layerGroupParams.attr_ctxt_ref_id_present_flag){
            setNewDependentUnitForAttribute(group_id, subgroup_id, layerGroupParams.refLayerGroupIdAttribute[group_id][subgroup_id], layerGroupParams.refSubgroupIdAttribute[group_id][subgroup_id]);
          }
          
          refArrayIdx4Context = getReferenceIdx(curArrayIdx);
          refArrayIdx4Parent = getParentIdx(curArrayIdx);
          

          _idx_uncoded_referenced_geom[refArrayIdx4Context][curArrayIdx] = curArrayIdx; 
          _idx_uncoded_children_geom[refArrayIdx4Parent][curArrayIdx] = curArrayIdx; 
        }
      }
    }
  }
  void LayerGroupHandler::setReferenceTree(LayerGroupStructureInventory& inventory, int slice_id){
  
      for (auto sl : inventory.slice_ids) {
        if (slice_id == sl.lgsi_slice_id) {
          for (int group_id = 0; group_id < sl.lgsi_num_layer_groups_minus1 + 1; group_id++) {
              for(int subgroup_id=0; subgroup_id<=sl.layerGroups[group_id].lgsi_num_subgroups_minus1; subgroup_id++)
              {
                auto& subgroup = sl.layerGroups[group_id].subgroups[subgroup_id];

          
                int curArrayIdx = 0;
                int refArrayIdx4Context = 0;
                int refArrayIdx4Parent = 0;


                Vec3<int> bboxMin = { 0, 0, 0 };
                Vec3<int> bboxMax = getRootBoxSize();
	    

                if(group_id==0){
          
                    setBox(curArrayIdx, bboxMin, bboxMax);  
                }else{

		              bboxMin = subgroup.lgsi_subgroupBboxOrigin;
		              bboxMax = bboxMin + subgroup.lgsi_subgroupBboxSize;
          
                  curArrayIdx = setNewDependentUnit(group_id, subgroup_id, group_id-1, subgroup.lgsi_parent_subgroup_id,bboxMin,bboxMax);
          
                  refArrayIdx4Context = getReferenceIdx(curArrayIdx);
                  refArrayIdx4Parent = getParentIdx(curArrayIdx);
          

                  _idx_uncoded_referenced_geom[refArrayIdx4Context][curArrayIdx] = curArrayIdx; 
                  _idx_uncoded_children_geom[refArrayIdx4Parent][curArrayIdx] = curArrayIdx; 
                }
              }
          }
        }
      }
  }
  
  
  void LayerGroupHandler::resetStoredCtxForGeometry(int arrayIdx){
    _nodesSaved[arrayIdx].reset(new pcc::ringbuf<PCCOctree3Node>);
    _ctxtMemSaved[arrayIdx].reset(new GeometryOctreeContexts);
    _phiBufferSaved[arrayIdx].reset(new std::vector<int>);
    _planarSaved[arrayIdx].reset(new OctreePlanarState);
    _planarEligibleKOctreeDepthSaved[arrayIdx].reset(new bool(false));
    _nodesBeforePlanarUpdateSaved[arrayIdx].reset(new int(1));
  }
  void LayerGroupHandler::loadCtxForGeometry(int arrayIdx,int refArrayIdx){
      _ctxtMemSaved[arrayIdx].reset(new GeometryOctreeContexts(*_ctxtMemSaved[refArrayIdx]));
      _phiBufferSaved[arrayIdx].reset(new std::vector<int>(*_phiBufferSaved[refArrayIdx]));
      _planarSaved[arrayIdx].reset(new OctreePlanarState(*_planarSaved[refArrayIdx]));
  }
  void LayerGroupHandler::resetStoredCtxForAttribute(int attrIdx, int arrayIdx){
    _ctxtMemAttrsSaved[attrIdx][arrayIdx].reset(new AttributeContexts);
  }
  void LayerGroupHandler::loadCtxForAttribute(int attrIdx, int arrayIdx,int refArrayIdx){
    _ctxtMemAttrsSaved[attrIdx][arrayIdx].reset(new AttributeContexts(*_ctxtMemAttrsSaved[attrIdx][refArrayIdx]));
  }

  void LayerGroupHandler::setSlicingParamGeom(const int group_id, const int subgroup_id, const int curArrayIdx, const int parentArrayIdx, const LayerGroupSlicingParams& layerGroupParams, GeometryGranularitySlicingParam& slicingParam){
    
		slicingParam.bboxMin = _bboxMinVector[curArrayIdx];
		slicingParam.bboxMax = _bboxMaxVector[curArrayIdx];
        
    slicingParam.layer_group_enabled_flag = true;
		slicingParam.numPointsInsubgroup = layerGroupParams.numPointsInSubgroups[group_id][subgroup_id] - layerGroupParams.numExcludedPoints[group_id][subgroup_id];

		slicingParam.startDepth = 0;
		if (group_id > 0)
			slicingParam.startDepth = getEndDepth(group_id - 1);
		slicingParam.endDepth = getEndDepth(group_id);

		slicingParam.buf.dcmNodesIdx.resize(layerGroupParams.numLayersPerLayerGroup[group_id]);

    if(group_id>0)
      slicingParam.buf.refNodes = _nodesSaved[parentArrayIdx].get();

    slicingParam.buf.curPhiBuffer = _phiBufferSaved[curArrayIdx].get();
		slicingParam.buf.curPlanar = _planarSaved[curArrayIdx].get(),
		slicingParam.planarEligibleKOctreeDepth = *_planarEligibleKOctreeDepthSaved[curArrayIdx].get();
    slicingParam.nodesBeforePlanarUpdate = *_nodesBeforePlanarUpdateSaved[curArrayIdx].get();
    slicingParam.buf.dcmNodesIdx.resize(layerGroupParams.numLayersPerLayerGroup[group_id]);

  
  }
  void LayerGroupHandler::setSlicingParamGeom(const int curArrayIdx, const GeometryBrickHeader& gbh, GeometryGranularitySlicingParam& slicingParam){
  
    slicingParam.layer_group_enabled_flag = true;
		slicingParam.bboxMin = _bboxMinVector[curArrayIdx];
		slicingParam.bboxMax = _bboxMaxVector[curArrayIdx];

		slicingParam.startDepth = 0;
		slicingParam.endDepth = getEndDepth(curArrayIdx);

    slicingParam.buf.refNodes = nullptr;
    slicingParam.buf.curPhiBuffer = _phiBufferSaved[curArrayIdx].get();
	  slicingParam.buf.curPlanar = _planarSaved[curArrayIdx].get(),
    slicingParam.nodesBeforePlanarUpdate = *_nodesBeforePlanarUpdateSaved[curArrayIdx].get();
    slicingParam.buf.planarEligibleKOctreeDepth_perLayer = gbh.planarEligibleKOctreeDepth;
    slicingParam.numPointsInsubgroup = gbh.footer.geom_num_points_minus1 + 1;

	slicingParam.buf.SubgroupOccNeighPatEq0P = _SubgroupOccNeighPatEq0P[curArrayIdx];
	slicingParam.buf.SubgroupOccNodeChildCntP = _SubgroupOccNodeChildCntP[curArrayIdx];
	slicingParam.buf.SubgroupOccNodeChildCntGP = _SubgroupOccNodeChildCntGP[curArrayIdx];
  }
  void LayerGroupHandler::setSlicingParamGeom(const int curArrayIdx, const int parentArrayIdx, const DependentGeometryDataUnitHeader& gbh, GeometryGranularitySlicingParam& slicingParam){
    slicingParam.layer_group_enabled_flag = true;
	  slicingParam.bboxMin = _bboxMinVector[curArrayIdx];
	  slicingParam.bboxMax = _bboxMaxVector[curArrayIdx];

	  slicingParam.startDepth = getEndDepth(gbh.layer_group_id - 1);
	  slicingParam.endDepth = getEndDepth(gbh.layer_group_id);

    slicingParam.buf.refNodes = _nodesSaved[parentArrayIdx].get();
    slicingParam.buf.curPhiBuffer = _phiBufferSaved[curArrayIdx].get();
	  slicingParam.buf.curPlanar = _planarSaved[curArrayIdx].get(),
    slicingParam.nodesBeforePlanarUpdate = *_nodesBeforePlanarUpdateSaved[curArrayIdx].get();
    slicingParam.buf.planarEligibleKOctreeDepth_perLayer = gbh.planarEligibleKOctreeDepth;
    slicingParam.numPointsInsubgroup = gbh.footer.num_nodes_minus1 + 1;

	slicingParam.buf.SubgroupOccNeighPatEq0P = _SubgroupOccNeighPatEq0P[parentArrayIdx];
	slicingParam.buf.SubgroupOccNodeChildCntP = _SubgroupOccNodeChildCntP[parentArrayIdx];
	slicingParam.buf.SubgroupOccNodeChildCntGP = _SubgroupOccNodeChildCntGP[parentArrayIdx];  
  }

  void LayerGroupHandler::storeSlicingParamGeom(const int curArrayIdx, const GeometryGranularitySlicingParam& slicingParam){
    _numDCMPointsSubgroup[curArrayIdx] = slicingParam.numDCMPointsSubgroup;
	  _planarEligibleKOctreeDepthSaved[curArrayIdx].reset(new bool(slicingParam.planarEligibleKOctreeDepth));
    _nodesBeforePlanarUpdateSaved[curArrayIdx].reset(new int(slicingParam.nodesBeforePlanarUpdate));
    
	  _bboxMinVector[curArrayIdx] = slicingParam.bboxMin;
	  _bboxMaxVector[curArrayIdx] = slicingParam.bboxMax;

	  _SubgroupOccNeighPatEq0P[curArrayIdx] = slicingParam.buf.SubgroupOccNeighPatEq0P;
	  _SubgroupOccNodeChildCntP[curArrayIdx] = slicingParam.buf.SubgroupOccNodeChildCntP;
	  _SubgroupOccNodeChildCntGP[curArrayIdx] = slicingParam.buf.SubgroupOccNodeChildCntGP;
  }

  void LayerGroupHandler::setSlicingParamAttr(
        const int attrIdx,
        const int group_id,
        DependentAttributeDataUnitHeader* abh_dep,
        PCCPointSet3* parentPointCloud, 
        bool subgroup_weight_adjustment_enabled_flag,
        int subgroup_weight_adjustment_method,
        std::vector<uint32_t>* subGroupPointIndexToOrgPredictorIndex,
        std::vector<uint64_t>* orgWeight,
        AttributeGranularitySlicingParam& slicingParam){

    slicingParam.layer_group_enabled_flag = true;
    if(group_id>0)
      slicingParam.is_dependent_unit = true;
    else
      slicingParam.is_dependent_unit = false;

    slicingParam.minGeomNodeSizeLog2=getMinNodeSize(attrIdx, group_id);
    slicingParam.maxLevel= getMaxLevel(attrIdx, group_id);

    if(slicingParam.is_dependent_unit){
      slicingParam.buf.pointCloudParent = parentPointCloud;
      slicingParam.buf.abh_dep = abh_dep;      
    }else{
      slicingParam.buf.pointCloudParent = NULL;
      slicingParam.buf.abh_dep = NULL; 
    }

    if(subgroup_weight_adjustment_enabled_flag){
      slicingParam.subgroup_weight_adjustment_enabled_flag=true;
      slicingParam.subgroup_weight_adjustment_method=subgroup_weight_adjustment_method;
      slicingParam.buf.subGroupPointIndexToOrgPredictorIndex = subGroupPointIndexToOrgPredictorIndex;
      slicingParam.buf.orgWeight = orgWeight;          
    }else{
      slicingParam.subgroup_weight_adjustment_enabled_flag=false;          
    }
  
  }


  void LayerGroupHandler::setSlicingParamAttrWithWeightAdjustment(
        const int attrIdx,
        const int curArrayIdx,
        const bool isReusable,
        const std::map<std::string, int>& attributeIdxMap,
        DependentAttributeDataUnitHeader* abh_dep,
        PCCPointSet3* parentPointCloud,
        const int subgroup_weight_adjustment_method,
        std::vector<std::vector<uint64_t>>& quantWeights,
        AttributeGranularitySlicingParam& slicingParam){
   
    auto key = getLayerGroupIds(curArrayIdx);
    int group_id = key.layerGroupID;

    std::vector<uint32_t>* subGroupPointIndexToOrgPredictorIndex;
    std::vector<uint64_t>* orgWeight;

    slicingParam.subgroup_weight_adjustment_enabled_flag=true;
    slicingParam.subgroup_weight_adjustment_method=subgroup_weight_adjustment_method;


    //skip weight adjustment in the leaf group
    if(group_id == getNumGroups()-1){
      slicingParam.subgroup_weight_adjustment_enabled_flag = false;
    }else{

      if(isReusable){
        int refAttrId = attributeIdxMap.begin()->second;
        orgWeight = &quantWeights[refAttrId];
        subGroupPointIndexToOrgPredictorIndex = (_subGroupPointIndexToOrgPredictorIndex[refAttrId][curArrayIdx]).get();


      }else{
        orgWeight = &quantWeights[attrIdx];
        subGroupPointIndexToOrgPredictorIndex = (_subGroupPointIndexToOrgPredictorIndex[attrIdx][curArrayIdx]).get();
            
      }   
        
    } 
    
    setSlicingParamAttr(attrIdx, group_id, abh_dep, parentPointCloud, 
    slicingParam.subgroup_weight_adjustment_enabled_flag, subgroup_weight_adjustment_method,  subGroupPointIndexToOrgPredictorIndex, orgWeight, slicingParam);
  
  }

  
  //used only for weight_adjustment_method==1
  void LayerGroupHandler::initializeDcmNodesIdx(LayerGroupSlicingParams& layerGroupParams){
  
	  int maxDepth = 0;
	  for (int i = 0; i <= layerGroupParams.numLayerGroupsMinus1; i++)
		  maxDepth += layerGroupParams.numLayersPerLayerGroup[i];

	  layerGroupParams.dcmNodesIdx.resize(maxDepth);
	  int layerIdx = 0;
	  for (int i = 0; i <= layerGroupParams.numLayerGroupsMinus1; i++)
		  for (int j = 0; j < layerGroupParams.numLayersPerLayerGroup[i]; j++, layerIdx++) {
			  layerGroupParams.dcmNodesIdx[layerIdx].resize(layerGroupParams.numSubgroupsMinus1[i] + 1);
		  }

	  _numPrevDCMPoints = 0;
  }
  
  //used only for weight_adjustment_method==1
  void LayerGroupHandler::setDcmNodesIdx(int curArrayIdx, LayerGroupSlicingParams& layerGroupParams, const GeometryGranularitySlicingParam& slicingParam){
  
		int group_id = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroup_id = getLayerGroupIds(curArrayIdx).subgroupID;

		for (int j = 0; j < layerGroupParams.numLayersPerLayerGroup[group_id]; j++) {
			assert(layerGroupParams.dcmNodesIdx[slicingParam.startDepth + j][subgroup_id].size() == 0);
			layerGroupParams.dcmNodesIdx[slicingParam.startDepth + j][subgroup_id].clear();
			for (int k = 0; k < slicingParam.buf.dcmNodesIdx[j].size(); k++) {
				layerGroupParams.dcmNodesIdx[slicingParam.startDepth + j][subgroup_id].push_back(_numPrevDCMPoints + slicingParam.buf.dcmNodesIdx[j][k]);
			}
		}

		_numPrevDCMPoints += _numDCMPointsSubgroup[curArrayIdx];        
  }

  bool LayerGroupHandler::computeWeightAdjustmentParametersInEncoder(
      const std::vector<const AttributeParameterSet*> aps, 
      std::vector<AttributeBrickHeader>& attr_abh, 
      const std::map<std::string, int> attributeIdxMap, 
      LayerGroupSlicingParams& lgsp, 
      const int weight_adjustment_method, 
      std::vector<std::vector<uint64_t>>& quantWeights,
      const PCCPointSet3& pointCloud, 
      const std::vector<std::vector<PCCPointSet3>>& subgroupPointCloud,
      AttributeInterPredParams& attrInterPredParams){
  
    AttributeLods lodsOrg;
    bool isReusable = true;
      
    std::vector<std::vector<uint32_t>> pointIndexToPredictorIndex;
    pointIndexToPredictorIndex.resize(attributeIdxMap.size());

    for (const auto& it : attributeIdxMap) {
      int attrIdx = it.second;
      const auto& attr_aps = *aps[attrIdx];

      if(weight_adjustment_method==0){

        if(lodsOrg.empty() || !lodsOrg.isReusable(attr_aps, attr_abh[attrIdx])){
            
          pointIndexToPredictorIndex[attrIdx] = lodsOrg.generate(attr_aps, attr_abh[attrIdx], pointCloud.getPointCount() - 1, 0, pointCloud, attrInterPredParams, getMaxDepth());

          isReusable = isReusable & lodsOrg.isReusable(attr_aps, attr_abh[attrIdx]);

          computeQuantizationWeights(lodsOrg.predictors, quantWeights[attrIdx], attr_aps.quant_neigh_weight);  

              
          for (int groupIndex = 0; groupIndex < lgsp.numLayerGroupsMinus1+1; groupIndex++)    
            for (int subgroupIndex = 0; subgroupIndex < lgsp.numSubgroupsMinus1[groupIndex] + 1; subgroupIndex++){
              initializeSubgroupPointIndexToOrgPredictorIndex(attrIdx,groupIndex,subgroupIndex,subgroupPointCloud[groupIndex][subgroupIndex].getPointCount(),pointIndexToPredictorIndex[attrIdx]);
              
            }
        }
      }else{

        if (lodsOrg.empty()) {

				  attr_abh[attrIdx].subgroup_weight_adjustment_enabled_flag = true;

				  lodsOrg.generate_forFullLayerGroupSlicingEncoder(attr_aps, attr_abh[attrIdx], pointCloud.getPointCount() - 1, 0, pointCloud, attrInterPredParams,
					  lgsp.layerGroupEnabledFlag,
					  getMaxDepth(),
					  getMaxDepth(),
					  lgsp.dcmNodesIdx,
					  lgsp.numLayersPerLayerGroup,
					  lgsp.subgrpBboxOrigin,
					  lgsp.subgrpBboxSize,
					  lgsp.sliceSelectionIndicationFlag,
					  lgsp.numberOfPointsPerLodPerSubgroups);


				  std::vector<uint64_t> quantWeights_outOfSubgroup;
				  if (lgsp.layerGroupEnabledFlag) {
					  quantWeights_outOfSubgroup.clear();
					  quantWeights_outOfSubgroup.resize(lodsOrg.predictors.size());
				  }

				  std::vector<uint64_t> quantWeights;
				  computeQuantizationWeights_forFullLayerGroupSlicingEncoder(
					  lodsOrg.predictors, quantWeights, attr_aps.quant_neigh_weight,
					  &quantWeights_outOfSubgroup, true, attrInterPredParams.enableAttrInterPred);

				  SQWA_coeff = SubgroupQuantizationWeightAdjustment_forFullLayerGroupSlicingEncoder(
					  quantWeights, 
					  lgsp.numLayerGroupsMinus1,
					  lgsp.numSubgroupsMinus1,
					  lgsp.numberOfPointsPerLodPerSubgroups,
					  lgsp.numLayersPerLayerGroup, 
					  quantWeights_outOfSubgroup, 
					  lodsOrg.numRefNodesInTheSameSubgroup);

			  }
        
      }
          

          
    }

     return isReusable;
  }

  
  void LayerGroupHandler::setGeometryHeader(const LayerGroupSlicingParams& layerGroupParams, GeometryBrickHeader& gbh){
    
		gbh.planarEligibleKOctreeDepth.resize(layerGroupParams.numLayersPerLayerGroup[0], 0);
		gbh.planarEligibleKOctreeDepth[0] = 0;
    gbh.numSubsequentSubgroups = _idx_uncoded_referenced_geom[0].size();

  }
  void LayerGroupHandler::setDependentGeometryHeader(int curArrayIdx, const LayerGroupSlicingParams& layerGroupParams, const GeometryBrickHeader& gbh, DependentGeometryDataUnitHeader& dep_gbh){
    
		int group_id = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroup_id = getLayerGroupIds(curArrayIdx).subgroupID;
		int refArrayIdx4Context = getReferenceIdx(curArrayIdx);
		int refArrayIdx4Parent = getParentIdx(curArrayIdx);

		dep_gbh.geom_parameter_set_id = gbh.geom_geom_parameter_set_id;
		dep_gbh.geom_slice_id = gbh.geom_slice_id;
		dep_gbh.layer_group_id = group_id;

		dep_gbh.subgroup_id = subgroup_id;
		dep_gbh.subgroupBboxOrigin = _bboxMinVector[curArrayIdx];
		dep_gbh.subgroupBboxSize = _bboxMaxVector[curArrayIdx] - _bboxMinVector[curArrayIdx];


		dep_gbh.ref_layer_group_id = layerGroupParams.refLayerGroupId[group_id][subgroup_id];
		dep_gbh.ref_subgroup_id = layerGroupParams.refSubgroupId[group_id][subgroup_id];

		if (_idx_uncoded_referenced_geom[curArrayIdx].size() > 0){
			dep_gbh.context_reference_indication_flag = true;
      dep_gbh.numSubsequentSubgroups = _idx_uncoded_referenced_geom[curArrayIdx].size();
    }
		else{
			dep_gbh.context_reference_indication_flag = false;
      dep_gbh.numSubsequentSubgroups = 0;
    }

		dep_gbh.planarEligibleKOctreeDepth.resize(layerGroupParams.numLayersPerLayerGroup[group_id], 0);
		dep_gbh.planarEligibleKOctreeDepth[0] = *_planarEligibleKOctreeDepthSaved[refArrayIdx4Parent];
  }

  int inline getQpDelta(
    const bool aps_slice_qp_deltas_present_flag,
    const int qpCoefDependentUnits,
    const int pointCount,
    const int numChildPoints){
  
    int qp_delta = 0;
    if(aps_slice_qp_deltas_present_flag && qpCoefDependentUnits>0){
      double average_followers = (double)numChildPoints/(double)pointCount;
            
      if(average_followers>0){
        double l2 = log2(average_followers);
        qp_delta = - int(l2 * qpCoefDependentUnits); 

        return qp_delta;
      }
    }
    return 0;
  }

  void LayerGroupHandler::setAttributeHeader(
    const int curArrayIdx, 
    const bool aps_slice_qp_deltas_present_flag,
    const int qpCoefDependentUnits, 
    const int pointCount, 
    const AttributeGranularitySlicingParam slicingParam,
    AttributeBrickHeader& abh){
    
		int groupIndex = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroupIndex = getLayerGroupIds(curArrayIdx).subgroupID;
          
    abh.attr_qp_delta_luma = getQpDelta(aps_slice_qp_deltas_present_flag, qpCoefDependentUnits, pointCount, _numChildPoints[curArrayIdx]);
    

    abh.subgroup_weight_adjustment_enabled_flag = false;
    if(slicingParam.subgroup_weight_adjustment_enabled_flag && groupIndex<getNumGroups()-1){
      abh.subgroup_weight_adjustment_enabled_flag = true;
              
      if(slicingParam.subgroup_weight_adjustment_method==1){
	      abh.subgroup_weight_adj_coeff_a=SQWA_coeff[groupIndex][subgroupIndex][0];
	      abh.subgroup_weight_adj_coeff_b=SQWA_coeff[groupIndex][subgroupIndex][1];              
      }
    }
  }

  void LayerGroupHandler::setDependentAttributeHeader(
    const int curArrayIdx, 
    const bool aps_slice_qp_deltas_present_flag,
    const int qpCoefDependentUnits, 
    const int qpLayerCoefDependentUnits,
    const int pointCount, 
    const AttributeGranularitySlicingParam slicingParam,
    const AttributeBrickHeader& abh,
    DependentAttributeDataUnitHeader& abh_dep){
    
		int groupIndex = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroupIndex = getLayerGroupIds(curArrayIdx).subgroupID;
    
    int refArrayIdx = getReferenceIdxAttribute(curArrayIdx);            
    auto refKey = getLayerGroupIds(refArrayIdx);
    int refGroupId = refKey.layerGroupID;
    int refSubgroupId = refKey.subgroupID;
    
    int qp_delta = 0;

    abh_dep.attr_attr_parameter_set_id = abh.attr_attr_parameter_set_id;
    abh_dep.attr_sps_attr_idx = abh.attr_sps_attr_idx;
    abh_dep.attr_geom_slice_id = abh.attr_geom_slice_id;
    abh_dep.layer_group_id = groupIndex;
    abh_dep.subgroup_id = subgroupIndex;
    abh_dep.ref_layer_group_id = refGroupId;
    abh_dep.ref_subgroup_id = refSubgroupId;
    abh_dep.attr_qp_delta_luma = 0;
    abh_dep.attr_qp_delta_chroma = 0;
    abh_dep.subgroup_weight_adjustment_enabled_flag = false;

    if(groupIndex<getNumGroups()-1){
            
      abh_dep.attr_qp_delta_luma = getQpDelta(aps_slice_qp_deltas_present_flag, qpCoefDependentUnits, pointCount, _numChildPoints[curArrayIdx]);
          
      if(slicingParam.subgroup_weight_adjustment_enabled_flag){
        abh_dep.subgroup_weight_adjustment_enabled_flag = true;
              
        if(slicingParam.subgroup_weight_adjustment_method==1){
	        abh_dep.subgroup_weight_adj_coeff_a=SQWA_coeff[groupIndex][subgroupIndex][0];
	        abh_dep.subgroup_weight_adj_coeff_b=SQWA_coeff[groupIndex][subgroupIndex][1];              
        }
      }

      if(qpLayerCoefDependentUnits>0){
        abh_dep.attr_layer_qp_delta_luma.resize(2);
        abh_dep.attr_layer_qp_delta_chroma.resize(2);
        abh_dep.attr_layer_qp_delta_luma = {qpLayerCoefDependentUnits,0};
      }
    }
  }
  
  void LayerGroupHandler::releaseGeometryEncoderResource(int curArrayIdx){
    
		int groupIndex = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroupIndex = getLayerGroupIds(curArrayIdx).subgroupID;
    
		int refArrayIdx4Context = getReferenceIdx(curArrayIdx);
		int refArrayIdx4Parent = getParentIdx(curArrayIdx);


		if (groupIndex > 0) {
			_idx_uncoded_referenced_geom[refArrayIdx4Context].erase(curArrayIdx);
			_idx_uncoded_children_geom[refArrayIdx4Parent].erase(curArrayIdx);

			if (_idx_uncoded_referenced_geom[refArrayIdx4Context].size() == 0)
				releaseCtxForGeometry(refArrayIdx4Context);
			if (_idx_uncoded_children_geom[refArrayIdx4Parent].size() == 0)
				releaseNodes(refArrayIdx4Parent);
		}
    
		if (_idx_uncoded_referenced_geom[curArrayIdx].size() == 0)
			releaseCtxForGeometry(curArrayIdx);
		if (_idx_uncoded_children_geom[curArrayIdx].size() == 0)
			releaseNodes(curArrayIdx);

  }

  void LayerGroupHandler::releaseAttributeEncoderResource(int attrIdx, int curArrayIdx, bool isReusable){
  
		int groupIndex = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroupIndex = getLayerGroupIds(curArrayIdx).subgroupID;
    
		int refArrayIdx4Context = getReferenceIdx(curArrayIdx);
		int refArrayIdx4Parent = getParentIdx(curArrayIdx);


    //release the stored contexts
    if(groupIndex>0){
    _idx_uncoded_referenced_attr[attrIdx][refArrayIdx4Context].erase(curArrayIdx);
    _idx_uncoded_children_attr[attrIdx][refArrayIdx4Parent].erase(curArrayIdx);

    if(_idx_uncoded_referenced_attr[attrIdx][refArrayIdx4Context].size()==0)
        releaseCtxForAttribute(attrIdx,refArrayIdx4Context);
    }
		if (_idx_uncoded_referenced_attr[attrIdx][curArrayIdx].size() == 0)
			releaseCtxForAttribute(attrIdx,curArrayIdx);
    

    if(!isReusable)
      releaseIndexes(attrIdx,curArrayIdx);

  }

  void LayerGroupHandler::releaseAttributeEncoderResource(
    int curArrayIdx,
    std::vector<std::vector<PCCPointSet3>>& subgroupPointCloud){
  
		int groupIndex = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroupIndex = getLayerGroupIds(curArrayIdx).subgroupID;
    
		int refArrayIdx4Parent = getParentIdx(curArrayIdx);
    
		int refGroupIndex = getLayerGroupIds(refArrayIdx4Parent).layerGroupID;
		int refSubgroupIndex = getLayerGroupIds(refArrayIdx4Parent).subgroupID;

    
    if(groupIndex == getNumGroups()-1){        
      subgroupPointCloud[groupIndex][subgroupIndex].clear();
    }

    if(groupIndex>0){
    
    bool finished = true;
    for (int i=0; i<_idx_uncoded_children_attr.size();i++) { 
        if(_idx_uncoded_children_attr[i][refArrayIdx4Parent].size()>0){
        finished = false;
        }
    }
    if(finished)
        subgroupPointCloud[refGroupIndex][refSubgroupIndex].clear();

    }

    for (int i=0; i<_subGroupPointIndexToOrgPredictorIndex.size();i++)
      releaseIndexes(i,curArrayIdx);
  }
  

  void LayerGroupHandler::releaseGeometryDecoderResource(int curArrayIdx, bool context_reference_indication_flag, int numSubsequentSubgroups){
  
		int groupIndex = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroupIndex = getLayerGroupIds(curArrayIdx).subgroupID;
    
		int refArrayIdx4Context = getReferenceIdx(curArrayIdx);
		int refArrayIdx4Parent = getParentIdx(curArrayIdx);

		if (numSubsequentSubgroups)
			_numSubsequentSubgroups[curArrayIdx] = numSubsequentSubgroups;


    if (isOutputLayer(groupIndex)) {
          
      releaseNodes(curArrayIdx);
      releaseCtxForGeometry(curArrayIdx);

    } else if(isRequiredLayer(groupIndex)){
    
	    if (!context_reference_indication_flag) {
        releaseCtxForGeometry(curArrayIdx);
      }
    }

    if(groupIndex>0){
      _numSubsequentSubgroups[refArrayIdx4Context]--;
	    if (_numSubsequentSubgroups[refArrayIdx4Context] == 0) {

        releaseCtxForGeometry(refArrayIdx4Context);
	    }
    }
  }
  
  void LayerGroupHandler::releaseAttributeDecoderResource(
    int attrIdx, 
    int curArrayIdx, 
    bool context_reference_indication_flag, 
    bool attr_ref_id_present_flag,
    std::vector<PCCPointSet3>& subgroupPointCloud){
  
		int groupIndex = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroupIndex = getLayerGroupIds(curArrayIdx).subgroupID;
    
		int refArrayIdx4Context = getReferenceIdxAttribute(curArrayIdx);
		int refArrayIdx4Parent = getParentIdx(curArrayIdx);

    if(groupIndex>0){

    
      if(!attr_ref_id_present_flag){
        _idx_uncoded_referenced_attr[attrIdx][refArrayIdx4Context].erase(curArrayIdx);
        if(_idx_uncoded_referenced_attr[attrIdx][refArrayIdx4Context].size()==0)
          releaseCtxForAttribute(attrIdx,refArrayIdx4Context);
      }

      if (isOutputLayer(groupIndex) 
        || ( attr_ref_id_present_flag && !context_reference_indication_flag))
        releaseCtxForAttribute(attrIdx,curArrayIdx);

    
      _idx_uncoded_children_attr[attrIdx][refArrayIdx4Parent].erase(curArrayIdx);
    
      bool finished = true;
      for (int i=0; i<_idx_uncoded_children_attr.size();i++) { 
        if(_idx_uncoded_children_attr[i][refArrayIdx4Parent].size()>0){
          finished = false;
        }
      }
      if(finished)
        subgroupPointCloud[refArrayIdx4Parent].clear();

    }

    
    if(groupIndex==getNumGroups()-1){
  
      //check if all attributes have been dedocded
      bool finished = true;
      for (int i=0; i<_idx_uncoded_children_attr.size();i++) { 
        if(_idx_uncoded_children_attr[i][refArrayIdx4Parent].find(curArrayIdx) !=_idx_uncoded_children_attr[i][refArrayIdx4Parent].end()) {
          finished = false;
        }
      }
      if(finished)
        subgroupPointCloud[curArrayIdx].clear();
    }
  }
  
  void LayerGroupHandler::releaseNodes(){
  
    _nodesSaved.clear();
    _planarEligibleKOctreeDepthSaved.clear();
    _nodesBeforePlanarUpdateSaved.clear();
  }
  void LayerGroupHandler::releaseNodes(int arrayIdx){
  
    _nodesSaved[arrayIdx].reset();
    _planarEligibleKOctreeDepthSaved[arrayIdx].reset();
    _nodesBeforePlanarUpdateSaved[arrayIdx].reset();

  }
  void LayerGroupHandler::releaseCtxForGeometry(){
  
    _ctxtMemSaved.clear();
    _phiBufferSaved.clear();
    _planarSaved.clear();

  }
  void LayerGroupHandler::releaseCtxForGeometry(int arrayIdx){
  
    _ctxtMemSaved[arrayIdx].reset();
    _phiBufferSaved[arrayIdx].reset();
    _planarSaved[arrayIdx].reset();

  }
  void LayerGroupHandler::releaseCtxForAttribute(){
    _ctxtMemAttrsSaved.clear();
  
  }
  void LayerGroupHandler::releaseCtxForAttribute(int attrIdx){
    _ctxtMemAttrsSaved[attrIdx].clear();
  
  }
  void LayerGroupHandler::releaseCtxForAttribute(int attrIdx, int arrayIdx){

    _ctxtMemAttrsSaved[attrIdx][arrayIdx].reset();
  
  }
  void LayerGroupHandler::releaseIndexes(){
    _subGroupPointIndexToOrgPredictorIndex.clear();
  
  }
  void LayerGroupHandler::releaseIndexes(int attrIdx){
    _subGroupPointIndexToOrgPredictorIndex[attrIdx].clear();
  
  }
  void LayerGroupHandler::releaseIndexes(int attrIdx,int arrayIdx){
    _subGroupPointIndexToOrgPredictorIndex[attrIdx][arrayIdx].reset();  
  }

  //make reference tree by the coresponding geometry info
  void LayerGroupHandler::initializeForAttribute(const int numAttr, const bool isEncoder){

    _ctxtMemAttrsSaved.clear();
    _ctxtMemAttrsSaved.resize(numAttr);
    _idx_uncoded_referenced_attr.resize(numAttr);
    _idx_uncoded_children_attr.resize(numAttr);


    for(int i=0; i<numAttr; i++){
      _ctxtMemAttrsSaved[i].resize(_num_subgroups);

      _idx_uncoded_referenced_attr[i].resize(_num_subgroups);
      _idx_uncoded_children_attr[i].resize(_num_subgroups);

      
      for (int curArrayIdx=0; curArrayIdx<_num_subgroups; curArrayIdx++) 
      {

        int refArrayIdx4Context = 0;
        int refArrayIdx4Parent = 0;

        if(curArrayIdx==0){
          
        }else{
          
          refArrayIdx4Context = getReferenceIdxAttribute(curArrayIdx);
          refArrayIdx4Parent = getParentIdx(curArrayIdx);

          _idx_uncoded_referenced_attr[i][refArrayIdx4Context][curArrayIdx] = curArrayIdx; 
          _idx_uncoded_children_attr[i][refArrayIdx4Parent][curArrayIdx] = curArrayIdx; 
        }
      }
    }
    

    _subGroupPointIndexToOrgPredictorIndex.clear();
    if(isEncoder){

      _subGroupPointIndexToOrgPredictorIndex.resize(numAttr);
    
      for(int i=0; i<numAttr; i++){
        _subGroupPointIndexToOrgPredictorIndex[i].resize(_num_subgroups);
      }
    
    }  
    
  }
  

  void LayerGroupHandler::buildLoDSettings(const AttributeParameterSet& aps, const int attrIdx){
      
    _lod_max_level[attrIdx].resize(_num_layer_groups);
    _lod_min_geom_node_size_log2[attrIdx].resize(_num_layer_groups);
    _lod_num_layers[attrIdx].resize(_num_layer_groups);
    _lod_acum_layer_id[attrIdx].resize(_num_layer_groups);
    
    int acumLayerId = 0;

    for (int groupIndex = 0; groupIndex < _num_layer_groups; groupIndex++){
      _lod_min_geom_node_size_log2[attrIdx][groupIndex] = _max_depth - _end_depth[groupIndex];   
      _lod_max_level[attrIdx][groupIndex] = _lod_min_geom_node_size_log2[attrIdx][groupIndex] + _end_depth[groupIndex] - _end_depth_of_parent[groupIndex];

            
      //Level parameters for overlapped LOD
      if(groupIndex>0)
          _lod_max_level[attrIdx][groupIndex]++;    



      _lod_num_layers[attrIdx][groupIndex] = _lod_max_level[attrIdx][groupIndex] - _lod_min_geom_node_size_log2[attrIdx][groupIndex];
      _lod_acum_layer_id[attrIdx][groupIndex] = acumLayerId;
      acumLayerId += _lod_num_layers[attrIdx][groupIndex];


    }
  }
  

  int LayerGroupHandler::setNewDependentUnit(const int layer_group_id, const int subgroup_id, const int ref_layer_group_id, const int ref_subgroup_id, Vec3<int> bboxMin, Vec3<int> bboxMax){
    //push idx of the current array to the idx map with group-id and subgroup-id
    LayerGroupKey curKey(layer_group_id, subgroup_id);
    if(_layerGroupIdxToSavedArrayIdx.count(curKey) == 0) {
      _layerGroupIdxToSavedArrayIdx[curKey] = _arrayCounter;
      _savedArrayIdxToLayerGroupIdx[_arrayCounter] = curKey;

      if(!_use_nonnormative_memory_handling){   
        _num_subgroups++;
        _nodesSaved.resize(_num_subgroups);
        _numDCMPointsSubgroup.resize(_num_subgroups);
        _indexExtractedDCMPointsSubgroup.resize(_num_subgroups);
        
        _ctxtMemSaved.resize(_num_subgroups);
        _phiBufferSaved.resize(_num_subgroups);
        _planarSaved.resize(_num_subgroups);
        _planarEligibleKOctreeDepthSaved.resize(_num_subgroups);
        _nodesBeforePlanarUpdateSaved.resize(_num_subgroups);
      
        _bboxMinVector.resize(_num_subgroups);
        _bboxMaxVector.resize(_num_subgroups);

        _idx_uncoded_referenced_geom.resize(_num_subgroups);
        _idx_uncoded_children_geom.resize(_num_subgroups);


        _visited.resize(_num_subgroups,false);
		_available_geom.resize(_num_subgroups);

        
        _parentIdxSubgroup.resize(_num_subgroups);
        _refIdxSubgroup.resize(_num_subgroups);
        _refIdxSubgroupAttribute.resize(_num_subgroups);
        _startIdxForEachOutputPoints.resize(_num_subgroups);

        _numSubsequentSubgroups.resize(_num_subgroups);

      }
	  _SubgroupOccNeighPatEq0P.resize(_num_subgroups);
	  _SubgroupOccNodeChildCntP.resize(_num_subgroups);
	  _SubgroupOccNodeChildCntGP.resize(_num_subgroups);

      _arrayCounter++;
    }
    int curArrayIdx = _layerGroupIdxToSavedArrayIdx[curKey];


    {
      //derive idx of the saved array by group-id and subgroup-id
      LayerGroupKey refKey(ref_layer_group_id, ref_subgroup_id);
      int refArrayIdx = _layerGroupIdxToSavedArrayIdx[refKey];

      _refIdxSubgroup[curArrayIdx] = refArrayIdx;
      _refIdxSubgroupAttribute[curArrayIdx] = refArrayIdx;
    }
    
    setBox(curArrayIdx, bboxMin, bboxMax);  
    
    if(layer_group_id>0){
      pcc::LayerGroupKey parentKey = checkBox(curArrayIdx,layer_group_id - 1);
	    assert(parentKey.subgroupID >= 0);

      int parentArrayIdx = _layerGroupIdxToSavedArrayIdx[parentKey];
      _parentIdxSubgroup[curArrayIdx] = parentArrayIdx;    
    }
    return curArrayIdx;
  }

  //only for attr_aps.attr_ref_id_present_flag==true
  int LayerGroupHandler::setNewDependentUnitForAttribute(const int layer_group_id, const int subgroup_id, const int ref_layer_group_id, const int ref_subgroup_id){
    //push idx of the current array to the idx map with group-id and subgroup-id
    LayerGroupKey curKey(layer_group_id, subgroup_id);
    
    int curArrayIdx = _layerGroupIdxToSavedArrayIdx[curKey];

    {
    //derive idx of the saved array by group-id and subgroup-id
    LayerGroupKey refKey(ref_layer_group_id, ref_subgroup_id);
    int refArrayIdx = _layerGroupIdxToSavedArrayIdx[refKey];
    
      _refIdxSubgroupAttribute[curArrayIdx] = refArrayIdx;
    }
    
    return curArrayIdx;
  }

  
  void LayerGroupHandler::setRoi(bool roiEnabledFlag, Vec3<double> roiOriginScaled, Vec3<int> roiSize){
  

      _roi_enable_flag = roiEnabledFlag;

      // ROI parameter setting for partial decoding
      if (_roi_enable_flag) {
        Vec3<int> bboxSize = _bboxMax - _bboxMin;
        for (int i = 0; i < 3; i++) {
          int Width = bboxSize[i];

          double originScaled = roiOriginScaled[i];
          if (originScaled < 0)
            originScaled = 0.0;
          else if (originScaled > 1)
            originScaled = 1;
            


          _roi_origin[i] = int(Width * originScaled + 0.5);
          _roi_size[i] = roiSize[i];

          
				  if (_roi_origin[i] >= Width)
					  _roi_origin[i] = Width - 1 - _roi_size[i];

          }
      }

        _roi_min = _roi_origin;
        _roi_max = _roi_origin + _roi_size;
      
  }
  bool LayerGroupHandler::checkRoi(int curArrayIdx){
  
    auto curBboxMin = _bboxMinVector[curArrayIdx];
    auto curBboxMax = _bboxMaxVector[curArrayIdx];

    return checkRoi(curBboxMin, curBboxMin+curBboxMax);
  }
  bool LayerGroupHandler::checkRoi(Vec3<int> bboxOrigin, Vec3<int> bboxSize){
  
    bool isRequiredSubGroup = true;

    auto curBboxMin = bboxOrigin;
    auto curBboxMax = bboxOrigin + bboxSize;


    if (_roi_enable_flag) {
      for (int i = 0; i < 3; i++) {
        if ((_roi_min[i] < curBboxMax[i] && _roi_max[i] >= curBboxMin[i])) {
          continue;
        } else {
          isRequiredSubGroup = false;
          break;
        }
      }
    }
    
    return isRequiredSubGroup;
  }

  PCCPointSet3 LayerGroupHandler::getDCMPoints(bool hasColour, bool hasReflectance,
    int curArrayIdx,
    PCCPointSet3& subgroupPointCloud){
    
      PCCPointSet3 dcmPoints;
      dcmPoints.clear();
      dcmPoints.addRemoveAttributes(hasColour, hasReflectance);
      dcmPoints.resize(_numDCMPointsSubgroup[curArrayIdx]);

      _indexExtractedDCMPointsSubgroup[curArrayIdx].resize(_numDCMPointsSubgroup[curArrayIdx]);

      {

        for(int index=0; index<_numDCMPointsSubgroup[curArrayIdx]; index++){
          dcmPoints[index] = subgroupPointCloud[index];
          _indexExtractedDCMPointsSubgroup[curArrayIdx][index] = index;
          
          if(hasColour)
            dcmPoints.setColor(index, subgroupPointCloud.getColor(index));
    
          if(hasReflectance)
            dcmPoints.setReflectance(index, subgroupPointCloud.getReflectance(index));
        }
      }
      

      return dcmPoints;
  }

  PCCPointSet3 LayerGroupHandler::getPoints(bool hasColour, bool hasReflectance,
    PCCPointSet3& subgroupPointCloud){
    
      if(!_roi_enable_flag){
        return subgroupPointCloud;
      }

      PCCPointSet3 outPoints;
      outPoints.clear();
      outPoints.addRemoveAttributes(hasColour, hasReflectance);
      outPoints.resize(subgroupPointCloud.getPointCount());

      int count=0;
      Box3<int> bbox(_roi_origin,_roi_origin+_roi_size);

      for(int index=0; index<subgroupPointCloud.getPointCount(); index++){

        {
          outPoints[count] = subgroupPointCloud[index];
          
          if(hasColour)
            outPoints.setColor(count, subgroupPointCloud.getColor(index));
    
          if(hasReflectance)
            outPoints.setReflectance(count, subgroupPointCloud.getReflectance(index));


          count++;
        }
      }
      outPoints.resize(count);

      return outPoints;
  }

  PCCPointSet3 LayerGroupHandler::getOutputPoints(bool hasColour, bool hasReflectance, 
        int curArrayIdx,
        PCCPointSet3& subgroupPointCloud){
    
		int group_id = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroup_id = getLayerGroupIds(curArrayIdx).subgroupID;
		PCCPointSet3 null;

  	if (isOutputLayer(group_id)) {
			return getPoints(hasColour, hasReflectance, subgroupPointCloud);
		}
		else if (isRequiredLayer(group_id)) {

			return getDCMPoints(hasColour, hasReflectance, curArrayIdx, subgroupPointCloud);
    }
    return null;
  }

  void LayerGroupHandler::countExcludedPoints(const int curArrayIdx, LayerGroupSlicingParams& layerGroupParams, PCCPointSet3& subgroupPointCloud){
  
		int group_id = getLayerGroupIds(curArrayIdx).layerGroupID;
		if (isRequiredLayer(group_id)) {

			//remove idcm points coded in paret subgroup from children subgroup
			for (int childGrpIdx = group_id + 1; childGrpIdx <= layerGroupParams.numLayerGroupsMinus1; childGrpIdx++) {
				for (int childSubGrpIdx = 0; childSubGrpIdx <= layerGroupParams.numSubgroupsMinus1[childGrpIdx]; childSubGrpIdx++) {
					Vec3<int> bboxMinChild = layerGroupParams.subgrpBboxOrigin[childGrpIdx][childSubGrpIdx];
					Vec3<int> bboxMaxChild = layerGroupParams.subgrpBboxOrigin[childGrpIdx][childSubGrpIdx] + layerGroupParams.subgrpBboxSize[childGrpIdx][childSubGrpIdx];

					for (int idx = 0; idx < _numDCMPointsSubgroup[curArrayIdx]; idx++) {
						auto& pos = subgroupPointCloud[idx];

						if ((pos.x() >= bboxMinChild.x() && pos.x() < bboxMaxChild.x()
							&& pos.y() >= bboxMinChild.y() && pos.y() < bboxMaxChild.y()
							&& pos.z() >= bboxMinChild.z() && pos.z() < bboxMaxChild.z())) {
							layerGroupParams.numExcludedPoints[childGrpIdx][childSubGrpIdx]++;
						}
					}
				}
			}
		}
  }
  
  PCCPointSet3 LayerGroupHandler::getParentPoints(
    int curArrayIdx, int parentArrayIdx, 
    PCCPointSet3& subgroupPointCloud){

    int pointCount = subgroupPointCloud.getPointCount() - _numDCMPointsSubgroup[parentArrayIdx];

    assert(pointCount>0);

    bool hasColour = subgroupPointCloud.hasColors();
    bool hasReflectance = subgroupPointCloud.hasReflectances();
    
    PCCPointSet3 parentPoints;
    parentPoints.clear();
    parentPoints.addRemoveAttributes(hasColour, hasReflectance);
    parentPoints.resize(pointCount);
    
        
    int count=0;
    auto bbox_min = _bboxMinVector[curArrayIdx];
    auto bbox_max = _bboxMaxVector[curArrayIdx];
        
    for(int index=_numDCMPointsSubgroup[parentArrayIdx]; index<subgroupPointCloud.getPointCount(); index++){
          
      auto& pos = subgroupPointCloud[index];
        
		  if ((pos.x() >= bbox_min.x() && pos.x() < bbox_max.x()
			&& pos.y() >= bbox_min.y() && pos.y() < bbox_max.y()
			&& pos.z() >= bbox_min.z() && pos.z() < bbox_max.z())) 
      {
        parentPoints[count] = subgroupPointCloud[index];
          
        if(hasColour)
          parentPoints.setColor(count, subgroupPointCloud.getColor(index));
    
        if(hasReflectance)
          parentPoints.setReflectance(count, subgroupPointCloud.getReflectance(index));


        count++;
      }
    }
    
    parentPoints.resize(count);

    return parentPoints;
  }

  int LayerGroupHandler::setEncodedAttribute(
    int curArrayIdx, 
    PCCPointSet3& pointCloud, 
    PCCPointSet3& subgroupPointCloud){
    
		int group_id = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroup_id = getLayerGroupIds(curArrayIdx).subgroupID;

    int numDCMPoints = subgroupPointCloud.getPointCount();
    if(group_id<getNumGroups()-1)
      numDCMPoints = _numDCMPointsSubgroup[curArrayIdx];
        
    size_t startIndex = getStartIdxForEachOutputPoints(curArrayIdx);

    for (int i=0; i<numDCMPoints; i++) {
      if(pointCloud.hasColors())
        pointCloud.setColor(startIndex + i, subgroupPointCloud.getColor(i));
          
      if(pointCloud.hasReflectances())
        pointCloud.setReflectance(startIndex + i, subgroupPointCloud.getReflectance(i));
    }

    return numDCMPoints;  
  }

  void LayerGroupHandler::setDecodedAttribute(
    int curArrayIdx, 
    PCCPointSet3& pointCloud, 
    PCCPointSet3& subgroupPointCloud,
    bool setColor,
    bool setReflectance){

    
		int group_id = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroup_id = getLayerGroupIds(curArrayIdx).subgroupID;
    
    size_t startIndex = getStartIdxForEachOutputPoints(curArrayIdx);

    if (isOutputLayer(group_id)){
      
      size_t pointCount = subgroupPointCloud.getPointCount();
      
      if(setColor)
        for(int index=0; index < pointCount; index++){
            pointCloud.setColor(index + startIndex, subgroupPointCloud.getColor(index));
        }
      
      if(setReflectance)
        for(int index=0; index < pointCount; index++){
            pointCloud.setReflectance(index + startIndex, subgroupPointCloud.getReflectance(index));
        }
    }
    else if(isRequiredLayer(group_id))
    {
      
      if(setColor)
        for(int index=0; index<_numDCMPointsSubgroup[curArrayIdx]; index++)
        {
          pointCloud.setColor(index + startIndex, subgroupPointCloud.getColor(getIdxDCMPoints(curArrayIdx, index)));
        }
         
      if(setReflectance)
        for(int index=0; index<_numDCMPointsSubgroup[curArrayIdx]; index++)
        {
          pointCloud.setReflectance(index + startIndex, subgroupPointCloud.getReflectance(getIdxDCMPoints(curArrayIdx, index)));
        }   
    }
  
  }

  int LayerGroupHandler::getNumGroups(){
    return _num_layer_groups;
  }
  int LayerGroupHandler::getNumSubGroups(){
    return _num_subgroups;
  }
  
  int LayerGroupHandler::getMaxDepth(){
    return _max_depth;
  }
  int LayerGroupHandler::getEndDepth(int groupIndex){
    return _end_depth[groupIndex];
  }
  int LayerGroupHandler::getNumRemainedLayers(int groupIndex){
    return _num_remained_layers[groupIndex];
  }
  
  int LayerGroupHandler::getBitshift(int groupIndex){
    return _bitshift[groupIndex];
  }
  
  int LayerGroupHandler::getMinNodeSize(int attrIdx,int groupIndex){
    return _lod_min_geom_node_size_log2[attrIdx][groupIndex];
  }
  int LayerGroupHandler::getMaxLevel(int attrIdx,int groupIndex){
    return _lod_max_level[attrIdx][groupIndex];
  }
  int LayerGroupHandler::getLodNumLayers(int attrIdx,int groupIndex){
    return _lod_num_layers[attrIdx][groupIndex];
  }
  int LayerGroupHandler::getLodAcumLayerId(int attrIdx,int groupIndex){
    return _lod_acum_layer_id[attrIdx][groupIndex];
  }

  bool LayerGroupHandler::LayerGroupHandler::isOutputLayer(int groupIndex){
    return (_id_output_layer_group == groupIndex)?true:false;
  }
  bool LayerGroupHandler::isRequiredLayer(int groupIndex){
    return (groupIndex<=_id_output_layer_group)?true:false;
  }

  int LayerGroupHandler::getTrueEndOfLayers(int maxDepthInGBH){
    
    int startDepth = (_num_layer_groups>1)?_end_depth[_num_layer_groups-2]:0;
    int endDepth = _end_depth[_num_layer_groups-1];
    
    assert(startDepth < maxDepthInGBH);
  
    if (endDepth < maxDepthInGBH || (endDepth > maxDepthInGBH && startDepth < maxDepthInGBH)) {
        return maxDepthInGBH;
    }else
        return  endDepth;
  }

  int LayerGroupHandler::getArrayId(int groupId, int subgroupId){  
      LayerGroupKey key(groupId, subgroupId);
      return _layerGroupIdxToSavedArrayIdx[key];
  }
  
  LayerGroupKey LayerGroupHandler::getLayerGroupIds(int currArayId){
      return _savedArrayIdxToLayerGroupIdx[currArayId];
  }

  int LayerGroupHandler::getReferenceIdx(int curArrayIdx){
  
    return _refIdxSubgroup[curArrayIdx];
  }
  int LayerGroupHandler::getReferenceIdxAttribute(int curArrayIdx){
  
    return _refIdxSubgroupAttribute[curArrayIdx];
  }
  int LayerGroupHandler::getParentIdx(int curArrayIdx){
  
    return _parentIdxSubgroup[curArrayIdx];
  }

  Vec3<int> LayerGroupHandler::getRootBoxSize(){
    return _rootBboxSize;
  }

  void LayerGroupHandler::setBox(int curArrayIdx, Vec3<int> min, Vec3<int> max){
    _bboxMinVector[curArrayIdx] = min;
    _bboxMaxVector[curArrayIdx] = max;
  }


  pcc::LayerGroupKey LayerGroupHandler::checkBox(int curArrayIdx, int refLayerGroupIdx){

    pcc::LayerGroupKey refKey(0,0);
    for(auto m: _layerGroupIdxToSavedArrayIdx){
      if( m.first.layerGroupID == refLayerGroupIdx) {
        if (_bboxMinVector[m.second][0] <= _bboxMinVector[curArrayIdx][0]
			  && _bboxMinVector[m.second][1] <= _bboxMinVector[curArrayIdx][1]
			  && _bboxMinVector[m.second][2] <= _bboxMinVector[curArrayIdx][2]

			  && _bboxMaxVector[m.second][0] > _bboxMinVector[curArrayIdx][0]
			  && _bboxMaxVector[m.second][1] > _bboxMinVector[curArrayIdx][1]
			  && _bboxMaxVector[m.second][2] > _bboxMinVector[curArrayIdx][2]) 
          refKey = m.first;
      }
    }
    return refKey;
  }

  bool LayerGroupHandler::checkBox(int curArrayIdx, point_t pos){

    if (pos.x() < _bboxMaxVector[curArrayIdx][0]
		&& pos.y()  < _bboxMaxVector[curArrayIdx][1]
		&& pos.z()  < _bboxMaxVector[curArrayIdx][2]

		&& pos.x() >= _bboxMinVector[curArrayIdx][0]
		&& pos.y() >= _bboxMinVector[curArrayIdx][1]
		&& pos.z() >= _bboxMinVector[curArrayIdx][2]) 
      return true;

    return false;
  }

  void LayerGroupHandler::setStartIdxForEachOutputPoints(int curArrayIdx, int idx){
    _startIdxForEachOutputPoints[curArrayIdx] = idx;
  }
  int LayerGroupHandler::getStartIdxForEachOutputPoints(int curArrayIdx){
    return _startIdxForEachOutputPoints[curArrayIdx];
  }

  int LayerGroupHandler::getNumDecodedDCMPoints(int curArrayIdx){
    return _numDCMPointsSubgroup[curArrayIdx];
  }

  int LayerGroupHandler::getNumExtractedDCMPoints(int curArrayIdx){
    return _indexExtractedDCMPointsSubgroup[curArrayIdx].size();
  }

  int LayerGroupHandler::getIdxDCMPoints(int curArrayIdx, int index){
    return _indexExtractedDCMPointsSubgroup[curArrayIdx][index];
  }

  bool LayerGroupHandler::hasCorrespondingGeometry(int layer_group_id, int subgroup_id){
    
    LayerGroupKey key(layer_group_id, subgroup_id);
    if(_layerGroupIdxToSavedArrayIdx.count(key) == 0){ 
      return false;
    }
    else
      return true;
  }

  void LayerGroupHandler::initializeSubgroupPointIndexToOrgPredictorIndex(const int numAttr,
    const int groupIndex, const int subgroupIndex,
    const int numberOfPoint,
    std::vector<uint32_t>& pointIndexToPredictorIndex){
    
    int curArrayIdx = getArrayId(groupIndex,subgroupIndex);
    _subGroupPointIndexToOrgPredictorIndex[numAttr][curArrayIdx].reset(new std::vector<uint32_t>(numberOfPoint));
    
      for(int index=0; index<numberOfPoint; index++){

        int idx = _subGroupPointIndexToEncodedPointIndex[curArrayIdx][index];
        (*_subGroupPointIndexToOrgPredictorIndex[numAttr][curArrayIdx])[index] = pointIndexToPredictorIndex[idx];
      }


  }

  
  void LayerGroupHandler::countChildPoints(
    const std::vector<int> numSubgroupsMinus1){
    _numChildPoints.clear();
    _numChildPoints.resize(_num_subgroups,0);
    
    for (int groupIndex = _num_layer_groups-1; groupIndex >=0 ; groupIndex--){
      for (int subgroupIndex = 0; subgroupIndex < numSubgroupsMinus1[groupIndex]+1; subgroupIndex++){
      
        int curArrayIdx = getArrayId(groupIndex,subgroupIndex);
        _numChildPoints[curArrayIdx] = _numDCMPointsSubgroup[curArrayIdx];
      }
    }

    for (int groupIndex = _num_layer_groups-1; groupIndex >0 ; groupIndex--){
      for (int subgroupIndex = 0; subgroupIndex < numSubgroupsMinus1[groupIndex]+1; subgroupIndex++){

        int curArrayIdx = getArrayId(groupIndex,subgroupIndex);
        int parentArrayIdx = getParentIdx(curArrayIdx);

        _numChildPoints[parentArrayIdx] += _numChildPoints[curArrayIdx];
        
      }        
    }
  }

  void LayerGroupHandler::setEncodedPointIndexToOrgPointIndex(int curArrayIdx, const std::vector<uint32_t>& encIndexToOrgIndex, const PCCPointSet3& subgroupPointCloud){
  
		int group_id = getLayerGroupIds(curArrayIdx).layerGroupID;
		int subgroup_id = getLayerGroupIds(curArrayIdx).subgroupID;

    ////store the original index for each dcm points
    _encodedPointIndexToOrgPointIndex.insert(_encodedPointIndexToOrgPointIndex.end(), encIndexToOrgIndex.begin(), encIndexToOrgIndex.end());
      
    //if it has attribute, store original index
    _subGroupPointIndexToOrgPointIndex[curArrayIdx].clear();
    _subGroupPointIndexToOrgPointIndex[curArrayIdx].resize(subgroupPointCloud.getPointCount(),-1);

    assert(encIndexToOrgIndex.size()==_numDCMPointsSubgroup[curArrayIdx]);

    for(int count=0; count<encIndexToOrgIndex.size();count++){
      _subGroupPointIndexToOrgPointIndex[curArrayIdx][count] = encIndexToOrgIndex[count];
    }

    
    int count = 0;
    int numNodes = _subGroupPointIndexToOrgPointIndex[curArrayIdx].size() - _numDCMPointsSubgroup[curArrayIdx];
    _nodeStart[curArrayIdx].resize(numNodes);
    _nodeEnd[curArrayIdx].resize(numNodes);
    if(_nodesSaved[curArrayIdx]){
      for(auto& node: *(_nodesSaved[curArrayIdx])){        
        //store node start/end
        _nodeStart[curArrayIdx][count] = node.start;
        _nodeEnd[curArrayIdx][count] = node.end;

        count++;

        if(count>=numNodes)
          break;
      }
    
    }
  }

  void LayerGroupHandler::setSubGroupPointIndexToEncodedPointIndex(const PCCPointSet3& pointCloud){
  
    //assign index to the encoded points to each points for each subgroups
    _orgPointIndexToEncodedPointIndex.resize(pointCloud.getPointCount(),-1);
    for(int count=0; count<_encodedPointIndexToOrgPointIndex.size();count++){
      int idx = _encodedPointIndexToOrgPointIndex[count];

      _orgPointIndexToEncodedPointIndex[idx] = count;
    }
    _encodedPointIndexToOrgPointIndex.clear();

    
    for(int curArrayIdx=0; curArrayIdx<getNumSubGroups(); curArrayIdx++){
      
      int offset = _numDCMPointsSubgroup[curArrayIdx];
      int count = 0;
      int numPoints = _subGroupPointIndexToOrgPointIndex[curArrayIdx].size();
      //use start/end indtesd of node
      if(_nodeStart[curArrayIdx].size()>0)
        for(int count=0; count<_nodeStart[curArrayIdx].size(); count++){        
          _subGroupPointIndexToOrgPointIndex[curArrayIdx][count+offset] = lodSamplingForNode(pointCloud, _nodeStart[curArrayIdx][count], _nodeEnd[curArrayIdx][count]);

        }
      _nodeStart[curArrayIdx].clear();
      _nodeEnd[curArrayIdx].clear();
    }

    for(int curArrayIdx=0; curArrayIdx<getNumSubGroups(); curArrayIdx++){
      
      int numPoints = _subGroupPointIndexToOrgPointIndex[curArrayIdx].size();
      _subGroupPointIndexToEncodedPointIndex[curArrayIdx].resize(numPoints,-1);
      

      for(int count=0; count<numPoints;count++){
        
        int idx = _subGroupPointIndexToOrgPointIndex[curArrayIdx][count];
        _subGroupPointIndexToEncodedPointIndex[curArrayIdx][count] = _orgPointIndexToEncodedPointIndex[idx];
        

      }

      _subGroupPointIndexToOrgPointIndex[curArrayIdx].clear();
    }

    _subGroupPointIndexToOrgPointIndex.clear();
    _orgPointIndexToEncodedPointIndex.clear();
    _nodeStart.clear();
    _nodeEnd.clear();

  }

  void LayerGroupHandler::setSubGroupAttribute(int curArrayIdx, PCCPointSet3& subgroupPointCloud, const PCCPointSet3& pointCloud){
      for(int index=0; index<subgroupPointCloud.getPointCount(); index++){

        int idx = _subGroupPointIndexToEncodedPointIndex[curArrayIdx][index];
        
        if(pointCloud.hasColors())
          subgroupPointCloud.setColor(index,pointCloud.getColor(idx));
        if(pointCloud.hasReflectances())
          subgroupPointCloud.setReflectance(index,pointCloud.getReflectance(idx));
      }

      _subGroupPointIndexToEncodedPointIndex[curArrayIdx].clear();
  
  }



void splitSubgroup(std::vector<Vec3<int>>& splitBboxOrigin, std::vector<Vec3<int>>& splitBboxSize, std::vector<int>& splitNumPoints, const std::vector<Vec3<int>> subgrpPointCloud,
	const Vec3<int> curBboxOrigin, const Vec3<int> initSubgroupBboxSize, const int maxNumPoint);

int splitOneDirection(std::vector<Vec3<int>>& splitBboxOrigin, std::vector<Vec3<int>>& splitBboxSize, std::vector<int>& splitNumPoints, const std::vector<int> numPointsInSplitedSubgroups, std::vector<std::vector<Vec3<int>>> splitSubgroupPointCloud,
	const Vec3<int> subOrigin_in, const Vec3<int> subSize_in, const Vec3<int> BestDirection, const int numDivMinus1, Vec3<int> posHigh, const int maxNumPoint)
{
	int divAxis = BestDirection[numDivMinus1];
	Vec3<int> subOrigin = subOrigin_in;
	Vec3<int> subSize = subSize_in;
	subSize[divAxis] /= 2;

	if (posHigh[numDivMinus1])
		subOrigin[divAxis] += subSize[divAxis];

	int mask = 0, mask_base = 0;
	for (int m = 0; m <= numDivMinus1; m++) {
		mask_base += 1 << (2 - BestDirection[m]);
		mask += posHigh[m] << (2 - BestDirection[m]);
	}

	int numPoints = 0;
	for (int k = 0; k < 8; k++) {
		if ((k & mask_base) == mask) {
			numPoints += numPointsInSplitedSubgroups[k];
		}
	}


	if (numPoints <= maxNumPoint) {
		if (numPoints) {
			splitBboxOrigin.push_back(subOrigin);
			splitBboxSize.push_back(subSize);
			splitNumPoints.push_back(numPoints);
		}

		return numPoints;
	}
	else if (numDivMinus1 < 2) {
		posHigh[numDivMinus1 + 1] = 0;
		splitOneDirection(splitBboxOrigin, splitBboxSize, splitNumPoints, numPointsInSplitedSubgroups, splitSubgroupPointCloud,
			subOrigin, subSize, BestDirection, numDivMinus1 + 1, posHigh, maxNumPoint);

		posHigh[numDivMinus1 + 1] = 1;
		splitOneDirection(splitBboxOrigin, splitBboxSize, splitNumPoints, numPointsInSplitedSubgroups, splitSubgroupPointCloud,
			subOrigin, subSize, BestDirection, numDivMinus1 + 1, posHigh, maxNumPoint);

		return 0;
	}
	else {
		splitSubgroup(splitBboxOrigin, splitBboxSize, splitNumPoints, splitSubgroupPointCloud[mask],
			subOrigin, subSize, maxNumPoint);
	}
	return -1;
}

void splitSubgroup(std::vector<Vec3<int>>& splitBboxOrigin, std::vector<Vec3<int>>& splitBboxSize, std::vector<int>& splitNumPoints, const std::vector<Vec3<int>> subgrpPointCloud,
	const Vec3<int> curBboxOrigin, const Vec3<int> initSubgroupBboxSize, const int maxNumPoint) {
	Vec3<int> center;
	for (int m = 0; m < 3; m++)
		center[m] = curBboxOrigin[m] + initSubgroupBboxSize[m] / 2;

	std::vector<std::vector<Vec3<int>>> splitSubgroupPointCloud;
	splitSubgroupPointCloud.resize(8);

	for (int k = 0; k < subgrpPointCloud.size(); k++) {
		auto pos = subgrpPointCloud[k];
		int splitIdx = 0;
		for (int m = 0; m < 3; m++)
			if (pos[m] >= center[m])
				splitIdx += 1 << (2 - m);

		splitSubgroupPointCloud[splitIdx].push_back(pos);
	}

	Vec3<int> score, sumLow, sumHigh;
	std::vector<int> numPointsInSplitedSubgroups;
	for (int m = 0; m < 3; m++) {
		int low = 0, high = 0;
		int mask = 1 << (2 - m);
		for (int k = 0; k < 8; k++) {
			numPointsInSplitedSubgroups.push_back(splitSubgroupPointCloud[k].size());
			if (!!(k & mask))
				high += numPointsInSplitedSubgroups[k];
			else
				low += numPointsInSplitedSubgroups[k];
		}
		sumLow[m] = low;
		sumHigh[m] = high;
		score[m] = abs(high - low);
	}

	Vec3<int> bestScore, BestDirection;
	for (int m = 0; m < 3; m++) {
		if (m == 0 || score[m] < bestScore[0]) {
			for (int k = m; k > 0; k--) {
				bestScore[k] = bestScore[k - 1];
				BestDirection[k] = BestDirection[k - 1];
			}
			bestScore[0] = score[m];
			BestDirection[0] = m;
		}
		else if (score[m] < bestScore[1]) {
			bestScore[2] = bestScore[1];
			BestDirection[2] = BestDirection[1];

			bestScore[1] = score[m];
			BestDirection[1] = m;
		}
		else {
			bestScore[m] = score[m];
			BestDirection[m] = m;
		}
	}


	Vec3<int> posHigh = -1;
	posHigh[0] = 0;
	splitOneDirection(splitBboxOrigin, splitBboxSize, splitNumPoints, numPointsInSplitedSubgroups, splitSubgroupPointCloud,
		curBboxOrigin, initSubgroupBboxSize, BestDirection, 0, posHigh, maxNumPoint);

	posHigh[0] = 1;
	splitOneDirection(splitBboxOrigin, splitBboxSize, splitNumPoints, numPointsInSplitedSubgroups, splitSubgroupPointCloud,
		curBboxOrigin, initSubgroupBboxSize, BestDirection, 0, posHigh, maxNumPoint);
}

//In this function, only local parameter shall be defined to avoid bugs due to parameter mismatch.
//Parameters in sps will be copied from them later. 
bool setLayerGroupParams(const PCCPointSet3& cloud, const GeometryParameterSet gps, const OctreeEncOpts geom, LayerGroupSlicingParams& params){
  
		Vec3<int> bound = cloud.computeBoundingBox().max - cloud.computeBoundingBox().min + 1;
		GeometryBrickHeader gbh_temp;
		for (int k = 0; k < 3; k++)
			gbh_temp.rootNodeSizeLog2[k] = ceillog2(std::max(2, bound[k]));

		if (!gps.qtbt_enabled_flag)
			gbh_temp.rootNodeSizeLog2 = gbh_temp.rootNodeSizeLog2.max();

		params.rootNodeSizeLog2 = gbh_temp.rootNodeSizeLog2;

		auto lvlNodeSizeLog2 = mkQtBtNodeSizeList(gps, geom.qtbt, gbh_temp);
		int minNodeSizeLog2 = gbh_temp.trisoupNodeSizeLog2(gps);

		lvlNodeSizeLog2.erase(
			std::remove_if(
				lvlNodeSizeLog2.begin(), lvlNodeSizeLog2.end(),
				[&](Vec3<int>& size) { return size < minNodeSizeLog2; }),
			lvlNodeSizeLog2.end());
		assert(lvlNodeSizeLog2.back() == minNodeSizeLog2);

		lvlNodeSizeLog2.emplace_back(lvlNodeSizeLog2.back());

		int maxDepth = lvlNodeSizeLog2.size() - 2;

		params.subgroupBboxSize_Cubic = (params.subgroupBboxSize_Cubic < (1 << gbh_temp.rootNodeSizeLog2.max()))
			? params.subgroupBboxSize_Cubic : 1 << gbh_temp.rootNodeSizeLog2.max();
		if (params.subgroupBboxSize_Cubic < 0)
			params.subgroupBboxSize_Cubic = 1 << (int)(gbh_temp.rootNodeSizeLog2.max() - 3);

		Vec3<int> initSubgroupBboxSize = params.subgroupBboxSize_Cubic;

		bool bits_signaling_flag = false;
		int depthRemained = maxDepth;
    
		params.numLayersPerLayerGroup.resize(params.numLayerGroupsMinus1 + 1);
    //Determine number of layers for each layer group
		for (int i = 0; i <= params.numLayerGroupsMinus1; i++) {
			if (params.numLayersPerLayerGroup[0] && depthRemained > 0) {
    
        //First, determine the depth of the root layer group.
        //The number of nodes in the root layer group must be less than or equal to the maximum number of points in a data unit.
				if (i == 0) {
					int numLayers = params.numLayersPerLayerGroup[i]+1;
					int numPoints = 1100000 * 2; //Dummy value to perform the first node count
					while (numPoints > 1100000) {
						numLayers--;

						auto shift = lvlNodeSizeLog2[numLayers];

						Vec3<int> rootLayerGroupBboxSize;
						for (int k = 0; k < 3; k++)
							rootLayerGroupBboxSize[k] = 1 << (gbh_temp.rootNodeSizeLog2[k] - shift[k]);

						std::vector<std::vector<std::vector<int>>> numPointsInVoxels;
						numPointsInVoxels.resize(rootLayerGroupBboxSize[0]);
						for (int m = 0; m < numPointsInVoxels.size(); m++) {
							numPointsInVoxels[m].resize(rootLayerGroupBboxSize[1]);
							for (int n = 0; n < numPointsInVoxels[m].size(); n++)
								numPointsInVoxels[m][n].resize(rootLayerGroupBboxSize[2]);
						}

						auto minQuantInput = cloud.computeBoundingBox().min;
						for (int m = 0; m < cloud.getPointCount(); m++) {
							Vec3<int> pos = cloud[m] - minQuantInput;

							for (int k = 0; k < 3; k++)
								pos[k] = pos[k] >> shift[k];
							numPointsInVoxels[pos[0]][pos[1]][pos[2]]++;
						}

						numPoints = 0;
						for (int m = 0; m < numPointsInVoxels.size(); m++)
							for (int n = 0; n < numPointsInVoxels[m].size(); n++)
								for (int p = 0; p < numPointsInVoxels[m][n].size(); p++)
									if (numPointsInVoxels[m][n][p])
										numPoints++;
					}
					if (numLayers < maxDepth)
						params.numLayersPerLayerGroup[i] = numLayers;
					else
						params.numLayersPerLayerGroup[i] = maxDepth;
				}
				else if (i < params.numLayerGroupsMinus1) 
          //Set number of layer for each layer-group ensuring that the total number of layers does not exceed the maximum number of layers
					params.numLayersPerLayerGroup[i] = (params.numLayersPerLayerGroup[i] <= depthRemained)
					? params.numLayersPerLayerGroup[i] : depthRemained;
				else
          //The finest layer group shall includes all of remained layers
					params.numLayersPerLayerGroup[i] = depthRemained;

				if (i < params.numLayerGroupsMinus1) {
					depthRemained -= (params.numLayersPerLayerGroup[i]);

					if (depthRemained <= 0) {
						params.numLayerGroupsMinus1 = i;

						break;
					}
				}
			}
			else {

        return false;
			
			}

      params.numLayersPerLayerGroup.resize(params.numLayerGroupsMinus1+1);

		}
    
    //Second, make subgroups by recursive splitting    
		params.numSubgroupsMinus1.resize(params.numLayerGroupsMinus1 + 1);

		params.refLayerGroupId.resize(params.numLayerGroupsMinus1 + 1);
		params.refSubgroupId.resize(params.numLayerGroupsMinus1 + 1);
		params.parentLayerGroupId.resize(params.numLayerGroupsMinus1 + 1);
		params.parentSubgroupId.resize(params.numLayerGroupsMinus1 + 1);
		params.subgrpBboxOrigin.resize(params.numLayerGroupsMinus1 + 1);
		params.subgrpBboxSize.resize(params.numLayerGroupsMinus1 + 1);
		params.numPointsInSubgroups.resize(params.numLayerGroupsMinus1 + 1);
    params.numExcludedPoints.resize(params.numLayerGroupsMinus1 + 1);
		params.refLayerGroupIdAttribute.resize(params.numLayerGroupsMinus1 + 1);
		params.refSubgroupIdAttribute.resize(params.numLayerGroupsMinus1 + 1);
      

		if (params.subgroupBboxSize_Cubic > 0) {

			Vec3<int> numPerAxis = 1;
			int numMaxSubgroups = 1;

			for (int k = 0; k < 3; k++) {
				numPerAxis[k] = std::ceil(double(1 << gbh_temp.rootNodeSizeLog2[k]) / initSubgroupBboxSize[k]);
				numMaxSubgroups *= numPerAxis[k];
			}

			std::vector<std::vector<Vec3<int>>> subgrpPointCloud;
			subgrpPointCloud.resize(numMaxSubgroups);
			auto minQuantInput = cloud.computeBoundingBox().min;
			for (int i = 0; i < cloud.getPointCount(); i++) {
				Vec3<int> subgroupIdx, pos;
				pos = cloud[i] - minQuantInput;
				for (int k = 0; k < 3; k++)
					subgroupIdx[k] = (int)(pos[k] / initSubgroupBboxSize[k]);

				int currentIdx = (subgroupIdx[0] * numPerAxis[1] + subgroupIdx[1]) * numPerAxis[2] + subgroupIdx[2];
				subgrpPointCloud[currentIdx].push_back(pos);
			}


			int maxXYZ = 0;
			int numMaxPointsInSubgroup = 1100000;
			for (int lyrGrpIdx = 0; lyrGrpIdx < params.numLayerGroupsMinus1 + 1; lyrGrpIdx++) {
				if (lyrGrpIdx == 0) {
					params.refLayerGroupId[lyrGrpIdx].push_back(0);
					params.refSubgroupId[lyrGrpIdx].push_back(0);

					params.parentLayerGroupId[lyrGrpIdx].push_back(0);
					params.parentSubgroupId[lyrGrpIdx].push_back(0);

					params.subgrpBboxOrigin[lyrGrpIdx].push_back({ 0, 0, 0 });
					params.subgrpBboxSize[lyrGrpIdx].push_back({ 1 << gbh_temp.rootNodeSizeLog2[0], 1 << gbh_temp.rootNodeSizeLog2[1], 1 << gbh_temp.rootNodeSizeLog2[2] });

					params.numPointsInSubgroups[lyrGrpIdx].push_back(cloud.getPointCount());
				  params.numExcludedPoints[lyrGrpIdx].push_back(0);
          
					params.refLayerGroupIdAttribute[lyrGrpIdx].push_back(0);
					params.refSubgroupIdAttribute[lyrGrpIdx].push_back(0);

				}
				else {
					int idx = 0;
					for (int i = 0; i < subgrpPointCloud.size(); i++) {
						if (subgrpPointCloud[i].size() > 0) {
							Vec3<int> curIdx;
							curIdx[2] = i % numPerAxis[2];
							curIdx[1] = (int)(i / numPerAxis[2]) % numPerAxis[1];
							curIdx[0] = (int)((int)(i / numPerAxis[2]) / numPerAxis[1]);

							Vec3<int> curBboxOrigin;
							for (int m = 0; m < 3; m++)
								curBboxOrigin[m] = initSubgroupBboxSize[m] * curIdx[m];

							std::vector<Vec3<int>> splitBboxOrigin, splitBboxSize;
							std::vector<int> splitNumPoints;
							if (subgrpPointCloud[i].size() > numMaxPointsInSubgroup) {
								splitSubgroup(splitBboxOrigin, splitBboxSize, splitNumPoints, subgrpPointCloud[i],
									curBboxOrigin, initSubgroupBboxSize, numMaxPointsInSubgroup);
							}
							else {
								splitBboxOrigin.push_back(curBboxOrigin);
								splitBboxSize.push_back(initSubgroupBboxSize);
								splitNumPoints.push_back(subgrpPointCloud[i].size());
							}

							for (int k = 0; k < splitBboxOrigin.size(); k++) {
								params.refLayerGroupId[lyrGrpIdx].push_back(lyrGrpIdx - 1);
								params.refSubgroupId[lyrGrpIdx].push_back(idx);
	
								params.refLayerGroupIdAttribute[lyrGrpIdx].push_back(lyrGrpIdx - 1);
								params.refSubgroupIdAttribute[lyrGrpIdx].push_back(idx);  
                
								params.parentLayerGroupId[lyrGrpIdx].push_back(lyrGrpIdx - 1);
								params.parentSubgroupId[lyrGrpIdx].push_back(idx);

								params.subgrpBboxOrigin[lyrGrpIdx].push_back(splitBboxOrigin[k]);
								params.subgrpBboxSize[lyrGrpIdx].push_back(splitBboxSize[k]);

								params.numPointsInSubgroups[lyrGrpIdx].push_back(splitNumPoints[k]);
				        params.numExcludedPoints[lyrGrpIdx].push_back(0);

								if (maxXYZ < splitBboxOrigin[k].max())
									maxXYZ = splitBboxOrigin[k].max();

								idx++;
							}
						}
					}
				}
			}
      
      params.subgroupBboxOrigin_bits_minus1 = numBits(maxXYZ) - 1;
      params.subgroupBboxSize_bits_minus1 = numBits(initSubgroupBboxSize.max()) - 1;
		}
		else {
      //In this case, all groups shall not be splitted into subgroups     

			for (int lyrGrpIdx = 0; lyrGrpIdx < params.numLayerGroupsMinus1 + 1; lyrGrpIdx++) {
				params.subgrpBboxOrigin[lyrGrpIdx].push_back({ 0, 0, 0 });
				params.subgrpBboxSize[lyrGrpIdx].push_back({ 1 << gbh_temp.rootNodeSizeLog2[0], 1 << gbh_temp.rootNodeSizeLog2[1], 1 << gbh_temp.rootNodeSizeLog2[2] });

				params.numPointsInSubgroups[lyrGrpIdx].push_back(cloud.getPointCount());
				params.numExcludedPoints[lyrGrpIdx].push_back(0);

				params.refSubgroupId[lyrGrpIdx].push_back(0);
				params.parentSubgroupId[lyrGrpIdx].push_back(0);

				if (lyrGrpIdx > 1)
					params.refLayerGroupId[lyrGrpIdx].push_back(lyrGrpIdx - 1);
				else
					params.refLayerGroupId[lyrGrpIdx].push_back(0);

        
				params.refLayerGroupIdAttribute[lyrGrpIdx].push_back(0);
				params.refSubgroupIdAttribute[lyrGrpIdx].push_back(0);
        

				params.parentLayerGroupId[lyrGrpIdx].push_back(lyrGrpIdx - 1);
			}
      
      params.subgroupBboxOrigin_bits_minus1 = 0;
      params.subgroupBboxSize_bits_minus1 = numBits((int)(1 << gbh_temp.rootNodeSizeLog2.max())) - 1;
		}

		for (int i = 0; i < params.numLayerGroupsMinus1 + 1; i++)
			params.numSubgroupsMinus1[i] = params.subgrpBboxOrigin[i].size() - 1;

	
    return true;
}
//==========================================================================
}