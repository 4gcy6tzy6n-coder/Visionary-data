#!/usr/bin/env python3
"""
基于3D信息生成智能描述
利用现有的语义和空间信息生成更丰富的描述
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict

def generate_smart_description(cell: Dict, pose: Dict) -> str:
    """
    基于cell的3D信息生成智能描述
    """
    objects = cell.get('objects', [])
    if not objects:
        return f"Location in {cell.get('id', 'unknown area')}"
    
    # 获取pose位置
    pose_location = pose.get('location', [0, 0, 0])
    if isinstance(pose_location, (list, tuple)) and len(pose_location) >= 2:
        pose_xy = np.array([float(pose_location[0]), float(pose_location[1])])
    else:
        pose_xy = np.array([0.0, 0.0])
    
    # 获取cell中心
    cell_center = cell.get('center', [0, 0, 0])
    cell_center_xy = np.array([float(cell_center[0]), float(cell_center[1])])
    
    # 计算pose相对于cell中心的位置
    relative_pos = pose_xy - cell_center_xy
    
    # 确定方向
    directions = []
    if relative_pos[0] > 1.0:
        directions.append("east")
    elif relative_pos[0] < -1.0:
        directions.append("west")
    
    if relative_pos[1] > 1.0:
        directions.append("north")
    elif relative_pos[1] < -1.0:
        directions.append("south")
    
    direction_str = " ".join(directions) if directions else "center"
    
    # 找到最近的对象
    nearest_obj = None
    min_distance = float('inf')
    
    for obj in objects:
        if isinstance(obj, dict):
            obj_center = obj.get('center', [0, 0, 0])
            if len(obj_center) >= 2:
                obj_xy = np.array([float(obj_center[0]), float(obj_center[1])])
                distance = np.linalg.norm(pose_xy - obj_xy)
                if distance < min_distance:
                    min_distance = distance
                    nearest_obj = obj
    
    # 生成描述
    if nearest_obj:
        semantic = nearest_obj.get('semantic', 'object')
        if isinstance(semantic, int):
            semantic_map = {
                0: 'unknown', 1: 'building', 2: 'car', 3: 'tree', 4: 'person',
                5: 'pole', 6: 'traffic sign', 7: 'wall', 8: 'road', 9: 'sidewalk'
            }
            semantic = semantic_map.get(semantic, 'object')
        
        # 根据距离生成描述
        if min_distance < 2.0:
            proximity = "very close to"
        elif min_distance < 5.0:
            proximity = "near"
        else:
            proximity = "in the area of"
        
        # 根据相对位置生成空间关系
        obj_center = nearest_obj.get('center', [0, 0, 0])
        if len(obj_center) >= 2:
            obj_xy = np.array([float(obj_center[0]), float(obj_center[1])])
            rel_to_obj = pose_xy - obj_xy
            
            if abs(rel_to_obj[0]) > abs(rel_to_obj[1]):
                if rel_to_obj[0] > 0:
                    spatial_rel = "to the right of"
                else:
                    spatial_rel = "to the left of"
            else:
                if rel_to_obj[1] > 0:
                    spatial_rel = "in front of"
                else:
                    spatial_rel = "behind"
            
            description = f"{proximity} the {semantic}, {spatial_rel} it, in the {direction_str} part of the area"
        else:
            description = f"{proximity} the {semantic} in the {direction_str} part of the area"
    else:
        description = f"Location in the {direction_str} part of {cell.get('id', 'area')}"
    
    return description


def enhance_dataset_with_smart_descriptions():
    """
    用智能描述增强数据集
    """
    print("="*80)
    print("基于3D信息生成智能描述")
    print("="*80)
    
    # 加载数据
    data_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_semantics_full")
    
    with open(data_path / "cells" / "cells.pkl", 'rb') as f:
        cells = pickle.load(f)
    with open(data_path / "poses" / "poses.pkl", 'rb') as f:
        poses = pickle.load(f)
    
    cells_dict = {cell['id']: cell for cell in cells if isinstance(cell, dict)}
    
    print(f"\n原始数据: {len(cells)} cells, {len(poses)} poses")
    
    # 生成智能描述
    enhanced_poses = []
    smart_desc_count = 0
    
    for pose in poses:
        if isinstance(pose, dict) and 'cell_id' in pose:
            cell_id = pose['cell_id']
            cell = cells_dict.get(cell_id)
            
            if cell:
                # 生成智能描述
                smart_desc = generate_smart_description(cell, pose)
                
                # 创建增强的pose
                enhanced_pose = pose.copy()
                enhanced_pose['original_description'] = pose.get('description', '')
                enhanced_pose['description'] = smart_desc
                enhanced_poses.append(enhanced_pose)
                smart_desc_count += 1
            else:
                enhanced_poses.append(pose)
        else:
            enhanced_poses.append(pose)
    
    print(f"\n生成智能描述: {smart_desc_count} poses")
    
    # 显示一些示例
    print("\n智能描述示例：")
    for i in range(min(5, len(enhanced_poses))):
        pose = enhanced_poses[i]
        print(f"\n  示例 {i+1}:")
        print(f"    原始: {pose.get('original_description', 'N/A')}")
        print(f"    智能: {pose.get('description', 'N/A')}")
    
    # 保存增强的数据
    output_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_enhanced")
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "cells.pkl", 'wb') as f:
        pickle.dump(cells, f)
    
    with open(output_path / "poses.pkl", 'wb') as f:
        pickle.dump(enhanced_poses, f)
    
    print(f"\n增强数据已保存到: {output_path}")
    print(f"  - cells.pkl: {len(cells)} cells")
    print(f"  - poses.pkl: {len(enhanced_poses)} poses")
    
    return output_path


if __name__ == '__main__':
    enhance_dataset_with_smart_descriptions()
