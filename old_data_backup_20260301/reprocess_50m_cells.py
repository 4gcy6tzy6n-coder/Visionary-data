#!/usr/bin/env python3
"""
重新处理数据 - 使用50m cell size
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image
from typing import List, Dict


def load_confidence_image(image_path: str) -> np.ndarray:
    """加载置信度图像"""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except:
        return None


def extract_semantic_from_confidence(confidence_map: np.ndarray) -> List[Dict]:
    """从置信度图中提取语义信息"""
    objects = []
    h, w = confidence_map.shape
    grid_size = 100
    
    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            grid = confidence_map[i:i+grid_size, j:j+grid_size]
            if grid.size == 0:
                continue
            
            mean_confidence = np.mean(grid)
            max_confidence = np.max(grid)
            
            if mean_confidence > 30:
                if max_confidence > 220:
                    semantic = 1
                elif max_confidence > 190:
                    semantic = 3
                elif max_confidence > 160:
                    semantic = 2
                elif max_confidence > 130:
                    semantic = 4
                elif max_confidence > 100:
                    semantic = 5
                else:
                    semantic = 8
                
                center_x = (j + grid_size / 2) / w
                center_y = (i + grid_size / 2) / h
                
                obj = {
                    'center': [float(center_x), float(center_y), 0.0],
                    'confidence': float(mean_confidence),
                    'max_confidence': float(max_confidence),
                    'semantic': semantic,
                    'grid_size': grid_size
                }
                objects.append(obj)
    
    return objects


def process_drive_50m(drive_dir: Path, cell_size_meters: float = 50.0) -> List[Dict]:
    """处理drive - 50m cells"""
    confidence_dir = drive_dir / "image_00" / "confidence"
    
    if not confidence_dir.exists():
        return []
    
    confidence_files = sorted(confidence_dir.glob("*.png"))
    print(f"  找到 {len(confidence_files)} 张图像")
    
    if len(confidence_files) == 0:
        return []
    
    # 每100帧创建一个50m cell (原来30帧=20m，现在100帧=50m)
    frames_per_cell = 100
    num_cells = len(confidence_files) // frames_per_cell
    
    cells = []
    
    for cell_idx in tqdm(range(num_cells), desc=f"处理 {drive_dir.name}"):
        start_idx = cell_idx * frames_per_cell
        end_idx = min(start_idx + frames_per_cell, len(confidence_files))
        cell_images = confidence_files[start_idx:end_idx]
        
        cell_objects = []
        valid_frames = 0
        
        for img_file in cell_images:
            confidence_map = load_confidence_image(str(img_file))
            if confidence_map is not None:
                objects = extract_semantic_from_confidence(confidence_map)
                cell_objects.extend(objects)
                valid_frames += 1
        
        if len(cell_objects) >= 10:
            cell_id = f"{drive_dir.name}_cell_{cell_idx:04d}"
            
            # 50m cell center
            cell_center_x = cell_idx * cell_size_meters
            cell_center_y = 0.0
            
            cell = {
                'id': cell_id,
                'center': [float(cell_center_x), float(cell_center_y), 0.0],
                'objects': cell_objects,
                'size': cell_size_meters,
                'drive': drive_dir.name,
                'frame_range': [start_idx, end_idx],
                'object_count': len(cell_objects),
                'valid_frames': valid_frames
            }
            cells.append(cell)
    
    return cells


def generate_poses_50m(cells: List[Dict], poses_per_cell: int = 50) -> List[Dict]:
    """生成poses - 50m cell内随机分布"""
    print("\n生成poses...")
    
    poses = []
    
    templates = [
        "{distance} meters from the {object}",
        "about {distance}m away from the {object}",
        "roughly {distance} meters to the {direction} of the {object}",
        "near the {object}, about {distance}m away",
        "to the {direction} of the {object}",
        "on the {direction} side of the {object}",
        "in front of the {object}",
        "behind the {object}",
        "between the {object1} and {object2}",
        "close to the {object}",
        "approaching the {object}",
        "passing by the {object}",
        "next to the {object}",
        "facing the {object}",
        "with the {object} on the {direction}"
    ]
    
    directions = ['left', 'right', 'front', 'back', 'north', 'south', 'east', 'west']
    distances = [5, 8, 10, 12, 15, 18, 20, 22, 25]
    
    semantic_map = {
        0: 'object', 1: 'building', 2: 'car', 3: 'tree', 4: 'person',
        5: 'pole', 6: 'traffic sign', 7: 'wall', 8: 'road', 9: 'sidewalk'
    }
    
    for cell in tqdm(cells, desc="生成poses"):
        cell_id = cell['id']
        cell_center = cell['center']
        cell_size = cell.get('size', 50.0)
        objects = cell.get('objects', [])
        
        # 按语义类别分组
        semantic_groups = {}
        for obj in objects:
            sem = obj.get('semantic', 0)
            if sem not in semantic_groups:
                semantic_groups[sem] = []
            semantic_groups[sem].append(obj)
        
        for i in range(poses_per_cell):
            # 在50m cell内生成位置 - 更大的范围
            angle = 2 * np.pi * i / poses_per_cell
            # 半径范围: 5m 到 22m (接近cell边缘)
            radius = np.random.uniform(5, cell_size/2 - 3)
            
            offset_x = radius * np.cos(angle)
            offset_y = radius * np.sin(angle)
            
            location = [
                float(cell_center[0] + offset_x),
                float(cell_center[1] + offset_y),
                0.0
            ]
            
            # 生成描述
            if len(semantic_groups) >= 2:
                sems = list(semantic_groups.keys())
                sem1, sem2 = np.random.choice(sems, 2, replace=False)
                obj_name1 = semantic_map.get(sem1, 'object')
                obj_name2 = semantic_map.get(sem2, 'object')
                
                template = np.random.choice(templates)
                
                if '{object1}' in template and '{object2}' in template:
                    description = template.format(
                        object1=obj_name1, object2=obj_name2,
                        distance=np.random.choice(distances),
                        direction=np.random.choice(directions)
                    )
                elif '{object}' in template:
                    description = template.format(
                        object=obj_name1,
                        distance=np.random.choice(distances),
                        direction=np.random.choice(directions)
                    )
                else:
                    description = f"near the {obj_name1}"
            elif len(semantic_groups) == 1:
                sem = list(semantic_groups.keys())[0]
                obj_name = semantic_map.get(sem, 'object')
                template = np.random.choice(templates)
                if '{object}' in template:
                    description = template.format(
                        object=obj_name,
                        distance=np.random.choice(distances),
                        direction=np.random.choice(directions)
                    )
                else:
                    description = f"near the {obj_name}"
            else:
                description = f"Location in {cell_id}"
            
            pose = {
                'cell_id': cell_id,
                'location': location,
                'description': description,
                'cell_center': cell_center
            }
            poses.append(pose)
    
    return poses


def main():
    print("="*80)
    print("重新处理数据 - 50m Cell Size")
    print("="*80)
    
    data_dir = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data_2d_semantics/train")
    output_dir = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_50m_cells")
    
    drives = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"\n找到 {len(drives)} 个drives")
    
    # 处理所有drives
    all_cells = []
    for drive in drives:
        print(f"\n{'='*60}")
        print(f"处理 {drive.name}...")
        print('='*60)
        cells = process_drive_50m(drive, cell_size_meters=50.0)
        all_cells.extend(cells)
        print(f"  ✓ 生成 {len(cells)} 个cells")
    
    print(f"\n{'='*80}")
    print(f"总共生成 {len(all_cells)} 个cells")
    print('='*80)
    
    # 生成poses
    poses = generate_poses_50m(all_cells, poses_per_cell=50)
    
    # 保存
    print("\n保存数据...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "cells.pkl", 'wb') as f:
        pickle.dump(all_cells, f)
    
    with open(output_path / "poses.pkl", 'wb') as f:
        pickle.dump(poses, f)
    
    # 统计
    stats = {
        'num_cells': len(all_cells),
        'num_poses': len(poses),
        'cell_size': 50.0,
        'avg_objects_per_cell': float(np.mean([c['object_count'] for c in all_cells])),
        'avg_desc_length': float(np.mean([len(p['description']) for p in poses]))
    }
    
    with open(output_path / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*80)
    print("数据处理完成!")
    print("="*80)
    print(f"\n输出: {output_dir}")
    print(f"  Cells: {len(all_cells)}")
    print(f"  Poses: {len(poses)}")
    print(f"  Cell Size: 50m")


if __name__ == '__main__':
    main()
