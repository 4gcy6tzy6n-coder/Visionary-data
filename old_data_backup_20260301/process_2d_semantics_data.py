#!/usr/bin/env python3
"""
处理 KITTI-360 2D语义置信度数据
从2D图像中提取语义信息并生成训练数据
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from PIL import Image
import json


def load_confidence_image(image_path: str) -> np.ndarray:
    """加载置信度图像"""
    img = Image.open(image_path)
    return np.array(img)


def extract_semantic_from_confidence(confidence_map: np.ndarray) -> List[Dict]:
    """
    从置信度图中提取语义信息
    置信度图是灰度图像，像素值表示语义类别的置信度
    """
    objects = []
    
    # 将图像分割成网格
    h, w = confidence_map.shape
    grid_size = 50  # 50x50像素的网格
    
    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            # 提取网格区域
            grid = confidence_map[i:i+grid_size, j:j+grid_size]
            
            if grid.size == 0:
                continue
            
            # 计算网格的统计信息
            mean_confidence = np.mean(grid)
            max_confidence = np.max(grid)
            
            # 只保留高置信度区域
            if mean_confidence > 50:  # 阈值
                # 根据置信度判断语义类别
                if max_confidence > 200:
                    semantic = 1  # building
                elif max_confidence > 150:
                    semantic = 3  # tree
                elif max_confidence > 100:
                    semantic = 2  # car
                else:
                    semantic = 8  # road
                
                # 计算网格中心（归一化坐标）
                center_x = (j + grid_size / 2) / w
                center_y = (i + grid_size / 2) / h
                
                obj = {
                    'center': [center_x, center_y, 0],
                    'confidence': float(mean_confidence),
                    'max_confidence': float(max_confidence),
                    'semantic': semantic,
                    'grid_size': grid_size
                }
                objects.append(obj)
    
    return objects


def process_drive_confidence_data(drive_dir: Path, 
                                  cell_size_meters: float = 10.0) -> List[Dict]:
    """
    处理一个drive的置信度数据
    """
    confidence_dir = drive_dir / "image_00" / "confidence"
    
    if not confidence_dir.exists():
        print(f"警告: 找不到目录 {confidence_dir}")
        return []
    
    # 获取所有置信度图像
    confidence_files = sorted(confidence_dir.glob("*.png"))
    print(f"  找到 {len(confidence_files)} 张置信度图像")
    
    if len(confidence_files) == 0:
        return []
    
    # 将图像分组到cells（按时间/帧分组）
    images_per_cell = 10  # 每个cell包含10帧图像
    num_cells = len(confidence_files) // images_per_cell
    
    cells = []
    
    for cell_idx in tqdm(range(num_cells), desc=f"处理 {drive_dir.name}"):
        start_idx = cell_idx * images_per_cell
        end_idx = start_idx + images_per_cell
        cell_images = confidence_files[start_idx:end_idx]
        
        # 收集这个cell的所有对象
        cell_objects = []
        
        for img_file in cell_images:
            try:
                confidence_map = load_confidence_image(str(img_file))
                objects = extract_semantic_from_confidence(confidence_map)
                cell_objects.extend(objects)
            except Exception as e:
                print(f"  错误处理 {img_file}: {e}")
                continue
        
        if len(cell_objects) > 0:
            # 创建cell
            cell_id = f"{drive_dir.name}_cell_{cell_idx:04d}"
            
            # 计算cell中心（假设车辆沿直线行驶）
            # 这里简化处理，实际应该使用GPS/SLAM位姿
            cell_center_x = cell_idx * cell_size_meters
            cell_center_y = 0.0  # 假设在y=0的直线上
            
            cell = {
                'id': cell_id,
                'center': [cell_center_x, cell_center_y, 0.0],
                'objects': cell_objects,
                'size': cell_size_meters,
                'drive': drive_dir.name,
                'frame_range': [start_idx, end_idx],
                'object_count': len(cell_objects)
            }
            cells.append(cell)
    
    return cells


def generate_poses_for_cells(cells: List[Dict], 
                             poses_per_cell: int = 20) -> List[Dict]:
    """为cells生成poses"""
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
        "close to the {object}"
    ]
    
    directions = ['left', 'right', 'front', 'back', 'north', 'south', 'east', 'west']
    distances = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    semantic_map = {
        0: 'object', 1: 'building', 2: 'car', 3: 'tree', 4: 'person',
        5: 'pole', 6: 'traffic sign', 7: 'wall', 8: 'road', 9: 'sidewalk'
    }
    
    for cell in tqdm(cells, desc="生成poses"):
        cell_id = cell['id']
        cell_center = cell['center']
        cell_size = cell.get('size', 10.0)
        objects = cell.get('objects', [])
        
        for i in range(poses_per_cell):
            # 在cell内随机生成位置
            offset_x = np.random.uniform(-cell_size/3, cell_size/3)
            offset_y = np.random.uniform(-cell_size/3, cell_size/3)
            
            location = [
                cell_center[0] + offset_x,
                cell_center[1] + offset_y,
                0
            ]
            
            # 生成描述
            if len(objects) >= 2:
                obj1, obj2 = np.random.choice(objects, 2, replace=False)
                semantic1 = semantic_map.get(obj1.get('semantic', 0), 'object')
                semantic2 = semantic_map.get(obj2.get('semantic', 0), 'object')
                
                template = np.random.choice(templates)
                
                if '{object1}' in template:
                    description = template.format(
                        object1=semantic1,
                        object2=semantic2,
                        distance=np.random.choice(distances),
                        direction=np.random.choice(directions)
                    )
                elif '{distance}' in template:
                    description = template.format(
                        object=semantic1,
                        distance=np.random.choice(distances),
                        direction=np.random.choice(directions)
                    )
                else:
                    description = template.format(
                        object=semantic1,
                        direction=np.random.choice(directions)
                    )
            elif len(objects) == 1:
                semantic = semantic_map.get(objects[0].get('semantic', 0), 'object')
                template = np.random.choice(templates)
                if '{distance}' in template and '{object1}' not in template:
                    description = template.format(
                        object=semantic,
                        distance=np.random.choice(distances),
                        direction=np.random.choice(directions)
                    )
                elif '{object1}' not in template:
                    description = template.format(
                        object=semantic,
                        direction=np.random.choice(directions)
                    )
                else:
                    description = f"near the {semantic}"
            else:
                description = f"Location in {cell_id}"
            
            pose = {
                'cell_id': cell_id,
                'location': location,
                'description': description,
                'cell_center': cell_center
            }
            
            poses.append(pose)
    
    print(f"生成 {len(poses)} 个poses")
    return poses


def main():
    print("="*80)
    print("处理 KITTI-360 2D语义置信度数据")
    print("="*80)
    
    # 数据路径
    data_dir = Path("/Volumes/MU90/data_2d_semantics/train")
    output_dir = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_2d_semantics")
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在 {data_dir}")
        print("请等待数据解压完成...")
        return
    
    # 获取所有drive
    drives = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"\n找到 {len(drives)} 个drives:")
    for drive in drives:
        print(f"  - {drive.name}")
    
    # 处理每个drive
    all_cells = []
    
    for drive in drives:
        print(f"\n处理 {drive.name}...")
        cells = process_drive_confidence_data(drive, cell_size_meters=10.0)
        all_cells.extend(cells)
        print(f"  生成 {len(cells)} 个cells")
    
    print(f"\n总共生成 {len(all_cells)} 个cells")
    
    if len(all_cells) == 0:
        print("错误: 没有生成任何cells")
        return
    
    # 生成poses
    poses = generate_poses_for_cells(all_cells, poses_per_cell=30)
    
    # 保存数据
    print("\n保存数据...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "cells.pkl", 'wb') as f:
        pickle.dump(all_cells, f)
    
    with open(output_path / "poses.pkl", 'wb') as f:
        pickle.dump(poses, f)
    
    # 保存统计信息
    stats = {
        'num_cells': len(all_cells),
        'num_poses': len(poses),
        'num_drives': len(drives),
        'avg_objects_per_cell': float(np.mean([c['object_count'] for c in all_cells])),
        'avg_desc_length': float(np.mean([len(p['description']) for p in poses])),
        'unique_descriptions': len(set(p['description'] for p in poses))
    }
    
    with open(output_path / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*80)
    print("数据处理完成!")
    print("="*80)
    print(f"\n输出目录: {output_dir}")
    print(f"  - cells.pkl: {len(all_cells)} cells")
    print(f"  - poses.pkl: {len(poses)} poses")
    print(f"\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
