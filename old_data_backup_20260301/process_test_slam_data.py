#!/usr/bin/env python3
"""
处理 data_3d_test_slam 数据集
提取点云数据，生成cells和poses用于训练
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import struct


def load_velodyne_points(bin_path: str) -> np.ndarray:
    """
    加载KITTI格式的velodyne点云数据 (.bin文件)
    每个点包含: x, y, z, intensity
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def extract_objects_from_pointcloud(points: np.ndarray, 
                                    z_threshold: float = 0.5,
                                    min_points: int = 50) -> List[Dict]:
    """
    从点云中提取对象
    使用简单的基于高度的分割
    """
    objects = []
    
    # 根据高度分割点云
    ground_mask = points[:, 2] < z_threshold
    non_ground = points[~ground_mask]
    
    if len(non_ground) < min_points:
        return objects
    
    # 简单的聚类：基于XY平面的网格分割
    grid_size = 2.0  # 2米网格
    x_min, x_max = non_ground[:, 0].min(), non_ground[:, 0].max()
    y_min, y_max = non_ground[:, 1].min(), non_ground[:, 1].max()
    
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    
    # 为每个网格创建对象
    for i in range(x_bins):
        for j in range(y_bins):
            x_start = x_min + i * grid_size
            x_end = x_start + grid_size
            y_start = y_min + j * grid_size
            y_end = y_start + grid_size
            
            mask = (
                (non_ground[:, 0] >= x_start) & (non_ground[:, 0] < x_end) &
                (non_ground[:, 1] >= y_start) & (non_ground[:, 1] < y_end)
            )
            
            grid_points = non_ground[mask]
            
            if len(grid_points) >= min_points:
                # 计算对象的中心
                center_x = grid_points[:, 0].mean()
                center_y = grid_points[:, 1].mean()
                center_z = grid_points[:, 2].mean()
                
                # 根据高度判断语义类型
                if center_z < 1.5:
                    semantic = 8  # road
                elif center_z < 2.0:
                    semantic = 9  # sidewalk
                elif center_z < 3.0:
                    semantic = 1  # building
                elif center_z < 5.0:
                    semantic = 3  # tree
                else:
                    semantic = 0  # unknown
                
                obj = {
                    'center': [center_x, center_y, center_z],
                    'points': len(grid_points),
                    'semantic': semantic,
                    'bbox': [
                        grid_points[:, 0].min(), grid_points[:, 0].max(),
                        grid_points[:, 1].min(), grid_points[:, 1].max(),
                        grid_points[:, 2].min(), grid_points[:, 2].max()
                    ]
                }
                objects.append(obj)
    
    return objects


def create_cells_from_pointclouds(data_dir: str, 
                                  cell_size: float = 20.0,
                                  max_cells: int = 1000) -> List[Dict]:
    """
    从点云数据创建cells
    """
    print("="*80)
    print("从点云数据创建Cells")
    print("="*80)
    
    data_path = Path(data_dir)
    velodyne_dir = data_path / "velodyne_points" / "data"
    
    if not velodyne_dir.exists():
        print(f"错误: 找不到目录 {velodyne_dir}")
        return []
    
    # 获取所有.bin文件
    bin_files = sorted(velodyne_dir.glob("*.bin"))
    print(f"\n找到 {len(bin_files)} 个点云文件")
    
    # 收集所有点的位置来确定场景范围
    all_points = []
    sample_files = bin_files[::max(1, len(bin_files)//100)]  # 采样100个文件
    
    print("\n分析场景范围...")
    for bin_file in sample_files:
        points = load_velodyne_points(str(bin_file))
        all_points.append(points[:, :3])  # 只取xyz
    
    all_points = np.vstack(all_points)
    
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    
    print(f"场景范围: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}]")
    
    # 创建网格cells
    cells = []
    cell_id = 0
    
    x = x_min
    while x < x_max and len(cells) < max_cells:
        y = y_min
        while y < y_max and len(cells) < max_cells:
            cell_center_x = x + cell_size / 2
            cell_center_y = y + cell_size / 2
            
            # 找到在这个cell范围内的点云文件
            cell_objects = []
            
            for bin_file in bin_files:
                points = load_velodyne_points(str(bin_file))
                
                # 检查这个点云是否在cell范围内
                mask = (
                    (points[:, 0] >= x) & (points[:, 0] < x + cell_size) &
                    (points[:, 1] >= y) & (points[:, 1] < y + cell_size)
                )
                
                if np.any(mask):
                    cell_points = points[mask]
                    objects = extract_objects_from_pointcloud(cell_points)
                    cell_objects.extend(objects)
            
            if len(cell_objects) > 0:
                cell = {
                    'id': f'cell_{cell_id}',
                    'center': [cell_center_x, cell_center_y, 0],
                    'objects': cell_objects,
                    'size': cell_size,
                    'bounds': [x, x + cell_size, y, y + cell_size]
                }
                cells.append(cell)
                cell_id += 1
                
                if cell_id % 100 == 0:
                    print(f"  已创建 {cell_id} 个cells...")
            
            y += cell_size
        x += cell_size
    
    print(f"\n总共创建 {len(cells)} 个cells")
    return cells


def generate_poses_for_cells(cells: List[Dict], 
                             poses_per_cell: int = 10) -> List[Dict]:
    """
    为每个cell生成poses
    """
    print("\n" + "="*80)
    print("生成Poses")
    print("="*80)
    
    poses = []
    
    # 描述模板
    distance_templates = [
        "{distance} meters from the {object}",
        "about {distance}m away from the {object}",
        "roughly {distance} meters to the {direction} of the {object}",
        "near the {object}, about {distance}m away",
        "approximately {distance}m from the {object} on the {direction}"
    ]
    
    spatial_templates = [
        "to the {direction} of the {object}",
        "on the {direction} side of the {object}",
        "{direction} of the {object}",
        "in front of the {object}",
        "behind the {object}"
    ]
    
    directions = ['left', 'right', 'front', 'back', 'north', 'south', 'east', 'west']
    distances = [2, 3, 5, 7, 10, 15]
    
    semantic_map = {
        0: 'object', 1: 'building', 2: 'car', 3: 'tree', 4: 'person',
        5: 'pole', 6: 'traffic sign', 7: 'wall', 8: 'road', 9: 'sidewalk'
    }
    
    for cell in cells:
        cell_id = cell['id']
        cell_center = cell['center']
        cell_size = cell.get('size', 20.0)
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
            if objects:
                # 选择最近的对象
                nearest_obj = min(objects, 
                    key=lambda o: np.linalg.norm(
                        np.array(o['center'][:2]) - np.array(location[:2])
                    )
                )
                
                semantic_id = nearest_obj.get('semantic', 0)
                semantic_name = semantic_map.get(semantic_id, 'object')
                
                # 计算相对位置
                obj_xy = np.array(nearest_obj['center'][:2])
                pose_xy = np.array(location[:2])
                rel_pos = pose_xy - obj_xy
                
                # 确定方向
                if abs(rel_pos[0]) > abs(rel_pos[1]):
                    direction = 'east' if rel_pos[0] > 0 else 'west'
                else:
                    direction = 'north' if rel_pos[1] > 0 else 'south'
                
                # 计算距离
                distance = int(np.linalg.norm(rel_pos))
                
                # 选择模板
                if np.random.random() < 0.5:
                    template = np.random.choice(distance_templates)
                    description = template.format(
                        distance=max(2, min(distance, 15)),
                        object=semantic_name,
                        direction=direction
                    )
                else:
                    template = np.random.choice(spatial_templates)
                    description = template.format(
                        object=semantic_name,
                        direction=direction
                    )
            else:
                description = f"Location in {cell_id}"
            
            pose = {
                'cell_id': cell_id,
                'location': location,
                'description': description,
                'cell_center': cell_center
            }
            
            poses.append(pose)
    
    print(f"\n生成 {len(poses)} 个poses")
    print(f"示例描述:")
    for i in range(min(5, len(poses))):
        print(f"  {i+1}. {poses[i]['description']}")
    
    return poses


def main():
    print("="*80)
    print("处理 data_3d_test_slam 数据集")
    print("="*80)
    
    # 数据路径
    data_dir = "/Users/yaoyingliang/Downloads/test_0/2013_05_28_drive_0008_sync"
    output_dir = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_test_slam"
    
    # 1. 创建cells
    print("\n步骤1: 从点云创建cells...")
    cells = create_cells_from_pointclouds(
        data_dir=data_dir,
        cell_size=20.0,
        max_cells=500
    )
    
    if not cells:
        print("错误: 无法创建cells")
        return
    
    # 2. 生成poses
    print("\n步骤2: 生成poses...")
    poses = generate_poses_for_cells(cells, poses_per_cell=20)
    
    # 3. 保存数据
    print("\n步骤3: 保存数据...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "cells.pkl", 'wb') as f:
        pickle.dump(cells, f)
    
    with open(output_path / "poses.pkl", 'wb') as f:
        pickle.dump(poses, f)
    
    print("\n" + "="*80)
    print("数据处理完成!")
    print("="*80)
    print(f"\n输出目录: {output_dir}")
    print(f"  - cells.pkl: {len(cells)} cells")
    print(f"  - poses.pkl: {len(poses)} poses")
    print(f"\n数据特征:")
    print(f"  - Cell大小: 20m x 20m")
    print(f"  - 每个cell平均对象数: {np.mean([len(c['objects']) for c in cells]):.1f}")
    print(f"  - 描述平均长度: {np.mean([len(p['description']) for p in poses]):.1f} 字符")


if __name__ == '__main__':
    main()
