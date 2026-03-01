#!/usr/bin/env python3
"""
完整处理 data_3d_test_slam 数据集 - 使用所有点云文件
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


def load_velodyne_points(bin_path: str) -> np.ndarray:
    """加载KITTI格式的velodyne点云数据 (.bin文件)"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def analyze_full_scene_range(data_dir: str) -> Tuple[float, float, float, float]:
    """分析所有点云文件的完整场景范围"""
    print("分析完整场景范围...")
    
    data_path = Path(data_dir)
    velodyne_dir = data_path / "velodyne_points" / "data"
    bin_files = sorted(velodyne_dir.glob("*.bin"))
    
    all_x, all_y = [], []
    
    for bin_file in tqdm(bin_files, desc="扫描点云文件"):
        points = load_velodyne_points(str(bin_file))
        all_x.extend(points[:, 0].tolist())
        all_y.extend(points[:, 1].tolist())
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    print(f"完整场景范围: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}]")
    print(f"场景尺寸: {x_max-x_min:.1f}m x {y_max-y_min:.1f}m")
    
    return x_min, x_max, y_min, y_max


def extract_objects_from_pointcloud(points: np.ndarray, 
                                    min_points: int = 30) -> List[Dict]:
    """从点云中提取对象 - 使用更细粒度的分割"""
    objects = []
    
    if len(points) < min_points:
        return objects
    
    # 根据高度分割
    ground_mask = points[:, 2] < 0.3  # 地面高度阈值
    non_ground = points[~ground_mask]
    
    if len(non_ground) < min_points:
        return objects
    
    # 更细的网格分割 (1m x 1m)
    grid_size = 1.0
    x_min, x_max = non_ground[:, 0].min(), non_ground[:, 0].max()
    y_min, y_max = non_ground[:, 1].min(), non_ground[:, 1].max()
    
    x_bins = max(1, int((x_max - x_min) / grid_size))
    y_bins = max(1, int((y_max - y_min) / grid_size))
    
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
                center_x = grid_points[:, 0].mean()
                center_y = grid_points[:, 1].mean()
                center_z = grid_points[:, 2].mean()
                
                # 更精细的语义分类
                z_mean = center_z
                z_std = grid_points[:, 2].std()
                
                if z_mean < 0.5:
                    semantic = 8  # road
                elif z_mean < 1.0:
                    semantic = 9  # sidewalk
                elif z_mean < 2.0 and z_std < 0.5:
                    semantic = 1  # building (flat, tall)
                elif z_mean < 3.0 and z_std > 0.5:
                    semantic = 3  # tree (irregular, medium height)
                elif z_mean < 1.5:
                    semantic = 2  # car (low, compact)
                else:
                    semantic = 0  # unknown
                
                obj = {
                    'center': [float(center_x), float(center_y), float(center_z)],
                    'points': len(grid_points),
                    'semantic': semantic,
                    'bbox': [
                        float(grid_points[:, 0].min()), float(grid_points[:, 0].max()),
                        float(grid_points[:, 1].min()), float(grid_points[:, 1].max()),
                        float(grid_points[:, 2].min()), float(grid_points[:, 2].max())
                    ]
                }
                objects.append(obj)
    
    return objects


def create_cells_from_all_pointclouds(data_dir: str, 
                                      cell_size: float = 10.0) -> List[Dict]:
    """从所有点云文件创建cells"""
    print("\n" + "="*80)
    print("从所有点云创建Cells")
    print("="*80)
    
    # 1. 分析完整场景范围
    x_min, x_max, y_min, y_max = analyze_full_scene_range(data_dir)
    
    data_path = Path(data_dir)
    velodyne_dir = data_path / "velodyne_points" / "data"
    bin_files = sorted(velodyne_dir.glob("*.bin"))
    
    print(f"\n总点云文件数: {len(bin_files)}")
    print(f"Cell大小: {cell_size}m x {cell_size}m")
    
    # 2. 创建网格
    cells = []
    cell_id = 0
    
    x = x_min
    while x < x_max:
        y = y_min
        while y < y_max:
            cell_center_x = x + cell_size / 2
            cell_center_y = y + cell_size / 2
            
            # 收集这个cell内的所有点云
            cell_objects = []
            cell_point_count = 0
            
            for bin_file in bin_files:
                points = load_velodyne_points(str(bin_file))
                
                # 检查点云是否在cell范围内
                mask = (
                    (points[:, 0] >= x) & (points[:, 0] < x + cell_size) &
                    (points[:, 1] >= y) & (points[:, 1] < y + cell_size)
                )
                
                if np.any(mask):
                    cell_points = points[mask]
                    cell_point_count += len(cell_points)
                    
                    # 提取对象
                    objects = extract_objects_from_pointcloud(cell_points)
                    cell_objects.extend(objects)
            
            # 只保留有足够对象的cell
            if len(cell_objects) >= 5:
                cell = {
                    'id': f'cell_{cell_id:04d}',
                    'center': [float(cell_center_x), float(cell_center_y), 0.0],
                    'objects': cell_objects,
                    'size': cell_size,
                    'bounds': [float(x), float(x + cell_size), float(y), float(y + cell_size)],
                    'point_count': cell_point_count,
                    'object_count': len(cell_objects)
                }
                cells.append(cell)
                cell_id += 1
                
                if cell_id % 50 == 0:
                    print(f"  已创建 {cell_id} 个cells...")
            
            y += cell_size
        x += cell_size
    
    print(f"\n总共创建 {len(cells)} 个cells")
    print(f"平均每个cell对象数: {np.mean([c['object_count'] for c in cells]):.1f}")
    
    return cells


def generate_poses_for_cells(cells: List[Dict], 
                             poses_per_cell: int = 50) -> List[Dict]:
    """为每个cell生成更多poses"""
    print("\n" + "="*80)
    print("生成Poses")
    print("="*80)
    
    poses = []
    
    distance_templates = [
        "{distance} meters from the {object}",
        "about {distance}m away from the {object}",
        "roughly {distance} meters to the {direction} of the {object}",
        "near the {object}, about {distance}m away",
        "approximately {distance}m from the {object} on the {direction}",
        "standing {distance} meters {direction} of the {object}",
        "located {distance}m to the {direction} of the {object}"
    ]
    
    spatial_templates = [
        "to the {direction} of the {object}",
        "on the {direction} side of the {object}",
        "{direction} of the {object}",
        "in front of the {object}",
        "behind the {object}",
        "next to the {object} on the {direction}",
        "close to the {object}, {direction} side"
    ]
    
    complex_templates = [
        "{distance} meters {direction} of the {object1}, near the {object2}",
        "between the {object1} and {object2}, about {distance}m from {object1}",
        "to the {direction} of the {object1}, close to the {object2}",
        "in the {direction} part of the area, {distance}m from the {object}",
        "near the {object1}, {distance} meters to the {direction}, by the {object2}"
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
            # 在cell内随机生成位置 (限制在中心区域)
            offset_x = np.random.uniform(-cell_size/2.5, cell_size/2.5)
            offset_y = np.random.uniform(-cell_size/2.5, cell_size/2.5)
            
            location = [
                cell_center[0] + offset_x,
                cell_center[1] + offset_y,
                0
            ]
            
            # 生成描述
            if len(objects) >= 2:
                template_type = np.random.choice(['distance', 'spatial', 'complex'], 
                                                  p=[0.4, 0.3, 0.3])
                
                if template_type == 'distance':
                    obj = np.random.choice(objects)
                    semantic_id = obj.get('semantic', 0)
                    semantic_name = semantic_map.get(semantic_id, 'object')
                    
                    template = np.random.choice(distance_templates)
                    description = template.format(
                        distance=np.random.choice(distances),
                        object=semantic_name,
                        direction=np.random.choice(directions)
                    )
                
                elif template_type == 'spatial':
                    obj = np.random.choice(objects)
                    semantic_id = obj.get('semantic', 0)
                    semantic_name = semantic_map.get(semantic_id, 'object')
                    
                    template = np.random.choice(spatial_templates)
                    description = template.format(
                        object=semantic_name,
                        direction=np.random.choice(directions)
                    )
                
                else:  # complex
                    obj1, obj2 = np.random.choice(objects, 2, replace=False)
                    semantic_id1 = obj1.get('semantic', 0)
                    semantic_id2 = obj2.get('semantic', 0)
                    semantic_name1 = semantic_map.get(semantic_id1, 'object')
                    semantic_name2 = semantic_map.get(semantic_id2, 'object')
                    
                    template = np.random.choice(complex_templates)
                    if '{object1}' in template:
                        description = template.format(
                            distance=np.random.choice(distances),
                            direction=np.random.choice(directions),
                            object1=semantic_name1,
                            object2=semantic_name2
                        )
                    else:
                        obj = np.random.choice(objects)
                        semantic_id = obj.get('semantic', 0)
                        semantic_name = semantic_map.get(semantic_id, 'object')
                        description = template.format(
                            distance=np.random.choice(distances),
                            direction=np.random.choice(directions),
                            object=semantic_name
                        )
            elif len(objects) == 1:
                obj = objects[0]
                semantic_id = obj.get('semantic', 0)
                semantic_name = semantic_map.get(semantic_id, 'object')
                
                template = np.random.choice(distance_templates + spatial_templates)
                if '{distance}' in template:
                    description = template.format(
                        distance=np.random.choice(distances),
                        object=semantic_name,
                        direction=np.random.choice(directions)
                    )
                else:
                    description = template.format(
                        object=semantic_name,
                        direction=np.random.choice(directions)
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
    
    # 显示示例
    print("\n描述示例:")
    unique_descs = set()
    for pose in poses:
        unique_descs.add(pose['description'])
        if len(unique_descs) <= 10:
            print(f"  - {pose['description']}")
    
    print(f"\n唯一描述数: {len(unique_descs)} / {len(poses)}")
    
    return poses


def main():
    print("="*80)
    print("完整处理 data_3d_test_slam 数据集")
    print("="*80)
    
    # 数据路径
    data_dir = "/Users/yaoyingliang/Downloads/test_0/2013_05_28_drive_0008_sync"
    output_dir = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_test_slam_full"
    
    # 1. 创建cells - 使用更小的cell size以获得更多cells
    print("\n步骤1: 从所有点云创建cells...")
    cells = create_cells_from_all_pointclouds(
        data_dir=data_dir,
        cell_size=10.0  # 10m cells
    )
    
    if not cells:
        print("错误: 无法创建cells")
        return
    
    # 2. 生成更多poses
    print("\n步骤2: 生成poses...")
    poses = generate_poses_for_cells(cells, poses_per_cell=50)
    
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
    print(f"  - Cell大小: 10m x 10m")
    print(f"  - 每个cell平均对象数: {np.mean([len(c['objects']) for c in cells]):.1f}")
    print(f"  - 描述平均长度: {np.mean([len(p['description']) for p in poses]):.1f} 字符")
    print(f"  - 唯一描述数: {len(set(p['description'] for p in poses))}")


if __name__ == '__main__':
    main()
