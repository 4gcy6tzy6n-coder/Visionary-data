"""
处理KITTI360 3D语义数据 - 完整版
利用48GB内存处理更多数据
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class SemanticDataProcessorFull:
    """处理3D语义点云数据 - 完整版"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
        # KITTI360语义标签映射
        self.semantic_labels = {
            0: 'unlabeled', 1: 'ego vehicle', 2: 'rectification border',
            3: 'out of roi', 4: 'static', 5: 'dynamic', 6: 'ground',
            7: 'road', 8: 'sidewalk', 9: 'parking', 10: 'rail track',
            11: 'building', 12: 'wall', 13: 'fence', 14: 'guard rail',
            15: 'bridge', 16: 'tunnel', 17: 'pole', 18: 'polegroup',
            19: 'traffic sign', 20: 'traffic light', 21: 'vegetation',
            22: 'terrain', 23: 'sky', 24: 'person', 25: 'rider',
            26: 'car', 27: 'truck', 28: 'bus', 29: 'caravan',
            30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle',
            34: 'garage', 35: 'gate', 36: 'stop', 37: 'smallpole',
            38: 'lamp', 39: 'trash bin', 40: 'vending machine',
            41: 'box', 42: 'unknown construction', 43: 'unknown vehicle',
            44: 'unknown object', 45: 'license plate'
        }
    
    def parse_static_ply_full(self, ply_file: Path, max_points: int = 1000000) -> np.ndarray:
        """解析static PLY文件 - 读取更多点"""
        try:
            with open(ply_file, 'rb') as f:
                # 读取头部
                header_lines = []
                while True:
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    header_lines.append(line)
                    if line == 'end_header':
                        break
                    if len(header_lines) > 200:
                        break
                
                # 解析头部
                num_vertices = 0
                for line in header_lines:
                    if line.startswith('element vertex'):
                        num_vertices = int(line.split()[-1])
                        break
                
                if num_vertices == 0:
                    return np.array([])
                
                # 读取更多点（利用48GB内存）
                read_vertices = min(num_vertices, max_points)
                
                # 数据格式
                dtype = np.dtype([
                    ('x', np.float32), ('y', np.float32), ('z', np.float32),
                    ('red', np.uint8), ('green', np.uint8), ('blue', np.uint8),
                    ('semantic', np.int32), ('instance', np.int32),
                    ('visible', np.uint8), ('confidence', np.float32)
                ])
                
                # 直接读取
                data = np.fromfile(f, dtype=dtype, count=read_vertices)
                
                # 转换为数组
                points = np.column_stack([
                    data['x'].astype(np.float64),
                    data['y'].astype(np.float64),
                    data['z'].astype(np.float64),
                    data['semantic'].astype(np.int32)
                ])
                
                # 过滤无效值
                if len(points) > 0:
                    valid_mask = (
                        np.isfinite(points[:, 0]) & 
                        np.isfinite(points[:, 1]) & 
                        np.isfinite(points[:, 2]) &
                        (points[:, 3] >= 0) & (points[:, 3] <= 45)
                    )
                    points = points[valid_mask]
                
                return points
                
        except Exception as e:
            logger.warning(f"  解析失败 {ply_file.name}: {e}")
            return np.array([])
    
    def extract_objects_from_points(self, points: np.ndarray) -> List[Dict]:
        """从点云提取物体"""
        if len(points) == 0:
            return []
        
        objects = []
        unique_labels = np.unique(points[:, 3])
        
        for label_id in unique_labels:
            if label_id == 0:  # 跳过未标记
                continue
            
            mask = points[:, 3] == label_id
            obj_points = points[mask]
            
            if len(obj_points) < 100:  # 过滤小物体（提高阈值）
                continue
            
            xyz = obj_points[:, :3]
            center = np.mean(xyz, axis=0)
            min_coords = np.min(xyz, axis=0)
            max_coords = np.max(xyz, axis=0)
            size = max_coords - min_coords
            
            # 过滤异常物体
            if np.any(size > 200) or np.any(np.isnan(center)):
                continue
            
            label_name = self.semantic_labels.get(int(label_id), f'unknown_{label_id}')
            
            objects.append({
                'label': label_name,
                'center': center.tolist(),
                'size': size.tolist(),
                'num_points': len(obj_points),
                'semantic_id': int(label_id)
            })
        
        return objects
    
    def process_scene(self, scene_dir: Path) -> List[Dict]:
        """处理一个场景的所有static PLY文件"""
        scene_name = scene_dir.name
        static_dir = scene_dir / "static"
        
        if not static_dir.exists():
            return []
        
        ply_files = list(static_dir.glob("*.ply"))
        logger.info(f"\n📁 处理场景: {scene_name}")
        logger.info(f"  找到 {len(ply_files)} 个static PLY文件")
        
        if len(ply_files) == 0:
            return []
        
        # 处理所有PLY文件（利用48GB内存）
        all_objects = []
        for ply_file in tqdm(ply_files, desc=f"  处理 {scene_name}"):
            points = self.parse_static_ply_full(ply_file, max_points=1000000)  # 读取100万点
            if len(points) > 0:
                objects = self.extract_objects_from_points(points)
                all_objects.extend(objects)
        
        logger.info(f"  ✅ 提取 {len(all_objects)} 个objects")
        return all_objects
    
    def process_all_scenes(self) -> Dict[str, List[Dict]]:
        """处理所有场景"""
        train_dir = self.data_path / "train"
        
        if not train_dir.exists():
            logger.error(f"训练目录不存在: {train_dir}")
            return {}
        
        scene_dirs = [d for d in train_dir.iterdir() if d.is_dir() and "drive" in d.name]
        logger.info(f"\n{'='*60}")
        logger.info(f"发现 {len(scene_dirs)} 个场景")
        logger.info(f"利用48GB内存处理全部数据...")
        logger.info(f"{'='*60}")
        
        scenes_data = {}
        
        for scene_dir in scene_dirs:
            objects = self.process_scene(scene_dir)
            if objects:
                scenes_data[scene_dir.name] = objects
        
        return scenes_data
    
    def create_cells(self, scenes_data: Dict[str, List[Dict]], cell_size: float = 10.0) -> List[Dict]:
        """创建cells"""
        logger.info(f"\n{'='*60}")
        logger.info("创建Cells")
        logger.info(f"{'='*60}")
        
        cells = []
        cell_id = 0
        
        for scene_name, objects in scenes_data.items():
            if not objects:
                continue
            
            centers = np.array([obj['center'] for obj in objects])
            
            # 过滤无效值
            valid_mask = np.all(np.isfinite(centers), axis=1)
            centers = centers[valid_mask]
            objects = [obj for i, obj in enumerate(objects) if valid_mask[i]]
            
            if len(centers) == 0:
                continue
            
            min_coords = np.min(centers, axis=0)
            max_coords = np.max(centers, axis=0)
            
            logger.info(f"\n📍 {scene_name}:")
            logger.info(f"  Objects: {len(objects)}")
            logger.info(f"  范围: X[{min_coords[0]:.1f}, {max_coords[0]:.1f}], "
                       f"Y[{min_coords[1]:.1f}, {max_coords[1]:.1f}]")
            
            # 创建grid
            range_x = max_coords[0] - min_coords[0]
            range_y = max_coords[1] - min_coords[1]
            
            num_x = min(100, max(1, int(range_x / cell_size) + 1))
            num_y = min(100, max(1, int(range_y / cell_size) + 1))
            
            x_starts = np.linspace(min_coords[0], max_coords[0] - cell_size, num_x)
            y_starts = np.linspace(min_coords[1], max_coords[1] - cell_size, num_y)
            
            scene_cells = 0
            for x in x_starts:
                for y in y_starts:
                    cell_objects = []
                    for obj in objects:
                        cx, cy = obj['center'][0], obj['center'][1]
                        if x <= cx < x + cell_size and y <= cy < y + cell_size:
                            cell_objects.append(obj)
                    
                    if cell_objects:
                        cell_centers = np.array([obj['center'] for obj in cell_objects])
                        cell_center = np.mean(cell_centers, axis=0)
                        
                        cells.append({
                            'id': f"{scene_name}_{cell_id:04d}",
                            'scene': scene_name,
                            'center': cell_center.tolist(),
                            'objects': cell_objects,
                            'bbox': [float(x), float(y), float(x + cell_size), float(y + cell_size)]
                        })
                        cell_id += 1
                        scene_cells += 1
            
            logger.info(f"  ✅ 创建了 {scene_cells} 个cells")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"总计: {len(cells)} 个cells")
        logger.info(f"{'='*60}")
        
        return cells
    
    def create_poses(self, cells: List[Dict], poses_per_cell: int = 20) -> List[Dict]:
        """创建poses - 增加数量"""
        logger.info(f"\n{'='*60}")
        logger.info("创建Poses")
        logger.info(f"{'='*60}")
        
        poses = []
        
        for cell in tqdm(cells, desc="创建poses"):
            objects = cell.get('objects', [])
            if not objects:
                continue
            
            cell_center = cell['center']
            
            for i in range(min(poses_per_cell, len(objects))):
                ref_obj = objects[i % len(objects)]
                
                templates = [
                    f"Near the {ref_obj['label']}",
                    f"Close to a {ref_obj['label']}",
                    f"By the {ref_obj['label']}",
                    f"Next to the {ref_obj['label']}",
                    f"In front of the {ref_obj['label']}",
                    f"Behind the {ref_obj['label']}",
                ]
                
                for desc in templates[:3]:
                    offset = np.random.randn(2) * 3.0
                    location = [
                        cell_center[0] + offset[0],
                        cell_center[1] + offset[1],
                        cell_center[2] if len(cell_center) > 2 else 0
                    ]
                    
                    poses.append({
                        'cell_id': cell['id'],
                        'description': desc,
                        'location': location,
                        'reference_object': ref_obj['label'],
                        'scene': cell['scene']
                    })
        
        logger.info(f"✅ 创建了 {len(poses)} 个poses")
        return poses
    
    def save_data(self, cells: List[Dict], poses: List[Dict], output_path: str):
        """保存数据"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cells_dir = output_path / "cells"
        cells_dir.mkdir(exist_ok=True)
        with open(cells_dir / "cells.pkl", 'wb') as f:
            pickle.dump(cells, f)
        
        poses_dir = output_path / "poses"
        poses_dir.mkdir(exist_ok=True)
        with open(poses_dir / "poses.pkl", 'wb') as f:
            pickle.dump(poses, f)
        
        stats = {
            'num_cells': len(cells),
            'num_poses': len(poses),
            'num_scenes': len(set(c['scene'] for c in cells)),
            'total_objects': sum(len(c['objects']) for c in cells)
        }
        
        with open(output_path / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("数据保存完成")
        logger.info(f"{'='*60}")
        logger.info(f"📁 输出: {output_path}")
        logger.info(f"📊 Cells: {stats['num_cells']}, Poses: {stats['num_poses']}")


def main():
    data_path = "/Users/yaoyingliang/Downloads/data_3d_semantics"
    output_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_semantics_full"
    
    logger.info("\n" + "="*60)
    logger.info("处理KITTI360 3D语义数据 - 完整版")
    logger.info("利用48GB内存处理全部数据")
    logger.info("="*60)
    
    processor = SemanticDataProcessorFull(data_path)
    
    scenes_data = processor.process_all_scenes()
    
    if not scenes_data:
        logger.error("没有处理到数据")
        return
    
    cells = processor.create_cells(scenes_data, cell_size=10.0)
    poses = processor.create_poses(cells, poses_per_cell=20)
    
    processor.save_data(cells, poses, output_path)
    
    logger.info("\n✅ 处理完成！")


if __name__ == "__main__":
    main()
