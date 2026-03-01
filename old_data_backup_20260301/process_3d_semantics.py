"""
处理KITTI360 3D语义数据
从22GB的PLY文件提取训练数据
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


class SemanticDataProcessor:
    """处理3D语义点云数据"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.scenes = {}
        
        # KITTI360语义标签映射
        self.semantic_labels = {
            0: 'unlabeled',
            1: 'ego vehicle',
            2: 'rectification border',
            3: 'out of roi',
            4: 'static',
            5: 'dynamic',
            6: 'ground',
            7: 'road',
            8: 'sidewalk',
            9: 'parking',
            10: 'rail track',
            11: 'building',
            12: 'wall',
            13: 'fence',
            14: 'guard rail',
            15: 'bridge',
            16: 'tunnel',
            17: 'pole',
            18: 'polegroup',
            19: 'traffic sign',
            20: 'traffic light',
            21: 'vegetation',
            22: 'terrain',
            23: 'sky',
            24: 'person',
            25: 'rider',
            26: 'car',
            27: 'truck',
            28: 'bus',
            29: 'caravan',
            30: 'trailer',
            31: 'train',
            32: 'motorcycle',
            33: 'bicycle',
            34: 'garage',
            35: 'gate',
            36: 'stop',
            37: 'smallpole',
            38: 'lamp',
            39: 'trash bin',
            40: 'vending machine',
            41: 'box',
            42: 'unknown construction',
            43: 'unknown vehicle',
            44: 'unknown object',
            45: 'license plate'
        }
    
    def parse_ply_file(self, ply_file: Path) -> np.ndarray:
        """解析PLY文件，返回点云数据 [N, 4] (x, y, z, semantic_label)"""
        try:
            with open(ply_file, 'rb') as f:
                # 读取PLY头
                header = []
                while True:
                    line = f.readline().decode('ascii').strip()
                    header.append(line)
                    if line == 'end_header':
                        break
                
                # 查找顶点数
                num_vertices = 0
                for line in header:
                    if line.startswith('element vertex'):
                        num_vertices = int(line.split()[-1])
                        break
                
                # 读取二进制数据
                dtype = np.dtype([
                    ('x', np.float32),
                    ('y', np.float32),
                    ('z', np.float32),
                    ('semantic', np.uint8),
                    ('instance', np.uint16)
                ])
                
                data = np.fromfile(f, dtype=dtype, count=num_vertices)
                
                # 转换为数组 [N, 4]
                points = np.column_stack([
                    data['x'], data['y'], data['z'], data['semantic']
                ])
                
                return points
        except Exception as e:
            logger.warning(f"解析 {ply_file} 失败: {e}")
            return np.array([])
    
    def extract_objects_from_points(self, points: np.ndarray) -> List[Dict]:
        """从点云提取物体"""
        if len(points) == 0:
            return []
        
        objects = []
        
        # 按语义标签分组
        unique_labels = np.unique(points[:, 3])
        
        for label_id in unique_labels:
            if label_id == 0:  # 跳过未标记
                continue
            
            mask = points[:, 3] == label_id
            obj_points = points[mask]
            
            if len(obj_points) < 10:  # 过滤小物体
                continue
            
            # 计算bounding box
            xyz = obj_points[:, :3]
            center = np.mean(xyz, axis=0)
            min_coords = np.min(xyz, axis=0)
            max_coords = np.max(xyz, axis=0)
            size = max_coords - min_coords
            
            # 过滤异常值
            if np.any(size > 100):  # 大于100米的物体可能是噪声
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
        """处理一个场景的所有PLY文件"""
        scene_name = scene_dir.name
        logger.info(f"\n📁 处理场景: {scene_name}")
        
        all_objects = []
        
        # 查找所有PLY文件
        ply_files = list(scene_dir.rglob("*.ply"))
        logger.info(f"  找到 {len(ply_files)} 个PLY文件")
        
        for ply_file in tqdm(ply_files[:50], desc=f"  处理 {scene_name}"):  # 限制处理数量
            points = self.parse_ply_file(ply_file)
            
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
        
        # 获取所有场景目录
        scene_dirs = [d for d in train_dir.iterdir() if d.is_dir() and "drive" in d.name]
        logger.info(f"\n{'='*60}")
        logger.info(f"发现 {len(scene_dirs)} 个场景")
        logger.info(f"{'='*60}")
        
        scenes_data = {}
        
        for scene_dir in scene_dirs:
            objects = self.process_scene(scene_dir)
            if objects:
                scenes_data[scene_dir.name] = objects
        
        return scenes_data
    
    def create_cells(self, scenes_data: Dict[str, List[Dict]], cell_size: float = 10.0) -> List[Dict]:
        """将场景分割成cells"""
        logger.info(f"\n{'='*60}")
        logger.info("创建Cells")
        logger.info(f"{'='*60}")
        
        cells = []
        cell_id = 0
        
        for scene_name, objects in scenes_data.items():
            if not objects:
                continue
            
            # 计算场景范围（过滤无效值）
            centers = np.array([obj['center'] for obj in objects])
            
            # 过滤NaN和无穷大
            valid_mask = np.all(np.isfinite(centers), axis=1)
            centers = centers[valid_mask]
            objects = [obj for i, obj in enumerate(objects) if valid_mask[i]]
            
            if len(centers) == 0:
                logger.warning(f"  ⚠️ {scene_name}: 没有有效的坐标")
                continue
            
            min_coords = np.min(centers, axis=0)
            max_coords = np.max(centers, axis=0)
            
            # 限制范围，防止数值溢出
            range_x = max_coords[0] - min_coords[0]
            range_y = max_coords[1] - min_coords[1]
            
            if range_x > 10000 or range_y > 10000:
                logger.warning(f"  ⚠️ {scene_name}: 坐标范围过大 ({range_x:.0f}, {range_y:.0f})，可能包含异常值")
                # 使用百分位数过滤异常值
                min_coords[0] = np.percentile(centers[:, 0], 1)
                max_coords[0] = np.percentile(centers[:, 0], 99)
                min_coords[1] = np.percentile(centers[:, 1], 1)
                max_coords[1] = np.percentile(centers[:, 1], 99)
            
            logger.info(f"\n📍 {scene_name}:")
            logger.info(f"  Objects: {len(objects)}")
            logger.info(f"  范围: X[{min_coords[0]:.1f}, {max_coords[0]:.1f}], "
                       f"Y[{min_coords[1]:.1f}, {max_coords[1]:.1f}]")
            
            # 创建grid（限制最大数量）
            range_x = max_coords[0] - min_coords[0]
            range_y = max_coords[1] - min_coords[1]
            
            num_x = min(100, max(1, int(range_x / cell_size) + 1))
            num_y = min(100, max(1, int(range_y / cell_size) + 1))
            
            x_range = np.linspace(min_coords[0], max_coords[0], num_x)
            y_range = np.linspace(min_coords[1], max_coords[1], num_y)
            
            scene_cells = 0
            for x in x_range[:-1]:
                for y in y_range[:-1]:
                    # 找到在这个cell范围内的objects
                    cell_objects = []
                    for obj in objects:
                        cx, cy, cz = obj['center']
                        if x <= cx < x + cell_size and y <= cy < y + cell_size:
                            cell_objects.append(obj)
                    
                    if cell_objects:
                        # 计算cell中心
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
    
    def create_poses(self, cells: List[Dict], poses_per_cell: int = 5) -> List[Dict]:
        """为每个cell创建poses（训练样本）"""
        logger.info(f"\n{'='*60}")
        logger.info("创建Poses")
        logger.info(f"{'='*60}")
        
        poses = []
        
        for cell in tqdm(cells, desc="创建poses"):
            objects = cell.get('objects', [])
            if not objects:
                continue
            
            cell_center = cell['center']
            
            # 为每个cell创建多个poses
            for i in range(min(poses_per_cell, len(objects))):
                # 选择一个物体作为参考
                ref_obj = objects[i % len(objects)]
                
                # 创建描述
                descriptions = self._generate_descriptions(ref_obj, cell['scene'])
                
                for desc in descriptions:
                    # 添加一些随机偏移
                    offset = np.random.randn(2) * 2.0  # 2米标准差
                    location = [
                        cell_center[0] + offset[0],
                        cell_center[1] + offset[1],
                        cell_center[2] if len(cell_center) > 2 else 0
                    ]
                    
                    poses.append({
                        'cell_id': cell['id'],
                        'description': desc,
                        'location': location,
                        'reference_object': ref_obj['label']
                    })
        
        logger.info(f"✅ 创建了 {len(poses)} 个poses")
        return poses
    
    def _generate_descriptions(self, obj: Dict, scene: str) -> List[str]:
        """生成自然语言描述"""
        label = obj['label']
        center = obj['center']
        
        templates = [
            f"Near the {label}",
            f"Close to a {label}",
            f"By the {label}",
            f"Next to the {label}",
            f"In front of the {label}",
            f"Behind the {label}",
            f"Left of the {label}",
            f"Right of the {label}",
        ]
        
        # 添加方向信息
        if center[0] > 0:
            templates.append(f"East of the {label}")
        else:
            templates.append(f"West of the {label}")
        
        if center[1] > 0:
            templates.append(f"North of the {label}")
        else:
            templates.append(f"South of the {label}")
        
        return templates[:3]  # 每个物体返回3个描述
    
    def save_data(self, cells: List[Dict], poses: List[Dict], output_path: str):
        """保存处理后的数据"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存cells
        cells_dir = output_path / "cells"
        cells_dir.mkdir(exist_ok=True)
        with open(cells_dir / "cells.pkl", 'wb') as f:
            pickle.dump(cells, f)
        
        # 保存poses
        poses_dir = output_path / "poses"
        poses_dir.mkdir(exist_ok=True)
        with open(poses_dir / "poses.pkl", 'wb') as f:
            pickle.dump(poses, f)
        
        # 保存统计信息
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
        logger.info(f"📁 输出路径: {output_path}")
        logger.info(f"📊 统计:")
        logger.info(f"  Cells: {stats['num_cells']}")
        logger.info(f"  Poses: {stats['num_poses']}")
        logger.info(f"  Scenes: {stats['num_scenes']}")
        logger.info(f"  Total Objects: {stats['total_objects']}")


def main():
    """主函数"""
    data_path = "/Users/yaoyingliang/Downloads/data_3d_semantics"
    output_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_semantics"
    
    logger.info("\n" + "="*60)
    logger.info("处理KITTI360 3D语义数据")
    logger.info("="*60)
    
    processor = SemanticDataProcessor(data_path)
    
    # 处理所有场景
    scenes_data = processor.process_all_scenes()
    
    if not scenes_data:
        logger.error("没有处理到任何场景数据")
        return
    
    # 创建cells
    cells = processor.create_cells(scenes_data, cell_size=10.0)
    
    # 创建poses
    poses = processor.create_poses(cells, poses_per_cell=5)
    
    # 保存数据
    processor.save_data(cells, poses, output_path)
    
    logger.info("\n✅ 处理完成！")


if __name__ == "__main__":
    main()
