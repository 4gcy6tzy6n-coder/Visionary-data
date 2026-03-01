"""
处理KITTI360的3D bounding box数据 V2
正确解析OpenCV格式的XML文件
"""

import os
import sys
import xml.etree.ElementTree as ET
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class KITTI360BBoxParserV2:
    """解析KITTI360的3D bounding box XML文件 (OpenCV格式)"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.scenes = {}
    
    def parse_xml(self, xml_file: Path) -> List[Dict]:
        """解析单个XML文件 (OpenCV格式)"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        objects = []
        
        # OpenCV格式：每个object是<object1>, <object2>等
        for child in root:
            if child.tag.startswith('object'):
                obj_data = {}
                
                # 获取标签
                label_elem = child.find('label')
                if label_elem is not None:
                    obj_data['label'] = label_elem.text
                else:
                    obj_data['label'] = 'unknown'
                
                # 获取变换矩阵
                transform = child.find('transform')
                if transform is not None:
                    data_elem = transform.find('data')
                    if data_elem is not None:
                        # 解析4x4变换矩阵
                        transform_data = [float(x) for x in data_elem.text.split()]
                        if len(transform_data) >= 16:
                            # 提取平移部分 (最后列的前3行)
                            tx, ty, tz = transform_data[3], transform_data[7], transform_data[11]
                            obj_data['center'] = [tx, ty, tz]
                        else:
                            obj_data['center'] = [0, 0, 0]
                    else:
                        obj_data['center'] = [0, 0, 0]
                else:
                    obj_data['center'] = [0, 0, 0]
                
                # 获取语义ID
                semantic_id = child.find('semanticId')
                if semantic_id is not None:
                    obj_data['semantic_id'] = int(semantic_id.text)
                else:
                    obj_data['semantic_id'] = -1
                
                # 获取实例ID
                instance_id = child.find('instanceId')
                if instance_id is not None:
                    obj_data['instance_id'] = int(instance_id.text)
                else:
                    obj_data['instance_id'] = -1
                
                objects.append(obj_data)
        
        return objects
    
    def process_all_scenes(self) -> Dict[str, List[Dict]]:
        """处理所有场景"""
        train_path = self.data_path / 'train'
        
        if not train_path.exists():
            logger.error(f"❌ 路径不存在: {train_path}")
            return {}
        
        logger.info(f"📁 处理场景数据: {train_path}")
        
        xml_files = sorted(train_path.glob('*.xml'))
        for xml_file in tqdm(xml_files, desc="解析XML"):
            scene_name = xml_file.stem
            
            objects = self.parse_xml(xml_file)
            self.scenes[scene_name] = objects
            
            if objects:
                logger.info(f"  {scene_name}: {len(objects)} objects")
        
        return self.scenes
    
    def create_cells(self, cell_size: float = 20.0) -> List[Dict]:
        """
        将场景分割成cells
        """
        cells = []
        cell_id = 0
        
        for scene_name, objects in self.scenes.items():
            if not objects:
                continue
            
            # 计算场景范围
            centers = np.array([obj['center'] for obj in objects])
            min_coords = np.min(centers, axis=0)
            max_coords = np.max(centers, axis=0)
            
            logger.info(f"\n  {scene_name}:")
            logger.info(f"    Objects: {len(objects)}")
            logger.info(f"    范围: X[{min_coords[0]:.1f}, {max_coords[0]:.1f}], Y[{min_coords[1]:.1f}, {max_coords[1]:.1f}]")
            
            # 创建grid
            x_range = np.arange(min_coords[0], max_coords[0] + cell_size, cell_size)
            y_range = np.arange(min_coords[1], max_coords[1] + cell_size, cell_size)
            
            scene_cells = 0
            for i, x in enumerate(x_range[:-1]):
                for j, y in enumerate(y_range[:-1]):
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
                            'bbox': [
                                float(x), float(y),
                                float(x + cell_size), float(y + cell_size)
                            ],
                            'cell_size': cell_size
                        })
                        cell_id += 1
                        scene_cells += 1
            
            logger.info(f"    Cells: {scene_cells}")
        
        logger.info(f"\n✅ 总共创建了 {len(cells)} 个cells")
        return cells
    
    def create_poses(self, num_poses_per_cell: int = 5) -> List[Dict]:
        """
        为每个cell创建poses（模拟查询）
        """
        poses = []
        
        for cell in self.cells:
            objects = cell['objects']
            if not objects:
                continue
            
            for i in range(min(num_poses_per_cell, len(objects))):
                # 选择一个object作为目标
                target_obj = objects[i % len(objects)]
                
                # 创建描述
                label = target_obj['label']
                scene = cell['scene']
                
                # 多种描述方式
                descriptions = [
                    f"Find the {label} in {scene}",
                    f"Locate the {label} near {scene}",
                    f"Where is the {label} in {scene}?",
                    f"Find location with {label} in {scene}",
                    f"Search for {label} in {scene}"
                ]
                description = descriptions[i % len(descriptions)]
                
                poses.append({
                    'cell_id': cell['id'],
                    'description': description,
                    'location': target_obj['center'],
                    'target_object': target_obj,
                    'scene': scene
                })
        
        logger.info(f"✅ 创建了 {len(poses)} 个poses")
        return poses
    
    def process_and_save(self, output_path: str):
        """处理并保存数据"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 解析所有场景
        self.process_all_scenes()
        
        # 创建cells
        self.cells = self.create_cells(cell_size=20.0)
        
        # 创建poses
        self.poses = self.create_poses(num_poses_per_cell=5)
        
        # 保存
        cells_file = output_path / 'cells' / 'cells.pkl'
        cells_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cells_file, 'wb') as f:
            pickle.dump(self.cells, f)
        
        poses_file = output_path / 'poses' / 'poses.pkl'
        poses_file.parent.mkdir(parents=True, exist_ok=True)
        with open(poses_file, 'wb') as f:
            pickle.dump(self.poses, f)
        
        logger.info(f"\n💾 数据已保存:")
        logger.info(f"  Cells: {cells_file}")
        logger.info(f"  Poses: {poses_file}")
        
        # 统计信息
        self._print_statistics()
    
    def _print_statistics(self):
        """打印统计信息"""
        logger.info("\n" + "="*60)
        logger.info("📊 数据统计")
        logger.info("="*60)
        logger.info(f"  场景数: {len(self.scenes)}")
        logger.info(f"  Cells: {len(self.cells)}")
        logger.info(f"  Poses: {len(self.poses)}")
        
        # 统计objects
        total_objects = sum(len(cell['objects']) for cell in self.cells)
        avg_objects = total_objects / len(self.cells) if self.cells else 0
        logger.info(f"  总Objects: {total_objects}")
        logger.info(f"  平均Objects/Cell: {avg_objects:.1f}")
        
        # 统计labels
        label_counts = {}
        for cell in self.cells:
            for obj in cell['objects']:
                label = obj['label']
                label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"\n  语义标签分布 (Top 10):")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"    {label}: {count}")


def main():
    """主函数"""
    # 输入路径
    input_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/kitti360_raw/data_3d_bboxes"
    
    # 输出路径
    output_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_bboxes"
    
    # 处理数据
    parser = KITTI360BBoxParserV2(input_path)
    parser.process_and_save(output_path)
    
    logger.info("\n✅ 数据处理完成！")


if __name__ == "__main__":
    main()
