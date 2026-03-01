#!/usr/bin/env python3
"""
KITTI360数据语义标签恢复与数据增强
1. 基于RGB颜色推断语义标签
2. 通过子采样增加cells数量
"""

import pickle
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


KITTI360_CLASSES = {
    'building': {
        'description': '建筑物、房屋、结构',
        'typical_colors': ['gray', 'brown', 'white', 'cream'],
        'rgb_ranges': {
            'gray': ((0.3, 0.4), (0.3, 0.4), (0.3, 0.4)),
            'brown': ((0.4, 0.6), (0.25, 0.4), (0.15, 0.25)),
            'white': ((0.7, 1.0), (0.7, 1.0), (0.7, 1.0)),
            'cream': ((0.9, 1.0), (0.85, 0.95), (0.75, 0.85)),
        }
    },
    'vegetation': {
        'description': '树木、植被、草地',
        'typical_colors': ['green', 'dark_green'],
        'rgb_ranges': {
            'green': ((0.1, 0.35), (0.4, 0.7), (0.1, 0.35)),
            'dark_green': ((0.05, 0.2), (0.25, 0.5), (0.05, 0.2)),
        }
    },
    'road': {
        'description': '道路、街道',
        'typical_colors': ['gray', 'dark_gray', 'black'],
        'rgb_ranges': {
            'gray': ((0.25, 0.4), (0.25, 0.4), (0.25, 0.4)),
            'dark_gray': ((0.15, 0.25), (0.15, 0.25), (0.15, 0.25)),
            'black': ((0.0, 0.15), (0.0, 0.15), (0.0, 0.15)),
        }
    },
    'sidewalk': {
        'description': '人行道',
        'typical_colors': ['light_gray', 'concrete'],
        'rgb_ranges': {
            'light_gray': ((0.5, 0.7), (0.5, 0.7), (0.5, 0.7)),
            'concrete': ((0.4, 0.55), (0.4, 0.55), (0.4, 0.55)),
        }
    },
    'traffic sign': {
        'description': '交通标志',
        'typical_colors': ['red', 'blue', 'white'],
        'rgb_ranges': {
            'red': ((0.7, 1.0), (0.0, 0.2), (0.0, 0.2)),
            'blue': ((0.0, 0.2), (0.3, 0.5), (0.6, 0.9)),
            'white': ((0.8, 1.0), (0.8, 1.0), (0.8, 1.0)),
        }
    },
    'traffic light': {
        'description': '交通灯',
        'typical_colors': ['red', 'green', 'yellow'],
        'rgb_ranges': {
            'red': ((0.8, 1.0), (0.0, 0.15), (0.0, 0.15)),
            'green': ((0.0, 0.15), (0.6, 0.9), (0.0, 0.2)),
            'yellow': ((0.9, 1.0), (0.7, 0.9), (0.0, 0.15)),
        }
    },
    'pole': {
        'description': '电线杆、路灯杆',
        'typical_colors': ['gray_metal', 'dark_gray'],
        'rgb_ranges': {
            'gray_metal': ((0.4, 0.6), (0.4, 0.6), (0.4, 0.6)),
            'dark_gray': ((0.2, 0.35), (0.2, 0.35), (0.2, 0.35)),
        }
    },
    'car': {
        'description': '车辆',
        'typical_colors': ['white_car', 'black_car', 'red_car', 'blue_car', 'silver'],
        'rgb_ranges': {
            'white_car': ((0.85, 1.0), (0.85, 1.0), (0.85, 1.0)),
            'black_car': ((0.0, 0.15), (0.0, 0.15), (0.0, 0.15)),
            'red_car': ((0.7, 0.95), (0.1, 0.25), (0.1, 0.25)),
            'blue_car': ((0.1, 0.25), (0.2, 0.4), (0.5, 0.8)),
            'silver': ((0.6, 0.8), (0.6, 0.8), (0.6, 0.8)),
        }
    },
    'terrain': {
        'description': '地形、草地',
        'typical_colors': ['light_green', 'brown_earth'],
        'rgb_ranges': {
            'light_green': ((0.3, 0.5), (0.4, 0.6), (0.2, 0.35)),
            'brown_earth': ((0.35, 0.5), (0.25, 0.4), (0.15, 0.25)),
        }
    },
    'fence': {
        'description': '围栏',
        'typical_colors': ['brown_fence', 'gray_fence'],
        'rgb_ranges': {
            'brown_fence': ((0.4, 0.6), (0.25, 0.4), (0.15, 0.25)),
            'gray_fence': ((0.35, 0.55), (0.35, 0.55), (0.35, 0.55)),
        }
    },
    'wall': {
        'description': '墙壁',
        'typical_colors': ['white_wall', 'gray_wall'],
        'rgb_ranges': {
            'white_wall': ((0.8, 0.95), (0.8, 0.95), (0.8, 0.95)),
            'gray_wall': ((0.4, 0.6), (0.4, 0.6), (0.4, 0.6)),
        }
    },
    'parking': {
        'description': '停车区域',
        'typical_colors': ['dark_parking', 'gray_parking'],
        'rgb_ranges': {
            'dark_parking': ((0.15, 0.25), (0.15, 0.25), (0.15, 0.25)),
            'gray_parking': ((0.3, 0.45), (0.3, 0.45), (0.3, 0.45)),
        }
    },
    'trash bin': {
        'description': '垃圾桶',
        'typical_colors': ['green_bin', 'blue_bin', 'gray_bin'],
        'rgb_ranges': {
            'green_bin': ((0.2, 0.35), (0.5, 0.7), (0.2, 0.35)),
            'blue_bin': ((0.15, 0.25), (0.35, 0.55), (0.5, 0.75)),
            'gray_bin': ((0.35, 0.5), (0.35, 0.5), (0.35, 0.5)),
        }
    },
    'vending machine': {
        'description': '自动售货机',
        'typical_colors': ['colorful_vending'],
        'rgb_ranges': {
            'colorful_vending': ((0.5, 0.9), (0.4, 0.8), (0.5, 0.9)),
        }
    },
}


def rgb_to_color_name(r: float, g: float, b: float) -> Tuple[str, float]:
    """将RGB颜色转换为颜色名称和置信度"""
    best_match = 'unknown'
    best_confidence = 0.0
    
    for color_name, rgb_ranges in {
        'white': ((0.85, 1.0), (0.85, 1.0), (0.85, 1.0)),
        'black': ((0.0, 0.1), (0.0, 0.1), (0.0, 0.1)),
        'gray': ((0.3, 0.5), (0.3, 0.5), (0.3, 0.5)),
        'red': ((0.6, 1.0), (0.0, 0.3), (0.0, 0.3)),
        'green': ((0.0, 0.3), (0.5, 0.8), (0.0, 0.3)),
        'blue': ((0.0, 0.2), (0.2, 0.5), (0.6, 0.9)),
        'yellow': ((0.8, 1.0), (0.7, 0.9), (0.0, 0.2)),
        'brown': ((0.4, 0.6), (0.25, 0.4), (0.15, 0.25)),
        'orange': ((0.9, 1.0), (0.4, 0.6), (0.0, 0.2)),
        'purple': ((0.5, 0.7), (0.2, 0.4), (0.6, 0.8)),
    }.items():
        r_min, r_max = rgb_ranges[0]
        g_min, g_max = rgb_ranges[1]
        b_min, b_max = rgb_ranges[2]
        
        if r_min <= r <= r_max and g_min <= g <= g_max and b_min <= b <= b_max:
            confidence = 1.0 - (abs(r - (r_min + r_max) / 2) / (r_max - r_min + 0.01) +
                               abs(g - (g_min + g_max) / 2) / (g_max - g_min + 0.01) +
                               abs(b - (b_min + b_max) / 2) / (b_max - b_min + 0.01)) / 3
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = color_name
    
    return best_match, best_confidence


def infer_semantic_label(r: float, g: float, b: float) -> Tuple[str, float]:
    """基于RGB颜色推断语义标签"""
    color_name, color_confidence = rgb_to_color_name(r, g, b)
    
    label_mapping = {
        'white': ['building', 'wall', 'sidewalk', 'car', 'trash bin'],
        'black': ['car', 'road', 'pole'],
        'gray': ['road', 'pole', 'building', 'wall', 'fence', 'sidewalk'],
        'red': ['traffic sign', 'car', 'trash bin'],
        'green': ['vegetation', 'trash bin', 'terrain'],
        'blue': ['traffic sign', 'trash bin', 'car'],
        'yellow': ['traffic light', 'vending machine', 'trash bin'],
        'brown': ['building', 'fence', 'terrain', 'wall'],
        'orange': ['traffic sign', 'car'],
        'purple': ['trash bin', 'vending machine'],
    }
    
    possible_labels = label_mapping.get(color_name, ['building', 'road', 'vegetation', 'pole'])
    
    label_scores = {}
    for label in possible_labels:
        label_scores[label] = color_confidence
    
    label_scores['vegetation'] = label_scores.get('vegetation', 0.0)
    if g > r + 0.2 and g > b + 0.2:
        label_scores['vegetation'] = max(label_scores['vegetation'], 0.7)
    if g > 0.5 and r < 0.3 and b < 0.3:
        label_scores['vegetation'] = max(label_scores['vegetation'], 0.8)
    
    label_scores['building'] = label_scores.get('building', 0.0)
    if 0.3 <= r <= 0.6 and 0.3 <= g <= 0.6 and 0.3 <= b <= 0.6:
        label_scores['building'] = max(label_scores['building'], 0.7)
    
    label_scores['road'] = label_scores.get('road', 0.0)
    if r < 0.3 and g < 0.3 and b < 0.3:
        label_scores['road'] = max(label_scores['road'], 0.6)
    
    label_scores['pole'] = label_scores.get('pole', 0.0)
    if 0.2 <= r <= 0.5 and 0.2 <= g <= 0.5 and 0.2 <= b <= 0.5:
        if max(r, g, b) - min(r, g, b) < 0.15:
            label_scores['pole'] = max(label_scores['pole'], 0.65)
    
    best_label = max(label_scores, key=label_scores.get)
    best_confidence = label_scores[best_label]
    
    return best_label, best_confidence


def augment_cell(cell: Dict, split_factor: int = 2, offset_range: float = 5.0) -> List[Dict]:
    """增强单个cell：通过子采样和扰动创建多个变体"""
    augmented_cells = []
    
    scene_name = cell.get('scene', 'unknown')
    objects = cell.get('objects', [])
    
    if not objects:
        return [cell]
    
    for i in range(split_factor):
        new_cell = {
            'id': f"{scene_name}_aug{i}_{random.randint(1000, 9999)}",
            'scene': scene_name,
            'objects': [],
            'augmented': True,
            'augmentation_type': 'perturbation'
        }
        
        for obj in objects:
            new_obj = obj.copy() if isinstance(obj, dict) else dict(obj)
            
            if 'center' in new_obj and isinstance(new_obj['center'], (list, tuple)):
                center = list(new_obj['center'])
                if len(center) >= 2:
                    noise_x = random.uniform(-offset_range, offset_range)
                    noise_y = random.uniform(-offset_range, offset_range)
                    center[0] += noise_x
                    center[1] += noise_y
                    new_obj['center'] = center
            
            new_cell['objects'].append(new_obj)
        
        augmented_cells.append(new_cell)
    
    return augmented_cells


class SemanticDataProcessor:
    """语义数据处理器"""
    
    def __init__(self, input_cells_path: str, output_dir: str):
        self.input_cells_path = input_cells_path
        self.output_dir = output_dir
        self.cells = []
        self.augmented_cells = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_cells(self) -> bool:
        """加载原始cells"""
        if not os.path.exists(self.input_cells_path):
            logger.error(f"Cells文件不存在: {self.input_cells_path}")
            return False
        
        with open(self.input_cells_path, 'rb') as f:
            self.cells = pickle.load(f)
        
        logger.info(f"加载了 {len(self.cells)} 个原始cells")
        return True
    
    def repair_semantic_labels(self) -> bool:
        """修复所有cells的语义标签"""
        if not self.cells:
            if not self.load_cells():
                return False
        
        label_stats = defaultdict(int)
        
        for cell in self.cells:
            objects = cell.get('objects', [])
            for obj in objects:
                if isinstance(obj, dict):
                    original_label = obj.get('label', 'unknown')
                    
                    color = obj.get('color', [])
                    if isinstance(color, (list, tuple, np.ndarray)) and len(color) >= 3:
                        r, g, b = color[0], color[1], color[2]
                        
                        if original_label == 'unknown' or not original_label:
                            new_label, confidence = infer_semantic_label(r, g, b)
                            obj['label'] = new_label
                            obj['label_confidence'] = confidence
                            obj['label_repaired'] = True
                        else:
                            obj['label_confidence'] = 1.0
                            obj['label_repaired'] = False
                        
                        label_stats[new_label] += 1
        
        logger.info(f"语义标签修复完成，标签分布: {dict(label_stats)}")
        return True
    
    def augment_data(self, augmentation_factor: int = 5) -> bool:
        """数据增强"""
        if not self.cells:
            if not self.load_cells():
                return False
        
        logger.info(f"开始数据增强，原始cells: {len(self.cells)}, 增强因子: {augmentation_factor}")
        
        self.augmented_cells = list(self.cells)
        
        for cell in self.cells:
            augmented_versions = augment_cell(cell, split_factor=augmentation_factor, offset_range=3.0)
            self.augmented_cells.extend(augmented_versions[1:])
        
        logger.info(f"增强完成，总cells: {len(self.augmented_cells)}")
        return True
    
    def create_semantic_index(self) -> Dict:
        """创建语义索引"""
        semantic_index = defaultdict(lambda: defaultdict(list))
        
        for cell in self.augmented_cells:
            scene = cell.get('scene', 'unknown')
            objects = cell.get('objects', [])
            
            for obj in objects:
                if isinstance(obj, dict):
                    label = obj.get('label', 'unknown')
                    color = obj.get('color', [])
                    center = obj.get('center', [])
                    
                    if isinstance(color, (list, tuple, np.ndarray)) and len(color) >= 3:
                        r, g, b = color[0], color[1], color[2]
                        color_name, _ = rgb_to_color_name(r, g, b)
                    else:
                        color_name = 'unknown'
                    
                    semantic_index[scene][label].append({
                        'cell_id': cell.get('id', 'unknown'),
                        'color': color_name,
                        'center': center,
                        'confidence': obj.get('label_confidence', 0.5)
                    })
        
        return dict(semantic_index)
    
    def save_results(self) -> Tuple[str, str, str]:
        """保存处理结果"""
        cells_output = os.path.join(self.output_dir, 'cells_augmented.pkl')
        index_output = os.path.join(self.output_dir, 'semantic_index.pkl')
        stats_output = os.path.join(self.output_dir, 'processing_stats.txt')
        
        with open(cells_output, 'wb') as f:
            pickle.dump(self.augmented_cells, f)
        logger.info(f"保存增强cells到: {cells_output}")
        
        semantic_index = self.create_semantic_index()
        with open(index_output, 'wb') as f:
            pickle.dump(semantic_index, f)
        logger.info(f"保存语义索引到: {index_output}")
        
        with open(stats_output, 'w') as f:
            f.write("KITTI360数据语义标签恢复与增强统计\n")
            f.write("="*60 + "\n\n")
            f.write(f"原始cells数量: {len(self.cells)}\n")
            f.write(f"增强后cells数量: {len(self.augmented_cells)}\n")
            f.write(f"增强倍数: {len(self.augmented_cells) / max(len(self.cells), 1):.1f}x\n\n")
            
            label_counts = defaultdict(int)
            for cell in self.augmented_cells:
                for obj in cell.get('objects', []):
                    if isinstance(obj, dict):
                        label = obj.get('label', 'unknown')
                        label_counts[label] += 1
            
            f.write("语义标签分布:\n")
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                f.write(f"  {label}: {count} ({count/len(self.augmented_cells)/10*100:.1f}%)\n")
            
            f.write("\n场景分布:\n")
            scene_counts = defaultdict(int)
            for cell in self.augmented_cells:
                scene = cell.get('scene', 'unknown')
                scene_counts[scene] += 1
            
            for scene, count in sorted(scene_counts.items()):
                f.write(f"  {scene}: {count}\n")
        
        logger.info(f"保存统计到: {stats_output}")
        
        return cells_output, index_output, stats_output


def main():
    """主函数"""
    print("="*80)
    print("KITTI360数据 语义标签恢复与数据增强")
    print("="*80)
    
    input_path = '/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired/cells/cells.pkl'
    output_dir = '/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_semantic_augmented'
    
    processor = SemanticDataProcessor(input_path, output_dir)
    
    print("\n[1/4] 加载原始数据...")
    if not processor.load_cells():
        print("❌ 加载失败")
        return
    
    print("\n[2/4] 修复语义标签...")
    processor.repair_semantic_labels()
    
    print("\n[3/4] 数据增强...")
    processor.augment_data(augmentation_factor=5)
    
    print("\n[4/4] 保存结果...")
    cells_file, index_file, stats_file = processor.save_results()
    
    print("\n" + "="*80)
    print("处理完成!")
    print("="*80)
    print(f"  原始cells: {len(processor.cells)}")
    print(f"  增强后cells: {len(processor.augmented_cells)}")
    print(f"  增强倍数: {len(processor.augmented_cells) / len(processor.cells):.1f}x")
    print(f"\n输出文件:")
    print(f"  - {cells_file}")
    print(f"  - {index_file}")
    print(f"  - {stats_file}")
    print("="*80)
    
    return processor


if __name__ == "__main__":
    main()
