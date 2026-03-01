#!/usr/bin/env python3
"""
修复数据集，为objects添加语义标签
由于原始KITTI360的3D语义数据(.ply)不可用，使用启发式方法根据颜色、大小等特征推断物体类型
"""

import pickle
import numpy as np
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# KITTI360语义标签映射（从原始论文）
KITTI360_LABELS = {
    7: 'road',
    8: 'sidewalk',
    9: 'parking',
    11: 'building',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    15: 'bridge',
    16: 'tunnel',
    17: 'pole',
    18: 'polegroup',
    19: 'traffic light',
    20: 'traffic sign',
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
    45: 'license plate',
}

# 颜色到物体类型的启发式映射（基于常见颜色）
COLOR_TO_CLASS_HEURISTICS = {
    # 绿色系 -> vegetation
    ('green',): 'vegetation',
    ('darkgreen',): 'vegetation',
    ('forest',): 'vegetation',
    ('lime',): 'vegetation',

    # 灰色系 -> building/road/wall
    ('gray', 'grey'): 'building',
    ('darkgray', 'darkgrey'): 'road',
    ('lightgray', 'lightgrey'): 'sidewalk',
    ('silver',): 'guard rail',

    # 蓝色系
    ('blue',): 'sky',  # 或者water（如果有）
    ('navy',): 'sky',
    ('lightblue',): 'sky',

    # 红色系
    ('red',): 'traffic sign',  # 或者stop
    ('darkred',): 'stop',
    ('maroon',): 'stop',

    # 黄色/橙色系
    ('yellow',): 'traffic sign',
    ('orange',): 'traffic sign',
    ('gold',): 'traffic sign',

    # 白色系
    ('white',): 'wall',
    ('whitesmoke',): 'wall',

    # 黑色系
    ('black',): 'road',

    # 棕色系
    ('brown',): 'terrain',
    ('tan',): 'terrain',
}

# 物体大小特征（高度阈值，单位：米）
SIZE_THRESHOLDS = {
    'vegetation': (2.0, 30.0),  # 树木高度范围
    'building': (3.0, 100.0),   # 建筑物
    'pole': (2.0, 15.0),        # 电线杆
    'traffic light': (3.0, 8.0),
    'traffic sign': (1.5, 5.0),
    'wall': (1.0, 5.0),
    'fence': (1.0, 3.0),
    'garage': (2.5, 10.0),
    'stop': (1.5, 3.0),
    'lamp': (3.0, 12.0),
    'trash bin': (0.8, 2.0),
    'box': (0.3, 2.0),
    'road': (0.0, 0.5),         # 路面接近地面
    'sidewalk': (0.0, 0.3),
    'terrain': (0.0, 2.0),
}


def rgb_to_color_name(rgb: np.ndarray) -> str:
    """将RGB值映射到最接近的颜色名称"""
    # 定义标准颜色
    colors = {
        'red': np.array([0.8, 0.2, 0.2]),
        'green': np.array([0.2, 0.7, 0.2]),
        'blue': np.array([0.2, 0.3, 0.8]),
        'yellow': np.array([0.9, 0.8, 0.2]),
        'orange': np.array([0.9, 0.5, 0.1]),
        'purple': np.array([0.6, 0.2, 0.7]),
        'pink': np.array([0.9, 0.5, 0.6]),
        'brown': np.array([0.6, 0.4, 0.2]),
        'gray': np.array([0.5, 0.5, 0.5]),
        'white': np.array([0.9, 0.9, 0.9]),
        'black': np.array([0.1, 0.1, 0.1]),
    }

    min_dist = float('inf')
    closest_color = 'unknown'

    for name, color in colors.items():
        dist = np.linalg.norm(rgb - color)
        if dist < min_dist:
            min_dist = dist
            closest_color = name

    return closest_color


def infer_label_from_features(obj: Dict) -> str:
    """根据物体特征推断语义标签"""
    # 获取颜色
    color = obj.get('color', [0.5, 0.5, 0.5])
    if isinstance(color, list):
        color = np.array(color)
    elif isinstance(color, np.ndarray):
        pass
    else:
        color = np.array([0.5, 0.5, 0.5])

    color_name = rgb_to_color_name(color)

    # 获取大小信息
    size = obj.get('size', [1.0, 1.0, 1.0])
    if isinstance(size, (list, tuple, np.ndarray)) and len(size) >= 3:
        height = float(size[2]) if size[2] > 0 else 1.0
    else:
        height = 1.0

    # 获取中心位置
    center = obj.get('center', [0, 0, 0])
    if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 3:
        z_pos = float(center[2])
    else:
        z_pos = 0.0

    # 启发式规则
    scores = defaultdict(float)

    # 基于颜色的推断
    if color_name in ['green', 'darkgreen']:
        scores['vegetation'] += 3.0
    elif color_name in ['gray', 'grey']:
        scores['building'] += 1.5
        scores['road'] += 1.0
        scores['wall'] += 1.0
    elif color_name in ['red', 'darkred', 'maroon']:
        scores['traffic sign'] += 2.0
        scores['stop'] += 1.5
    elif color_name in ['yellow', 'orange', 'gold']:
        scores['traffic sign'] += 2.0
        scores['traffic light'] += 1.0
    elif color_name in ['blue', 'navy']:
        scores['sky'] += 1.0  # 可能不是物体
    elif color_name in ['white', 'whitesmoke']:
        scores['wall'] += 1.5
        scores['building'] += 1.0
    elif color_name in ['brown', 'tan']:
        scores['terrain'] += 2.0
        scores['building'] += 0.5

    # 基于高度的推断
    if height > 10:
        scores['building'] += 2.0
        scores['vegetation'] += 1.0
    elif height > 5:
        scores['vegetation'] += 2.0
        scores['building'] += 1.0
        scores['pole'] += 1.0
    elif height > 2:
        scores['pole'] += 1.5
        scores['traffic light'] += 1.0
        scores['lamp'] += 1.0
        scores['vegetation'] += 1.0
    elif height > 1:
        scores['traffic sign'] += 1.5
        scores['stop'] += 1.0
        scores['trash bin'] += 1.0
        scores['box'] += 0.5
    else:
        scores['road'] += 1.0
        scores['sidewalk'] += 1.0
        scores['terrain'] += 0.5

    # 基于位置（z坐标）的推断
    if z_pos < 0.5:
        scores['road'] += 1.0
        scores['sidewalk'] += 1.0
        scores['terrain'] += 0.5

    # 选择得分最高的标签
    if scores:
        best_label = max(scores, key=scores.get)
        confidence = scores[best_label] / sum(scores.values()) if sum(scores.values()) > 0 else 0
    else:
        best_label = 'unknown'
        confidence = 0.0

    return best_label, confidence, color_name


def repair_cell_objects(cell: Dict) -> Dict:
    """修复单个cell中的所有objects，添加语义标签"""
    if 'objects' not in cell:
        return cell

    repaired_objects = []
    for obj in cell['objects']:
        # 推断标签
        label, confidence, color_name = infer_label_from_features(obj)

        # 创建新的object（保留原有字段）
        new_obj = obj.copy()
        new_obj['label'] = label
        new_obj['label_confidence'] = round(confidence, 2)
        new_obj['inferred_color_name'] = color_name

        # 如果原始数据有class_name字段，也更新它
        if 'class_name' in new_obj:
            new_obj['class_name'] = label

        repaired_objects.append(new_obj)

    # 创建新的cell
    repaired_cell = cell.copy()
    repaired_cell['objects'] = repaired_objects

    return repaired_cell


def repair_dataset(input_path: str, output_path: str):
    """修复整个数据集"""
    print(f"Loading data from: {input_path}")

    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} items")

    # 修复每个cell
    repaired_data = []
    label_distribution = defaultdict(int)

    for i, item in enumerate(data):
        if isinstance(item, dict):
            repaired_item = repair_cell_objects(item)
        else:
            # 如果是对象类型，转换为字典处理
            repaired_item = item

        repaired_data.append(repaired_item)

        # 统计标签分布
        if isinstance(repaired_item, dict) and 'objects' in repaired_item:
            for obj in repaired_item['objects']:
                if isinstance(obj, dict):
                    label = obj.get('label', 'unknown')
                    label_distribution[label] += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(data)} items")

    # 保存修复后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(repaired_data, f)

    print(f"\nSaved repaired data to: {output_path}")
    print(f"\nLabel distribution:")
    for label, count in sorted(label_distribution.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    return repaired_data


def main():
    """主函数"""
    # 输入输出路径
    cells_input = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired/cells/cells.pkl"
    cells_output = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_semantic/cells/cells.pkl"

    # 修复cells数据
    print("=" * 60)
    print("Repairing cells data with semantic labels...")
    print("=" * 60)
    repair_dataset(cells_input, cells_output)

    # 同时复制poses数据（不需要修改）
    poses_input = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired/poses/poses.pkl"
    poses_output = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_semantic/poses/poses.pkl"

    print("\n" + "=" * 60)
    print("Copying poses data...")
    print("=" * 60)

    os.makedirs(os.path.dirname(poses_output), exist_ok=True)
    with open(poses_input, 'rb') as f:
        poses = pickle.load(f)
    with open(poses_output, 'wb') as f:
        pickle.dump(poses, f)
    print(f"Copied {len(poses)} poses to: {poses_output}")

    print("\n" + "=" * 60)
    print("Data repair completed!")
    print("=" * 60)
    print(f"\nNew dataset location: /Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_semantic/")
    print("You can now update the config to use this new dataset.")


if __name__ == "__main__":
    main()
