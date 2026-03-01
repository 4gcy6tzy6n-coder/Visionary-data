#!/usr/bin/env python3
"""
数据集修复脚本
将RGB颜色映射到预定义颜色名称，提升匹配精度
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 预定义颜色映射（基于Text2Loc-one的COLOR_NAMES）
COLOR_MAP = {
    'dark-green': [0.1, 0.4, 0.1],
    'gray': [0.5, 0.5, 0.5],
    'gray-green': [0.4, 0.5, 0.4],
    'bright-gray': [0.7, 0.7, 0.7],
    'black': [0.1, 0.1, 0.1],
    'green': [0.2, 0.6, 0.2],
    'beige': [0.8, 0.7, 0.5],
}

# 扩展颜色映射（用于更精确的颜色识别）
EXTENDED_COLOR_MAP = {
    'red': [0.8, 0.2, 0.2],
    'green': [0.2, 0.8, 0.2],
    'blue': [0.2, 0.2, 0.8],
    'yellow': [0.9, 0.9, 0.2],
    'white': [0.9, 0.9, 0.9],
    'black': [0.1, 0.1, 0.1],
    'gray': [0.5, 0.5, 0.5],
    'orange': [0.9, 0.5, 0.2],
    'pink': [0.9, 0.6, 0.7],
    'purple': [0.6, 0.2, 0.8],
    'brown': [0.6, 0.4, 0.2],
}


def rgb_to_color_name(rgb: np.ndarray) -> str:
    """
    将RGB数组映射到最近的颜色名称
    
    Args:
        rgb: RGB颜色数组 [r, g, b]
        
    Returns:
        颜色名称
    """
    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)
    
    min_dist = float('inf')
    best_color = 'unknown'
    
    # 使用扩展颜色映射
    for color_name, color_rgb in EXTENDED_COLOR_MAP.items():
        dist = np.linalg.norm(rgb - np.array(color_rgb))
        if dist < min_dist:
            min_dist = dist
            best_color = color_name
    
    # 如果距离太大，认为是unknown
    if min_dist > 0.5:
        return 'unknown'
    
    return best_color


def repair_object(obj: Dict) -> Dict:
    """
    修复单个object的数据
    
    Args:
        obj: 原始object字典
        
    Returns:
        修复后的object字典
    """
    repaired = obj.copy()
    
    # 修复颜色：将RGB转换为颜色名称
    if 'color' in obj:
        color = obj['color']
        if isinstance(color, (list, tuple, np.ndarray)):
            color_name = rgb_to_color_name(np.array(color))
            repaired['color_name'] = color_name
            logger.debug(f"RGB {color} -> {color_name}")
    
    # 保持原始label（即使它是unknown）
    # 后续可以通过其他方式推断label
    
    return repaired


def repair_cell(cell: Dict) -> Dict:
    """
    修复单个cell的数据
    
    Args:
        cell: 原始cell字典
        
    Returns:
        修复后的cell字典
    """
    repaired = cell.copy()
    
    if 'objects' in cell:
        repaired_objects = []
        for obj in cell['objects']:
            if isinstance(obj, dict):
                repaired_obj = repair_object(obj)
                repaired_objects.append(repaired_obj)
            else:
                repaired_objects.append(obj)
        repaired['objects'] = repaired_objects
    
    return repaired


def repair_dataset(data_path: Path, output_path: Path) -> bool:
    """
    修复整个数据集
    
    Args:
        data_path: 原始数据路径
        output_path: 输出路径
        
    Returns:
        是否成功
    """
    logger.info("=" * 80)
    logger.info("开始修复数据集")
    logger.info("=" * 80)
    
    # 加载原始数据
    cells_file = data_path / "cells" / "cells.pkl"
    poses_file = data_path / "poses" / "poses.pkl"
    
    if not cells_file.exists():
        logger.error(f"Cells文件不存在: {cells_file}")
        return False
    
    if not poses_file.exists():
        logger.error(f"Poses文件不存在: {poses_file}")
        return False
    
    try:
        # 加载cells
        logger.info(f"加载cells: {cells_file}")
        with open(cells_file, 'rb') as f:
            cells = pickle.load(f)
        logger.info(f"  加载了 {len(cells)} 个cells")
        
        # 加载poses
        logger.info(f"加载poses: {poses_file}")
        with open(poses_file, 'rb') as f:
            poses = pickle.load(f)
        logger.info(f"  加载了 {len(poses)} 个poses")
        
        # 修复cells
        logger.info("\n修复cells...")
        repaired_cells = []
        total_objects = 0
        color_distribution = {}
        
        for i, cell in enumerate(cells):
            if isinstance(cell, dict):
                repaired_cell = repair_cell(cell)
                repaired_cells.append(repaired_cell)
                
                # 统计颜色分布
                if 'objects' in repaired_cell:
                    for obj in repaired_cell['objects']:
                        if isinstance(obj, dict) and 'color_name' in obj:
                            color_name = obj['color_name']
                            color_distribution[color_name] = color_distribution.get(color_name, 0) + 1
                            total_objects += 1
            else:
                repaired_cells.append(cell)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  已修复 {i + 1}/{len(cells)} 个cells")
        
        logger.info(f"\n修复完成!")
        logger.info(f"  总objects: {total_objects}")
        logger.info(f"  颜色分布:")
        for color, count in sorted(color_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_objects * 100 if total_objects > 0 else 0
            logger.info(f"    {color}: {count} ({percentage:.1f}%)")
        
        # 保存修复后的数据
        logger.info("\n保存修复后的数据...")
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "cells").mkdir(exist_ok=True)
        (output_path / "poses").mkdir(exist_ok=True)
        
        with open(output_path / "cells" / "cells.pkl", 'wb') as f:
            pickle.dump(repaired_cells, f)
        logger.info(f"  保存cells: {output_path / 'cells' / 'cells.pkl'}")
        
        with open(output_path / "poses" / "poses.pkl", 'wb') as f:
            pickle.dump(poses, f)
        logger.info(f"  保存poses: {output_path / 'poses' / 'poses.pkl'}")
        
        logger.info("\n✅ 数据集修复完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 数据路径
    original_data_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all")
    repaired_data_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired")
    
    # 修复数据集
    success = repair_dataset(original_data_path, repaired_data_path)
    
    if success:
        logger.info(f"\n修复后的数据保存在: {repaired_data_path}")
        logger.info("可以使用修复后的数据重新运行实验")
    else:
        logger.error("数据修复失败")


if __name__ == "__main__":
    main()
