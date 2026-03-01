"""
Text2Loc-main 集成评估器
直接使用 Text2Loc-main 的代码和模型进行评估
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加 Text2Loc-main 到路径
TEXT2LOC_MAIN_PATH = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc-main")
if str(TEXT2LOC_MAIN_PATH) not in sys.path:
    sys.path.insert(0, str(TEXT2LOC_MAIN_PATH))

class Text2LocMainEvaluator:
    """
    使用 Text2Loc-main 原始代码进行评估
    """
    
    def __init__(self, data_path: str = None):
        """
        初始化评估器
        
        Args:
            data_path: 数据集路径
        """
        if data_path is None:
            data_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all"
        
        self.data_path = Path(data_path)
        self.cells = []
        self.poses = []
        
        # 加载数据
        self._load_data()
        
        logger.info(f"✅ Text2LocMainEvaluator 初始化完成")
        logger.info(f"   Cells: {len(self.cells)}")
        logger.info(f"   Poses: {len(self.poses)}")
    
    def _load_data(self):
        """加载数据"""
        cells_file = self.data_path / "cells" / "cells.pkl"
        poses_file = self.data_path / "poses" / "poses.pkl"
        
        if cells_file.exists():
            with open(cells_file, 'rb') as f:
                self.cells = pickle.load(f)
            logger.info(f"📁 加载 {len(self.cells)} cells")
        
        if poses_file.exists():
            with open(poses_file, 'rb') as f:
                self.poses = pickle.load(f)
            logger.info(f"📁 加载 {len(self.poses)} poses")
    
    def evaluate_baseline(self) -> Dict[str, Any]:
        """
        评估基线方法：随机选择cell中心
        
        Returns:
            评估指标
        """
        logger.info("\n" + "="*60)
        logger.info("Text2Loc-main 基线评估 (随机选择)")
        logger.info("="*60)
        
        if not self.cells or not self.poses:
            logger.error("❌ 数据未加载")
            return {}
        
        # 获取所有cell的中心
        cell_centers = []
        for cell in self.cells:
            if isinstance(cell, dict):
                center = cell.get('center', [0, 0])
                if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                    cell_centers.append([float(center[0]), float(center[1])])
        
        if not cell_centers:
            logger.error("❌ 无法获取cell中心")
            return {}
        
        cell_centers = np.array(cell_centers)
        
        # 评估每个pose
        distances = []
        results = []
        
        for pose in self.poses:
            if isinstance(pose, dict):
                location = pose.get('location', [0, 0])
                if isinstance(location, (list, tuple, np.ndarray)) and len(location) >= 2:
                    gt_x, gt_y = float(location[0]), float(location[1])
                    
                    # 随机选择一个cell中心作为预测
                    pred_idx = np.random.randint(len(cell_centers))
                    pred_x, pred_y = cell_centers[pred_idx]
                    
                    # 计算距离
                    dist = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                    distances.append(dist)
                    
                    results.append({
                        'gt': [gt_x, gt_y],
                        'pred': [pred_x, pred_y],
                        'error': dist
                    })
        
        if not distances:
            logger.error("❌ 没有有效的评估结果")
            return {}
        
        distances = np.array(distances)
        
        # 计算指标
        metrics = {
            'avg_error': float(np.mean(distances)),
            'median_error': float(np.median(distances)),
            'min_error': float(np.min(distances)),
            'max_error': float(np.max(distances)),
            'std_error': float(np.std(distances)),
            'acc_1m': float(np.mean(distances <= 1.0) * 100),
            'acc_3m': float(np.mean(distances <= 3.0) * 100),
            'acc_5m': float(np.mean(distances <= 5.0) * 100),
            'acc_10m': float(np.mean(distances <= 10.0) * 100),
            'acc_20m': float(np.mean(distances <= 20.0) * 100),
            'total_samples': len(distances)
        }
        
        # 打印结果
        logger.info(f"\n📊 评估结果:")
        logger.info(f"  样本数: {metrics['total_samples']}")
        logger.info(f"  平均误差: {metrics['avg_error']:.2f}m")
        logger.info(f"  中位数误差: {metrics['median_error']:.2f}m")
        logger.info(f"  最小误差: {metrics['min_error']:.2f}m")
        logger.info(f"  最大误差: {metrics['max_error']:.2f}m")
        logger.info(f"  标准差: {metrics['std_error']:.2f}m")
        logger.info(f"\n📈 准确率:")
        logger.info(f"  1m内: {metrics['acc_1m']:.1f}%")
        logger.info(f"  3m内: {metrics['acc_3m']:.1f}%")
        logger.info(f"  5m内: {metrics['acc_5m']:.1f}%")
        logger.info(f"  10m内: {metrics['acc_10m']:.1f}%")
        logger.info(f"  20m内: {metrics['acc_20m']:.1f}%")
        
        return metrics
    
    def evaluate_nearest_cell(self) -> Dict[str, Any]:
        """
        评估最近cell方法：选择距离最近的cell中心
        
        Returns:
            评估指标
        """
        logger.info("\n" + "="*60)
        logger.info("Text2Loc-main 最近Cell评估 (Oracle)")
        logger.info("="*60)
        
        if not self.cells or not self.poses:
            logger.error("❌ 数据未加载")
            return {}
        
        # 获取所有cell的中心和场景
        cell_data = []
        for cell in self.cells:
            if isinstance(cell, dict):
                center = cell.get('center', [0, 0])
                scene = cell.get('scene', 'unknown')
                if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                    cell_data.append({
                        'center': [float(center[0]), float(center[1])],
                        'scene': scene,
                        'cell_id': cell.get('id', 'unknown')
                    })
        
        if not cell_data:
            logger.error("❌ 无法获取cell数据")
            return {}
        
        # 评估每个pose
        distances = []
        results = []
        
        for pose in self.poses:
            if isinstance(pose, dict):
                location = pose.get('location', [0, 0])
                cell_id = pose.get('cell_id', 'unknown')
                pose_scene = cell_id.split('_')[0] if '_' in cell_id else 'unknown'
                
                if isinstance(location, (list, tuple, np.ndarray)) and len(location) >= 2:
                    gt_x, gt_y = float(location[0]), float(location[1])
                    
                    # 找到最近的cell（同场景）
                    min_dist = float('inf')
                    best_cell = None
                    
                    for cell in cell_data:
                        # 只考虑同场景的cell
                        if cell['scene'] == pose_scene or pose_scene in cell['cell_id']:
                            cx, cy = cell['center']
                            dist = np.sqrt((gt_x - cx)**2 + (gt_y - cy)**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_cell = cell
                    
                    if best_cell:
                        distances.append(min_dist)
                        results.append({
                            'gt': [gt_x, gt_y],
                            'pred': best_cell['center'],
                            'error': min_dist,
                            'cell_id': best_cell['cell_id']
                        })
        
        if not distances:
            logger.error("❌ 没有有效的评估结果")
            return {}
        
        distances = np.array(distances)
        
        # 计算指标
        metrics = {
            'avg_error': float(np.mean(distances)),
            'median_error': float(np.median(distances)),
            'min_error': float(np.min(distances)),
            'max_error': float(np.max(distances)),
            'std_error': float(np.std(distances)),
            'acc_1m': float(np.mean(distances <= 1.0) * 100),
            'acc_3m': float(np.mean(distances <= 3.0) * 100),
            'acc_5m': float(np.mean(distances <= 5.0) * 100),
            'acc_10m': float(np.mean(distances <= 10.0) * 100),
            'acc_20m': float(np.mean(distances <= 20.0) * 100),
            'total_samples': len(distances)
        }
        
        # 打印结果
        logger.info(f"\n📊 评估结果 (最近Cell - Oracle):")
        logger.info(f"  样本数: {metrics['total_samples']}")
        logger.info(f"  平均误差: {metrics['avg_error']:.2f}m")
        logger.info(f"  中位数误差: {metrics['median_error']:.2f}m")
        logger.info(f"  最小误差: {metrics['min_error']:.2f}m")
        logger.info(f"  最大误差: {metrics['max_error']:.2f}m")
        logger.info(f"  标准差: {metrics['std_error']:.2f}m")
        logger.info(f"\n📈 准确率:")
        logger.info(f"  1m内: {metrics['acc_1m']:.1f}%")
        logger.info(f"  3m内: {metrics['acc_3m']:.1f}%")
        logger.info(f"  5m内: {metrics['acc_5m']:.1f}%")
        logger.info(f"  10m内: {metrics['acc_10m']:.1f}%")
        logger.info(f"  20m内: {metrics['acc_20m']:.1f}%")
        
        return metrics


def run_comparison():
    """运行对比评估"""
    evaluator = Text2LocMainEvaluator()
    
    # 基线评估（随机）
    baseline_metrics = evaluator.evaluate_baseline()
    
    # Oracle评估（最近cell）
    oracle_metrics = evaluator.evaluate_nearest_cell()
    
    # 对比
    logger.info("\n" + "="*60)
    logger.info("对比总结")
    logger.info("="*60)
    
    if baseline_metrics and oracle_metrics:
        logger.info(f"\n随机选择 vs 最近Cell (Oracle):")
        logger.info(f"  平均误差: {baseline_metrics['avg_error']:.2f}m → {oracle_metrics['avg_error']:.2f}m")
        logger.info(f"  5m准确率: {baseline_metrics['acc_5m']:.1f}% → {oracle_metrics['acc_5m']:.1f}%")
        
        improvement = (baseline_metrics['avg_error'] - oracle_metrics['avg_error']) / baseline_metrics['avg_error'] * 100
        logger.info(f"  提升: {improvement:.1f}%")
    
    return baseline_metrics, oracle_metrics


if __name__ == "__main__":
    run_comparison()
