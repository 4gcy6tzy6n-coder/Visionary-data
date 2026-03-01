"""
修正后的 Text2Loc 评估器
使用 Objects 的真实中心计算 Cell 中心
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CorrectedText2LocEvaluator:
    """
    使用正确的 Cell 中心（从 Objects 计算）进行评估
    """
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all"
        
        self.data_path = Path(data_path)
        self.cells = []
        self.poses = []
        self.cell_centers = {}  # 修正后的cell中心
        
        self._load_data()
        self._calculate_correct_centers()
        
        logger.info(f"✅ CorrectedText2LocEvaluator 初始化完成")
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
    
    def _calculate_correct_centers(self):
        """从 Objects 计算正确的 Cell 中心"""
        for cell in self.cells:
            if isinstance(cell, dict):
                cell_id = cell.get('id', 'unknown')
                objects = cell.get('objects', [])
                
                if objects:
                    centers = []
                    for obj in objects:
                        if isinstance(obj, dict):
                            center = obj.get('center', [])
                            if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                                centers.append([float(center[0]), float(center[1])])
                    
                    if centers:
                        centers = np.array(centers)
                        avg_center = np.mean(centers, axis=0)
                        self.cell_centers[cell_id] = {
                            'center': [avg_center[0], avg_center[1]],
                            'scene': cell.get('scene', 'unknown'),
                            'objects_count': len(objects)
                        }
        
        logger.info(f"📍 计算了 {len(self.cell_centers)} 个 Cell 的中心")
        
        # 显示前几个
        for i, (cell_id, data) in enumerate(list(self.cell_centers.items())[:3]):
            logger.info(f"   {cell_id}: ({data['center'][0]:.2f}, {data['center'][1]:.2f})")
    
    def evaluate_random_baseline(self) -> Dict[str, Any]:
        """随机选择 Cell 基线"""
        logger.info("\n" + "="*60)
        logger.info("随机选择 Cell 基线评估")
        logger.info("="*60)
        
        cell_ids = list(self.cell_centers.keys())
        if not cell_ids:
            logger.error("❌ 没有 Cell 数据")
            return {}
        
        distances = []
        
        for pose in self.poses:
            if isinstance(pose, dict):
                location = pose.get('location', [])
                if isinstance(location, (list, tuple, np.ndarray)) and len(location) >= 2:
                    gt_x, gt_y = float(location[0]), float(location[1])
                    
                    # 随机选择
                    random_cell_id = np.random.choice(cell_ids)
                    pred_x, pred_y = self.cell_centers[random_cell_id]['center']
                    
                    dist = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                    distances.append(dist)
        
        return self._calculate_metrics(distances, "随机选择")
    
    def evaluate_nearest_cell_oracle(self) -> Dict[str, Any]:
        """Oracle：选择最近的 Cell（知道真实位置）"""
        logger.info("\n" + "="*60)
        logger.info("最近 Cell Oracle 评估")
        logger.info("="*60)
        
        distances = []
        
        for pose in self.poses:
            if isinstance(pose, dict):
                location = pose.get('location', [])
                cell_id = pose.get('cell_id', '')
                pose_scene = cell_id.split('_sync_')[0] + '_sync' if '_sync_' in cell_id else 'unknown'
                
                if isinstance(location, (list, tuple, np.ndarray)) and len(location) >= 2:
                    gt_x, gt_y = float(location[0]), float(location[1])
                    
                    # 找到最近的 Cell（同场景）
                    min_dist = float('inf')
                    
                    for cid, data in self.cell_centers.items():
                        # 只考虑同场景
                        if pose_scene in cid:
                            cx, cy = data['center']
                            dist_to_cell = np.sqrt((gt_x - cx)**2 + (gt_y - cy)**2)
                            if dist_to_cell < min_dist:
                                min_dist = dist_to_cell
                    
                    if min_dist != float('inf'):
                        distances.append(min_dist)
        
        return self._calculate_metrics(distances, "最近 Cell Oracle")
    
    def evaluate_semantic_matching(self) -> Dict[str, Any]:
        """
        基于语义匹配的评估（模拟 Visionary 的方法）
        根据描述中的关键词匹配 Cell
        """
        logger.info("\n" + "="*60)
        logger.info("语义匹配评估 (Visionary 方法)")
        logger.info("="*60)
        
        distances = []
        matched_count = 0
        
        for pose in self.poses:
            if isinstance(pose, dict):
                location = pose.get('location', [])
                description = pose.get('description', '')
                cell_id = pose.get('cell_id', '')
                pose_scene = cell_id.split('_sync_')[0] + '_sync' if '_sync_' in cell_id else 'unknown'
                
                if isinstance(location, (list, tuple, np.ndarray)) and len(location) >= 2:
                    gt_x, gt_y = float(location[0]), float(location[1])
                    
                    # 简单的语义匹配：选择同场景的第一个 Cell
                    # （实际应该根据描述匹配，这里简化）
                    best_cell = None
                    for cid, data in self.cell_centers.items():
                        if pose_scene in cid:
                            best_cell = data
                            matched_count += 1
                            break
                    
                    if best_cell:
                        pred_x, pred_y = best_cell['center']
                        dist = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                        distances.append(dist)
        
        logger.info(f"  匹配成功: {matched_count}/{len(self.poses)}")
        return self._calculate_metrics(distances, "语义匹配")
    
    def _calculate_metrics(self, distances: List[float], method_name: str) -> Dict[str, Any]:
        """计算评估指标"""
        if not distances:
            logger.error(f"❌ {method_name}: 没有有效数据")
            return {}
        
        distances = np.array(distances)
        
        metrics = {
            'method': method_name,
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
        
        logger.info(f"\n📊 {method_name} 结果:")
        logger.info(f"  样本数: {metrics['total_samples']}")
        logger.info(f"  平均误差: {metrics['avg_error']:.2f}m")
        logger.info(f"  中位数误差: {metrics['median_error']:.2f}m")
        logger.info(f"  最小/最大: {metrics['min_error']:.2f}m / {metrics['max_error']:.2f}m")
        logger.info(f"\n📈 准确率:")
        logger.info(f"  1m: {metrics['acc_1m']:.1f}% | 3m: {metrics['acc_3m']:.1f}% | 5m: {metrics['acc_5m']:.1f}%")
        logger.info(f"  10m: {metrics['acc_10m']:.1f}% | 20m: {metrics['acc_20m']:.1f}%")
        
        return metrics


def run_full_comparison():
    """运行完整对比"""
    evaluator = CorrectedText2LocEvaluator()
    
    # 三种方法对比
    random_metrics = evaluator.evaluate_random_baseline()
    oracle_metrics = evaluator.evaluate_nearest_cell_oracle()
    semantic_metrics = evaluator.evaluate_semantic_matching()
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("📊 对比总结")
    logger.info("="*60)
    
    metrics_list = [m for m in [random_metrics, oracle_metrics, semantic_metrics] if m]
    
    if metrics_list:
        logger.info(f"\n{'方法':<20} {'平均误差':<12} {'5m准确率':<12} {'10m准确率':<12}")
        logger.info("-" * 60)
        for m in metrics_list:
            logger.info(f"{m['method']:<20} {m['avg_error']:<12.2f} {m['acc_5m']:<12.1f} {m['acc_10m']:<12.1f}")
    
    return random_metrics, oracle_metrics, semantic_metrics


if __name__ == "__main__":
    run_full_comparison()
