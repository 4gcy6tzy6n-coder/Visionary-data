#!/usr/bin/env python3
"""
Text2Loc Visionary - 增强数据评估脚本
使用语义标签恢复和数据增强后的数据进行全面评估
"""

import sys
import os
import pickle
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

sys.path.insert(0, '/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary')

from api.text2loc_adapter import Text2LocAdapter


class EnhancedDataEvaluator:
    """增强数据评估器"""
    
    def __init__(self):
        self.adapter = None
        self.test_cases = []
        
    def initialize_with_augmented_data(self) -> bool:
        """使用增强数据初始化适配器"""
        augmented_path = '/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_semantic_augmented/cells_augmented.pkl'
        
        if not os.path.exists(augmented_path):
            print(f"❌ 增强数据不存在: {augmented_path}")
            return False
        
        print(f"📁 加载增强数据: {augmented_path}")
        
        with open(augmented_path, 'rb') as f:
            augmented_cells = pickle.load(f)
        
        print(f"✅ 加载了 {len(augmented_cells)} 个增强cells")
        
        self.adapter = Text2LocAdapter()
        self.adapter.cells = {'default': augmented_cells}
        
        label_stats = {}
        for cell in augmented_cells:
            for obj in cell.get('objects', []):
                if isinstance(obj, dict):
                    label = obj.get('label', 'unknown')
                    label_stats[label] = label_stats.get(label, 0) + 1
        
        print(f"📊 语义标签分布: {label_stats}")
        
        return True
    
    def create_test_cases(self) -> List[Dict]:
        """创建测试用例"""
        test_cases = [
            # 场景0000
            {'query': 'Find the building', 'scene': '2013_05_28_drive_0000_sync', 'ground_truth': (12.63, -13.07)},
            {'query': 'Find the traffic sign', 'scene': '2013_05_28_drive_0000_sync', 'ground_truth': (0.70, -12.00)},
            {'query': 'Find the vegetation near building', 'scene': '2013_05_28_drive_0000_sync', 'ground_truth': (5.0, -8.0)},
            
            # 场景0002
            {'query': 'Find the lamp post', 'scene': '2013_05_28_drive_0002_sync', 'ground_truth': (7.86, 3.16)},
            {'query': 'Find the road', 'scene': '2013_05_28_drive_0002_sync', 'ground_truth': (5.0, 0.0)},
            {'query': 'Find the pole', 'scene': '2013_05_28_drive_0002_sync', 'ground_truth': (-4.23, 1.45)},
            
            # 场景0003
            {'query': 'Find the tree', 'scene': '2013_05_28_drive_0003_sync', 'ground_truth': (-9.41, 14.33)},
            {'query': 'Find the building nearby', 'scene': '2013_05_28_drive_0003_sync', 'ground_truth': (0.0, 5.0)},
            {'query': 'Find the road intersection', 'scene': '2013_05_28_drive_0003_sync', 'ground_truth': (5.0, -5.0)},
            
            # 泛化查询
            {'query': 'Find a building', 'scene': None, 'ground_truth': (0.0, 0.0)},
            {'query': 'Find a traffic sign', 'scene': None, 'ground_truth': (0.0, 0.0)},
            {'query': 'Find vegetation', 'scene': None, 'ground_truth': (0.0, 0.0)},
            {'query': 'Find a pole', 'scene': None, 'ground_truth': (0.0, 0.0)},
            {'query': 'Find the car', 'scene': None, 'ground_truth': (0.0, 0.0)},
            
            # 带场景的查询
            {'query': 'Find building in 2013_05_28_drive_0000_sync', 'scene': '2013_05_28_drive_0000_sync', 'ground_truth': (12.63, -13.07)},
            {'query': 'Find tree in scene 2013_05_28_drive_0003_sync', 'scene': '2013_05_28_drive_0003_sync', 'ground_truth': (-9.41, 14.33)},
        ]
        
        import random
        random.shuffle(test_cases)
        self.test_cases = test_cases
        
        return test_cases
    
    def evaluate(self) -> Dict:
        """运行评估"""
        if not self.adapter:
            if not self.initialize_with_augmented_data():
                return {'error': '初始化失败'}
        
        self.create_test_cases()
        
        print(f"\n{'='*80}")
        print("Text2Loc Visionary - 增强数据评估")
        print(f"{'='*80}")
        print(f"测试用例数: {len(self.test_cases)}")
        print(f"Cells数量: {len(self.adapter.cells.get('default', []))}")
        
        results = []
        
        for i, test_case in enumerate(self.test_cases):
            query = test_case['query']
            gt = test_case['ground_truth']
            
            try:
                location_results = self.adapter.find_location(
                    query=query,
                    direction='none',
                    color='none',
                    obj='none',
                    top_k=3
                )
                
                if location_results and len(location_results) > 0:
                    best = location_results[0]
                    pred_x = best.get('x', 0)
                    pred_y = best.get('y', 0)
                    confidence = best.get('confidence', 0)
                    score = best.get('score', 0)
                    cell_id = best.get('cell_id', 'unknown')
                    matched_scene = best.get('scene', 'unknown')
                else:
                    pred_x, pred_y = 0, 0
                    confidence = 0
                    score = 0
                    cell_id = 'none'
                    matched_scene = 'unknown'
                
                if gt != (0.0, 0.0):
                    error = np.sqrt((pred_x - gt[0])**2 + (pred_y - gt[1])**2)
                else:
                    error = None
                
                results.append({
                    'query': query,
                    'ground_truth': gt,
                    'predicted': (round(pred_x, 2), round(pred_y, 2)),
                    'error_m': round(error, 2) if error else None,
                    'confidence': confidence,
                    'score': score,
                    'cell_id': cell_id,
                    'matched_scene': matched_scene
                })
                
                status = '✓' if error and error < 10 else ('✗' if error else '?')
                print(f"[{i+1}/{len(self.test_cases)}] {status} \"{query[:40]}\" -> ({pred_x:.1f}, {pred_y:.1f}) | 误差: {error:.1f}m" if error else f"[{i+1}/{len(self.test_cases)}] ? \"{query[:40]}\"")
                
            except Exception as e:
                print(f"[{i+1}/{len(self.test_cases)}] ✗ 错误: {e}")
                results.append({
                    'query': query,
                    'error': str(e)
                })
        
        return self._calculate_metrics(results)
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """计算评估指标"""
        successful = [r for r in results if r.get('error_m') is not None]
        total = len(results)
        
        if not successful:
            return {'total': total, 'success': 0, 'success_rate': 0.0}
        
        errors = [r['error_m'] for r in successful]
        n = len(errors)
        
        sorted_errors = sorted(errors)
        avg_error = sum(errors) / n
        median_error = sorted_errors[n // 2]
        min_error = sorted_errors[0]
        max_error = sorted_errors[-1]
        std_error = np.std(errors) if n > 1 else 0
        
        acc_1m = sum(1 for e in errors if e <= 1) / n * 100
        acc_3m = sum(1 for e in errors if e <= 3) / n * 100
        acc_5m = sum(1 for e in errors if e <= 5) / n * 100
        acc_10m = sum(1 for e in errors if e <= 10) / n * 100
        acc_20m = sum(1 for e in errors if e <= 20) / n * 100
        
        metrics = {
            'total': total,
            'success': n,
            'success_rate': n / total * 100,
            'avg_error': round(avg_error, 2),
            'median_error': round(median_error, 2),
            'min_error': round(min_error, 2),
            'max_error': round(max_error, 2),
            'std_error': round(std_error, 2),
            'acc_1m': round(acc_1m, 1),
            'acc_3m': round(acc_3m, 1),
            'acc_5m': round(acc_5m, 1),
            'acc_10m': round(acc_10m, 1),
            'acc_20m': round(acc_20m, 1),
            'detailed_results': sorted(successful, key=lambda x: x['error_m'] or 999)[:5]
        }
        
        return metrics
    
    def print_report(self, metrics: Dict):
        """打印评估报告"""
        print(f"\n{'='*80}")
        print("评估报告 - 增强数据")
        print(f"{'='*80}")
        
        print(f"\n【基础指标】")
        print(f"  总查询数: {metrics['total']}")
        print(f"  成功查询: {metrics['success']}")
        print(f"  成功率: {metrics['success_rate']:.1f}%")
        
        print(f"\n【距离误差统计】")
        print(f"  平均误差: {metrics['avg_error']:.2f}m")
        print(f"  中位数误差: {metrics['median_error']:.2f}m")
        print(f"  最小误差: {metrics['min_error']:.2f}m")
        print(f"  最大误差: {metrics['max_error']:.2f}m")
        print(f"  标准差: {metrics['std_error']:.2f}m")
        
        print(f"\n【不同阈值内的准确率】")
        print(f"  1米内准确率: {metrics['acc_1m']:.1f}%")
        print(f"  3米内准确率: {metrics['acc_3m']:.1f}%")
        print(f"  5米内准确率: {metrics['acc_5m']:.1f}%")
        print(f"  10米内准确率: {metrics['acc_10m']:.1f}%")
        print(f"  20米内准确率: {metrics['acc_20m']:.1f}%")
        
        print(f"\n【与Text2Loc-one基线对比】")
        print(f"  Text2Loc-one基线平均误差: 19.75m")
        improvement = (19.75 - metrics['avg_error']) / 19.75 * 100
        if improvement > 0:
            print(f"  Visionary误差: {metrics['avg_error']:.2f}m → 改善: +{improvement:.1f}%")
        else:
            print(f"  Visionary误差: {metrics['avg_error']:.2f}m → 变化: {improvement:.1f}%")
        
        print(f"  Text2Loc-one基线5m准确率: 13.3%")
        print(f"  Visionary 5m准确率: {metrics['acc_5m']:.1f}%")
        print(f"  5m准确率变化: {metrics['acc_5m'] - 13.3:+.1f}个百分点")
        
        if 'detailed_results' in metrics:
            print(f"\n【最佳定位结果】")
            for i, r in enumerate(metrics['detailed_results'][:3]):
                print(f"  {i+1}. 查询: \"{r['query'][:40]}\"")
                print(f"     误差: {r['error_m']:.2f}m, 置信度: {r['confidence']:.3f}")
        
        print(f"\n{'='*80}")
        print("核心指标总结")
        print(f"{'='*80}")
        print(f"  成功率: {metrics['success_rate']:.1f}%")
        print(f"  平均误差: {metrics['avg_error']:.2f}m")
        print(f"  5米内准确率: {metrics['acc_5m']:.1f}%")
        print(f"  10米内准确率: {metrics['acc_10m']:.1f}%")
        print(f"{'='*80}")
        
        return metrics


def main():
    """主函数"""
    print("="*80)
    print("Text2Loc Visionary - 增强数据评估")
    print("使用语义标签恢复和数据增强后的数据进行评估")
    print("="*80)
    
    evaluator = EnhancedDataEvaluator()
    metrics = evaluator.evaluate()
    
    if 'error' in metrics:
        print(f"❌ 评估失败: {metrics['error']}")
        return
    
    evaluator.print_report(metrics)
    
    return metrics


if __name__ == "__main__":
    main()
