#!/usr/bin/env python3
"""
Text2Loc Visionary - 基于Poses的精确定位系统
直接使用poses数据构建定位索引，而不是cells
"""

import pickle
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PoseBasedLocator:
    """基于Pose的定位器"""
    
    SCENE_NAMES = [
        '2013_05_28_drive_0000_sync',
        '2013_05_28_drive_0002_sync',
        '2013_05_28_drive_0003_sync',
        '2013_05_28_drive_0004_sync',
        '2013_05_28_drive_0005_sync',
        '2013_05_28_drive_0006_sync',
        '2013_05_28_drive_0007_sync',
        '2013_05_28_drive_0009_sync',
        '2013_05_28_drive_0010_sync',
    ]
    
    OBJECT_PATTERNS = {
        'traffic sign': ['traffic sign', 'sign', 'road sign', 'stop sign', 'street sign'],
        'traffic light': ['traffic light', 'traffic signal', 'signal', 'light'],
        'building': ['building', 'buildings', 'house', 'structure', 'apartment'],
        'vegetation': ['tree', 'trees', 'vegetation', 'plant', 'grass', 'park'],
        'car': ['car', 'cars', 'vehicle', 'vehicles', 'automobile'],
        'pole': ['pole', 'poles', 'lamp post', 'street light', 'light pole'],
        'road': ['road', 'roads', 'street', 'drive', 'lane', 'intersection'],
        'sidewalk': ['sidewalk', 'pavement', 'footpath'],
        'parking': ['parking', 'parking lot', 'car park'],
        'wall': ['wall', 'fence'],
    }
    
    COLOR_PATTERNS = {
        'red': ['red', 'crimson', 'scarlet'],
        'blue': ['blue', 'navy', 'azure'],
        'green': ['green', 'emerald'],
        'yellow': ['yellow', 'gold', 'amber'],
        'white': ['white', 'snow', 'cream'],
        'black': ['black', 'dark'],
        'gray': ['gray', 'grey', 'silver'],
        'brown': ['brown', 'tan'],
    }
    
    DIRECTION_PATTERNS = {
        'left': ['left', 'west', 'port side'],
        'right': ['right', 'east', 'starboard'],
        'front': ['front', 'ahead', 'forward', 'north'],
        'back': ['back', 'behind', 'south'],
    }
    
    def __init__(self, poses_path: str = None, cells_path: str = None):
        """初始化定位器"""
        self.poses_path = poses_path or '/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired/poses/poses.pkl'
        self.cells_path = cells_path or '/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired/cells/cells.pkl'
        
        self.poses = []
        self.scene_poses = defaultdict(list)
        self.semantic_index = defaultdict(list)  # scene -> label -> poses
        
        self._load_data()
    
    def _load_data(self):
        """加载poses和cells数据"""
        if not os.path.exists(self.poses_path):
            logger.error(f"Poses文件不存在: {self.poses_path}")
            return
        
        with open(self.poses_path, 'rb') as f:
            self.poses = pickle.load(f)
        
        logger.info(f"加载了 {len(self.poses)} 个poses")
        
        for pose in self.poses:
            if isinstance(pose, dict):
                scene = pose.get('scene', 'unknown')
                pose_id = pose.get('id', 'unknown')
                location = pose.get('location', [])
                descriptions = pose.get('descriptions', [])
                
                self.scene_poses[scene].append(pose)
                
                desc_text = ''
                for desc in descriptions:
                    if isinstance(desc, dict):
                        desc_text = desc.get('text', '')
                    elif isinstance(desc, str):
                        desc_text = desc
                    break
                
                desc_lower = desc_text.lower()
                for label, patterns in self.OBJECT_PATTERNS.items():
                    for pattern in patterns:
                        if pattern in desc_lower:
                            self.semantic_index[scene][label].append(pose)
                            break
        
        logger.info(f"场景分布: {dict((k, len(v)) for k, v in self.scene_poses.items())}")
    
    def parse_query(self, query: str) -> Dict:
        """解析查询"""
        query_lower = query.lower()
        
        result = {
            'scene': None,
            'object': None,
            'color': None,
            'direction': None,
            'location_type': None,
        }
        
        for scene in self.SCENE_NAMES:
            if scene.lower() in query_lower:
                result['scene'] = scene
                break
        
        for obj, patterns in self.OBJECT_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    result['object'] = obj
                    break
            if result['object']:
                break
        
        for color, patterns in self.COLOR_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    result['color'] = color
                    break
            if result['color']:
                break
        
        for direction, patterns in self.DIRECTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    result['direction'] = direction
                    break
            if result['direction']:
                break
        
        return result
    
    def locate(self, query: str, top_k: int = 5) -> Dict:
        """执行定位"""
        nlu_result = self.parse_query(query)
        
        scene = nlu_result.get('scene')
        obj = nlu_result.get('object')
        color = nlu_result.get('color')
        direction = nlu_result.get('direction')
        
        candidates = []
        
        target_scenes = [scene] if scene else list(self.scene_poses.keys())
        
        for s in target_scenes:
            if s not in self.scene_poses:
                continue
            
            scene_pose_list = self.scene_poses[s]
            
            if obj and obj in self.semantic_index.get(s, {}):
                obj_poses = self.semantic_index[s][obj]
                for pose in obj_poses:
                    candidates.append({
                        'pose': pose,
                        'score': 0.85,
                        'match_type': 'semantic',
                        'scene_bonus': 1.5 if scene == s else 1.0
                    })
            
            for pose in scene_pose_list:
                pose_id = pose.get('id', 'unknown')
                if any(c['pose'].get('id') == pose_id for c in candidates if c.get('match_type') == 'semantic'):
                    continue
                
                score = 0.3
                descriptions = pose.get('descriptions', [])
                desc_text = ''
                for desc in descriptions:
                    if isinstance(desc, dict):
                        desc_text = desc.get('text', '')
                    elif isinstance(desc, str):
                        desc_text = desc
                    break
                
                desc_lower = desc_text.lower()
                
                if obj:
                    for label, patterns in self.OBJECT_PATTERNS.items():
                        for pattern in patterns:
                            if pattern in desc_lower:
                                score = max(score, 0.6)
                                break
                
                if color:
                    for color_pattern, patterns in self.COLOR_PATTERNS.items():
                        for pattern in patterns:
                            if pattern in desc_lower:
                                score = max(score, 0.5)
                                break
                
                if direction:
                    for dir_pattern, patterns in self.DIRECTION_PATTERNS.items():
                        for pattern in patterns:
                            if pattern in desc_lower:
                                score = max(score, 0.45)
                                break
                
                candidates.append({
                    'pose': pose,
                    'score': score,
                    'match_type': 'text',
                    'scene_bonus': 1.5 if scene == s else 1.0
                })
        
        for c in candidates:
            c['score'] = c['score'] * c.get('scene_bonus', 1.0)
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        results = []
        for i, c in enumerate(candidates[:top_k]):
            pose = c['pose']
            location = pose.get('location', [0, 0, 0])
            
            if isinstance(location, (list, tuple, np.ndarray)) and len(location) >= 2:
                x, y = float(location[0]), float(location[1])
            else:
                x, y = 0.0, 0.0
            
            results.append({
                'rank': i + 1,
                'pose_id': pose.get('id', 'unknown'),
                'scene': pose.get('scene', 'unknown'),
                'x': round(x, 2),
                'y': round(y, 2),
                'score': round(c['score'], 3),
                'confidence': round(c['score'] * 0.8 + 0.2, 3),
                'match_type': c.get('match_type', 'unknown')
            })
        
        return {
            'status': 'success',
            'query': query,
            'nlu_result': nlu_result,
            'results': results,
            'total_candidates': len(candidates)
        }


class PoseBasedEvaluator:
    """基于Poses的评估器"""
    
    def __init__(self):
        self.locator = PoseBasedLocator()
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[Dict]:
        """创建测试用例"""
        test_cases = []
        
        scenarios = [
            ('2013_05_28_drive_0000_sync', '城市住宅区', [
                ('Find the traffic sign', (5.43, -10.13)),
                ('Find the car', (32.33, 9.15)),
                ('Find the pole', (-27.42, 11.38)),
                ('Find the tree near intersection', (5.43, -10.13)),
                ('I am looking for a red car', (32.33, 9.15)),
            ]),
            ('2013_05_28_drive_0002_sync', '商业区街道', [
                ('Find the shop', (30.72, 4.32)),
                ('Find the lamp near the street', (24.24, 8.94)),
                ('Find the sign', (-26.61, -26.21)),
                ('Find the building near the road', (24.24, 8.94)),
            ]),
            ('2013_05_28_drive_0003_sync', '郊区道路', [
                ('Find the tree on the left', (-5.65, 35.73)),
                ('Find the house near the road', (10.82, 4.34)),
                ('Find the sign at the junction', (-5.65, 35.73)),
            ]),
            ('2013_05_28_drive_0004_sync', '住宅区道路', [
                ('Find the car on the left side', (17.86, 21.11)),
                ('Find the building near the street', (17.86, 21.11)),
            ]),
            ('2013_05_28_drive_0005_sync', '商业区', [
                ('Find the car in this area', (9.17, -0.25)),
                ('Find the building nearby', (9.17, -0.25)),
                ('Find the tree near here', (9.17, -0.25)),
            ]),
        ]
        
        for scene, desc, queries in scenarios:
            for query, (gt_x, gt_y) in queries:
                test_cases.append({
                    'query': query,
                    'ground_truth': {'x': gt_x, 'y': gt_y},
                    'scene': scene
                })
        
        import random
        random.shuffle(test_cases)
        return test_cases
    
    def evaluate(self) -> Dict:
        """运行评估"""
        results = []
        
        logger.info(f"使用Pose-Based定位器进行评估，共 {len(self.test_cases)} 个测试用例...")
        
        for i, test_case in enumerate(self.test_cases):
            query = test_case['query']
            gt = test_case['ground_truth']
            
            response = self.locator.locate(query, top_k=5)
            api_results = response.get('results', [])
            
            if api_results:
                best = api_results[0]
                pred_x, pred_y = best.get('x', 0), best.get('y', 0)
                confidence = best.get('confidence', 0)
                score = best.get('score', 0)
                match_type = best.get('match_type', 'unknown')
            else:
                pred_x, pred_y = 0, 0
                confidence = 0
                score = 0
                match_type = 'none'
            
            error = np.sqrt((pred_x - gt['x'])**2 + (pred_y - gt['y'])**2)
            
            results.append({
                'query': query[:50],
                'ground_truth': gt,
                'predicted': {'x': round(pred_x, 2), 'y': round(pred_y, 2)},
                'error_m': round(error, 2),
                'confidence': confidence,
                'score': score,
                'match_type': match_type,
                'scene_match': best.get('scene') == test_case['scene'] if api_results else False
            })
            
            if (i + 1) % 5 == 0:
                current_metrics = self._calc_metrics(results)
                logger.info(f"进度 {i+1}/{len(self.test_cases)} - "
                          f"平均误差: {current_metrics['avg_error']:.2f}m, "
                          f"5m准确率: {current_metrics['acc_5m']:.1f}%")
        
        return self._calc_metrics(results, detailed=True)
    
    def _calc_metrics(self, results: List[Dict], detailed: bool = False) -> Dict:
        """计算指标"""
        successful = [r for r in results if r.get('error_m') is not None]
        
        if not successful:
            return {'total': len(results), 'success': 0, 'success_rate': 0.0}
        
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
        
        scene_matches = sum(1 for r in successful if r.get('scene_match')) / n * 100
        
        semantic_matches = sum(1 for r in successful if r.get('match_type') == 'semantic') / n * 100
        
        metrics = {
            'total': len(results),
            'success': len(successful),
            'success_rate': len(successful) / len(results) * 100,
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
            'scene_match_rate': round(scene_matches, 1),
            'semantic_match_rate': round(semantic_matches, 1)
        }
        
        if detailed:
            metrics['detailed_results'] = sorted(
                successful, key=lambda x: x['error_m'] or float('inf')
            )[:5]
        
        return metrics
    
    def print_report(self, metrics: Dict):
        """打印报告"""
        print("\n" + "="*80)
        print("Text2Loc Visionary - Pose-Based 精确定位评估")
        print("="*80)
        
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
        
        print(f"\n【匹配质量】")
        print(f"  场景正确匹配: {metrics['scene_match_rate']:.1f}%")
        print(f"  语义匹配率: {metrics['semantic_match_rate']:.1f}%")
        
        print(f"\n【与Text2Loc-one基线对比】")
        print(f"  Text2Loc-one基线平均误差: 19.75m → Visionary优化后: {metrics['avg_error']:.2f}m")
        improvement = (19.75 - metrics['avg_error']) / 19.75 * 100
        if improvement > 0:
            print(f"  误差改善: +{improvement:.1f}%")
        else:
            print(f"  误差变化: {improvement:.1f}% (负值表示误差增加)")
        print(f"  Text2Loc-one基线5m准确率: 13.3% → Visionary优化后: {metrics['acc_5m']:.1f}%")
        print(f"  5m准确率变化: {metrics['acc_5m'] - 13.3:+.1f}个百分点")
        
        if 'detailed_results' in metrics:
            print(f"\n【最佳定位结果】")
            for i, r in enumerate(metrics['detailed_results'][:3]):
                print(f"  {i+1}. 查询: \"{r['query']}\"")
                print(f"     误差: {r['error_m']:.2f}m, 置信度: {r['confidence']:.3f}, 匹配类型: {r['match_type']}")
        
        print("="*80)
        
        return metrics


def main():
    """主函数"""
    print("="*80)
    print("Text2Loc Visionary - 基于Poses的精确定位系统")
    print("="*80)
    
    evaluator = PoseBasedEvaluator()
    metrics = evaluator.evaluate()
    
    evaluator.print_report(metrics)
    
    print("\n" + "="*80)
    print("核心指标总结")
    print("="*80)
    print(f"  成功率: {metrics['success_rate']:.1f}%")
    print(f"  平均误差: {metrics['avg_error']:.2f}m")
    print(f"  5米内准确率: {metrics['acc_5m']:.1f}%")
    print(f"  10米内准确率: {metrics['acc_10m']:.1f}%")
    print(f"  场景匹配率: {metrics['scene_match_rate']:.1f}%")
    print("="*80)
    
    return metrics


if __name__ == "__main__":
    main()
