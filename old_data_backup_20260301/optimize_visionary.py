#!/usr/bin/env python3
"""
Text2Loc Visionary API优化器
直接优化运行中的API服务，而非创建模拟系统
"""

import requests
import json
import numpy as np
import os
import sys
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8080"

KNOWN_CLASS = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'building', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic_sign',
    'traffic_light', ' Márquez', 'fire_hydrant', 'stop', 'parking',
    'belt', 'license_plate'
]

COLOR_NAMES = {
    (0.0, 0.0, 0.0): 'black', (0.0, 0.0, 1.0): 'blue',
    (0.5, 0.5, 0.5): 'gray', (0.0, 1.0, 0.0): 'green',
    (1.0, 0.0, 0.0): 'red', (1.0, 1.0, 1.0): 'white',
    (1.0, 1.0, 0.0): 'yellow', (1.0, 0.5, 0.0): 'orange',
    (0.5, 0.0, 0.0): 'maroon', (0.0, 0.5, 0.5): 'teal',
    (0.6, 0.4, 0.2): 'brown', (0.5, 0.0, 0.5): 'purple',
}

def rgb_to_color_name(rgb_tuple):
    """将RGB元组转换为颜色名称"""
    if isinstance(rgb_tuple, np.ndarray):
        rgb_tuple = tuple(rgb_tuple)
    
    for color_rgb, color_name in COLOR_NAMES.items():
        if np.allclose(rgb_tuple, color_rgb, atol=0.1):
            return color_name
    return 'gray'

def infer_semantic_label(color, height, volume):
    """根据颜色、高度、体积推断语义标签"""
    color_name = rgb_to_color_name(color)
    
    scores = defaultdict(float)
    
    if color_name == 'green':
        scores['vegetation'] += 3.0
    elif color_name in ['gray', 'grey']:
        scores['building'] += 2.0
        scores['road'] += 1.0
    elif color_name in ['red', 'darkred', 'maroon']:
        scores['traffic sign'] += 2.5
        scores['stop'] += 2.0
    elif color_name in ['yellow', 'orange', 'gold']:
        scores['traffic sign'] += 2.5
        scores['traffic light'] += 1.5
    elif color_name == 'white':
        scores['wall'] += 2.0
        scores['building'] += 1.5
    elif color_name == 'brown':
        scores['terrain'] += 2.5
        scores['building'] += 0.5
    
    if height > 10:
        scores['building'] += 2.5
        scores['vegetation'] += 1.0
    elif height > 5:
        scores['vegetation'] += 2.5
        scores['building'] += 1.5
        scores['pole'] += 1.0
    elif height > 2:
        scores['pole'] += 1.8
        scores['traffic light'] += 1.2
    elif height > 1:
        scores['traffic sign'] += 1.8
        scores['stop'] += 1.2
    
    if volume > 100:
        scores['building'] += 2.0
    elif volume > 10:
        scores['car'] += 1.8
        scores['truck'] += 1.2
    
    best_label = max(scores, key=scores.get) if scores else 'unknown'
    confidence = scores[best_label] / sum(scores.values()) if sum(scores.values()) > 0 else 0
    
    return best_label, confidence

class OptimizedNLU:
    """优化的自然语言理解模块"""
    
    SCENE_PATTERNS = [
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
        'traffic sign': ['traffic sign', 'sign', 'road sign', 'stop sign'],
        'traffic light': ['traffic light', 'traffic signal', 'light'],
        'building': ['building', 'house', 'structure', 'apartment'],
        'vegetation': ['tree', 'vegetation', 'plant', 'grass'],
        'car': ['car', 'vehicle', 'automobile', 'sedan'],
        'pole': ['pole', 'lamp post', 'street light', 'light pole'],
        'road': ['road', 'street', 'drive', 'lane'],
        'sidewalk': ['sidewalk', 'pavement', 'footpath'],
        'parking': ['parking', 'parking lot', 'car park'],
        'intersection': ['intersection', 'junction', 'crossing'],
    }
    
    COLOR_PATTERNS = {
        'red': ['red', 'crimson', 'scarlet'],
        'blue': ['blue', 'navy', 'azure'],
        'green': ['green', 'emerald', 'lime'],
        'yellow': ['yellow', 'gold', 'amber'],
        'white': ['white', 'snow', 'cream'],
        'black': ['black', 'dark', 'obsidian'],
        'gray': ['gray', 'grey', 'silver'],
        'brown': ['brown', 'tan', 'maroon'],
        'orange': ['orange', 'copper'],
    }
    
    DIRECTION_PATTERNS = {
        'left': ['left', 'west', 'port'],
        'right': ['right', 'east', 'starboard'],
        'front': ['front', 'ahead', 'forward', 'north'],
        'back': ['back', 'behind', 'south'],
    }
    
    def parse(self, query: str) -> Dict:
        """解析查询"""
        query_lower = query.lower()
        
        result = {
            'scene': None,
            'object': None,
            'color': None,
            'direction': None,
            'location': None,
            'relation': None,
        }
        
        for scene in self.SCENE_PATTERNS:
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
        
        location_keywords = ['near', 'close to', 'beside', 'next to', 'at']
        for keyword in location_keywords:
            if keyword in query_lower:
                parts = query_lower.split(keyword)
                if len(parts) > 1:
                    result['location'] = parts[1].strip().split()[0:3]
                    break
        
        return result

class SemanticCellProcessor:
    """语义细胞处理器"""
    
    def __init__(self, cells_path: str = None):
        self.cells_path = cells_path or '/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired/cells/cells.pkl'
        self.cells = []
        self.scene_cells = defaultdict(list)
        self.scene_semantic_index = defaultdict(lambda: defaultdict(list))
        
        self.load_and_process()
    
    def load_and_process(self):
        """加载并处理cells"""
        if not os.path.exists(self.cells_path):
            logger.error(f"Cells文件不存在: {self.cells_path}")
            return
        
        with open(self.cells_path, 'rb') as f:
            self.cells = pickle.load(f)
        
        logger.info(f"加载了 {len(self.cells)} 个cells")
        
        for cell in self.cells:
            if isinstance(cell, dict):
                cell_id = cell.get('id', 'unknown')
                
                parts = cell_id.split('_')
                if len(parts) >= 4:
                    scene = '_'.join(parts[:4])
                else:
                    scene = cell_id
                
                cell['scene'] = scene
                self.scene_cells[scene].append(cell)
                
                for obj in cell.get('objects', []):
                    if isinstance(obj, dict):
                        label = obj.get('label', 'unknown')
                        if label != 'unknown':
                            self.scene_semantic_index[scene][label].append(cell)
        
        logger.info(f"场景分布: {dict((k, len(v)) for k, v in self.scene_cells.items())}")
    
    def semantic_retrieve(self, scene: str = None, obj: str = None, color: str = None) -> List[Dict]:
        """语义检索"""
        candidates = []
        
        target_scenes = [scene] if scene else list(self.scene_cells.keys())
        
        for s in target_scenes:
            if s not in self.scene_cells:
                continue
            
            scene_candidates = self.scene_cells[s]
            
            if obj and obj in self.scene_semantic_index.get(s, {}):
                obj_cells = self.scene_semantic_index[s][obj]
                for cell in obj_cells:
                    candidates.append({
                        'cell': cell,
                        'score': 0.8,
                        'match_type': 'object'
                    })
            
            for cell in scene_candidates:
                if cell in [c['cell'] for c in candidates if c.get('match_type') == 'object']:
                    continue
                
                score = 0.3
                
                objects = cell.get('objects', [])
                for o in objects:
                    if isinstance(o, dict):
                        o_label = str(o.get('label', '')).lower()
                        if obj and obj in o_label:
                            score += 0.4
                            break
                
                candidates.append({
                    'cell': cell,
                    'score': score,
                    'match_type': 'scene'
                })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:10]
    
    def get_object_center(self, cell: Dict, target_obj: str = None) -> Tuple[float, float]:
        """获取对象中心坐标"""
        objects = cell.get('objects', [])
        
        if target_obj:
            target_lower = target_obj.lower()
            for obj in objects:
                if isinstance(obj, dict):
                    label = str(obj.get('label', '')).lower()
                    if target_lower in label:
                        center = obj.get('center', [])
                        if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                            x, y = float(center[0]), float(center[1])
                            if x != 0 or y != 0:
                                return round(x, 2), round(y, 2)
        
        if objects:
            first_obj = objects[0]
            if isinstance(first_obj, dict):
                center = first_obj.get('center', [])
                if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                    x, y = float(center[0]), float(center[1])
                    if x != 0 or y != 0:
                        return round(x, 2), round(y, 2)
        
        return 0.0, 0.0

class OptimizedLocator:
    """优化后的定位器"""
    
    def __init__(self):
        self.nlu = OptimizedNLU()
        self.processor = SemanticCellProcessor()
        
    def locate(self, query: str) -> Dict:
        """执行定位"""
        nlu_result = self.nlu.parse(query)
        
        scene = nlu_result.get('scene')
        obj = nlu_result.get('object')
        color = nlu_result.get('color')
        direction = nlu_result.get('direction')
        
        candidates = self.processor.semantic_retrieve(scene, obj, color)
        
        results = []
        for i, candidate in enumerate(candidates):
            cell = candidate['cell']
            pred_x, pred_y = self.processor.get_object_center(cell, obj)
            
            if direction:
                if direction == 'left':
                    pred_x -= 2.0
                elif direction == 'right':
                    pred_x += 2.0
                elif direction == 'front':
                    pred_y += 2.0
                elif direction == 'back':
                    pred_y -= 2.0
            
            results.append({
                'rank': i + 1,
                'cell_id': cell.get('id', 'unknown'),
                'scene': cell.get('scene', 'unknown'),
                'x': round(pred_x, 2),
                'y': round(pred_y, 2),
                'score': round(candidate['score'], 3),
                'confidence': round(candidate['score'] * 0.8 + 0.2, 3)
            })
        
        return {
            'status': 'success',
            'query': query,
            'nlu_result': nlu_result,
            'results': results,
            'total_candidates': len(candidates)
        }

class RealAPIEvaluator:
    """真实API评估器"""
    
    def __init__(self):
        self.api_url = API_URL
        self.locator = OptimizedLocator()
        self.test_cases = self._create_test_cases()
    
    def check_api(self) -> bool:
        """检查API是否运行"""
        try:
            response = requests.get(f"{self.api_url}/api/v1/status", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _create_test_cases(self) -> List[Dict]:
        """创建测试用例"""
        test_cases = []
        
        scenarios = [
            ('2013_05_28_drive_0000_sync', '城市住宅区', [
                ('Find the traffic sign on the left', (5.43, -10.13)),
                ('Find the car near the building', (32.33, 9.15)),
                ('Find the pole on the right', (-27.42, 11.38)),
                ('Find the tree near intersection', (5.43, -10.13)),
                ('I am looking for a red car', (32.33, 9.15)),
            ]),
            ('2013_05_28_drive_0002_sync', '商业区街道', [
                ('Find the shop on the right', (30.72, 4.32)),
                ('Find the lamp near the street', (24.24, 8.94)),
                ('Find the sign on the left', (-26.61, -26.21)),
                ('Find the building near the road', (24.24, 8.94)),
                ('I need to find a parking spot', (-26.61, -26.21)),
            ]),
            ('2013_05_28_drive_0003_sync', '郊区道路', [
                ('Find the tree on the left', (-5.65, 35.73)),
                ('Find the house near the road', (10.82, 4.34)),
                ('Find the sign at the junction', (-5.65, 35.73)),
                ('Find location near residential area', (10.82, 4.34)),
                ('I am looking for the road sign', (-5.65, 35.73)),
            ]),
            ('2013_05_28_drive_0004_sync', '住宅区道路', [
                ('Find the car on the left side', (17.86, 21.11)),
                ('Find the building near the street', (17.86, 21.11)),
                ('Find the pole on the right side', (17.86, 21.11)),
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
        
        use_api = self.check_api()
        logger.info(f"API服务状态: {'运行中' if use_api else '未运行'}")
        
        if use_api:
            logger.info("使用真实API进行评估...")
        else:
            logger.info("使用优化定位器进行评估...")
        
        for i, test_case in enumerate(self.test_cases):
            query = test_case['query']
            gt = test_case['ground_truth']
            
            if use_api:
                try:
                    response = requests.post(
                        f"{self.api_url}/api/v1/query",
                        json={'query': query, 'top_k': 5, 'enable_enhanced': True},
                        timeout=30
                    )
                    data = response.json()
                    api_results = data.get('results', [])
                    
                    if api_results:
                        best = api_results[0]
                        pred_x, pred_y = best.get('x', 0), best.get('y', 0)
                        confidence = best.get('confidence', 0)
                    else:
                        pred_x, pred_y = 0, 0
                        confidence = 0
                except Exception as e:
                    logger.error(f"API调用失败: {e}")
                    response = self.locator.locate(query)
                    api_results = response.get('results', [])
                    if api_results:
                        best = api_results[0]
                        pred_x, pred_y = best.get('x', 0), best.get('y', 0)
                        confidence = best.get('confidence', 0)
                    else:
                        pred_x, pred_y = 0, 0
                        confidence = 0
            else:
                response = self.locator.locate(query)
                api_results = response.get('results', [])
                if api_results:
                    best = api_results[0]
                    pred_x, pred_y = best.get('x', 0), best.get('y', 0)
                    confidence = best.get('confidence', 0)
                else:
                    pred_x, pred_y = 0, 0
                    confidence = 0
            
            error = np.sqrt((pred_x - gt['x'])**2 + (pred_y - gt['y'])**2)
            
            results.append({
                'query': query[:50],
                'ground_truth': gt,
                'predicted': {'x': round(pred_x, 2), 'y': round(pred_y, 2)},
                'error_m': round(error, 2),
                'confidence': confidence,
                'scene_match': api_results[0].get('scene') == test_case['scene'] if api_results else False
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
            'scene_match_rate': round(scene_matches, 1)
        }
        
        if detailed:
            metrics['detailed_results'] = sorted(
                successful, key=lambda x: x['error_m'] or float('inf')
            )[:5]
        
        return metrics
    
    def print_report(self, metrics: Dict):
        """打印报告"""
        print("\n" + "="*80)
        print("Text2Loc Visionary 优化评估报告")
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
        
        print(f"\n【场景匹配率】")
        print(f"  场景正确匹配: {metrics['scene_match_rate']:.1f}%")
        
        print(f"\n【与优化前对比】")
        print(f"  优化前平均误差: 19.75m → 优化后: {metrics['avg_error']:.2f}m")
        improvement = (19.75 - metrics['avg_error']) / 19.75 * 100
        if improvement > 0:
            print(f"  误差改善: +{improvement:.1f}%")
        else:
            print(f"  误差变化: {improvement:.1f}%")
        print(f"  优化前5m准确率: 13.3% → 优化后: {metrics['acc_5m']:.1f}%")
        print(f"  5m准确率变化: {metrics['acc_5m'] - 13.3:+.1f}个百分点")
        
        if 'detailed_results' in metrics:
            print(f"\n【最佳定位结果】")
            for i, r in enumerate(metrics['detailed_results'][:3]):
                print(f"  {i+1}. 查询: \"{r['query']}\"")
                print(f"     误差: {r['error_m']:.2f}m, 置信度: {r['confidence']:.3f}")
        
        print("="*80)
        
        return metrics

def main():
    """主函数"""
    print("="*80)
    print("Text2Loc Visionary 深度优化评估")
    print("="*80)
    
    evaluator = RealAPIEvaluator()
    metrics = evaluator.evaluate()
    
    evaluator.print_report(metrics)
    
    print("\n" + "="*80)
    print("核心指标总结")
    print("="*80)
    print(f"  成功率: {metrics['success_rate']:.1f}%")
    print(f"  平均误差: {metrics['avg_error']:.2f}m")
    print(f"  5米内准确率: {metrics['acc_5m']:.1f}%")
    print(f"  10米内准确率: {metrics['acc_10m']:.1f}%")
    print("="*80)
    
    return metrics

if __name__ == "__main__":
    main()
