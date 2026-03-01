#!/usr/bin/env python3
"""
深度优化版本：超越Text2Loc-one的性能

优化策略：
1. 语义标签修复 - 从RGB推断语义类别
2. 场景感知检索 - 优先匹配同场景的cell
3. 对象级精确定位 - 使用对象中心而非cell中心
4. 多尺度匹配 - 结合粗检索和细匹配
5. 自适应阈值 - 根据查询类型调整匹配策略
"""

import os
import sys
import json
import pickle
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KNOWN_CLASS = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'building', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic_sign',
    'traffic_light', ' Márquez', 'fire_hydrant', 'stop', 'parking',
    'belt', 'license_plate'
]

COLOR_NAMES = {
    (0.0, 0.0, 0.0): 'black',
    (0.0, 0.0, 1.0): 'blue',
    (0.5, 0.5, 0.5): 'gray',
    (0.0, 1.0, 0.0): 'green',
    (1.0, 0.0, 0.0): 'red',
    (1.0, 1.0, 1.0): 'white',
    (1.0, 1.0, 0.0): 'yellow',
    (1.0, 0.5, 0.0): 'orange',
    (0.5, 0.0, 0.0): 'maroon',
    (0.0, 0.5, 0.5): 'teal',
    (0.6, 0.4, 0.2): 'brown',
    (0.5, 0.0, 0.5): 'purple',
}

def rgb_to_color_name(rgb_tuple):
    """将RGB元组转换为颜色名称"""
    if isinstance(rgb_tuple, np.ndarray):
        rgb_tuple = tuple(rgb_tuple)
    
    for color_rgb, color_name in COLOR_NAMES.items():
        if np.allclose(rgb_tuple, color_rgb, atol=0.1):
            return color_name
    return 'gray'

def infer_semantic_label(obj: Dict) -> Tuple[str, float]:
    """根据物体特征推断语义标签"""
    color = obj.get('color', [0.5, 0.5, 0.5])
    if isinstance(color, np.ndarray):
        color_arr = color
    elif isinstance(color, list):
        color_arr = np.array(color)
    else:
        color_arr = np.array([0.5, 0.5, 0.5])
    
    color_name = rgb_to_color_name(color_arr)
    
    size = obj.get('size', [1.0, 1.0, 1.0])
    if isinstance(size, (list, tuple, np.ndarray)) and len(size) >= 3:
        height = float(size[2]) if size[2] > 0 else 1.0
        width = float(size[0]) if size[0] > 0 else 1.0
        length = float(size[1]) if size[1] > 0 else 1.0
    else:
        height, width, length = 1.0, 1.0, 1.0
    
    center = obj.get('center', [0, 0, 0])
    if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 3:
        z_pos = float(center[2])
    else:
        z_pos = 0.0
    
    scores = defaultdict(float)
    
    if color_name == 'green':
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
    elif color_name == 'white':
        scores['wall'] += 1.5
        scores['building'] += 1.0
    elif color_name == 'brown':
        scores['terrain'] += 2.0
        scores['building'] += 0.5
    
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
    elif height > 1:
        scores['traffic sign'] += 1.5
        scores['stop'] += 1.0
        scores['trash bin'] += 1.0
    else:
        scores['road'] += 1.0
        scores['sidewalk'] += 1.0
    
    if z_pos < 0.5:
        scores['road'] += 1.0
        scores['sidewalk'] += 1.0
    
    volume = width * length * height
    if volume > 100:
        scores['building'] += 1.5
        scores['vegetation'] += 1.0
    elif volume > 10:
        scores['car'] += 1.5
        scores['truck'] += 1.0
    elif volume > 1:
        scores['traffic sign'] += 1.0
        scores['pole'] += 0.5
    else:
        scores['traffic cone'] += 1.0
        scores['barrier'] += 0.5
    
    if scores:
        best_label = max(scores, key=scores.get)
        confidence = scores[best_label] / sum(scores.values()) if sum(scores.values()) > 0 else 0
    else:
        best_label = 'unknown'
        confidence = 0.0
    
    return best_label, confidence

class SemanticRepairer:
    """语义标签修复器"""
    
    def __init__(self, cells_path: str = None):
        self.cells_path = cells_path or '/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired/cells/cells.pkl'
        self.cells = []
        self.semantic_cells = []
        self.scene_mapping = defaultdict(list)
        
    def load_cells(self):
        """加载cells数据"""
        if not os.path.exists(self.cells_path):
            logger.error(f"Cells文件不存在: {self.cells_path}")
            return False
        
        with open(self.cells_path, 'rb') as f:
            self.cells = pickle.load(f)
        
        logger.info(f"加载了 {len(self.cells)} 个cells")
        return True
    
    def repair_all_cells(self):
        """修复所有cells的语义标签"""
        if not self.cells:
            if not self.load_cells():
                return False
        
        for cell in self.cells:
            if isinstance(cell, dict):
                cell_id = cell.get('id', 'unknown')
                scene_name = cell_id.split('_')[0] if '_' in cell_id else cell_id
                
                objects = cell.get('objects', [])
                repaired_objects = []
                
                for obj in objects:
                    if isinstance(obj, dict):
                        original_label = obj.get('label', 'unknown')
                        if original_label == 'unknown' or not original_label:
                            new_label, confidence = infer_semantic_label(obj)
                            obj['label'] = new_label
                            obj['label_confidence'] = confidence
                            obj['label_repaired'] = True
                        else:
                            obj['label_repaired'] = False
                            obj['label_confidence'] = 1.0
                        
                        repaired_objects.append(obj)
                    else:
                        repaired_objects.append(obj)
                
                cell['objects'] = repaired_objects
                cell['scene'] = scene_name
                self.semantic_cells.append(cell)
                self.scene_mapping[scene_name].append(cell)
        
        logger.info(f"修复了 {len(self.semantic_cells)} 个cells")
        logger.info(f"场景分布: {dict((k, len(v)) for k, v in self.scene_mapping.items())}")
        return True
    
    def save_repaired_cells(self, output_path: str = None):
        """保存修复后的cells"""
        if not self.semantic_cells:
            logger.error("没有可保存的修复数据")
            return False
        
        output_path = output_path or self.cells_path.replace('.pkl', '_semantic.pkl')
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.semantic_cells, f)
        
        logger.info(f"修复数据已保存至: {output_path}")
        return output_path

class SceneAwareRetriever:
    """场景感知检索器"""
    
    def __init__(self, cells: List[Dict]):
        self.all_cells = cells
        self.scene_cells = defaultdict(list)
        self.scene_index = defaultdict(dict)
        
        for cell in cells:
            if isinstance(cell, dict):
                scene = cell.get('scene', 'unknown')
                cell_id = cell.get('id', 'unknown')
                self.scene_cells[scene].append(cell)
                self.scene_index[scene][cell_id] = cell
    
    def get_scene_from_query(self, query: str) -> Optional[str]:
        """从查询中提取场景名称"""
        query_lower = query.lower()
        
        scenes = ['2013_05_28_drive_0000_sync', '2013_05_28_drive_0002_sync', 
                  '2013_05_28_drive_0003_sync', '2013_05_28_drive_0004_sync',
                  '2013_05_28_drive_0005_sync', '2013_05_28_drive_0006_sync',
                  '2013_05_28_drive_0007_sync', '2013_05_28_drive_0009_sync',
                  '2013_05_28_drive_0010_sync']
        
        for scene in scenes:
            if scene.lower() in query_lower:
                return scene
            scene_short = scene.split('_')[0] + '_' + scene.split('_')[2]
            if scene_short in query_lower:
                return scene
        
        return None
    
    def retrieve_with_scene_priority(self, query: str, nlu_result: Dict, top_k: int = 10) -> List[Dict]:
        """场景优先检索"""
        target_scene = self.get_scene_from_query(query)
        
        object_query = nlu_result.get('object', '')
        color_query = nlu_result.get('color', '')
        direction_query = nlu_result.get('direction', '')
        
        candidates = []
        
        if target_scene and target_scene in self.scene_cells:
            scene_candidates = self.scene_cells[target_scene]
            for cell in scene_candidates:
                score = self._calculate_match_score(cell, object_query, color_query, direction_query)
                candidates.append({
                    'cell': cell,
                    'score': score * 1.5,
                    'scene_match': True
                })
            
            other_scenes = [s for s in self.scene_cells.keys() if s != target_scene]
            for scene in other_scenes:
                for cell in self.scene_cells[scene]:
                    score = self._calculate_match_score(cell, object_query, color_query, direction_query)
                    candidates.append({
                        'cell': cell,
                        'score': score * 0.8,
                        'scene_match': False
                    })
        else:
            for scene in self.scene_cells:
                for cell in self.scene_cells[scene]:
                    score = self._calculate_match_score(cell, object_query, color_query, direction_query)
                    candidates.append({
                        'cell': cell,
                        'score': score,
                        'scene_match': False
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]
    
    def _calculate_match_score(self, cell: Dict, obj: str, color: str, direction: str) -> float:
        """计算匹配分数"""
        score = 0.0
        
        objects = cell.get('objects', [])
        if not objects:
            return 0.05
        
        obj_lower = obj.lower() if obj else ''
        color_lower = color.lower() if color else ''
        direction_lower = direction.lower() if direction else ''
        
        best_obj_match = 0.0
        for obj_item in objects:
            if isinstance(obj_item, dict):
                label = str(obj_item.get('label', '')).lower()
                class_name = str(obj_item.get('class_name', '')).lower()
                
                if obj_lower == label or obj_lower == class_name:
                    best_obj_match = max(best_obj_match, 0.5)
                elif obj_lower in label or label in obj_lower:
                    best_obj_match = max(best_obj_match, 0.4)
                elif any(word in label for word in obj_lower.split()):
                    best_obj_match = max(best_obj_match, 0.25)
        
        score += best_obj_match
        
        if color_lower:
            best_color_match = 0.0
            for obj_item in objects:
                if isinstance(obj_item, dict):
                    obj_color = obj_item.get('color', [])
                    obj_color_name = rgb_to_color_name(obj_color)
                    if color_lower == obj_color_name:
                        best_color_match = max(best_color_match, 0.35)
            score += best_color_match
        
        if direction_lower:
            cell_id = cell.get('id', '')
            if direction_lower in ['left', 'west']:
                if 'left' in cell_id.lower():
                    score += 0.15
            elif direction_lower in ['right', 'east']:
                if 'right' in cell_id.lower() or 'east' in cell_id.lower():
                    score += 0.15
        
        if score == 0:
            score = 0.10
        
        matched_categories = sum(1 for val in [obj, color, direction] if val and val != 'none')
        if matched_categories >= 3:
            score *= 1.2
        elif matched_categories == 2:
            score *= 1.1
        
        return min(score, 1.0)

class PrecisePositioning:
    """精确定位器"""
    
    @staticmethod
    def get_object_center(cell: Dict, target_label: str = None) -> Tuple[float, float]:
        """获取对象中心坐标"""
        objects = cell.get('objects', [])
        
        if target_label:
            target_lower = target_label.lower()
            for obj in objects:
                if isinstance(obj, dict):
                    label = str(obj.get('label', '')).lower()
                    if target_lower == label or target_lower in label:
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
        
        cell_id = cell.get('id', '')
        parts = cell_id.split('_')
        if len(parts) >= 4:
            try:
                idx = int(parts[-1])
                base_x = -30 + (idx % 10) * 6
                base_y = -30 + (idx // 10) * 6
                return round(base_x + 3, 2), round(base_y + 3, 2)
            except:
                pass
        
        return 0.0, 0.0
    
    @staticmethod
    def refine_position(pred_x: float, pred_y: float, cell: Dict, 
                        query: str, nlu_result: Dict) -> Tuple[float, float]:
        """精修位置"""
        if nlu_result is None:
            return pred_x, pred_y
        
        direction = str(nlu_result.get('direction', '')).lower() if nlu_result.get('direction') else ''
        obj = str(nlu_result.get('object', '')).lower() if nlu_result.get('object') else ''
        
        objects = cell.get('objects', [])
        
        target_obj = None
        for o in objects:
            if isinstance(o, dict):
                label = str(o.get('label', '')).lower()
                if obj in label or label in obj:
                    target_obj = o
                    break
        
        if target_obj:
            center = target_obj.get('center', [])
            if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                obj_x, obj_y = float(center[0]), float(center[1])
                if obj_x != 0 or obj_y != 0:
                    return round(obj_x, 2), round(obj_y, 2)
        
        if direction in ['left', 'west']:
            pred_x -= 3.0
        elif direction in ['right', 'east']:
            pred_x += 3.0
        elif direction in ['front', 'north']:
            pred_y += 3.0
        elif direction in ['back', 'south']:
            pred_y -= 3.0
        
        return round(pred_x, 2), round(pred_y, 2)


class OptimizedText2Loc:
    """优化后的Text2Loc系统"""
    
    def __init__(self):
        self.semantic_repairer = SemanticRepairer()
        self.retriever = None
        self.positioning = PrecisePositioning()
        self.nlu_engine = None
        
        self.load_and_repair_data()
    
    def load_and_repair_data(self):
        """加载并修复数据"""
        logger.info("加载并修复语义标签...")
        
        if self.semantic_repairer.repair_all_cells():
            self.retriever = SceneAwareRetriever(self.semantic_repairer.semantic_cells)
            logger.info(f"初始化场景感知检索器，包含 {len(self.semantic_repairer.semantic_cells)} 个cells")
    
    def parse_query(self, query: str) -> Dict:
        """解析查询"""
        result = {
            'scene': None,
            'object': None,
            'color': None,
            'direction': None,
            'relation': None,
            'distance': None
        }
        
        query_lower = query.lower()
        
        scenes = ['2013_05_28_drive_0000_sync', '2013_05_28_drive_0002_sync', 
                  '2013_05_28_drive_0003_sync', '2013_05_28_drive_0004_sync',
                  '2013_05_28_drive_0005_sync', '2013_05_28_drive_0006_sync',
                  '2013_05_28_drive_0007_sync', '2013_05_28_drive_0009_sync',
                  '2013_05_28_drive_0010_sync']
        
        for scene in scenes:
            if scene.lower() in query_lower:
                result['scene'] = scene
                break
        
        for obj in ['car', 'building', 'tree', 'traffic sign', 'traffic light', 
                    'pole', 'lamp', 'sign', 'road', 'sidewalk', 'vegetation']:
            if obj in query_lower:
                result['object'] = obj
                break
        
        for color in ['red', 'blue', 'green', 'yellow', 'white', 'black', 
                      'gray', 'orange', 'brown']:
            if color in query_lower:
                result['color'] = color
                break
        
        for direction in ['left', 'right', 'front', 'back', 'east', 'west', 
                          'north', 'south']:
            if direction in query_lower:
                result['direction'] = direction
                break
        
        return result
    
    def locate(self, query: str, top_k: int = 5) -> Dict:
        """定位查询"""
        if not self.retriever:
            return {'error': '系统未初始化'}
        
        nlu_result = self.parse_query(query)
        if nlu_result is None:
            nlu_result = {'scene': None, 'object': None, 'color': None, 'direction': None, 'relation': None, 'distance': None}
        
        candidates = self.retriever.retrieve_with_scene_priority(query, nlu_result, top_k)
        
        results = []
        for i, candidate in enumerate(candidates):
            cell = candidate['cell']
            score = candidate['score']
            
            pred_x, pred_y = self.positioning.get_object_center(
                cell, nlu_result.get('object', '')
            )
            
            pred_x, pred_y = self.positioning.refine_position(
                pred_x, pred_y, cell, query, nlu_result
            )
            
            results.append({
                'rank': i + 1,
                'cell_id': cell.get('id', 'unknown'),
                'scene': cell.get('scene', 'unknown'),
                'x': pred_x,
                'y': pred_y,
                'score': round(score, 4),
                'confidence': round(score * 0.9 + 0.1, 3)
            })
        
        return {
            'status': 'success',
            'query': query,
            'nlu_result': nlu_result,
            'results': results,
            'total_candidates': len(candidates)
        }


class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self):
        self.system = OptimizedText2Loc()
        self.test_cases = self._create_test_cases()
    
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
        
        random.shuffle(test_cases)
        logger.info(f"创建了 {len(test_cases)} 个测试用例")
        return test_cases
    
    def evaluate(self) -> Dict:
        """运行评估"""
        if not self.test_cases:
            return {'error': '没有测试用例'}
        
        results = []
        
        logger.info(f"运行 {len(self.test_cases)} 个测试...")
        
        for i, test_case in enumerate(self.test_cases):
            query = test_case['query']
            gt = test_case['ground_truth']
            
            response = self.system.locate(query, top_k=5)
            
            if response.get('status') == 'success' and response.get('results'):
                best = response['results'][0]
                pred_x, pred_y = best['x'], best['y']
                confidence = best['confidence']
                
                error = np.sqrt((pred_x - gt['x'])**2 + (pred_y - gt['y'])**2)
                
                results.append({
                    'query': query[:50],
                    'ground_truth': gt,
                    'predicted': {'x': pred_x, 'y': pred_y},
                    'error_m': round(error, 2),
                    'confidence': confidence,
                    'scene_match': response['results'][0].get('scene') == test_case['scene'],
                    'cell_id': response['results'][0].get('cell_id')
                })
            else:
                results.append({
                    'query': query[:50],
                    'ground_truth': gt,
                    'predicted': None,
                    'error_m': None,
                    'confidence': 0,
                    'scene_match': False,
                    'cell_id': None
                })
            
            if (i + 1) % 5 == 0:
                current_metrics = self._calc_metrics(results)
                logger.info(f"进度 {i+1}/{len(self.test_cases)} - "
                          f"平均误差: {current_metrics['avg_error']:.2f}m, "
                          f"5m准确率: {current_metrics['acc_5m']:.1f}%")
        
        return self._calc_metrics(results, detailed=True)
    
    def _calc_metrics(self, results: List[Dict], detailed: bool = False) -> Dict:
        """计算评估指标"""
        successful = [r for r in results if r.get('error_m') is not None]
        
        if not successful:
            return {
                'total': len(results),
                'success': 0,
                'success_rate': 0.0,
                'avg_error': float('inf'),
                'acc_1m': 0.0,
                'acc_3m': 0.0,
                'acc_5m': 0.0,
                'acc_10m': 0.0,
                'acc_20m': 0.0
            }
        
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
        print("深度优化版本评估报告")
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
        
        print(f"\n【与Text2Loc-one对比】")
        print(f"  优化前平均误差: 19.75m → 优化后: {metrics['avg_error']:.2f}m")
        improvement = (19.75 - metrics['avg_error']) / 19.75 * 100
        print(f"  误差改善: {improvement:.1f}%")
        print(f"  优化前5m准确率: 13.3% → 优化后: {metrics['acc_5m']:.1f}%")
        print(f"  5m准确率提升: +{metrics['acc_5m'] - 13.3:.1f}个百分点")
        
        if 'detailed_results' in metrics:
            print(f"\n【最佳定位结果】")
            for i, r in enumerate(metrics['detailed_results'][:3]):
                print(f"  {i+1}. 查询: \"{r['query']}\"")
                print(f"     误差: {r['error_m']:.2f}m, 置信度: {r['confidence']:.3f}")
        
        print("="*80)
    
    def save_report(self, metrics: Dict):
        """保存报告"""
        os.makedirs('/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/experiment_results', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_path = f'/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/experiment_results/deep_optimization_report_{timestamp}.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 深度优化版本评估报告\n\n")
            f.write(f"生成时间: {timestamp}\n\n")
            f.write("## 基础指标\n\n")
            f.write(f"- 总查询数: {metrics['total']}\n")
            f.write(f"- 成功查询: {metrics['success']}\n")
            f.write(f"- 成功率: {metrics['success_rate']:.1f}%\n\n")
            f.write("## 距离误差统计\n\n")
            f.write(f"- 平均误差: {metrics['avg_error']:.2f}m\n")
            f.write(f"- 中位数误差: {metrics['median_error']:.2f}m\n")
            f.write(f"- 最小误差: {metrics['min_error']:.2f}m\n")
            f.write(f"- 最大误差: {metrics['max_error']:.2f}m\n")
            f.write(f"- 标准差: {metrics['std_error']:.2f}m\n\n")
            f.write("## 不同阈值内的准确率\n\n")
            f.write(f"- 1米内准确率: {metrics['acc_1m']:.1f}%\n")
            f.write(f"- 3米内准确率: {metrics['acc_3m']:.1f}%\n")
            f.write(f"- 5米内准确率: {metrics['acc_5m']:.1f}%\n")
            f.write(f"- 10米内准确率: {metrics['acc_10m']:.1f}%\n")
            f.write(f"- 20米内准确率: {metrics['acc_20m']:.1f}%\n\n")
            f.write("## 与Text2Loc-one对比\n\n")
            f.write(f"- 优化前平均误差: 19.75m → 优化后: {metrics['avg_error']:.2f}m\n")
            improvement = (19.75 - metrics['avg_error']) / 19.75 * 100
            f.write(f"- 误差改善: {improvement:.1f}%\n")
            f.write(f"- 优化前5m准确率: 13.3% → 优化后: {metrics['acc_5m']:.1f}%\n")
            f.write(f"- 5m准确率提升: +{metrics['acc_5m'] - 13.3:.1f}个百分点\n")
        
        print(f"\n📄 报告已保存至: {report_path}")


def main():
    """主函数"""
    from datetime import datetime
    
    print("="*80)
    print("深度优化版本：超越Text2Loc-one")
    print("="*80)
    
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.evaluate()
    
    evaluator.print_report(metrics)
    evaluator.save_report(metrics)
    
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
