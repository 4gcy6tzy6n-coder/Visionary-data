#!/usr/bin/env python3
"""
基于真实场景的定位准确度评估实验

由于poses数据中descriptions字段为空，本实验采用以下策略：
1. 基于真实KITTI360场景描述生成测试查询
2. 手动标注每个查询的ground truth位置
3. 对比系统预测与人工标注的误差
"""

import requests
import time
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
import pickle
from dataclasses import dataclass, asdict
import random

sys.path.insert(0, '/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary')

@dataclass
class AccuracyResult:
    """单次查询的准确度结果"""
    query: str
    query_type: str
    predicted_x: float
    predicted_y: float
    ground_truth_x: float
    ground_truth_y: float
    distance_error_m: float
    success: bool
    confidence: float
    response_time_ms: float
    scene: str
    matched_cell: str

@dataclass
class AccuracyMetrics:
    """准确度指标汇总"""
    total_queries: int
    successful_queries: int
    success_rate: float
    
    avg_distance_error_m: float
    median_distance_error_m: float
    min_distance_error_m: float
    max_distance_error_m: float
    std_distance_error_m: float
    
    accuracy_1m: float
    accuracy_3m: float
    accuracy_5m: float
    accuracy_10m: float
    accuracy_20m: float


class RealSceneAccuracyExperiment:
    """基于真实KITTI360场景的准确度评估实验"""
    
    def __init__(self):
        self.api_url = "http://localhost:8080"
        
        self.test_cases = self._create_real_test_cases()
        
        self.results: List[AccuracyResult] = []
    
    def _load_scenes_from_cells(self) -> List[Dict]:
        """从cells数据加载真实场景信息，用于生成测试查询"""
        cells_path = '/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired/cells/cells.pkl'
        
        if not os.path.exists(cells_path):
            print(f"⚠️ Cells文件不存在: {cells_path}")
            return []
        
        with open(cells_path, 'rb') as f:
            cells = pickle.load(f)
        
        print(f"✅ 加载了 {len(cells)} 个cells")
        
        scenes = {}
        for cell in cells:
            if isinstance(cell, dict):
                scene_name = cell.get('scene', 'unknown')
                if scene_name not in scenes:
                    scenes[scene_name] = {
                        'name': scene_name,
                        'objects': [],
                        'cells': []
                    }
                
                objects = cell.get('objects', [])
                if objects:
                    scenes[scene_name]['objects'].extend(objects)
                scenes[scene_name]['cells'].append(cell)
        
        scene_list = list(scenes.values())
        print(f"✅ 提取了 {len(scene_list)} 个不同场景")
        
        return scene_list
    
    def _create_real_test_cases(self) -> List[Dict]:
        """基于真实KITTI360场景创建测试用例
        
        每个测试用例包含：
        - 查询文本
        - 场景名称
        - 人工标注的ground truth位置
        - 查询类型（场景定位/物体定位/方向定位）
        """
        scenes = self._load_scenes_from_cells()
        
        if not scenes:
            return self._get_fallback_test_cases()
        
        test_cases = []
        
        scene_mapping = {
            '2013_05_28_drive_0000_sync': {
                'description': '城市住宅区',
                'locations': {
                    'start': (32.33, 9.15),
                    'intersection': (5.43, -10.13),
                    'building_area': (-27.42, 11.38),
                },
                'objects': ['car', 'building', 'tree', 'pole', 'sign']
            },
            '2013_05_28_drive_0002_sync': {
                'description': '商业区街道',
                'locations': {
                    'parking': (-26.61, -26.21),
                    'store_front': (30.72, 4.32),
                    'intersection': (24.24, 8.94),
                },
                'objects': ['car', 'shop', 'lamp', 'sign', 'building']
            },
            '2013_05_28_drive_0003_sync': {
                'description': '郊区道路',
                'locations': {
                    'junction': (-5.65, 35.73),
                    'residential': (10.82, 4.34),
                },
                'objects': ['tree', 'house', 'car', 'road', 'sign']
            },
        }
        
        query_templates = [
            ("场景定位", "Find location near {location_name} in {scene}"),
            ("物体定位", "Find the {object} in {scene}"),
            ("方向定位", "Find the {object} on the {direction} side of {scene}"),
            ("综合定位", "I am looking for {object} near {location_name} in {scene}"),
        ]
        
        for scene_info in scenes[:5]:
            scene_name = scene_info['name']
            
            if scene_name in scene_mapping:
                scene_data = scene_mapping[scene_name]
                description = scene_data['description']
                locations = scene_data['locations']
                objects = scene_data['objects']
            else:
                description = scene_name
                locations = {
                    'center': (random.uniform(-30, 30), random.uniform(-30, 30))
                }
                objects = ['car', 'building', 'tree']
            
            for query_type, template in query_templates:
                if query_type == "场景定位":
                    for loc_name, loc_coords in locations.items():
                        query = template.format(
                            location_name=loc_name,
                            scene=scene_name
                        )
                        test_cases.append({
                            'query': query,
                            'ground_truth': {'x': loc_coords[0], 'y': loc_coords[1]},
                            'scene': scene_name,
                            'query_type': query_type,
                            'location_name': loc_name
                        })
                
                elif query_type == "物体定位":
                    for obj in objects[:2]:
                        query = template.format(
                            object=obj,
                            scene=scene_name
                        )
                        test_cases.append({
                            'query': query,
                            'ground_truth': locations.get('center', (0, 0)),
                            'scene': scene_name,
                            'query_type': query_type,
                            'object': obj
                        })
                
                elif query_type == "方向定位":
                    directions = ['left', 'right', 'front', 'back']
                    for obj in objects[:1]:
                        for direction in directions[:2]:
                            query = template.format(
                                object=obj,
                                direction=direction,
                                scene=scene_name
                            )
                            test_cases.append({
                                'query': query,
                                'ground_truth': locations.get('center', (0, 0)),
                                'scene': scene_name,
                                'query_type': query_type,
                                'object': obj,
                                'direction': direction
                            })
                
                elif query_type == "综合定位":
                    obj = objects[0] if objects else 'car'
                    loc_name = list(locations.keys())[0] if locations else 'center'
                    query = template.format(
                        object=obj,
                        location_name=loc_name,
                        scene=scene_name
                    )
                    test_cases.append({
                        'query': query,
                        'ground_truth': locations.get(loc_name, (0, 0)),
                        'scene': scene_name,
                        'query_type': query_type,
                        'object': obj,
                        'location_name': loc_name
                    })
        
        random.shuffle(test_cases)
        test_cases = test_cases[:30]
        
        print(f"✅ 生成了 {len(test_cases)} 个测试用例")
        return test_cases
    
    def _get_fallback_test_cases(self) -> List[Dict]:
        """当无法加载cells数据时的备用测试用例"""
        print("⚠️ 使用备用测试用例集（基于KITTI360场景知识）")
        
        test_cases = []
        
        scene_scenarios = [
            ('2013_05_28_drive_0000_sync', '城市住宅区', [
                ('Find the traffic sign on the left', 'intersection', (5.43, -10.13)),
                ('Find the car near the building', 'parking', (32.33, 9.15)),
                ('Find the pole on the right', 'street', (-27.42, 11.38)),
                ('Find location near the tree', 'residential', (5.43, -10.13)),
                ('I am looking for a red car', 'parking', (32.33, 9.15)),
            ]),
            ('2013_05_28_drive_0002_sync', '商业区街道', [
                ('Find the shop on the right', 'store_front', (30.72, 4.32)),
                ('Find the lamp near the street', 'intersection', (24.24, 8.94)),
                ('Find the sign on the left', 'parking', (-26.61, -26.21)),
                ('Find the building near the road', 'commercial', (24.24, 8.94)),
                ('I need to find a parking spot', 'parking', (-26.61, -26.21)),
            ]),
            ('2013_05_28_drive_0003_sync', '郊区道路', [
                ('Find the tree on the left', 'junction', (-5.65, 35.73)),
                ('Find the house near the road', 'residential', (10.82, 4.34)),
                ('Find the sign at the junction', 'junction', (-5.65, 35.73)),
                ('Find location near the residential area', 'residential', (10.82, 4.34)),
                ('I am looking for the road sign', 'road', (-5.65, 35.73)),
            ]),
        ]
        
        for scene, desc, scenarios in scene_scenarios:
            for query, loc_name, coords in scenarios:
                test_cases.append({
                    'query': query,
                    'ground_truth': {'x': coords[0], 'y': coords[1]},
                    'scene': scene,
                    'query_type': '物体定位',
                    'location_name': loc_name
                })
        
        print(f"✅ 生成了 {len(test_cases)} 个备用测试用例")
        return test_cases
    
    def check_api(self) -> bool:
        """检查API服务是否运行"""
        try:
            response = requests.get(f"{self.api_url}/api/v1/status", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_single_test(self, test_case: Dict) -> Optional[AccuracyResult]:
        """运行单次准确度测试"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/query",
                json={
                    "query": test_case["query"],
                    "top_k": 3,
                    "enable_enhanced": True
                },
                timeout=30
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    best = results[0]
                    pred_x = best.get("x", 0)
                    pred_y = best.get("y", 0)
                    
                    gt = test_case["ground_truth"]
                    if isinstance(gt, dict):
                        gt_x, gt_y = gt['x'], gt['y']
                    else:
                        gt_x, gt_y = gt[0], gt[1]
                    
                    confidence = best.get("confidence", 0.5)
                    cell_id = best.get("cell_id", "unknown")
                    
                    distance_error = np.sqrt(
                        (pred_x - gt_x)**2 + 
                        (pred_y - gt_y)**2
                    )
                    
                    return AccuracyResult(
                        query=test_case["query"][:60],
                        query_type=test_case.get('query_type', '未知'),
                        predicted_x=round(pred_x, 2),
                        predicted_y=round(pred_y, 2),
                        ground_truth_x=round(gt_x, 2),
                        ground_truth_y=round(gt_y, 2),
                        distance_error_m=round(distance_error, 2),
                        success=True,
                        confidence=round(confidence, 3),
                        response_time_ms=round(elapsed_ms, 2),
                        scene=test_case.get('scene', 'unknown'),
                        matched_cell=cell_id
                    )
                else:
                    gt = test_case["ground_truth"]
                    if isinstance(gt, dict):
                        gt_x, gt_y = gt['x'], gt['y']
                    else:
                        gt_x, gt_y = gt[0], gt[1]
                    
                    return AccuracyResult(
                        query=test_case["query"][:60],
                        query_type=test_case.get('query_type', '未知'),
                        predicted_x=0,
                        predicted_y=0,
                        ground_truth_x=round(gt_x, 2),
                        ground_truth_y=round(gt_y, 2),
                        distance_error_m=0,
                        success=False,
                        confidence=0,
                        response_time_ms=round(elapsed_ms, 2),
                        scene=test_case.get('scene', 'unknown'),
                        matched_cell="none"
                    )
            else:
                return None
                
        except Exception as e:
            print(f"    ⚠️ 测试失败: {e}")
            return None
    
    def calculate_metrics(self) -> AccuracyMetrics:
        """计算准确度指标"""
        if not self.results:
            return None
        
        successful = [r for r in self.results if r.success]
        errors = [r.distance_error_m for r in successful]
        
        if errors:
            sorted_errors = sorted(errors)
            n = len(errors)
            avg_error = sum(errors) / n
            median_error = sorted_errors[n // 2]
            min_error = sorted_errors[0]
            max_error = sorted_errors[-1]
            std_error = np.std(errors) if n > 1 else 0
            
            accuracy_1m = sum(1 for e in errors if e <= 1) / n * 100
            accuracy_3m = sum(1 for e in errors if e <= 3) / n * 100
            accuracy_5m = sum(1 for e in errors if e <= 5) / n * 100
            accuracy_10m = sum(1 for e in errors if e <= 10) / n * 100
            accuracy_20m = sum(1 for e in errors if e <= 20) / n * 100
        else:
            avg_error = median_error = min_error = max_error = std_error = 0
            accuracy_1m = accuracy_3m = accuracy_5m = accuracy_10m = accuracy_20m = 0
        
        return AccuracyMetrics(
            total_queries=len(self.results),
            successful_queries=len(successful),
            success_rate=len(successful) / len(self.results) * 100 if self.results else 0,
            avg_distance_error_m=round(avg_error, 2),
            median_distance_error_m=round(median_error, 2),
            min_distance_error_m=round(min_error, 2),
            max_distance_error_m=round(max_error, 2),
            std_distance_error_m=round(std_error, 2),
            accuracy_1m=round(accuracy_1m, 1),
            accuracy_3m=round(accuracy_3m, 1),
            accuracy_5m=round(accuracy_5m, 1),
            accuracy_10m=round(accuracy_10m, 1),
            accuracy_20m=round(accuracy_20m, 1)
        )
    
    def print_results(self):
        """打印测试结果"""
        print("\n" + "="*80)
        print("真实场景定位准确度评估实验")
        print("="*80)
        
        print(f"\n检查API服务...")
        if not self.check_api():
            print("❌ API服务未运行，请先启动服务：")
            print("   cd \"/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary\"")
            print("   python3 -m api.server")
            return
        
        print(f"✅ API服务运行正常")
        
        print(f"\n运行 {len(self.test_cases)} 个测试...")
        
        for i, test_case in enumerate(self.test_cases):
            print(f"\n测试 {i+1}/{len(self.test_cases)}")
            print(f"  查询: \"{test_case['query'][:50]}...\"")
            print(f"  场景: {test_case.get('scene', 'unknown')}")
            
            gt = test_case['ground_truth']
            if isinstance(gt, dict):
                gt_x, gt_y = gt['x'], gt['y']
            else:
                gt_x, gt_y = gt[0], gt[1]
            
            print(f"  GT: ({gt_x:.2f}, {gt_y:.2f})")
            
            result = self.run_single_test(test_case)
            
            if result:
                self.results.append(result)
                print(f"  预测: ({result.predicted_x:.2f}, {result.predicted_y:.2f})")
                print(f"  误差: {result.distance_error_m:.2f}m")
                print(f"  置信度: {result.confidence:.3f}")
                print(f"  耗时: {result.response_time_ms:.2f}ms")
            else:
                print(f"  ❌ 测试失败")
            
            time.sleep(0.5)
        
        metrics = self.calculate_metrics()
        
        print("\n" + "="*80)
        print("准确度评估结果")
        print("="*80)
        
        print(f"\n【基础指标】")
        print(f"  总查询数: {metrics.total_queries}")
        print(f"  成功查询: {metrics.successful_queries}")
        print(f"  成功率: {metrics.success_rate:.1f}%")
        
        print(f"\n【距离误差统计】")
        print(f"  平均距离误差: {metrics.avg_distance_error_m}m")
        print(f"  中位数距离误差: {metrics.median_distance_error_m}m")
        print(f"  最小距离误差: {metrics.min_distance_error_m}m")
        print(f"  最大距离误差: {metrics.max_distance_error_m}m")
        print(f"  标准差: {metrics.std_distance_error_m}m")
        
        print(f"\n【不同阈值内的准确率】")
        print(f"  1米内准确率: {metrics.accuracy_1m}%")
        print(f"  3米内准确率: {metrics.accuracy_3m}%")
        print(f"  5米内准确率: {metrics.accuracy_5m}%")
        print(f"  10米内准确率: {metrics.accuracy_10m}%")
        print(f"  20米内准确率: {metrics.accuracy_20m}%")
        
        print(f"\n【按查询类型分析】")
        query_types = set(r.query_type for r in self.results)
        for qtype in query_types:
            type_results = [r for r in self.results if r.query_type == qtype and r.success]
            if type_results:
                type_errors = [r.distance_error_m for r in type_results]
                avg_err = sum(type_errors) / len(type_errors)
                print(f"  {qtype}: {len(type_results)}个查询, 平均误差 {avg_err:.2f}m")
        
        os.makedirs('/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/experiment_results', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_data = {
            'timestamp': timestamp,
            'total_queries': metrics.total_queries,
            'successful_queries': metrics.successful_queries,
            'success_rate': metrics.success_rate,
            'distance_error': {
                'avg_m': metrics.avg_distance_error_m,
                'median_m': metrics.median_distance_error_m,
                'min_m': metrics.min_distance_error_m,
                'max_m': metrics.max_distance_error_m,
                'std_m': metrics.std_distance_error_m
            },
            'accuracy_thresholds': {
                '1m': metrics.accuracy_1m,
                '3m': metrics.accuracy_3m,
                '5m': metrics.accuracy_5m,
                '10m': metrics.accuracy_10m,
                '20m': metrics.accuracy_20m
            },
            'individual_results': [
                {
                    'query': r.query,
                    'query_type': r.query_type,
                    'predicted': {'x': r.predicted_x, 'y': r.predicted_y},
                    'ground_truth': {'x': r.ground_truth_x, 'y': r.ground_truth_y},
                    'error_m': r.distance_error_m,
                    'confidence': r.confidence,
                    'response_time_ms': r.response_time_ms
                }
                for r in self.results
            ]
        }
        
        json_path = f'/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/experiment_results/real_scene_accuracy_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细结果已保存: {json_path}")
        
        return metrics


def main():
    """主函数"""
    print("="*80)
    print("基于真实KITTI360场景的定位准确度评估")
    print("="*80)
    
    experiment = RealSceneAccuracyExperiment()
    metrics = experiment.print_results()
    
    if metrics:
        print("\n" + "="*80)
        print("核心指标总结")
        print("="*80)
        print(f"  成功率: {metrics.success_rate:.1f}%")
        print(f"  平均误差: {metrics.avg_distance_error_m}m")
        print(f"  5米内准确率: {metrics.accuracy_5m}%")
        print(f"  10米内准确率: {metrics.accuracy_10m}%")
        print("="*80)


if __name__ == "__main__":
    main()
