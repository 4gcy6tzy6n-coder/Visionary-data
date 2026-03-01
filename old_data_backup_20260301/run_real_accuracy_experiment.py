#!/usr/bin/env python3
"""
真实定位准确度评估实验
使用poses数据中的真实位置作为ground truth
对比预测坐标与真实坐标的误差
"""

import requests
import time
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
import sys
import os
import pickle
from dataclasses import dataclass, asdict

# 添加项目路径
sys.path.insert(0, '/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary')

@dataclass
class AccuracyResult:
    """单次查询的准确度结果"""
    query: str
    predicted_x: float
    predicted_y: float
    ground_truth_x: float
    ground_truth_y: float
    distance_error_m: float
    success: bool
    confidence: float
    response_time_ms: float

@dataclass
class AccuracyMetrics:
    """准确度指标汇总"""
    total_queries: int
    successful_queries: int
    success_rate: float
    
    # 距离误差统计
    avg_distance_error_m: float
    median_distance_error_m: float
    min_distance_error_m: float
    max_distance_error_m: float
    std_distance_error_m: float
    
    # 不同阈值内的准确率
    accuracy_1m: float
    accuracy_3m: float
    accuracy_5m: float
    accuracy_10m: float


class RealAccuracyExperiment:
    """真实定位准确度实验"""
    
    def __init__(self, dataset_path: str = None):
        self.api_url = "http://localhost:8080"
        
        # 加载poses数据作为ground truth - 使用修复后的数据集
        if dataset_path is None:
            dataset_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired"
        
        self.dataset_path = dataset_path
        self.test_cases = self._load_real_test_cases()
    
    def _load_real_test_cases(self) -> List[Dict]:
        """从poses数据加载真实测试用例 - 使用正确的k360_repaired数据结构"""
        poses_path = os.path.join(self.dataset_path, "poses", "poses.pkl")
        
        if not os.path.exists(poses_path):
            print(f"⚠️ Poses文件不存在: {poses_path}")
            return []
        
        with open(poses_path, 'rb') as f:
            poses = pickle.load(f)
        
        print(f"✅ 从 {poses_path} 加载了 {len(poses)} 个poses")
        
        # 从poses构建测试用例 - k360_repaired格式
        test_cases = []
        for i, pose in enumerate(poses[:10]):  # 使用前10个poses
            if isinstance(pose, dict):
                # 获取pose的真实坐标 - k360_repaired使用location字段(numpy数组)
                location = pose.get('location')
                if isinstance(location, np.ndarray) and len(location) >= 2:
                    gt_x = float(location[0])
                    gt_y = float(location[1])
                elif isinstance(location, (list, tuple)) and len(location) >= 2:
                    gt_x = float(location[0])
                    gt_y = float(location[1])
                else:
                    print(f"  ⚠️ Pose {i} 坐标格式错误: {type(location)}")
                    continue
                
                # 获取描述 - k360_repaired格式
                descriptions = pose.get('descriptions', [])
                if descriptions and isinstance(descriptions[0], dict):
                    query = descriptions[0].get('text', '')
                    if not query:
                        query = f"Find location near {pose.get('scene', 'unknown')}"
                elif descriptions:
                    query = str(descriptions[0])
                else:
                    query = f"Find location near {pose.get('scene', 'unknown')}"
                
                scene = pose.get('scene', 'unknown')
                pose_id = pose.get('id', f'unknown_{i}')
                
                test_cases.append({
                    'query': query,
                    'ground_truth': {'x': gt_x, 'y': gt_y},
                    'scene': scene,
                    'pose_id': pose_id
                })
                
                print(f"  ✅ 测试用例 {i+1}: scene={scene}, GT=({gt_x:.2f}, {gt_y:.2f}), query=\"{query[:50]}...\"")
        
        print(f"✅ 生成了 {len(test_cases)} 个有效测试用例")
        return test_cases
    
    def check_api(self) -> bool:
        """检查API服务是否运行"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_single_test(self, test_case: Dict) -> AccuracyResult:
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
            
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    best = results[0]
                    pred_x = best.get("x", 0)
                    pred_y = best.get("y", 0)
                    gt = test_case["ground_truth"]
                    
                    # 计算欧氏距离误差
                    distance_error = np.sqrt((pred_x - gt["x"])**2 + (pred_y - gt["y"])**2)
                    
                    return AccuracyResult(
                        query=test_case["query"][:50],
                        predicted_x=round(pred_x, 2),
                        predicted_y=round(pred_y, 2),
                        ground_truth_x=round(gt["x"], 2),
                        ground_truth_y=round(gt["y"], 2),
                        distance_error_m=round(distance_error, 2),
                        success=True,
                        confidence=best.get("confidence", 0),
                        response_time_ms=round(elapsed, 2)
                    )
            
            # 失败情况
            return AccuracyResult(
                query=test_case["query"][:50],
                predicted_x=0, predicted_y=0,
                ground_truth_x=test_case["ground_truth"]["x"],
                ground_truth_y=test_case["ground_truth"]["y"],
                distance_error_m=999,
                success=False,
                confidence=0,
                response_time_ms=round(elapsed, 2)
            )
                
        except Exception as e:
            return AccuracyResult(
                query=test_case["query"][:50],
                predicted_x=0, predicted_y=0,
                ground_truth_x=test_case["ground_truth"]["x"],
                ground_truth_y=test_case["ground_truth"]["y"],
                distance_error_m=999,
                success=False,
                confidence=0,
                response_time_ms=0
            )
    
    def calculate_metrics(self, results: List[AccuracyResult]) -> AccuracyMetrics:
        """计算准确度指标"""
        successful = [r for r in results if r.success]
        total = len(results)
        success_count = len(successful)
        
        if not successful:
            return AccuracyMetrics(
                total_queries=total, successful_queries=0, success_rate=0,
                avg_distance_error_m=999, median_distance_error_m=999,
                min_distance_error_m=999, max_distance_error_m=999, std_distance_error_m=0,
                accuracy_1m=0, accuracy_3m=0, accuracy_5m=0, accuracy_10m=0
            )
        
        # 距离误差统计
        distance_errors = [r.distance_error_m for r in successful]
        
        # 不同阈值内的准确率
        accuracy_1m = sum(1 for r in successful if r.distance_error_m <= 1.0) / total * 100
        accuracy_3m = sum(1 for r in successful if r.distance_error_m <= 3.0) / total * 100
        accuracy_5m = sum(1 for r in successful if r.distance_error_m <= 5.0) / total * 100
        accuracy_10m = sum(1 for r in successful if r.distance_error_m <= 10.0) / total * 100
        
        return AccuracyMetrics(
            total_queries=total,
            successful_queries=success_count,
            success_rate=round(success_count / total * 100, 2),
            avg_distance_error_m=round(np.mean(distance_errors), 2),
            median_distance_error_m=round(np.median(distance_errors), 2),
            min_distance_error_m=round(np.min(distance_errors), 2),
            max_distance_error_m=round(np.max(distance_errors), 2),
            std_distance_error_m=round(np.std(distance_errors), 2),
            accuracy_1m=round(accuracy_1m, 2),
            accuracy_3m=round(accuracy_3m, 2),
            accuracy_5m=round(accuracy_5m, 2),
            accuracy_10m=round(accuracy_10m, 2)
        )
    
    def run_experiment(self) -> Dict:
        """运行完整准确度实验"""
        print("=" * 80)
        print("真实定位准确度评估实验")
        print("=" * 80)
        print()
        
        # 检查API
        print("检查API服务...")
        if not self.check_api():
            print("❌ API服务未运行")
            return {"error": "API not running"}
        print("✅ API服务运行正常")
        print()
        
        if not self.test_cases:
            print("❌ 没有可用的测试用例")
            return {"error": "No test cases"}
        
        # 运行测试
        print(f"运行 {len(self.test_cases)} 个真实测试...")
        print("-" * 80)
        
        results = []
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n测试 {i}/{len(self.test_cases)}")
            print(f"  查询: {test_case['query'][:60]}...")
            print(f"  真实坐标: ({test_case['ground_truth']['x']:.2f}, {test_case['ground_truth']['y']:.2f})")
            
            result = self.run_single_test(test_case)
            results.append(result)
            
            if result.success:
                print(f"  预测坐标: ({result.predicted_x:.2f}, {result.predicted_y:.2f})")
                print(f"  距离误差: {result.distance_error_m:.2f}m")
            else:
                print(f"  ❌ 失败")
        
        # 计算指标
        metrics = self.calculate_metrics(results)
        
        # 打印汇总
        print("\n" + "=" * 80)
        print("真实准确度评估结果")
        print("=" * 80)
        
        print(f"\n【基础指标】")
        print(f"  总查询数: {metrics.total_queries}")
        print(f"  成功查询: {metrics.successful_queries}")
        print(f"  成功率: {metrics.success_rate}%")
        
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
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/experiment_results"
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            "timestamp": timestamp,
            "experiment_type": "real_accuracy_evaluation",
            "dataset": "k360_semantic",
            "metrics": asdict(metrics),
            "detailed_results": [asdict(r) for r in results]
        }
        
        json_path = f"{output_dir}/real_accuracy_experiment_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 详细结果已保存: {json_path}")
        
        md_path = f"{output_dir}/real_accuracy_experiment_{timestamp}.md"
        self._generate_markdown_report(report, md_path)
        print(f"📄 Markdown报告已保存: {md_path}")
        
        return report
    
    def _generate_markdown_report(self, report: Dict, output_path: str):
        """生成Markdown格式报告"""
        metrics = report["metrics"]
        results = report["detailed_results"]
        
        md = f"""# 真实定位准确度评估报告

**实验时间**: {report['timestamp']}
**数据集**: {report['dataset']}
**实验类型**: {report['experiment_type']}

## 1. 实验概述

本实验使用poses数据中的真实位置作为ground truth，评估Text2Loc Visionary系统的真实定位准确度。

## 2. 总体指标

| 指标 | 数值 |
|------|------|
| 总查询数 | {metrics['total_queries']} |
| 成功查询 | {metrics['successful_queries']} |
| **成功率** | **{metrics['success_rate']}%** |

## 3. 距离误差统计

| 指标 | 数值 |
|------|------|
| 平均距离误差 | **{metrics['avg_distance_error_m']} m** |
| 中位数距离误差 | {metrics['median_distance_error_m']} m |
| 最小距离误差 | {metrics['min_distance_error_m']} m |
| 最大距离误差 | {metrics['max_distance_error_m']} m |
| 标准差 | {metrics['std_distance_error_m']} m |

## 4. 不同阈值内的准确率

| 距离阈值 | 准确率 |
|----------|--------|
| **1米内** | **{metrics['accuracy_1m']}%** |
| **3米内** | **{metrics['accuracy_3m']}%** |
| **5米内** | **{metrics['accuracy_5m']}%** |
| **10米内** | **{metrics['accuracy_10m']}%** |

## 5. 详细结果

| # | 查询 | 预测坐标 | 真实坐标 | 误差(m) | 置信度 |
|---|------|----------|----------|---------|--------|
"""
        
        for i, r in enumerate(results, 1):
            status = "✅" if r["success"] else "❌"
            pred = f"({r['predicted_x']:.2f}, {r['predicted_y']:.2f})" if r["success"] else "-"
            gt = f"({r['ground_truth_x']:.2f}, {r['ground_truth_y']:.2f})"
            dist = f"{r['distance_error_m']:.2f}" if r["success"] else "-"
            conf = f"{r['confidence']:.4f}" if r["success"] else "-"
            
            md += f"| {i} | {r['query'][:30]}... | {pred} | {gt} | {dist} | {conf} |\n"
        
        md += f"""
## 6. 结论

### 6.1 真实定位精度
- 平均距离误差: **{metrics['avg_distance_error_m']}m**
- 5米内准确率: **{metrics['accuracy_5m']}%**

### 6.2 系统性能
- 成功率: **{metrics['success_rate']}%**
- 测试基于真实poses数据

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        with open(output_path, 'w') as f:
            f.write(md)


def main():
    """主函数"""
    experiment = RealAccuracyExperiment()
    report = experiment.run_experiment()
    
    if "error" not in report:
        print("\n✅ 真实准确度实验完成!")
        metrics = report['metrics']
        print(f"\n核心指标:")
        print(f"  平均距离误差: {metrics['avg_distance_error_m']}m")
        print(f"  5米内准确率: {metrics['accuracy_5m']}%")
        print(f"  成功率: {metrics['success_rate']}%")
    else:
        print(f"\n❌ 实验失败: {report['error']}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
