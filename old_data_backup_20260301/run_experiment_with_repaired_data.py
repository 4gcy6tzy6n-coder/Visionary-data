#!/usr/bin/env python3
"""
使用修复后的数据运行对比实验
验证数据修复对定位精度的影响
"""

import os
import sys
import json
import time
import psutil
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import requests

# 设置路径
VISIONARY_PATH = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary")
REPAIRED_DATA_PATH = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_repaired")
ORIGINAL_DATA_PATH = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all")
sys.path.insert(0, str(VISIONARY_PATH))


@dataclass
class TestSample:
    """测试样本"""
    query: str
    ground_truth_x: float
    ground_truth_y: float
    scene_name: str
    cell_id: str


@dataclass
class ExperimentResult:
    """实验结果"""
    sample: TestSample
    system: str
    success: bool
    response_time_ms: float
    predicted_x: float
    predicted_y: float
    distance_error_m: float
    cpu_percent: float
    memory_mb: float
    error_message: str = ""


class ValidationExperiment:
    """验证实验执行器"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.api_url = "http://localhost:8080"
        self.cells = []
        self.poses = []
        self.test_samples = []
        self.results = []
        
    def load_data(self) -> bool:
        """加载KITTI360数据"""
        print("=" * 80)
        print(f"加载数据: {self.data_path}")
        print("=" * 80)
        
        cells_file = self.data_path / "cells" / "cells.pkl"
        poses_file = self.data_path / "poses" / "poses.pkl"
        
        if not cells_file.exists():
            print(f"❌ Cells文件不存在: {cells_file}")
            return False
        
        if not poses_file.exists():
            print(f"❌ Poses文件不存在: {poses_file}")
            return False
        
        try:
            with open(cells_file, 'rb') as f:
                self.cells = pickle.load(f)
            print(f"✅ 加载了 {len(self.cells)} 个cells")
            
            with open(poses_file, 'rb') as f:
                self.poses = pickle.load(f)
            print(f"✅ 加载了 {len(self.poses)} 个poses")
            
            return True
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return False
    
    def generate_test_samples(self, num_samples: int = 30) -> List[TestSample]:
        """生成测试样本"""
        print("\n" + "=" * 80)
        print("生成测试样本")
        print("=" * 80)
        
        samples = []
        
        for i, pose in enumerate(self.poses):
            if len(samples) >= num_samples:
                break
            
            # 处理字典格式的pose
            if isinstance(pose, dict):
                location = pose.get('location', [0, 0, 0])
                gt_x, gt_y = float(location[0]), float(location[1])
                cell_id = pose.get('cell_id', f"cell_{i}")
                scene_name = pose.get('scene', 'unknown')
            else:
                if hasattr(pose, 'pose_w'):
                    gt_x, gt_y = float(pose.pose_w[0]), float(pose.pose_w[1])
                else:
                    continue
                cell_id = pose.cell_id if hasattr(pose, 'cell_id') else f"cell_{i}"
                scene_name = pose.scene if hasattr(pose, 'scene') else "unknown"
            
            # 生成查询文本
            query_templates = [
                "the red car on the left",
                "white building in front",
                "green tree on the right side",
                "blue vehicle behind",
                "pedestrian crossing ahead",
                "traffic light on the corner",
                "parked car near the sidewalk",
                "bus stop on the left",
                "road sign ahead",
                "bicycle on the right",
                "truck in the distance",
                "storefront on the corner",
                "street lamp on the left",
                "crosswalk in front",
                "parking lot on the right",
                "gas station ahead",
                "bus on the left side",
                "motorcycle parked nearby",
                "fence along the road",
                "garage entrance ahead",
                "building with glass windows",
                "tree line on the right",
                "sidewalk on the left",
                "intersection ahead",
                "roundabout on the right",
                "bridge in the distance",
                "tunnel entrance ahead",
                "construction site on the left",
                "park on the right side",
                "school zone ahead",
            ]
            query = query_templates[i % len(query_templates)]
            
            sample = TestSample(
                query=query[:100],
                ground_truth_x=gt_x,
                ground_truth_y=gt_y,
                scene_name=scene_name,
                cell_id=str(cell_id)
            )
            
            samples.append(sample)
        
        print(f"✅ 生成了 {len(samples)} 个测试样本")
        self.test_samples = samples
        return samples
    
    def run_visionary_tests(self) -> List[ExperimentResult]:
        """运行Visionary系统测试"""
        print("\n" + "=" * 80)
        print("测试 Text2Loc Visionary")
        print("=" * 80)
        
        results = []
        process = psutil.Process()
        
        for i, sample in enumerate(self.test_samples, 1):
            print(f"\n[{i}/{len(self.test_samples)}] 查询: {sample.query[:40]}...")
            
            cpu_before = process.cpu_percent(interval=0.05)
            mem_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.api_url}/api/v1/query",
                    json={"query": sample.query},
                    timeout=30
                )
                
                response_time = (time.time() - start_time) * 1000
                cpu_after = process.cpu_percent(interval=0.05)
                mem_after = process.memory_info().rss / 1024 / 1024
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('status') == 'success' and data.get('results'):
                        result_data = data['results'][0]
                        pred_x = float(result_data.get('x', 0.0))
                        pred_y = float(result_data.get('y', 0.0))
                        success = True
                        error_msg = ""
                    else:
                        pred_x, pred_y = 0.0, 0.0
                        success = False
                        error_msg = data.get('error', 'No results')
                else:
                    pred_x, pred_y = 0.0, 0.0
                    success = False
                    error_msg = f"HTTP {response.status_code}"
                    
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                cpu_after = process.cpu_percent(interval=0.05)
                mem_after = process.memory_info().rss / 1024 / 1024
                pred_x, pred_y = 0.0, 0.0
                success = False
                error_msg = str(e)
            
            distance_error = np.sqrt((pred_x - sample.ground_truth_x)**2 + 
                                      (pred_y - sample.ground_truth_y)**2)
            
            result = ExperimentResult(
                sample=sample,
                system='visionary',
                success=success,
                response_time_ms=response_time,
                predicted_x=pred_x,
                predicted_y=pred_y,
                distance_error_m=distance_error,
                cpu_percent=(cpu_before + cpu_after) / 2,
                memory_mb=(mem_before + mem_after) / 2,
                error_message=error_msg
            )
            
            results.append(result)
            
            status = "✅" if success else "❌"
            print(f"    {status} 响应: {response_time:.1f}ms")
            print(f"    📍 预测: ({pred_x:.2f}, {pred_y:.2f})")
            print(f"    🎯 真实: ({sample.ground_truth_x:.2f}, {sample.ground_truth_y:.2f})")
            print(f"    📏 误差: {distance_error:.2f}m")
            
        return results
    
    def calculate_metrics(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """计算指标"""
        total = len(results)
        successful = [r for r in results if r.success]
        
        if not successful:
            return {
                'total': total,
                'success_count': 0,
                'success_rate': 0,
                'avg_response_time_ms': 0,
                'avg_distance_error_m': 999,
                'accuracy_1m': 0,
                'accuracy_3m': 0,
                'accuracy_5m': 0,
                'accuracy_10m': 0,
            }
        
        success_count = len(successful)
        success_rate = success_count / total * 100
        
        response_times = [r.response_time_ms for r in successful]
        distance_errors = [r.distance_error_m for r in successful]
        
        return {
            'total': total,
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_response_time_ms': np.mean(response_times),
            'min_response_time_ms': np.min(response_times),
            'max_response_time_ms': np.max(response_times),
            'avg_distance_error_m': np.mean(distance_errors),
            'median_distance_error_m': np.median(distance_errors),
            'min_distance_error_m': np.min(distance_errors),
            'max_distance_error_m': np.max(distance_errors),
            'std_distance_error_m': np.std(distance_errors),
            'accuracy_1m': sum(1 for r in successful if r.distance_error_m <= 1.0) / total * 100,
            'accuracy_3m': sum(1 for r in successful if r.distance_error_m <= 3.0) / total * 100,
            'accuracy_5m': sum(1 for r in successful if r.distance_error_m <= 5.0) / total * 100,
            'accuracy_10m': sum(1 for r in successful if r.distance_error_m <= 10.0) / total * 100,
        }
    
    def generate_report(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """生成实验报告"""
        metrics = self.calculate_metrics(results)
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_path': str(self.data_path),
                'test_samples': len(self.test_samples),
            },
            'metrics': metrics,
            'detailed_results': [
                {
                    'sample': asdict(r.sample),
                    'system': r.system,
                    'success': r.success,
                    'response_time_ms': r.response_time_ms,
                    'predicted_x': r.predicted_x,
                    'predicted_y': r.predicted_y,
                    'distance_error_m': r.distance_error_m,
                    'error_message': r.error_message,
                }
                for r in results
            ]
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], suffix: str = ""):
        """保存报告"""
        output_dir = Path("experiment_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_dir / f"validation_experiment_{suffix}_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ 报告已保存: {json_file}")
        
        # 生成Markdown报告
        md_file = output_dir / f"validation_experiment_{suffix}_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(f"# 验证实验报告 ({suffix})\n\n")
            f.write(f"**数据路径**: {report['metadata']['data_path']}\n\n")
            f.write(f"**测试样本数**: {report['metadata']['test_samples']}\n\n")
            
            f.write("## 实验结果\n\n")
            metrics = report['metrics']
            f.write(f"- **成功率**: {metrics['success_rate']:.1f}%\n")
            f.write(f"- **平均响应时间**: {metrics['avg_response_time_ms']:.2f}ms\n")
            f.write(f"- **平均距离误差**: {metrics['avg_distance_error_m']:.2f}m\n")
            f.write(f"- **中位数距离误差**: {metrics['median_distance_error_m']:.2f}m\n")
            f.write(f"- **1米内准确率**: {metrics['accuracy_1m']:.1f}%\n")
            f.write(f"- **3米内准确率**: {metrics['accuracy_3m']:.1f}%\n")
            f.write(f"- **5米内准确率**: {metrics['accuracy_5m']:.1f}%\n")
            f.write(f"- **10米内准确率**: {metrics['accuracy_10m']:.1f}%\n")
        
        print(f"✅ Markdown报告: {md_file}")
    
    def print_summary(self, report: Dict[str, Any]):
        """打印摘要"""
        print("\n" + "=" * 80)
        print("实验结果摘要")
        print("=" * 80)
        
        metrics = report['metrics']
        print(f"\n📊 测试样本数: {report['metadata']['test_samples']}")
        print(f"📁 数据路径: {report['metadata']['data_path']}")
        print()
        print(f"成功率: {metrics['success_rate']:.1f}%")
        print(f"平均响应时间: {metrics['avg_response_time_ms']:.2f}ms")
        print(f"平均距离误差: {metrics['avg_distance_error_m']:.2f}m")
        print(f"中位数距离误差: {metrics['median_distance_error_m']:.2f}m")
        print()
        print(f"1米内准确率: {metrics['accuracy_1m']:.1f}%")
        print(f"3米内准确率: {metrics['accuracy_3m']:.1f}%")
        print(f"5米内准确率: {metrics['accuracy_5m']:.1f}%")
        print(f"10米内准确率: {metrics['accuracy_10m']:.1f}%")


def main():
    """主函数"""
    print("=" * 80)
    print("Text2Loc Visionary 验证实验")
    print("对比原始数据和修复后的数据")
    print("=" * 80)
    
    # 检查API服务
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code != 200:
            print("❌ API服务未运行，请先启动服务")
            print("   cd \"/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary\" && python3 -m api.server")
            return
    except Exception as e:
        print(f"❌ 无法连接到API服务: {e}")
        print("   请先启动服务:")
        print("   cd \"/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary\" && python3 -m api.server")
        return
    
    print("✅ API服务运行正常\n")
    
    # 实验1：使用原始数据
    print("\n" + "=" * 80)
    print("实验1: 使用原始数据")
    print("=" * 80)
    
    exp_original = ValidationExperiment(ORIGINAL_DATA_PATH)
    if not exp_original.load_data():
        print("❌ 原始数据加载失败")
        return
    
    exp_original.generate_test_samples(30)
    results_original = exp_original.run_visionary_tests()
    report_original = exp_original.generate_report(results_original)
    exp_original.save_report(report_original, "original_data")
    exp_original.print_summary(report_original)
    
    # 实验2：使用修复后的数据
    print("\n" + "=" * 80)
    print("实验2: 使用修复后的数据")
    print("=" * 80)
    
    exp_repaired = ValidationExperiment(REPAIRED_DATA_PATH)
    if not exp_repaired.load_data():
        print("❌ 修复数据加载失败")
        return
    
    exp_repaired.generate_test_samples(30)
    results_repaired = exp_repaired.run_visionary_tests()
    report_repaired = exp_repaired.generate_report(results_repaired)
    exp_repaired.save_report(report_repaired, "repaired_data")
    exp_repaired.print_summary(report_repaired)
    
    # 对比结果
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    
    orig_metrics = report_original['metrics']
    rep_metrics = report_repaired['metrics']
    
    print(f"\n{'指标':<30} {'原始数据':<15} {'修复后数据':<15} {'改进':<15}")
    print("-" * 80)
    print(f"{'平均距离误差':<30} {orig_metrics['avg_distance_error_m']:>10.2f}m {rep_metrics['avg_distance_error_m']:>10.2f}m {(orig_metrics['avg_distance_error_m'] - rep_metrics['avg_distance_error_m']):>+10.2f}m")
    print(f"{'5米内准确率':<30} {orig_metrics['accuracy_5m']:>10.1f}% {rep_metrics['accuracy_5m']:>10.1f}% {(rep_metrics['accuracy_5m'] - orig_metrics['accuracy_5m']):>+10.1f}pp")
    print(f"{'10米内准确率':<30} {orig_metrics['accuracy_10m']:>10.1f}% {rep_metrics['accuracy_10m']:>10.1f}% {(rep_metrics['accuracy_10m'] - orig_metrics['accuracy_10m']):>+10.1f}pp")
    
    print("\n✅ 验证实验完成!")


if __name__ == "__main__":
    main()
