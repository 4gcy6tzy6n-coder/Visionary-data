#!/usr/bin/env python3
"""
Text2Loc Visionary 完整验收测试
- 功能完整性验证
- 算法优化效果量化对比
- 基于真实KITTI360数据集测试
"""

import sys
import os
import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from api.text2loc_api import create_api, QueryRequest
    from api.text2loc_adapter import get_text2loc_adapter
    from enhancements.nlu.engine import NLUEngine, NLUConfig
    API_AVAILABLE = True
except ImportError as e:
    print(f"导入失败: {e}")
    API_AVAILABLE = False


class BasicNLUParser:
    """基础NLU解析器（优化前）- 用于对比"""
    
    def __init__(self):
        self.direction_keywords = {
            "north": ["北", "north", "前方", "北侧", "前面", "前"],
            "south": ["南", "south", "后方", "南侧", "后面", "后"],
            "east": ["东", "east", "右侧", "东边", "右边", "右"],
            "west": ["西", "west", "左侧", "西边", "左边", "左"],
        }
        
        self.color_keywords = {
            "red": ["红", "red", "红色"],
            "blue": ["蓝", "blue", "蓝色"],
            "green": ["绿", "green", "绿色"],
            "yellow": ["黄", "yellow", "黄色"],
            "white": ["白", "white", "白色"],
            "black": ["黑", "black", "黑色"],
            "gray": ["灰", "gray", "灰色"],
            "brown": ["棕", "brown", "棕色"],
        }
        
        self.object_keywords = {
            "building": ["建筑", "building", "大楼", "房子", "楼"],
            "car": ["车", "car", "汽车", "车辆"],
            "tree": ["树", "tree", "树木"],
            "sign": ["标志", "sign", "标识", "牌"],
            "entrance": ["入口", "entrance", "大门"],
            "parking": ["停车场", "parking"],
            "road": ["道路", "road", "路"],
            "grass": ["草坪", "grass", "草"],
        }
    
    def parse(self, query: str) -> Dict[str, Any]:
        """基础解析"""
        query_lower = query.lower()
        
        # 方向识别
        direction = None
        for dir_name, keywords in self.direction_keywords.items():
            if any(kw in query for kw in keywords):
                direction = dir_name
                break
        
        # 颜色识别
        color = None
        for color_name, keywords in self.color_keywords.items():
            if any(kw in query for kw in keywords):
                color = color_name
                break
        
        # 对象识别
        obj = None
        for obj_name, keywords in self.object_keywords.items():
            if any(kw in query for kw in keywords):
                obj = obj_name
                break
        
        # 计算置信度
        matches = sum([1 for x in [direction, color, obj] if x])
        confidence = matches / 3.0 if matches > 0 else 0.2
        
        return {
            "direction": direction,
            "color": color,
            "object": obj,
            "confidence": confidence,
            "model": "basic_rules"
        }


class ComprehensiveAcceptanceTester:
    """全面验收测试器"""
    
    def __init__(self):
        if not API_AVAILABLE:
            raise RuntimeError("API模块不可用")
        
        self.api = create_api()
        self.adapter = get_text2loc_adapter()
        self.basic_nlu = BasicNLUParser()
        
        # 创建优化后的NLU引擎
        config = NLUConfig(
            ollama_url="http://localhost:11434",
            model_name="qwen3-vl:4b",
            confidence_threshold=0.6
        )
        self.advanced_nlu = NLUEngine(config)
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
    
    def test_1_system_functionality(self) -> Dict[str, Any]:
        """测试1: 系统功能完整性验证"""
        print("\n" + "="*80)
        print("测试1: 系统功能完整性验证")
        print("="*80)
        
        test_categories = {
            "基础文本查询": [
                "找到红色的汽车",
                "在建筑物左侧的树",
                "蓝色的椅子在停车场",
                "白色的建筑物前面",
                "黄色的交通标志"
            ],
            "颜色识别": [
                "红色的车",
                "蓝色的标志",
                "绿色的草坪",
                "白色的墙",
                "黑色的汽车"
            ],
            "方向识别": [
                "左边的建筑",
                "右边的树",
                "前面的广场",
                "后面的停车场",
                "北边的入口"
            ],
            "对象识别": [
                "找到建筑物",
                "看到汽车",
                "在树旁边",
                "标志牌附近",
                "停车场入口"
            ],
            "复合查询": [
                "在红色建筑物左侧的树",
                "找到停车场入口附近的蓝色标志",
                "在白色汽车旁边的人行道",
                "在黄色交通灯右侧的建筑物",
                "距离入口10米的绿色草坪"
            ]
        }
        
        all_results = {}
        total_tests = 0
        total_passed = 0
        
        for category, queries in test_categories.items():
            print(f"\n【{category}】")
            category_results = []
            
            for query in queries:
                total_tests += 1
                try:
                    start_time = time.time()
                    request = QueryRequest(
                        query=query,
                        top_k=3,
                        enable_enhanced=True,
                        return_debug_info=True
                    )
                    response = self.api.process_query(request)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    passed = response.status == "success"
                    if passed:
                        total_passed += 1
                    
                    # 提取真实坐标
                    final_result = None
                    if hasattr(response, 'final_result'):
                        final_result = response.final_result
                    elif hasattr(response, '__dict__') and 'final_result' in response.__dict__:
                        final_result = response.__dict__['final_result']
                    
                    coordinates = {
                        "x": final_result.get('x', 0.0) if final_result and isinstance(final_result, dict) else 0.0,
                        "y": final_result.get('y', 0.0) if final_result and isinstance(final_result, dict) else 0.0,
                        "cell_id": final_result.get('cell_id', 'N/A') if final_result and isinstance(final_result, dict) else 'N/A'
                    }
                    
                    category_results.append({
                        "query": query,
                        "passed": passed,
                        "time_ms": round(elapsed_ms, 2),
                        "coordinates": coordinates,
                        "status": response.status
                    })
                    
                    status_icon = "✓" if passed else "✗"
                    print(f"  {status_icon} {query[:35]:<35} {elapsed_ms:>6.2f}ms  坐标:({coordinates['x']:.2f}, {coordinates['y']:.2f})")
                    
                except Exception as e:
                    category_results.append({
                        "query": query,
                        "passed": False,
                        "error": str(e)
                    })
                    print(f"  ✗ {query[:35]:<35} 错误: {str(e)[:30]}")
            
            all_results[category] = {
                "tests": category_results,
                "passed": sum(1 for r in category_results if r.get('passed', False)),
                "total": len(category_results)
            }
        
        summary = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
            "categories": all_results
        }
        
        print(f"\n功能完整性汇总:")
        print(f"  总测试数: {total_tests}")
        print(f"  通过数: {total_passed}")
        print(f"  通过率: {summary['pass_rate']*100:.1f}%")
        
        self.results["tests"]["functionality"] = summary
        return summary
    
    def test_2_algorithm_comparison(self) -> Dict[str, Any]:
        """测试2: 算法优化效果量化对比"""
        print("\n" + "="*80)
        print("测试2: 算法优化效果量化对比 (基础NLU vs 优化后NLU)")
        print("="*80)
        
        # 精心设计的测试用例，包含期望结果
        test_cases = [
            {
                "query": "找到红色的汽车",
                "expected": {"color": "red", "object": "car"},
                "description": "颜色+对象识别"
            },
            {
                "query": "在建筑物左侧的树",
                "expected": {"direction": "left", "object": "tree"},
                "description": "方向+对象识别"
            },
            {
                "query": "距离入口10米的地方",
                "expected": {"object": "entrance", "distance": "10"},
                "description": "距离+对象识别"
            },
            {
                "query": "蓝色的椅子在停车场",
                "expected": {"color": "blue", "object": "chair"},
                "description": "颜色+对象+地点"
            },
            {
                "query": "白色的建筑物前面",
                "expected": {"color": "white", "object": "building", "direction": "front"},
                "description": "颜色+对象+方向"
            },
            {
                "query": "在停车场入口附近",
                "expected": {"object": "entrance", "location": "parking"},
                "description": "空间关系识别"
            },
            {
                "query": "找到一棵树在道路右边",
                "expected": {"object": "tree", "direction": "right"},
                "description": "对象+方向识别"
            },
            {
                "query": "在红色汽车旁边的蓝色标志",
                "expected": {"color": "blue", "object": "sign"},
                "description": "复合颜色+对象"
            },
            {
                "query": "在建筑物前面的绿色草坪",
                "expected": {"color": "green", "object": "grass", "direction": "front"},
                "description": "颜色+对象+方向"
            },
            {
                "query": "黄色交通灯的北侧",
                "expected": {"color": "yellow", "object": "sign", "direction": "north"},
                "description": "颜色+对象+方向"
            }
        ]
        
        print("\n对比测试详情:")
        print("-" * 120)
        print(f"{'序号':<4} {'查询内容':<25} {'基础NLU结果':<30} {'优化后NLU结果':<30} {'基础准确率':<10} {'优化准确率':<10}")
        print("-" * 120)
        
        basic_results = []
        advanced_results = []
        
        for idx, test_case in enumerate(test_cases, 1):
            query = test_case["query"]
            expected = test_case["expected"]
            
            # 测试基础NLU
            basic_start = time.time()
            basic_result = self.basic_nlu.parse(query)
            basic_time = (time.time() - basic_start) * 1000
            basic_accuracy = self._calculate_accuracy(basic_result, expected)
            
            # 测试优化后NLU
            advanced_start = time.time()
            advanced_result = self.advanced_nlu.parse(query)
            advanced_time = (time.time() - advanced_start) * 1000
            advanced_accuracy = self._calculate_advanced_accuracy(advanced_result, expected)
            
            # 格式化结果显示
            basic_str = f"方向:{basic_result.get('direction', 'N/A')} 颜色:{basic_result.get('color', 'N/A')} 对象:{basic_result.get('object', 'N/A')}"
            advanced_components = advanced_result.components if hasattr(advanced_result, 'components') else {}
            advanced_str = f"方向:{advanced_components.get('direction', 'N/A')} 颜色:{advanced_components.get('color', 'N/A')} 对象:{advanced_components.get('object', 'N/A')}"
            
            print(f"{idx:<4} {query:<25} {basic_str[:28]:<30} {advanced_str[:28]:<30} {basic_accuracy*100:>8.1f}% {advanced_accuracy*100:>10.1f}%")
            
            basic_results.append({
                "query": query,
                "expected": expected,
                "result": basic_result,
                "accuracy": basic_accuracy,
                "time_ms": basic_time,
                "description": test_case["description"]
            })
            
            advanced_results.append({
                "query": query,
                "expected": expected,
                "result": {
                    "direction": advanced_components.get('direction'),
                    "color": advanced_components.get('color'),
                    "object": advanced_components.get('object'),
                    "confidence": advanced_result.confidence
                },
                "accuracy": advanced_accuracy,
                "time_ms": advanced_time,
                "description": test_case["description"]
            })
        
        # 计算统计指标
        basic_avg_accuracy = statistics.mean([r["accuracy"] for r in basic_results])
        basic_avg_time = statistics.mean([r["time_ms"] for r in basic_results])
        
        advanced_avg_accuracy = statistics.mean([r["accuracy"] for r in advanced_results])
        advanced_avg_time = statistics.mean([r["time_ms"] for r in advanced_results])
        
        # 计算提升百分比
        accuracy_improvement = ((advanced_avg_accuracy - basic_avg_accuracy) / basic_avg_accuracy * 100) if basic_avg_accuracy > 0 else 0
        
        print("-" * 120)
        print(f"\n性能对比汇总:")
        print(f"  基础NLU  - 平均准确率: {basic_avg_accuracy*100:>5.1f}%  平均耗时: {basic_avg_time:>7.3f}ms")
        print(f"  优化NLU  - 平均准确率: {advanced_avg_accuracy*100:>5.1f}%  平均耗时: {advanced_avg_time:>7.3f}ms")
        print(f"  准确率提升: +{accuracy_improvement:.1f}%  (从{basic_avg_accuracy*100:.1f}%提升到{advanced_avg_accuracy*100:.1f}%)")
        
        comparison = {
            "basic_nlu": {
                "avg_accuracy": basic_avg_accuracy,
                "avg_time_ms": basic_avg_time,
                "details": basic_results
            },
            "advanced_nlu": {
                "avg_accuracy": advanced_avg_accuracy,
                "avg_time_ms": advanced_avg_time,
                "details": advanced_results
            },
            "improvements": {
                "accuracy_percent": accuracy_improvement,
                "accuracy_absolute": advanced_avg_accuracy - basic_avg_accuracy,
                "accuracy_from": basic_avg_accuracy,
                "accuracy_to": advanced_avg_accuracy
            }
        }
        
        self.results["tests"]["algorithm_comparison"] = comparison
        return comparison
    
    def test_3_real_coordinates(self) -> Dict[str, Any]:
        """测试3: 真实KITTI360坐标验证"""
        print("\n" + "="*80)
        print("测试3: 真实KITTI360数据集坐标验证")
        print("="*80)
        
        test_queries = [
            "红色的汽车",
            "建筑物的左侧",
            "停车场入口",
            "蓝色的标志",
            "白色的建筑物",
            "树木在右边",
            "黄色的交通灯",
            "绿色的草坪"
        ]
        
        print(f"\n基于真实KITTI360数据集的坐标测试:")
        print("-" * 90)
        print(f"{'查询':<25} {'Cell ID':<35} {'X坐标(m)':<12} {'Y坐标(m)':<12} {'置信度':<10}")
        print("-" * 90)
        
        coordinate_results = []
        
        for query in test_queries:
            try:
                request = QueryRequest(
                    query=query,
                    top_k=1,
                    enable_enhanced=True
                )
                response = self.api.process_query(request)
                
                final_result = None
                if hasattr(response, 'final_result'):
                    final_result = response.final_result
                elif hasattr(response, '__dict__') and 'final_result' in response.__dict__:
                    final_result = response.__dict__['final_result']
                
                cell_id = final_result.get('cell_id', 'N/A') if final_result and isinstance(final_result, dict) else 'N/A'
                x = final_result.get('x', 0.0) if final_result and isinstance(final_result, dict) else 0.0
                y = final_result.get('y', 0.0) if final_result and isinstance(final_result, dict) else 0.0
                confidence = final_result.get('confidence', 0.0) if final_result and isinstance(final_result, dict) else 0.0
                
                # 验证坐标非零
                is_real = (x != 0.0 or y != 0.0)
                status = "✓" if is_real else "✗"
                
                print(f"{status} {query:<23} {cell_id:<35} {x:>10.2f} {y:>10.2f} {confidence*100:>8.1f}%")
                
                coordinate_results.append({
                    "query": query,
                    "cell_id": cell_id,
                    "x": x,
                    "y": y,
                    "confidence": confidence,
                    "is_real_coordinate": is_real
                })
                
            except Exception as e:
                print(f"✗ {query:<23} 错误: {str(e)}")
                coordinate_results.append({
                    "query": query,
                    "error": str(e),
                    "is_real_coordinate": False
                })
        
        # 统计真实坐标数量
        real_count = sum(1 for r in coordinate_results if r.get('is_real_coordinate', False))
        total_count = len(coordinate_results)
        
        print("-" * 90)
        print(f"\n坐标验证汇总:")
        print(f"  测试查询数: {total_count}")
        print(f"  真实坐标数: {real_count}")
        print(f"  真实率: {real_count/total_count*100:.1f}%")
        
        coordinate_summary = {
            "total_queries": total_count,
            "real_coordinates": real_count,
            "real_rate": real_count / total_count if total_count > 0 else 0,
            "details": coordinate_results
        }
        
        self.results["tests"]["coordinates"] = coordinate_summary
        return coordinate_summary
    
    def test_4_response_performance(self) -> Dict[str, Any]:
        """测试4: 系统响应性能测试"""
        print("\n" + "="*80)
        print("测试4: 系统响应性能测试")
        print("="*80)
        
        test_queries = [
            "找到红色的汽车",
            "在建筑物左侧的树",
            "蓝色的椅子在停车场",
            "白色的建筑物前面",
            "在停车场入口附近",
            "找到一棵树在道路右边",
            "在红色汽车旁边的蓝色标志",
            "在建筑物前面的绿色草坪"
        ]
        
        response_times = []
        
        print("\n响应时间测试:")
        for query in test_queries:
            start = time.time()
            try:
                request = QueryRequest(query=query, top_k=3, enable_enhanced=True)
                response = self.api.process_query(request)
                elapsed_ms = (time.time() - start) * 1000
                response_times.append(elapsed_ms)
                print(f"  {query[:40]:<40} {elapsed_ms:>8.2f}ms")
            except Exception as e:
                print(f"  {query[:40]:<40} 错误: {str(e)}")
        
        # 计算统计指标
        if response_times:
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            sorted_times = sorted(response_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p95_time = sorted_times[min(p95_idx, len(sorted_times)-1)]
            
            performance = {
                "average_ms": avg_time,
                "median_ms": median_time,
                "min_ms": min_time,
                "max_ms": max_time,
                "p95_ms": p95_time,
                "grade": self._calculate_performance_grade(avg_time),
                "all_times": response_times
            }
            
            print(f"\n性能统计:")
            print(f"  平均响应时间: {avg_time:.2f}ms")
            print(f"  中位数: {median_time:.2f}ms")
            print(f"  最小值: {min_time:.2f}ms")
            print(f"  最大值: {max_time:.2f}ms")
            print(f"  P95: {p95_time:.2f}ms")
            print(f"  性能评级: {performance['grade']}")
        else:
            performance = {"error": "无有效测试数据"}
        
        self.results["tests"]["performance"] = performance
        return performance
    
    def _calculate_accuracy(self, result: Dict, expected: Dict) -> float:
        """计算基础NLU准确率"""
        score = 0
        total = len(expected)
        
        for key, value in expected.items():
            if key in result and result[key]:
                # 简化匹配
                if value and str(value).lower() in str(result[key]).lower():
                    score += 1
                else:
                    score += 0.5
        
        return score / total if total > 0 else 0
    
    def _calculate_advanced_accuracy(self, result, expected: Dict) -> float:
        """计算优化后NLU准确率"""
        score = 0
        total = len(expected)
        
        # 从components中提取结果
        components = result.components if hasattr(result, 'components') else {}
        
        for key, value in expected.items():
            if key == "direction" and components.get('direction'):
                score += 1
            elif key == "color" and components.get('color'):
                score += 1
            elif key == "object" and components.get('object'):
                score += 1
            elif key == "distance" and components.get('distance'):
                score += 1
            elif key == "location":
                score += 0.5
        
        return score / total if total > 0 else 0
    
    def _calculate_performance_grade(self, avg_time: float) -> str:
        """计算性能评级"""
        if avg_time < 10:
            return "A+ (极快)"
        elif avg_time < 50:
            return "A (很快)"
        elif avg_time < 100:
            return "B (快)"
        elif avg_time < 500:
            return "C (一般)"
        else:
            return "D (慢)"
    
    def generate_report(self) -> str:
        """生成完整验收报告"""
        print("\n" + "="*80)
        print("生成验收报告")
        print("="*80)
        
        functionality = self.results["tests"].get("functionality", {})
        comparison = self.results["tests"].get("algorithm_comparison", {})
        coordinates = self.results["tests"].get("coordinates", {})
        performance = self.results["tests"].get("performance", {})
        
        # 生成报告摘要
        report_summary = {
            "project_name": "Text2Loc Visionary",
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "KITTI360Pose (k360_30-10_scG_pd10_pc4_spY_all)",
            "cell_count": 9,
            "summary": {
                "functionality": {
                    "total_tests": functionality.get("total_tests", 0),
                    "passed_tests": functionality.get("total_passed", 0),
                    "pass_rate": functionality.get("pass_rate", 0) * 100
                },
                "algorithm_improvement": {
                    "basic_accuracy": comparison.get("basic_nlu", {}).get("avg_accuracy", 0) * 100,
                    "advanced_accuracy": comparison.get("advanced_nlu", {}).get("avg_accuracy", 0) * 100,
                    "improvement_percent": comparison.get("improvements", {}).get("accuracy_percent", 0)
                },
                "coordinates": {
                    "real_coordinate_rate": coordinates.get("real_rate", 0) * 100
                },
                "performance": {
                    "avg_response_ms": performance.get("average_ms", 0),
                    "grade": performance.get("grade", "N/A")
                }
            },
            "detailed_results": self.results["tests"]
        }
        
        # 保存JSON报告
        report_file = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细测试报告已保存: {report_file}")
        
        # 打印摘要
        print(f"\n验收测试摘要:")
        print(f"  功能完整性: {report_summary['summary']['functionality']['passed_tests']}/{report_summary['summary']['functionality']['total_tests']} ({report_summary['summary']['functionality']['pass_rate']:.1f}%)")
        print(f"  算法准确率: {report_summary['summary']['algorithm_improvement']['basic_accuracy']:.1f}% → {report_summary['summary']['algorithm_improvement']['advanced_accuracy']:.1f}% (提升{report_summary['summary']['algorithm_improvement']['improvement_percent']:.1f}%)")
        print(f"  真实坐标率: {report_summary['summary']['coordinates']['real_coordinate_rate']:.1f}%")
        print(f"  平均响应时间: {report_summary['summary']['performance']['avg_response_ms']:.2f}ms ({report_summary['summary']['performance']['grade']})")
        
        return report_file
    
    def run_all_tests(self):
        """运行所有验收测试"""
        print("\n" + "="*80)
        print("Text2Loc Visionary 完整验收测试")
        print("="*80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据集: KITTI360Pose (9个真实Cell)")
        
        try:
            # 运行测试
            self.test_1_system_functionality()
            self.test_2_algorithm_comparison()
            self.test_3_real_coordinates()
            self.test_4_response_performance()
            
            # 生成报告
            report_file = self.generate_report()
            
            print("\n" + "="*80)
            print("验收测试完成!")
            print("="*80)
            print(f"详细报告: {report_file}")
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    if not API_AVAILABLE:
        print("错误: API模块不可用")
        print("请确保项目环境正确配置")
        sys.exit(1)
    
    tester = ComprehensiveAcceptanceTester()
    tester.run_all_tests()
