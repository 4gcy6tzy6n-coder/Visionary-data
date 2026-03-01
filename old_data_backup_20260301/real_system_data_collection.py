#!/usr/bin/env python3
"""
Text2Loc 整体系统真实数据采集实验
通过实际运行系统收集端到端的真实性能和精度数据
"""

import sys
import os
import time
import json
import requests
import statistics
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class RealSystemDataCollector:
    """真实系统数据采集器"""
    
    def __init__(self, server_url: str = "http://localhost:5050"):
        self.server_url = server_url
        self.test_queries = self._prepare_realistic_test_queries()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "server_info": {},
            "test_results": [],
            "aggregated_metrics": {}
        }
    
    def _prepare_realistic_test_queries(self) -> List[str]:
        """准备真实场景测试查询集"""
        return [
            # 基础定位查询
            "找到红色的汽车",
            "在建筑物左侧的树", 
            "蓝色的标志牌",
            "白色的建筑物",
            "绿色的草坪",
            "黄色的交通灯",
            
            # 方向定位查询
            "道路右边的停车场",
            "前方的入口",
            "后面的灰色建筑",
            "北侧的蓝色汽车",
            "东边的红色卡车",
            
            # 复杂语义查询
            "在红色建筑物左侧的树",
            "找到停车场入口附近的蓝色标志",
            "在白色汽车旁边的人行道",
            "在黄色交通灯右侧的建筑物",
            "距离入口10米的绿色草坪",
            "在建筑物前面的红色汽车旁边",
            "蓝色的椅子在停车场东边",
            "灰色的房子在道路南侧",
            "在大树北面的小车",
            "白色墙壁左侧的黑色门",
            
            # 模糊语义查询
            "附近有个红色的东西",
            "好像在左边有什么",
            "大概是建筑物周围",
            "不清楚具体位置但有棵树",
            " somewhere around the building"
        ]
    
    def check_server_status(self) -> bool:
        """检查服务器状态"""
        try:
            response = requests.get(f"{self.server_url}/api/v1/status", timeout=5)
            if response.status_code == 200:
                status_data = response.json()
                self.results["server_info"] = status_data
                print("✅ 服务器状态正常")
                print(f"   模型: {status_data.get('model', 'N/A')}")
                print(f"   运行时间: {status_data.get('uptime', 'N/A')}秒")
                return True
            else:
                print(f"❌ 服务器返回状态码: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 无法连接服务器: {e}")
            return False
    
    def run_single_query_test(self, query: str) -> Dict[str, Any]:
        """执行单次查询测试并收集数据"""
        print(f"\n测试查询: '{query}'")
        print("-" * 50)
        
        try:
            start_time = time.time()
            
            # 发送查询请求
            payload = {
                "query": query,
                "top_k": 3,
                "enable_enhanced": True,
                "return_debug_info": True
            }
            
            response = requests.post(
                f"{self.server_url}/api/v1/query",
                json=payload,
                timeout=10
            )
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            if response.status_code == 200:
                result_data = response.json()
                
                # 提取关键信息
                nlu_result = result_data.get('nlu_result', {})
                final_result = result_data.get('final_result', {})
                candidates = result_data.get('candidates', [])
                
                # 计算NLU解析质量
                nlu_components = nlu_result.get('components', {})
                nlu_confidence = nlu_result.get('confidence', 0)
                
                # 提取坐标信息
                x_coord = final_result.get('x', 0.0) if final_result else 0.0
                y_coord = final_result.get('y', 0.0) if final_result else 0.0
                cell_id = final_result.get('cell_id', 'N/A') if final_result else 'N/A'
                
                # 判断是否为真实坐标（非默认值）
                is_real_coordinate = not (abs(x_coord) < 0.01 and abs(y_coord) < 0.01)
                
                test_result = {
                    "query": query,
                    "status": "success",
                    "total_time_ms": round(total_time, 2),
                    "nlu_time_ms": round(nlu_result.get('parse_time', 0) * 1000, 2),
                    "nlu_confidence": round(nlu_confidence, 2),
                    "nlu_components": nlu_components,
                    "coordinates": {
                        "x": round(x_coord, 2),
                        "y": round(y_coord, 2),
                        "cell_id": cell_id,
                        "is_real": is_real_coordinate
                    },
                    "candidates_count": len(candidates),
                    "response_data": result_data
                }
                
                print(f"  ✅ 成功")
                print(f"     总耗时: {test_result['total_time_ms']}ms")
                print(f"     NLU耗时: {test_result['nlu_time_ms']}ms")
                print(f"     NLU置信度: {test_result['nlu_confidence']}")
                print(f"     坐标: ({test_result['coordinates']['x']}, {test_result['coordinates']['y']})")
                print(f"     真实坐标: {'是' if is_real_coordinate else '否'}")
                print(f"     候选数: {test_result['candidates_count']}")
                
                return test_result
                
            else:
                error_msg = response.text if response.text else f"HTTP {response.status_code}"
                test_result = {
                    "query": query,
                    "status": "failed",
                    "error": error_msg,
                    "total_time_ms": round(total_time, 2)
                }
                print(f"  ❌ 失败: {error_msg}")
                return test_result
                
        except Exception as e:
            test_result = {
                "query": query,
                "status": "error",
                "error": str(e),
                "total_time_ms": 0
            }
            print(f"  ❌ 错误: {e}")
            return test_result
    
    def run_comprehensive_test(self):
        """运行完整的系统测试"""
        print("=" * 80)
        print("Text2Loc 整体系统真实数据采集实验")
        print("=" * 80)
        
        # 1. 检查服务器状态
        if not self.check_server_status():
            print("❌ 服务器不可用，终止测试")
            return None
        
        # 2. 执行测试查询
        print(f"\n开始执行 {len(self.test_queries)} 个测试查询...")
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\n[{i}/{len(self.test_queries)}]", end="")
            result = self.run_single_query_test(query)
            self.results["test_results"].append(result)
            
            # 短暂延迟避免请求过于密集
            time.sleep(0.5)
        
        # 3. 计算聚合指标
        self._calculate_aggregated_metrics()
        
        # 4. 保存结果
        self._save_results()
        
        # 5. 打印汇总
        self._print_summary()
        
        return self.results
    
    def _calculate_aggregated_metrics(self):
        """计算聚合指标"""
        successful_results = [r for r in self.results["test_results"] if r["status"] == "success"]
        
        if not successful_results:
            self.results["aggregated_metrics"] = {"error": "无成功结果"}
            return
        
        # 性能指标
        total_times = [r["total_time_ms"] for r in successful_results]
        nlu_times = [r["nlu_time_ms"] for r in successful_results if r["nlu_time_ms"] > 0]
        confidences = [r["nlu_confidence"] for r in successful_results]
        
        # 坐标指标
        real_coords = [1 for r in successful_results if r["coordinates"]["is_real"]]
        coord_x_values = [r["coordinates"]["x"] for r in successful_results]
        coord_y_values = [r["coordinates"]["y"] for r in successful_results]
        
        # NLU组件覆盖率
        direction_coverage = sum(1 for r in successful_results if r["nlu_components"].get("direction") != "none")
        color_coverage = sum(1 for r in successful_results if r["nlu_components"].get("color") != "none")
        object_coverage = sum(1 for r in successful_results if r["nlu_components"].get("object") != "none")
        
        self.results["aggregated_metrics"] = {
            "total_queries": len(self.test_queries),
            "successful_queries": len(successful_results),
            "success_rate": round(len(successful_results) / len(self.test_queries) * 100, 1),
            
            "performance": {
                "avg_total_time_ms": round(statistics.mean(total_times), 2),
                "median_total_time_ms": round(statistics.median(total_times), 2),
                "min_total_time_ms": round(min(total_times), 2),
                "max_total_time_ms": round(max(total_times), 2),
                "std_total_time_ms": round(statistics.stdev(total_times), 2) if len(total_times) > 1 else 0,
                
                "avg_nlu_time_ms": round(statistics.mean(nlu_times), 2) if nlu_times else 0,
                "avg_nlu_confidence": round(statistics.mean(confidences), 2)
            },
            
            "coordinates": {
                "real_coordinate_rate": round(len(real_coords) / len(successful_results) * 100, 1),
                "coord_x_range": {
                    "min": round(min(coord_x_values), 2),
                    "max": round(max(coord_x_values), 2),
                    "avg": round(statistics.mean(coord_x_values), 2)
                },
                "coord_y_range": {
                    "min": round(min(coord_y_values), 2),
                    "max": round(max(coord_y_values), 2),
                    "avg": round(statistics.mean(coord_y_values), 2)
                }
            },
            
            "nlu_coverage": {
                "direction_recognition_rate": round(direction_coverage / len(successful_results) * 100, 1),
                "color_recognition_rate": round(color_coverage / len(successful_results) * 100, 1),
                "object_recognition_rate": round(object_coverage / len(successful_results) * 100, 1)
            }
        }
    
    def _save_results(self):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"text2loc_real_system_test_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 测试结果已保存到: {filename}")
        
        # 同时保存简化版CSV格式便于分析
        self._save_csv_summary()
    
    def _save_csv_summary(self):
        """保存CSV格式摘要"""
        csv_filename = "text2loc_test_summary.csv"
        
        with open(csv_filename, 'w', encoding='utf-8') as f:
            f.write("Query,Status,TotalTime(ms),NLUConfidence,X,Y,IsReal,Direction,Color,Object\n")
            
            for result in self.results["test_results"]:
                query = result["query"].replace(",", "，")  # 处理CSV分隔符
                status = result["status"]
                time_ms = result.get("total_time_ms", "")
                confidence = result.get("nlu_confidence", "")
                coords = result.get("coordinates", {})
                x = coords.get("x", "")
                y = coords.get("y", "")
                is_real = "Yes" if coords.get("is_real", False) else "No"
                components = result.get("nlu_components", {})
                direction = components.get("direction", "")
                color = components.get("color", "")
                obj = components.get("object", "")
                
                f.write(f"{query},{status},{time_ms},{confidence},{x},{y},{is_real},{direction},{color},{obj}\n")
        
        print(f"📊 CSV摘要已保存到: {csv_filename}")
    
    def _print_summary(self):
        """打印测试汇总"""
        print("\n" + "=" * 80)
        print("测试汇总报告")
        print("=" * 80)
        
        metrics = self.results["aggregated_metrics"]
        
        if "error" in metrics:
            print(f"❌ {metrics['error']}")
            return
        
        print(f"总查询数: {metrics['total_queries']}")
        print(f"成功数: {metrics['successful_queries']}")
        print(f"成功率: {metrics['success_rate']}%")
        
        print(f"\n性能指标:")
        perf = metrics["performance"]
        print(f"  平均总耗时: {perf['avg_total_time_ms']}ms")
        print(f"  NLU平均耗时: {perf['avg_nlu_time_ms']}ms")
        print(f"  NLU平均置信度: {perf['avg_nlu_confidence']}")
        
        print(f"\n坐标质量:")
        coords = metrics["coordinates"]
        print(f"  真实坐标率: {coords['real_coordinate_rate']}%")
        print(f"  X坐标范围: {coords['coord_x_range']['min']} ~ {coords['coord_x_range']['max']}m")
        print(f"  Y坐标范围: {coords['coord_y_range']['min']} ~ {coords['coord_y_range']['max']}m")
        
        print(f"\nNLU覆盖度:")
        nlu = metrics["nlu_coverage"]
        print(f"  方向识别率: {nlu['direction_recognition_rate']}%")
        print(f"  颜色识别率: {nlu['color_recognition_rate']}%")
        print(f"  对象识别率: {nlu['object_recognition_rate']}%")

if __name__ == "__main__":
    collector = RealSystemDataCollector()
    results = collector.run_comprehensive_test()
