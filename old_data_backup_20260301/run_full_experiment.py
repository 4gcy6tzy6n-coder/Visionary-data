"""
全面实验 - 收集所有实验数据用于报告生成
"""

import requests
import json
import time
import os
import pickle
from datetime import datetime

class FullExperiment:
    def __init__(self, base_url="http://localhost:8080/api/v1"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "experiments": {}
        }
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def collect_system_info(self):
        """收集系统信息"""
        self.log("收集系统信息...")
        
        # API配置
        try:
            response = requests.get(f"{self.base_url}/config", timeout=5)
            if response.status_code == 200:
                self.results["system_info"]["api_config"] = response.json()
        except:
            pass
        
        # 数据信息
        data_path = os.path.expanduser("~/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all")
        
        # Cells
        try:
            with open(os.path.join(data_path, "cells", "cells.pkl"), 'rb') as f:
                cells = pickle.load(f)
            self.results["system_info"]["cells_count"] = len(cells)
            self.results["system_info"]["cells_details"] = [
                {"id": c["id"], "objects": len(c.get("objects", []))} 
                for c in cells[:5]
            ]
        except Exception as e:
            self.log(f"读取cells失败: {e}")
        
        # Poses
        try:
            with open(os.path.join(data_path, "poses", "poses.pkl"), 'rb') as f:
                poses = pickle.load(f)
            self.results["system_info"]["poses_count"] = len(poses)
            self.results["system_info"]["poses_details"] = [
                {"id": p["id"], "location": p.get("location", [])[:3].tolist() if hasattr(p.get("location"), 'tolist') else p.get("location", [])[:3]} 
                for p in poses[:5]
            ]
        except Exception as e:
            self.log(f"读取poses失败: {e}")
        
        # 原始数据
        raw_path = os.path.join(data_path, "raw_data")
        if os.path.exists(raw_path):
            self.results["system_info"]["raw_data"] = os.listdir(raw_path)
        
        self.log(f"系统信息收集完成")
        self.log(f"  - Cells: {self.results['system_info'].get('cells_count', 0)}")
        self.log(f"  - Poses: {self.results['system_info'].get('poses_count', 0)}")
    
    def run_nlu_tests(self):
        """NLU解析测试"""
        self.log("\n运行NLU解析测试...")
        
        test_cases = [
            {"query": "我在红色大楼的北侧", "expected": {"direction": "北侧", "color": "红色", "object": "大楼"}},
            {"query": "在绿色树木的东边", "expected": {"direction": "东边", "color": "绿色", "object": "树木"}},
            {"query": "靠近蓝色标志", "expected": {"direction": "靠近", "color": "蓝色", "object": "标志"}},
            {"query": "前方50米有白色建筑物", "expected": {"direction": "前方", "distance": "50米", "color": "白色", "object": "建筑物"}},
            {"query": "位于十字路口的东南角", "expected": {"direction": "东南角", "object": "十字路口"}},
            {"query": "在停车场的入口处", "expected": {"direction": "入口处", "object": "停车场"}},
            {"query": "靠近加油站的便利店", "expected": {"direction": "靠近", "object": "便利店", "reference": "加油站"}},
            {"query": "在人行横道的对面", "expected": {"direction": "对面", "object": "人行横道"}}
        ]
        
        results = []
        for test in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": test["query"], "top_k": 3, "enable_enhanced": True, "return_debug_info": True},
                    timeout=60
                )
                elapsed = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "query": test["query"],
                        "success": result.get("status") == "success",
                        "time_ms": elapsed,
                        "has_results": "results" in result and result["results"],
                        "nlu_result": result.get("nlu_result", {}),
                        "top_result": result.get("results", [{}])[0] if result.get("results") else None
                    })
                else:
                    results.append({
                        "query": test["query"],
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    })
            except Exception as e:
                results.append({
                    "query": test["query"],
                    "success": False,
                    "error": str(e)
                })
        
        self.results["experiments"]["nlu_tests"] = results
        success_count = sum(1 for r in results if r["success"])
        self.log(f"NLU测试完成: {success_count}/{len(results)} 成功")
        
        return results
    
    def run_performance_tests(self):
        """性能测试"""
        self.log("\n运行性能测试...")
        
        queries = [
            "我在红色大楼的北侧",
            "在绿色树木的东边",
            "靠近蓝色标志",
            "前方有白色建筑物",
            "位于十字路口"
        ]
        
        times = []
        for query in queries:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": query, "top_k": 3},
                    timeout=60
                )
                elapsed = (time.time() - start_time) * 1000
                times.append(elapsed)
            except:
                pass
        
        if times:
            perf_stats = {
                "count": len(times),
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "times": times
            }
            self.results["experiments"]["performance"] = perf_stats
            self.log(f"性能测试完成: 平均 {perf_stats['avg_ms']:.2f} ms")
        
        return times
    
    def run_cache_tests(self):
        """缓存效果测试"""
        self.log("\n运行缓存效果测试...")
        
        query = "我在红色大楼的北侧"
        times = []
        
        # 连续查询3次
        for i in range(3):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": query, "top_k": 3},
                    timeout=60
                )
                elapsed = (time.time() - start_time) * 1000
                times.append({"run": i+1, "time_ms": elapsed})
            except:
                pass
        
        self.results["experiments"]["cache_tests"] = times
        if len(times) >= 2:
            speedup = times[0]["time_ms"] / times[-1]["time_ms"] if times[-1]["time_ms"] > 0 else 0
            self.log(f"缓存测试完成: 首次 {times[0]['time_ms']:.2f} ms, 缓存后 {times[-1]['time_ms']:.2f} ms (加速 {speedup:.1f}x)")
        
        return times
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("="*70)
        print("Text2Loc Visionary 全面实验")
        print("="*70)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. 系统信息
        self.collect_system_info()
        
        # 2. NLU测试
        self.run_nlu_tests()
        
        # 3. 性能测试
        self.run_performance_tests()
        
        # 4. 缓存测试
        self.run_cache_tests()
        
        # 保存结果
        output_file = "full_experiment_data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("实验数据收集完成")
        print("="*70)
        print(f"数据已保存到: {output_file}")
        
        return self.results

if __name__ == "__main__":
    experiment = FullExperiment()
    results = experiment.run_all_experiments()
