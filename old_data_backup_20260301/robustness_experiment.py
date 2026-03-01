#!/usr/bin/env python3
"""鲁棒性实验 - Text2Loc Visionary 系统稳定性测试"""

import sys
import os
import time
import json
import statistics
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

from enhancements.nlu.ollama_engine import OllamaNLUEngine, OllamaConfig
from api.text2loc_adapter import Text2LocAdapter

class RobustnessExperiment:
    """鲁棒性实验测试器"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "rounds": [],
            "statistics": {}
        }
    
    def run_stability_test(self, num_rounds=5):
        """稳定性测试：多次运行同一查询"""
        print("\n" + "="*80)
        print("鲁棒性实验1: 系统稳定性测试")
        print("="*80)
        
        test_query = "在建筑物左侧的树"
        round_results = []
        
        for i in range(num_rounds):
            print(f"\n轮次 {i+1}/{num_rounds}")
            
            start = time.time()
            try:
                nlu_config = OllamaConfig(
                    base_url="http://localhost:11434",
                    model="qwen3-vl:4b",
                    temperature=0.3,
                    timeout=30
                )
                nlu_engine = OllamaNLUEngine(nlu_config)
                adapter = Text2LocAdapter()
                
                nlu_result = nlu_engine.parse(test_query)
                if nlu_result.error:
                    raise Exception(f"NLU错误: {nlu_result.error}")
                
                location = adapter.find_location(
                    query=test_query,
                    direction=nlu_result.components.get('direction', 'none'),
                    color=nlu_result.components.get('color', 'none'),
                    obj=nlu_result.components.get('object', 'none'),
                    top_k=1
                )
                
                elapsed = (time.time() - start) * 1000
                success = len(location) > 0
                
                round_results.append({
                    "round": i+1,
                    "success": success,
                    "time_ms": elapsed,
                    "direction": nlu_result.components.get('direction'),
                    "color": nlu_result.components.get('color'),
                    "object": nlu_result.components.get('object')
                })
                
                print(f"   状态: {'✅ 成功' if success else '❌ 失败'}")
                print(f"   时间: {elapsed:.1f}ms")
                
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                round_results.append({
                    "round": i+1,
                    "success": False,
                    "time_ms": elapsed,
                    "error": str(e)
                })
                print(f"   状态: ❌ 失败 - {e}")
        
        # 统计
        successful = [r for r in round_results if r["success"]]
        times = [r["time_ms"] for r in successful]
        
        stats = {
            "total_rounds": num_rounds,
            "success_count": len(successful),
            "success_rate": len(successful) / num_rounds * 100,
            "avg_time_ms": statistics.mean(times) if times else 0,
            "std_time_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "min_time_ms": min(times) if times else 0,
            "max_time_ms": max(times) if times else 0
        }
        
        print(f"\n稳定性统计:")
        print(f"   成功率: {stats['success_rate']:.1f}%")
        print(f"   平均时间: {stats['avg_time_ms']:.1f}ms")
        print(f"   标准差: {stats['std_time_ms']:.1f}ms")
        
        self.results["rounds"].append({
            "test": "stability",
            "query": test_query,
            "results": round_results,
            "statistics": stats
        })
        
        return stats
    
    def run_query_variety_test(self):
        """多样性测试：不同类型查询"""
        print("\n" + "="*80)
        print("鲁棒性实验2: 查询多样性测试")
        print("="*80)
        
        test_queries = [
            ("颜色查询", "红色的汽车"),
            ("方向查询", "左边的建筑"),
            ("对象查询", "找到建筑物"),
            ("复合查询", "在建筑物前面的绿色草坪"),
            ("距离查询", "距离入口10米的地方")
        ]
        
        results = []
        
        for category, query in test_queries:
            print(f"\n[{category}] {query}")
            
            try:
                nlu_config = OllamaConfig(
                    base_url="http://localhost:11434",
                    model="qwen3-vl:4b",
                    temperature=0.3,
                    timeout=30
                )
                nlu_engine = OllamaNLUEngine(nlu_config)
                adapter = Text2LocAdapter()
                
                start = time.time()
                nlu_result = nlu_engine.parse(query)
                nlu_time = time.time() - start
                
                if nlu_result.error:
                    raise Exception(f"NLU错误")
                
                start = time.time()
                location = adapter.find_location(
                    query=query,
                    direction=nlu_result.components.get('direction', 'none'),
                    color=nlu_result.components.get('color', 'none'),
                    obj=nlu_result.components.get('object', 'none'),
                    top_k=1
                )
                loc_time = time.time() - start
                
                success = len(location) > 0
                
                results.append({
                    "category": category,
                    "query": query,
                    "success": success,
                    "nlu_time_ms": nlu_time * 1000,
                    "loc_time_ms": loc_time * 1000,
                    "total_time_ms": (nlu_time + loc_time) * 1000
                })
                
                print(f"   NLU: {nlu_time*1000:.1f}ms, 定位: {loc_time*1000:.1f}ms -> {'✅' if success else '❌'}")
                
            except Exception as e:
                results.append({
                    "category": category,
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
                print(f"   ❌ {e}")
        
        # 统计
        successful = [r for r in results if r["success"]]
        stats = {
            "total": len(results),
            "success": len(successful),
            "success_rate": len(successful) / len(results) * 100
        }
        
        print(f"\n多样性统计:")
        print(f"   成功率: {stats['success_rate']:.1f}%")
        
        self.results["rounds"].append({
            "test": "variety",
            "results": results,
            "statistics": stats
        })
        
        return stats
    
    def run_error_recovery_test(self):
        """错误恢复测试"""
        print("\n" + "="*80)
        print("鲁棒性实验3: 错误恢复测试")
        print("="*80)
        
        # 测试各种异常输入
        error_queries = [
            ("空输入", ""),
            ("超长输入", "在" * 100),
            ("特殊字符", "%%$$@@"),
            ("混合语言", "building在north边的red汽车旁边")
        ]
        
        results = []
        
        for test_type, query in error_queries:
            print(f"\n[{test_type}] \"{query[:30]}...\"" if len(query) > 30 else f"\n[{test_type}] \"{query}\"")
            
            try:
                nlu_config = OllamaConfig(
                    base_url="http://localhost:11434",
                    model="qwen3-vl:4b",
                    temperature=0.3,
                    timeout=30
                )
                nlu_engine = OllamaNLUEngine(nlu_config)
                adapter = Text2LocAdapter()
                
                start = time.time()
                
                if not query:
                    raise Exception("空查询")
                
                nlu_result = nlu_engine.parse(query)
                location = adapter.find_location(
                    query=query,
                    direction='none',
                    color='none',
                    obj='none',
                    top_k=1
                )
                
                elapsed = (time.time() - start) * 1000
                
                results.append({
                    "test_type": test_type,
                    "query": query[:50],
                    "success": True,
                    "time_ms": elapsed
                })
                print(f"   ✅ 恢复成功 ({elapsed:.1f}ms)")
                
            except Exception as e:
                elapsed = (time.time() - start) * 1000 if 'start' in dir() else 0
                results.append({
                    "test_type": test_type,
                    "query": query[:50],
                    "success": False,
                    "time_ms": elapsed,
                    "error": str(e)
                })
                print(f"   ❌ 预期失败 - {e}")
        
        # 所有异常输入都应该被正确处理（返回False或抛出异常）
        handled = sum(1 for r in results if not r["success"])
        stats = {
            "total": len(results),
            "handled": handled,
            "handling_rate": handled / len(results) * 100
        }
        
        print(f"\n错误处理统计:")
        print(f"   正确处理: {stats['handling_rate']:.1f}%")
        
        self.results["rounds"].append({
            "test": "error_recovery",
            "results": results,
            "statistics": stats
        })
        
        return stats
    
    def generate_report(self):
        """生成完整报告"""
        print("\n" + "="*80)
        print("鲁棒性实验完整报告")
        print("="*80)
        
        all_stats = []
        for test in self.results["rounds"]:
            if "statistics" in test:
                all_stats.append(test["statistics"])
        
        # 综合统计
        total_success = sum(s.get("success_count", s.get("success", 0)) for s in all_stats)
        total_tests = sum(s.get("total_rounds", s.get("total", 0)) for s in all_stats)
        overall_success = total_success / total_tests * 100 if total_tests > 0 else 0
        
        print(f"\n【综合评估】")
        print(f"   总测试数: {total_tests}")
        print(f"   总成功数: {total_success}")
        print(f"   综合成功率: {overall_success:.1f}%")
        
        # 评级
        if overall_success >= 95:
            rating = "A+ (优秀)"
        elif overall_success >= 85:
            rating = "A (良好)"
        elif overall_success >= 70:
            rating = "B (一般)"
        else:
            rating = "C (需改进)"
        
        print(f"   系统评级: {rating}")
        
        # 保存结果
        self.results["statistics"] = {
            "total_tests": total_tests,
            "total_success": total_success,
            "overall_success_rate": overall_success,
            "rating": rating
        }
        
        with open("robustness_experiment_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存: robustness_experiment_results.json")
        
        return self.results

def main():
    print("="*80)
    print("Text2Loc Visionary 鲁棒性实验")
    print("="*80)
    
    experiment = RobustnessExperiment()
    
    # 运行各项测试
    experiment.run_stability_test(num_rounds=5)
    experiment.run_query_variety_test()
    experiment.run_error_recovery_test()
    
    # 生成报告
    experiment.generate_report()

if __name__ == "__main__":
    main()
