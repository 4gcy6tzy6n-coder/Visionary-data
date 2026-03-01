"""
大规模实验 - 确保50+次独立API调用测试
每个测试都是独立的API请求，真实验证系统性能
"""

import requests
import json
import time
import os
import pickle
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class MassiveExperiments:
    def __init__(self, base_url="http://localhost:8080/api/v1"):
        self.base_url = base_url
        self.all_results = []
        self.lock = threading.Lock()
        self.test_counter = 0
        
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}.get(level, "ℹ️")
        print(f"[{timestamp}] {prefix} {message}")
    
    def run_single_api_test(self, test_id, query, category, expected_status=200):
        """运行单个API测试并记录结果"""
        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query, "top_k": 3},
                timeout=60
            )
            elapsed = (time.time() - start) * 1000
            
            result = {
                "test_id": test_id,
                "query": query,
                "category": category,
                "response_time_ms": elapsed,
                "status_code": response.status_code,
                "success": response.status_code == expected_status,
                "timestamp": datetime.now().isoformat()
            }
            
            with self.lock:
                self.all_results.append(result)
                self.test_counter += 1
            
            status = "✅" if result["success"] else "❌"
            self.log(f"{status} 测试 {test_id}: [{category}] '{query[:20]}...' - {elapsed:.2f}ms")
            return result
            
        except Exception as e:
            result = {
                "test_id": test_id,
                "query": query,
                "category": category,
                "response_time_ms": 0,
                "status_code": -1,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            with self.lock:
                self.all_results.append(result)
                self.test_counter += 1
            self.log(f"❌ 测试 {test_id}: [{category}] 失败 - {e}", "ERROR")
            return result
    
    # ==================== 测试套件 1: 基础方向 (10个) ====================
    def test_basic_directions(self):
        """基础方向测试 - 10个独立测试"""
        self.log("\n" + "="*70)
        self.log("测试套件 1: 基础方向 (10个测试)")
        self.log("="*70)
        
        tests = [
            ("D01", "北", "基础方向"),
            ("D02", "南", "基础方向"),
            ("D03", "东", "基础方向"),
            ("D04", "西", "基础方向"),
            ("D05", "东北", "基础方向"),
            ("D06", "东南", "基础方向"),
            ("D07", "西北", "基础方向"),
            ("D08", "西南", "基础方向"),
            ("D09", "上", "基础方向"),
            ("D10", "下", "基础方向"),
        ]
        
        for test_id, query, category in tests:
            self.run_single_api_test(test_id, query, category)
    
    # ==================== 测试套件 2: 颜色+物体 (15个) ====================
    def test_color_objects(self):
        """颜色物体测试 - 15个独立测试"""
        self.log("\n" + "="*70)
        self.log("测试套件 2: 颜色+物体 (15个测试)")
        self.log("="*70)
        
        tests = [
            ("C01", "红色大楼", "颜色物体"),
            ("C02", "蓝色汽车", "颜色物体"),
            ("C03", "白色建筑物", "颜色物体"),
            ("C04", "绿色树木", "颜色物体"),
            ("C05", "黑色标志", "颜色物体"),
            ("C06", "灰色墙壁", "颜色物体"),
            ("C07", "黄色信号灯", "颜色物体"),
            ("C08", "橙色交通锥", "颜色物体"),
            ("C09", "紫色招牌", "颜色物体"),
            ("C10", "粉色花朵", "颜色物体"),
            ("C11", "棕色木箱", "颜色物体"),
            ("C12", "银色汽车", "颜色物体"),
            ("C13", "金色装饰", "颜色物体"),
            ("C14", "青色门", "颜色物体"),
            ("C15", "米色墙面", "颜色物体"),
        ]
        
        for test_id, query, category in tests:
            self.run_single_api_test(test_id, query, category)
    
    # ==================== 测试套件 3: 复合位置描述 (15个) ====================
    def test_complex_locations(self):
        """复合位置测试 - 15个独立测试"""
        self.log("\n" + "="*70)
        self.log("测试套件 3: 复合位置描述 (15个测试)")
        self.log("="*70)
        
        tests = [
            ("L01", "我在红色大楼的北侧", "复合位置"),
            ("L02", "在绿色树木的东边", "复合位置"),
            ("L03", "靠近蓝色标志", "复合位置"),
            ("L04", "前方有白色建筑物", "复合位置"),
            ("L05", "位于十字路口", "复合位置"),
            ("L06", "在停车场的入口处", "复合位置"),
            ("L07", "靠近加油站的便利店", "复合位置"),
            ("L08", "在人行横道的对面", "复合位置"),
            ("L09", "东南方向的建筑物", "复合位置"),
            ("L10", "西北角的标志", "复合位置"),
            ("L11", "在红色汽车和蓝色卡车之间", "复合位置"),
            ("L12", "靠近白色墙壁的角落", "复合位置"),
            ("L13", "在绿色草坪的南侧", "复合位置"),
            ("L14", "位于灰色大楼的东侧入口", "复合位置"),
            ("L15", "在黑色大门的前面", "复合位置"),
        ]
        
        for test_id, query, category in tests:
            self.run_single_api_test(test_id, query, category)
    
    # ==================== 测试套件 4: 相对位置 (10个) ====================
    def test_relative_positions(self):
        """相对位置测试 - 10个独立测试"""
        self.log("\n" + "="*70)
        self.log("测试套件 4: 相对位置 (10个测试)")
        self.log("="*70)
        
        tests = [
            ("R01", "靠近入口", "相对位置"),
            ("R02", "在对面", "相对位置"),
            ("R03", "旁边", "相对位置"),
            ("R04", "附近", "相对位置"),
            ("R05", "中间位置", "相对位置"),
            ("R06", "在左边", "相对位置"),
            ("R07", "在右边", "相对位置"),
            ("R08", "在前面", "相对位置"),
            ("R09", "在后面", "相对位置"),
            ("R10", "在之间", "相对位置"),
        ]
        
        for test_id, query, category in tests:
            self.run_single_api_test(test_id, query, category)
    
    # ==================== 测试套件 5: 边界情况 (10个) ====================
    def test_boundary_cases(self):
        """边界测试 - 10个独立测试"""
        self.log("\n" + "="*70)
        self.log("测试套件 5: 边界情况 (10个测试)")
        self.log("="*70)
        
        tests = [
            ("B01", "北", "边界-最短"),
            ("B02", "红", "边界-最短"),
            ("B03", "我在一个非常大的红色建筑物的北侧靠近入口的地方有很多车辆和行人", "边界-最长"),
            ("B04", "红色@大楼#北侧$", "边界-特殊字符"),
            ("B05", "在50米外的红色建筑物旁边100米处", "边界-数字"),
            ("B06", "在red大楼的blue汽车旁边", "边界-中英文"),
            ("B07", "北北北北北北北北北", "边界-重复"),
            ("B08", "在不存在的地方", "边界-无效"),
            ("B09", "某个地方", "边界-模糊"),
            ("B10", "东南西北上下左右前后", "边界-复杂"),
        ]
        
        for test_id, query, category in tests:
            self.run_single_api_test(test_id, query, category)
    
    # ==================== 测试套件 6: 错误处理 (5个) ====================
    def test_error_handling(self):
        """错误处理测试 - 5个独立测试"""
        self.log("\n" + "="*70)
        self.log("测试套件 6: 错误处理 (5个测试)")
        self.log("="*70)
        
        # 空查询测试
        self.log("测试 E01: 空查询")
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": "", "top_k": 3},
                timeout=10
            )
            result = {
                "test_id": "E01",
                "query": "(空)",
                "category": "错误处理",
                "response_time_ms": 0,
                "status_code": response.status_code,
                "success": response.status_code == 400,
                "timestamp": datetime.now().isoformat()
            }
            self.all_results.append(result)
            self.test_counter += 1
            self.log(f"{'✅' if result['success'] else '⚠️'} 测试 E01: 空查询处理 - 状态码 {response.status_code}")
        except Exception as e:
            self.log(f"❌ 测试 E01: 失败 - {e}", "ERROR")
        
        # 其他错误测试
        tests = [
            ("E02", "   ", "错误处理"),
            ("E03", "<script>alert('xss')</script>", "错误处理-XSS"),
            ("E04", "SELECT * FROM table", "错误处理-SQL"),
            ("E05", "'; DROP TABLE users; --", "错误处理-注入"),
        ]
        
        for test_id, query, category in tests:
            self.run_single_api_test(test_id, query, category)
    
    # ==================== 测试套件 7: 性能压力 (10个) ====================
    def test_performance_stress(self):
        """性能压力测试 - 10个独立测试"""
        self.log("\n" + "="*70)
        self.log("测试套件 7: 性能压力 (10个测试)")
        self.log("="*70)
        
        tests = [
            ("P01", "红色大楼北侧", "性能测试"),
            ("P02", "绿色树木东边", "性能测试"),
            ("P03", "蓝色标志附近", "性能测试"),
            ("P04", "白色建筑物前方", "性能测试"),
            ("P05", "十字路口东南角", "性能测试"),
            ("P06", "停车场入口", "性能测试"),
            ("P07", "加油站便利店", "性能测试"),
            ("P08", "人行横道对面", "性能测试"),
            ("P09", "建筑物东南方向", "性能测试"),
            ("P10", "标志西北角", "性能测试"),
        ]
        
        for test_id, query, category in tests:
            self.run_single_api_test(test_id, query, category)
    
    # ==================== 测试套件 8: 并发测试 (5个) ====================
    def test_concurrent(self):
        """并发测试 - 5个同时进行的测试"""
        self.log("\n" + "="*70)
        self.log("测试套件 8: 并发测试 (5个并发)")
        self.log("="*70)
        
        tests = [
            ("T01", "并发查询1-红色大楼", "并发测试"),
            ("T02", "并发查询2-蓝色汽车", "并发测试"),
            ("T03", "并发查询3-绿色树木", "并发测试"),
            ("T04", "并发查询4-白色墙壁", "并发测试"),
            ("T05", "并发查询5-黑色大门", "并发测试"),
        ]
        
        def run_concurrent(test_id, query, category):
            return self.run_single_api_test(test_id, query, category)
        
        # 使用线程池并发执行
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for test_id, query, category in tests:
                future = executor.submit(run_concurrent, test_id, query, category)
                futures.append(future)
            
            # 等待所有完成
            for future in as_completed(futures):
                future.result()
    
    # ==================== 运行所有测试 ====================
    def run_all_experiments(self):
        """运行所有实验"""
        print("\n" + "="*70)
        print("Text2Loc Visionary - 大规模实验 (50+次独立API调用)")
        print("="*70)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"目标: 确保50+次独立API测试")
        print("="*70)
        
        start_time = time.time()
        
        # 运行所有测试套件
        self.test_basic_directions()      # 10个
        self.test_color_objects()          # 15个
        self.test_complex_locations()      # 15个
        self.test_relative_positions()     # 10个
        self.test_boundary_cases()         # 10个
        self.test_error_handling()         # 5个
        self.test_performance_stress()     # 10个
        self.test_concurrent()             # 5个
        
        total_time = time.time() - start_time
        
        # 统计分析
        self.analyze_results(total_time)
        
        return self.all_results
    
    def analyze_results(self, total_time):
        """分析实验结果"""
        self.log("\n" + "="*70)
        self.log("实验结果统计分析")
        self.log("="*70)
        
        total = len(self.all_results)
        passed = sum(1 for r in self.all_results if r.get("success"))
        failed = total - passed
        
        # 响应时间统计
        response_times = [r["response_time_ms"] for r in self.all_results if r.get("response_time_ms", 0) > 0]
        
        if response_times:
            avg_time = statistics.mean(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            median_time = statistics.median(response_times)
        else:
            avg_time = min_time = max_time = median_time = 0
        
        # 按类别统计
        category_stats = {}
        for r in self.all_results:
            cat = r.get("category", "Unknown")
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0}
            category_stats[cat]["total"] += 1
            if r.get("success"):
                category_stats[cat]["passed"] += 1
        
        # 打印统计
        self.log(f"\n总测试数: {total}")
        self.log(f"通过: {passed} ({passed/total*100:.1f}%)")
        self.log(f"失败: {failed}")
        self.log(f"总耗时: {total_time/60:.1f} 分钟")
        
        self.log(f"\n响应时间统计:")
        self.log(f"  平均: {avg_time:.2f} ms")
        self.log(f"  最小: {min_time:.2f} ms")
        self.log(f"  最大: {max_time:.2f} ms")
        self.log(f"  中位数: {median_time:.2f} ms")
        
        self.log(f"\n分类统计:")
        for cat, stats in sorted(category_stats.items()):
            rate = stats["passed"] / stats["total"] * 100
            self.log(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        
        # 保存结果
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "success_rate": passed/total*100 if total > 0 else 0,
                "total_time_seconds": total_time
            },
            "statistics": {
                "response_time": {
                    "avg_ms": avg_time,
                    "min_ms": min_time,
                    "max_ms": max_time,
                    "median_ms": median_time
                },
                "category_breakdown": category_stats
            },
            "all_results": self.all_results
        }
        
        with open("massive_experiment_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        self.log(f"\n结果已保存: massive_experiment_results.json")
        
        # 最终结论
        print("\n" + "="*70)
        if total >= 50:
            print(f"🎉 成功完成 {total} 次独立API测试！")
        else:
            print(f"⚠️ 仅完成 {total} 次测试，未达到50+目标")
        print(f"成功率: {passed/total*100:.1f}%")
        print("="*70)


if __name__ == "__main__":
    experiment = MassiveExperiments()
    results = experiment.run_all_experiments()
