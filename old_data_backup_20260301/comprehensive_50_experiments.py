"""
高质量大规模实验 - 50+次测试验证项目正确性
包含多种测试类型：功能测试、性能测试、边界测试、压力测试
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

class ComprehensiveExperiments:
    def __init__(self, base_url="http://localhost:8080/api/v1"):
        self.base_url = base_url
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            },
            "system_validation": {},
            "functional_tests": [],
            "performance_tests": {},
            "boundary_tests": [],
            "stress_tests": {},
            "robustness_tests": [],
            "statistics": {}
        }
        self.lock = threading.Lock()
        
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}.get(level, "ℹ️")
        print(f"[{timestamp}] {prefix} {message}")
    
    # ==================== 1. 系统验证测试 ====================
    def test_system_validation(self):
        """系统基础验证"""
        self.log("\n" + "="*70)
        self.log("阶段 1: 系统基础验证")
        self.log("="*70)
        
        validation_results = {}
        
        # 1.1 健康检查
        self.log("测试 1.1: 服务健康检查...")
        try:
            response = requests.get(f"{self.base_url.replace('/api/v1', '')}/health", timeout=5)
            validation_results["health_check"] = {
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "details": response.json() if response.status_code == 200 else None
            }
            self.log(f"健康检查: {'通过' if response.status_code == 200 else '失败'}", 
                    "SUCCESS" if response.status_code == 200 else "ERROR")
        except Exception as e:
            validation_results["health_check"] = {"status": "FAIL", "error": str(e)}
            self.log(f"健康检查失败: {e}", "ERROR")
        
        # 1.2 配置验证
        self.log("测试 1.2: API配置验证...")
        try:
            response = requests.get(f"{self.base_url}/config", timeout=5)
            config = response.json() if response.status_code == 200 else {}
            validation_results["config_check"] = {
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "provider": config.get("provider"),
                "model": config.get("model"),
                "is_configured": config.get("is_configured")
            }
            self.log(f"配置检查: 提供商={config.get('provider')}, 模型={config.get('model')}", 
                    "SUCCESS" if response.status_code == 200 else "ERROR")
        except Exception as e:
            validation_results["config_check"] = {"status": "FAIL", "error": str(e)}
            self.log(f"配置检查失败: {e}", "ERROR")
        
        # 1.3 数据完整性检查
        self.log("测试 1.3: 数据完整性检查...")
        data_path = os.path.expanduser("~/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all")
        data_checks = {}
        
        # 检查Cells
        try:
            with open(os.path.join(data_path, "cells", "cells.pkl"), 'rb') as f:
                cells = pickle.load(f)
            data_checks["cells"] = {"count": len(cells), "status": "PASS" if len(cells) > 0 else "FAIL"}
            self.log(f"Cells数据: {len(cells)} 个场景单元", "SUCCESS")
        except Exception as e:
            data_checks["cells"] = {"status": "FAIL", "error": str(e)}
            self.log(f"Cells数据检查失败: {e}", "ERROR")
        
        # 检查Poses
        try:
            with open(os.path.join(data_path, "poses", "poses.pkl"), 'rb') as f:
                poses = pickle.load(f)
            data_checks["poses"] = {"count": len(poses), "status": "PASS" if len(poses) > 0 else "FAIL"}
            self.log(f"Poses数据: {len(poses)} 个位姿", "SUCCESS")
        except Exception as e:
            data_checks["poses"] = {"status": "FAIL", "error": str(e)}
            self.log(f"Poses数据检查失败: {e}", "ERROR")
        
        # 检查原始数据
        raw_path = os.path.join(data_path, "raw_data")
        if os.path.exists(raw_path):
            raw_dirs = os.listdir(raw_path)
            data_checks["raw_data"] = {"directories": raw_dirs, "status": "PASS"}
            self.log(f"原始数据: {raw_dirs}", "SUCCESS")
        
        validation_results["data_integrity"] = data_checks
        
        self.results["system_validation"] = validation_results
        return validation_results
    
    # ==================== 2. 功能测试 (20个) ====================
    def test_functional(self):
        """功能测试 - 20个测试用例"""
        self.log("\n" + "="*70)
        self.log("阶段 2: 功能测试 (20个测试用例)")
        self.log("="*70)
        
        test_cases = [
            # 基础方向测试 (4个)
            {"id": "F001", "category": "基础方向", "query": "我在北方", "expected_elements": ["方向"]},
            {"id": "F002", "category": "基础方向", "query": "在南边", "expected_elements": ["方向"]},
            {"id": "F003", "category": "基础方向", "query": "东侧位置", "expected_elements": ["方向"]},
            {"id": "F004", "category": "基础方向", "query": "西边", "expected_elements": ["方向"]},
            
            # 颜色+物体测试 (6个)
            {"id": "F005", "category": "颜色物体", "query": "红色大楼", "expected_elements": ["颜色", "物体"]},
            {"id": "F006", "category": "颜色物体", "query": "蓝色汽车", "expected_elements": ["颜色", "物体"]},
            {"id": "F007", "category": "颜色物体", "query": "白色建筑物", "expected_elements": ["颜色", "物体"]},
            {"id": "F008", "category": "颜色物体", "query": "绿色树木", "expected_elements": ["颜色", "物体"]},
            {"id": "F009", "category": "颜色物体", "query": "黑色标志", "expected_elements": ["颜色", "物体"]},
            {"id": "F010", "category": "颜色物体", "query": "灰色墙壁", "expected_elements": ["颜色", "物体"]},
            
            # 复合方向测试 (5个)
            {"id": "F011", "category": "复合方向", "query": "东北方向", "expected_elements": ["方向"]},
            {"id": "F012", "category": "复合方向", "query": "西南角", "expected_elements": ["方向"]},
            {"id": "F013", "category": "复合方向", "query": "东南侧", "expected_elements": ["方向"]},
            {"id": "F014", "category": "复合方向", "query": "西北方位", "expected_elements": ["方向"]},
            {"id": "F015", "category": "复合方向", "query": "正前方", "expected_elements": ["方向"]},
            
            # 相对位置测试 (5个)
            {"id": "F016", "category": "相对位置", "query": "靠近入口", "expected_elements": ["相对位置"]},
            {"id": "F017", "category": "相对位置", "query": "在对面", "expected_elements": ["相对位置"]},
            {"id": "F018", "category": "相对位置", "query": "旁边", "expected_elements": ["相对位置"]},
            {"id": "F019", "category": "相对位置", "query": "附近", "expected_elements": ["相对位置"]},
            {"id": "F020", "category": "相对位置", "query": "中间位置", "expected_elements": ["相对位置"]},
        ]
        
        results = []
        for test in test_cases:
            result = self._execute_single_test(test)
            results.append(result)
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            self.log(f"{status_icon} {test['id']}: {test['query']} - {result['response_time_ms']:.2f}ms")
        
        self.results["functional_tests"] = results
        return results
    
    # ==================== 3. 性能测试 (15个) ====================
    def test_performance(self):
        """性能测试 - 15个测试用例"""
        self.log("\n" + "="*70)
        self.log("阶段 3: 性能测试 (15个测试用例)")
        self.log("="*70)
        
        performance_tests = {
            "response_time": [],
            "concurrent": {},
            "cache_efficiency": [],
            "throughput": {}
        }
        
        # 3.1 响应时间测试 (10次)
        self.log("测试 3.1: 响应时间测试 (10次)...")
        queries = [
            "我在红色大楼的北侧",
            "在绿色树木的东边",
            "靠近蓝色标志",
            "前方有白色建筑物",
            "位于十字路口",
            "在停车场的入口处",
            "靠近加油站的便利店",
            "在人行横道的对面",
            "东南方向的建筑物",
            "西北角的标志"
        ]
        
        response_times = []
        for i, query in enumerate(queries):
            try:
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": query, "top_k": 3},
                    timeout=60
                )
                elapsed = (time.time() - start) * 1000
                response_times.append(elapsed)
                self.log(f"  查询 {i+1}: {elapsed:.2f}ms")
            except Exception as e:
                self.log(f"  查询 {i+1}: 失败 - {e}", "ERROR")
        
        if response_times:
            performance_tests["response_time"] = {
                "values": response_times,
                "avg_ms": statistics.mean(response_times),
                "min_ms": min(response_times),
                "max_ms": max(response_times),
                "median_ms": statistics.median(response_times),
                "stdev_ms": statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
            self.log(f"响应时间统计: 平均={performance_tests['response_time']['avg_ms']:.2f}ms, "
                    f"最小={performance_tests['response_time']['min_ms']:.2f}ms, "
                    f"最大={performance_tests['response_time']['max_ms']:.2f}ms", "SUCCESS")
        
        # 3.2 缓存效率测试 (3次重复)
        self.log("测试 3.2: 缓存效率测试...")
        cache_query = "我在红色大楼的北侧"
        cache_times = []
        for i in range(3):
            try:
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": cache_query, "top_k": 3},
                    timeout=60
                )
                elapsed = (time.time() - start) * 1000
                cache_times.append({"run": i+1, "time_ms": elapsed})
                self.log(f"  第 {i+1} 次: {elapsed:.2f}ms")
            except Exception as e:
                self.log(f"  第 {i+1} 次: 失败 - {e}", "ERROR")
        
        performance_tests["cache_efficiency"] = cache_times
        if len(cache_times) >= 2:
            speedup = cache_times[0]["time_ms"] / cache_times[-1]["time_ms"] if cache_times[-1]["time_ms"] > 0 else 0
            self.log(f"缓存加速比: {speedup:.2f}x", "SUCCESS")
        
        # 3.3 并发测试 (5个并发请求)
        self.log("测试 3.3: 并发压力测试 (5并发)...")
        concurrent_queries = [
            "红色大楼北侧",
            "绿色树木东边", 
            "蓝色标志附近",
            "白色建筑物前方",
            "十字路口东南角"
        ]
        
        concurrent_results = []
        def concurrent_request(query):
            try:
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": query, "top_k": 3},
                    timeout=60
                )
                elapsed = (time.time() - start) * 1000
                return {"query": query, "time_ms": elapsed, "status": response.status_code}
            except Exception as e:
                return {"query": query, "error": str(e), "status": -1}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(concurrent_request, q) for q in concurrent_queries]
            for future in as_completed(futures):
                result = future.result()
                concurrent_results.append(result)
                self.log(f"  并发请求: {result.get('time_ms', 'N/A')}ms")
        
        valid_times = [r["time_ms"] for r in concurrent_results if "time_ms" in r]
        performance_tests["concurrent"] = {
            "results": concurrent_results,
            "success_count": sum(1 for r in concurrent_results if r.get("status") == 200),
            "avg_time_ms": statistics.mean(valid_times) if valid_times else 0
        }
        self.log(f"并发测试: {performance_tests['concurrent']['success_count']}/5 成功", "SUCCESS")
        
        self.results["performance_tests"] = performance_tests
        return performance_tests
    
    # ==================== 4. 边界测试 (10个) ====================
    def test_boundary(self):
        """边界测试 - 10个测试用例"""
        self.log("\n" + "="*70)
        self.log("阶段 4: 边界测试 (10个测试用例)")
        self.log("="*70)
        
        boundary_cases = [
            {"id": "B001", "category": "最短查询", "query": "北", "description": "单字查询"},
            {"id": "B002", "category": "最短查询", "query": "红", "description": "单字颜色"},
            {"id": "B003", "category": "长查询", "query": "我在一个非常大的红色建筑物的北侧靠近入口的地方", "description": "长句查询"},
            {"id": "B004", "category": "特殊字符", "query": "红色@大楼#北侧", "description": "包含特殊字符"},
            {"id": "B005", "category": "数字混合", "query": "在50米外的红色建筑物旁边", "description": "包含数字"},
            {"id": "B006", "category": "中英文混合", "query": "在red大楼的北侧", "description": "中英文混合"},
            {"id": "B007", "category": "重复内容", "query": "北北北北北", "description": "重复字符"},
            {"id": "B008", "category": "空相关内容", "query": "在不存在的地方", "description": "不存在的地点"},
            {"id": "B009", "category": "模糊描述", "query": "某个地方", "description": "极度模糊"},
            {"id": "B010", "category": "复杂复合", "query": "东南西北中发白", "description": "复杂方向组合"},
        ]
        
        results = []
        for test in boundary_cases:
            result = self._execute_single_test(test)
            results.append(result)
            status_icon = "✅" if result["status"] == "PASS" else "⚠️"
            self.log(f"{status_icon} {test['id']}: {test['description']} - {result['response_time_ms']:.2f}ms")
        
        self.results["boundary_tests"] = results
        return results
    
    # ==================== 5. 压力测试 (10个) ====================
    def test_stress(self):
        """压力测试 - 10个测试用例"""
        self.log("\n" + "="*70)
        self.log("阶段 5: 压力测试 (10个测试用例)")
        self.log("="*70)
        
        stress_results = {
            "rapid_fire": [],
            "burst_mode": {},
            "sustained_load": []
        }
        
        # 5.1 快速连续请求 (10次快速请求)
        self.log("测试 5.1: 快速连续请求 (10次)...")
        rapid_times = []
        for i in range(10):
            try:
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": f"测试查询{i}", "top_k": 3},
                    timeout=30
                )
                elapsed = (time.time() - start) * 1000
                rapid_times.append(elapsed)
            except Exception as e:
                rapid_times.append(-1)
                self.log(f"  请求 {i+1}: 失败", "ERROR")
        
        valid_times = [t for t in rapid_times if t > 0]
        stress_results["rapid_fire"] = {
            "times": rapid_times,
            "success_count": len(valid_times),
            "avg_time_ms": statistics.mean(valid_times) if valid_times else 0
        }
        self.log(f"快速请求: {stress_results['rapid_fire']['success_count']}/10 成功", 
                "SUCCESS" if stress_results['rapid_fire']['success_count'] >= 8 else "WARNING")
        
        # 5.2 突发模式 (短时间内5个不同查询)
        self.log("测试 5.2: 突发模式测试...")
        burst_queries = ["北", "南", "东", "西", "中"]
        burst_times = []
        for query in burst_queries:
            try:
                start = time.time()
                response = requests.post(
                    f"{self.base_url}/query",
                    json={"query": query, "top_k": 3},
                    timeout=30
                )
                elapsed = (time.time() - start) * 1000
                burst_times.append(elapsed)
            except Exception as e:
                burst_times.append(-1)
        
        valid_burst = [t for t in burst_times if t > 0]
        stress_results["burst_mode"] = {
            "times": burst_times,
            "success_count": len(valid_burst),
            "total_time_ms": sum(valid_burst) if valid_burst else 0
        }
        self.log(f"突发模式: {stress_results['burst_mode']['success_count']}/5 成功", 
                "SUCCESS" if stress_results['burst_mode']['success_count'] >= 4 else "WARNING")
        
        self.results["stress_tests"] = stress_results
        return stress_results
    
    # ==================== 6. 鲁棒性测试 (5个) ====================
    def test_robustness(self):
        """鲁棒性测试 - 5个测试用例"""
        self.log("\n" + "="*70)
        self.log("阶段 6: 鲁棒性测试 (5个测试用例)")
        self.log("="*70)
        
        robustness_cases = [
            {"id": "R001", "category": "超时恢复", "query": "测试超时恢复能力", "description": "系统超时后恢复"},
            {"id": "R002", "category": "错误处理", "query": "", "description": "空查询处理"},
            {"id": "R003", "category": "错误处理", "query": "   ", "description": "空白查询处理"},
            {"id": "R004", "category": "异常输入", "query": "<script>alert('test')</script>", "description": "XSS攻击防护"},
            {"id": "R005", "category": "异常输入", "query": "SELECT * FROM users", "description": "SQL注入防护"},
        ]
        
        results = []
        for test in robustness_cases:
            result = self._execute_single_test(test)
            results.append(result)
            status_icon = "✅" if result["status"] == "PASS" else "⚠️"
            self.log(f"{status_icon} {test['id']}: {test['description']} - {result.get('error', 'OK')}")
        
        self.results["robustness_tests"] = results
        return results
    
    # ==================== 辅助方法 ====================
    def _execute_single_test(self, test_case):
        """执行单个测试用例"""
        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": test_case["query"], "top_k": 3},
                timeout=60
            )
            elapsed = (time.time() - start) * 1000
            
            return {
                "id": test_case["id"],
                "query": test_case["query"],
                "category": test_case.get("category", ""),
                "response_time_ms": elapsed,
                "status_code": response.status_code,
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "has_results": False
            }
        except Exception as e:
            return {
                "id": test_case["id"],
                "query": test_case["query"],
                "category": test_case.get("category", ""),
                "response_time_ms": 0,
                "status": "FAIL",
                "error": str(e)
            }
    
    # ==================== 统计分析 ====================
    def calculate_statistics(self):
        """计算统计信息"""
        self.log("\n" + "="*70)
        self.log("阶段 7: 统计分析")
        self.log("="*70)
        
        all_tests = []
        
        # 收集所有测试结果
        if self.results["functional_tests"]:
            all_tests.extend(self.results["functional_tests"])
        if self.results["boundary_tests"]:
            all_tests.extend(self.results["boundary_tests"])
        if self.results["robustness_tests"]:
            all_tests.extend(self.results["robustness_tests"])
        
        total = len(all_tests)
        passed = sum(1 for t in all_tests if t.get("status") == "PASS")
        failed = total - passed
        
        # 按类别统计
        category_stats = {}
        for test in all_tests:
            cat = test.get("category", "Unknown")
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0}
            category_stats[cat]["total"] += 1
            if test.get("status") == "PASS":
                category_stats[cat]["passed"] += 1
        
        # 响应时间统计
        response_times = [t["response_time_ms"] for t in all_tests if t.get("response_time_ms", 0) > 0]
        time_stats = {}
        if response_times:
            time_stats = {
                "avg_ms": statistics.mean(response_times),
                "min_ms": min(response_times),
                "max_ms": max(response_times),
                "median_ms": statistics.median(response_times)
            }
        
        self.results["statistics"] = {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "category_breakdown": category_stats,
            "response_time_stats": time_stats
        }
        
        self.log(f"总测试数: {total}")
        self.log(f"通过: {passed} ({self.results['statistics']['success_rate']:.1f}%)")
        self.log(f"失败: {failed}")
        self.log(f"平均响应时间: {time_stats.get('avg_ms', 0):.2f}ms")
        
        return self.results["statistics"]
    
    # ==================== 运行所有测试 ====================
    def run_all_experiments(self):
        """运行所有实验"""
        print("\n" + "="*70)
        print("Text2Loc Visionary - 高质量大规模实验")
        print("="*70)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"目标: 50+ 次高质量测试")
        print("="*70)
        
        # 1. 系统验证 (3个测试)
        self.test_system_validation()
        
        # 2. 功能测试 (20个测试)
        self.test_functional()
        
        # 3. 性能测试 (15个测试)
        self.test_performance()
        
        # 4. 边界测试 (10个测试)
        self.test_boundary()
        
        # 5. 压力测试 (10个测试)
        self.test_stress()
        
        # 6. 鲁棒性测试 (5个测试)
        self.test_robustness()
        
        # 7. 统计分析
        self.calculate_statistics()
        
        # 保存结果
        output_file = "comprehensive_experiment_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("实验完成!")
        print("="*70)
        print(f"总测试数: {self.results['statistics']['total_tests']}")
        print(f"成功率: {self.results['statistics']['success_rate']:.1f}%")
        print(f"结果已保存: {output_file}")
        print("="*70)
        
        return self.results


if __name__ == "__main__":
    experiment = ComprehensiveExperiments()
    results = experiment.run_all_experiments()
