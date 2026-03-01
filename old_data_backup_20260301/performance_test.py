"""
Text2Loc Visionary 快速性能测试脚本

这个脚本提供快速的性能测试，用于验证系统关键性能指标。
包括响应时间、内存使用、并发能力和稳定性测试。
"""

import time
import sys
import os
import numpy as np
from typing import Dict, List, Any, Optional
import threading
import statistics
import psutil
import gc

# 尝试导入增强模块，失败时使用模拟模式
try:
    from enhancements.nlu.engine import NLUEngine, NLUConfig
    from enhancements.vector_db.embedding_client import EmbeddingClient
    from enhancements.integration.adapter import Text2LocAdapter, IntegrationConfig
    from enhancements.integration.config_manager import ConfigManager
    from api.monitoring import SystemMonitor
    MODULES_LOADED = True
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"⚠️  模块导入失败，将使用模拟模式: {e}")
    MODULES_LOADED = False


class PerformanceTester:
    """性能测试器"""

    def __init__(self, mock_mode=True):
        """
        初始化性能测试器

        Args:
            mock_mode: 是否使用模拟模式
        """
        self.mock_mode = mock_mode
        self.results = {}
        self.process = psutil.Process()
        self.modules_loaded = MODULES_LOADED

        if self.modules_loaded:
            print("初始化测试组件...")
            try:
                # 初始化NLU引擎
                nlu_config = NLUConfig(mock_mode=mock_mode)
                self.nlu_engine = NLUEngine(nlu_config)

                # 初始化向量检索客户端
                self.vector_client = EmbeddingClient(mock_mode=mock_mode)

                # 初始化适配器
                self.adapter_config = IntegrationConfig(mock_mode=mock_mode)
                self.adapter = Text2LocAdapter(self.adapter_config)

                # 初始化监控器
                self.monitor = SystemMonitor()

                print("✅ 测试组件初始化完成")
            except Exception as e:
                print(f"❌ 组件初始化失败: {e}")
                self.modules_loaded = False

    def run_test(self, test_name: str, test_func, *args, **kwargs) -> Dict[str, Any]:
        """
        运行单个测试并记录结果

        Args:
            test_name: 测试名称
            test_func: 测试函数
            *args, **kwargs: 测试函数参数

        Returns:
            测试结果字典
        """
        print(f"\n{'='*60}")
        print(f"运行测试: {test_name}")
        print(f"{'='*60}")

        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time

            test_result = {
                "name": test_name,
                "status": "PASSED",
                "duration": duration,
                "data": result,
                "error": None
            }

            print(f"✅ 测试通过 - 用时: {duration:.3f}秒")
            return test_result

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            test_result = {
                "name": test_name,
                "status": "FAILED",
                "duration": duration,
                "data": None,
                "error": str(e)
            }

            print(f"❌ 测试失败 - 用时: {duration:.3f}秒")
            print(f"   错误: {e}")
            return test_result

    def test_nlu_response_time(self) -> Dict[str, Any]:
        """测试NLU响应时间"""
        if not self.modules_loaded:
            return {"status": "SKIPPED", "reason": "模块未加载"}

        test_queries = [
            "红色大楼的北侧",
            "停车场的东边约10米",
            "灯柱的上方",
            "交通标志的前方",
            "建筑物的西南角",
            "蓝色汽车的右边",
            "绿色树木的前方",
            "黄色房子的后面",
            "白色围墙的左侧",
            "黑色屋顶的上方"
        ]

        times = []
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            try:
                result = self.nlu_engine.parse_all(query)
                end_time = time.time()
                query_time = (end_time - start_time) * 1000  # 转换为毫秒
                times.append(query_time)
                print(f"  [{i}/{len(test_queries)}] '{query[:15]}...' - {query_time:.1f}ms")
            except Exception as e:
                print(f"  [{i}/{len(test_queries)}] '{query[:15]}...' - 错误: {e}")
                times.append(1000)  # 错误时使用默认值

        if not times:
            return {"status": "FAILED", "reason": "所有查询都失败"}

        result = {
            "query_count": len(test_queries),
            "avg_time_ms": statistics.mean(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "p95_time_ms": sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times),
            "times_ms": times
        }

        print(f"  NLU平均响应时间: {result['avg_time_ms']:.1f}ms")
        print(f"  NLU最小响应时间: {result['min_time_ms']:.1f}ms")
        print(f"  NLU最大响应时间: {result['max_time_ms']:.1f}ms")

        return result

    def test_vector_embedding_performance(self) -> Dict[str, Any]:
        """测试向量嵌入性能"""
        if not self.modules_loaded:
            return {"status": "SKIPPED", "reason": "模块未加载"}

        # 测试单文本嵌入
        single_text = "这是一个测试文本用于向量嵌入"
        start_time = time.time()
        try:
            embedding = self.vector_client.embed_text(single_text)
            single_time = (time.time() - start_time) * 1000
            print(f"  单文本嵌入: {single_time:.1f}ms")
        except Exception as e:
            print(f"  单文本嵌入失败: {e}")
            single_time = 1000  # 默认值

        # 测试批量嵌入
        batch_texts = [f"测试文本{i}用于批量嵌入性能测试" for i in range(10)]
        start_time = time.time()
        try:
            embeddings = self.vector_client.embed_batch(batch_texts)
            batch_time = (time.time() - start_time) * 1000
            avg_batch_time = batch_time / len(batch_texts)
            print(f"  批量嵌入(10个): {batch_time:.1f}ms")
            print(f"  平均每个: {avg_batch_time:.1f}ms")
        except Exception as e:
            print(f"  批量嵌入失败: {e}")
            batch_time = 10000
            avg_batch_time = 1000

        result = {
            "single_embedding_ms": single_time,
            "batch_embedding_ms": batch_time,
            "avg_per_text_ms": avg_batch_time,
            "embedding_dimension": len(embedding) if 'embedding' in locals() else 0
        }

        return result

    def test_adapter_query_performance(self) -> Dict[str, Any]:
        """测试适配器查询性能"""
        if not self.modules_loaded:
            return {"status": "SKIPPED", "reason": "模块未加载"}

        # 创建测试数据
        query = "我站在红色大楼的北侧约5米处"

        candidates = []
        for i in range(20):
            candidates.append({
                "cell_id": f"cell_{i:03d}",
                "description": f"位置{i}的描述信息",
                "pose_id": f"pose_{i:03d}",
                "object_label": ["building", "parking", "light", "tree", "sign"][i % 5],
                "object_color": ["red", "blue", "green", "yellow", "white"][i % 5],
                "direction": ["north", "south", "east", "west", "above"][i % 5]
            })

        # 执行多次查询
        query_times = []
        for i in range(10):
            start_time = time.time()
            try:
                result = self.adapter.process_query(f"{query}_{i}", candidates)
                query_time = (time.time() - start_time) * 1000
                query_times.append(query_time)

                # 验证结果结构
                if not all(key in result for key in ["query_analysis", "retrieval_results", "final_result"]):
                    print(f"  查询{i}: 结果结构不完整")
            except Exception as e:
                print(f"  查询{i}失败: {e}")
                query_times.append(1000)  # 默认值

        if not query_times:
            return {"status": "FAILED", "reason": "所有查询都失败"}

        result = {
            "query_count": len(query_times),
            "candidate_count": len(candidates),
            "avg_query_time_ms": statistics.mean(query_times),
            "min_query_time_ms": min(query_times),
            "max_query_time_ms": max(query_times),
            "p95_query_time_ms": sorted(query_times)[int(len(query_times) * 0.95)],
            "query_times_ms": query_times
        }

        print(f"  平均查询时间: {result['avg_query_time_ms']:.1f}ms")
        print(f"  P95查询时间: {result['p95_query_time_ms']:.1f}ms")
        print(f"  候选数量: {result['candidate_count']}")

        return result

    def test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用"""
        print("  测量内存使用情况...")

        # 初始内存
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # 创建多个对象
        objects = []
        if self.modules_loaded:
            for i in range(5):
                config = IntegrationConfig(mock_mode=self.mock_mode)
                adapter = Text2LocAdapter(config)
                objects.append(adapter)

        # 使用后内存
        gc.collect()
        after_creation_memory = self.process.memory_info().rss / 1024 / 1024

        # 清理
        del objects
        gc.collect()
        final_memory = self.process.memory_info().rss / 1024 / 1024

        result = {
            "initial_memory_mb": round(initial_memory, 2),
            "after_creation_mb": round(after_creation_memory, 2),
            "final_memory_mb": round(final_memory, 2),
            "memory_increase_mb": round(after_creation_memory - initial_memory, 2),
            "memory_reclaimed_mb": round(after_creation_memory - final_memory, 2)
        }

        print(f"  初始内存: {result['initial_memory_mb']} MB")
        print(f"  创建对象后: {result['after_creation_mb']} MB")
        print(f"  清理后内存: {result['final_memory_mb']} MB")
        print(f"  内存增加: {result['memory_increase_mb']} MB")
        print(f"  内存回收: {result['memory_reclaimed_mb']} MB")

        return result

    def test_concurrent_performance(self, num_threads=10) -> Dict[str, Any]:
        """测试并发性能"""
        if not self.modules_loaded:
            return {"status": "SKIPPED", "reason": "模块未加载"}

        print(f"  测试{num_threads}个并发查询...")

        query = "并发测试查询"
        candidates = [
            {
                "cell_id": "cell_001",
                "description": "测试位置",
                "object_label": "building",
                "object_color": "red",
                "direction": "north"
            }
        ]

        results = []
        errors = []
        lock = threading.Lock()

        def worker(worker_id):
            try:
                start_time = time.time()
                result = self.adapter.process_query(f"{query}_{worker_id}", candidates)
                end_time = time.time()
                query_time = (end_time - start_time) * 1000

                with lock:
                    results.append({
                        "worker_id": worker_id,
                        "query_time": query_time,
                        "success": True
                    })
            except Exception as e:
                with lock:
                    errors.append({
                        "worker_id": worker_id,
                        "error": str(e)
                    })

        # 创建并启动线程
        threads = []
        test_start = time.time()

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        test_end = time.time()
        total_time = (test_end - test_start) * 1000  # 毫秒

        if results:
            query_times = [r["query_time"] for r in results]
            result_data = {
                "thread_count": num_threads,
                "successful_queries": len(results),
                "failed_queries": len(errors),
                "total_time_ms": total_time,
                "avg_query_time_ms": statistics.mean(query_times) if query_times else 0,
                "max_query_time_ms": max(query_times) if query_times else 0,
                "qps": len(results) / (total_time / 1000) if total_time > 0 else 0
            }
        else:
            result_data = {
                "thread_count": num_threads,
                "successful_queries": 0,
                "failed_queries": len(errors),
                "total_time_ms": total_time,
                "avg_query_time_ms": 0,
                "max_query_time_ms": 0,
                "qps": 0
            }

        print(f"  成功查询: {result_data['successful_queries']}")
        print(f"  失败查询: {result_data['failed_queries']}")
        print(f"  总时间: {result_data['total_time_ms']:.1f}ms")
        print(f"  平均查询时间: {result_data['avg_query_time_ms']:.1f}ms")
        print(f"  QPS: {result_data['qps']:.1f}")

        return result_data

    def test_monitoring_performance(self) -> Dict[str, Any]:
        """测试监控系统性能"""
        if not self.modules_loaded:
            return {"status": "SKIPPED", "reason": "模块未加载"}

        print("  测试监控系统...")

        # 记录多个查询
        start_time = time.time()
        query_count = 100

        for i in range(query_count):
            query_id = f"perf_test_{i}"
            duration = np.random.uniform(10, 100)
            success = np.random.random() > 0.1  # 90%成功率

            self.monitor.record_query(query_id, duration, success, {"test": True})

            # 随机记录模块操作
            if i % 10 == 0:
                self.monitor.record_module_operation("nlu", "parse_direction",
                                                    np.random.uniform(5, 20), True)
            if i % 7 == 0:
                self.monitor.record_module_operation("vector_db", "embed_text",
                                                    np.random.uniform(3, 15), True)

        end_time = time.time()
        record_time = (end_time - start_time) * 1000

        # 获取系统状态
        status_start = time.time()
        status = self.monitor.get_system_status()
        status_time = (time.time() - status_start) * 1000

        result = {
            "queries_recorded": query_count,
            "record_time_ms": record_time,
            "avg_record_time_ms": record_time / query_count if query_count > 0 else 0,
            "status_query_time_ms": status_time,
            "system_status": status.get("status", "unknown"),
            "uptime_seconds": status.get("uptime_seconds", 0)
        }

        print(f"  记录{query_count}个查询用时: {record_time:.1f}ms")
        print(f"  平均每个: {result['avg_record_time_ms']:.1f}ms")
        print(f"  状态查询: {status_time:.1f}ms")
        print(f"  系统状态: {result['system_status']}")

        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有性能测试"""
        print("\n" + "="*80)
        print("Text2Loc Visionary 性能测试套件")
        print("="*80)
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模拟模式: {self.mock_mode}")
        print(f"模块加载: {'成功' if self.modules_loaded else '失败'}")

        if not self.modules_loaded:
            print("⚠️  部分测试将跳过，因为模块未加载")

        overall_start = time.time()

        # 定义要运行的测试
        tests = [
            ("NLU响应时间测试", self.test_nlu_response_time),
            ("向量嵌入性能测试", self.test_vector_embedding_performance),
            ("适配器查询性能测试", self.test_adapter_query_performance),
            ("内存使用测试", self.test_memory_usage),
            ("并发性能测试", self.test_concurrent_performance),
            ("监控系统性能测试", self.test_monitoring_performance),
        ]

        # 运行测试
        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            self.results[test_name] = result

        overall_end = time.time()
        total_time = overall_end - overall_start

        # 生成摘要
        summary = self.generate_summary(total_time)

        print("\n" + "="*80)
        print("性能测试完成")
        print("="*80)
        self.print_summary(summary)

        return {
            "summary": summary,
            "detailed_results": self.results,
            "total_time": total_time
        }

    def generate_summary(self, total_time: float) -> Dict[str, Any]:
        """生成测试摘要"""
        passed = sum(1 for r in self.results.values() if r["status"] == "PASSED")
        failed = sum(1 for r in self.results.values() if r["status"] == "FAILED")
        skipped = sum(1 for r in self.results.values() if r["status"] == "SKIPPED")

        # 提取关键性能指标
        key_metrics = {}

        if "NLU响应时间测试" in self.results and self.results["NLU响应时间测试"]["status"] == "PASSED":
            nlu_data = self.results["NLU响应时间测试"]["data"]
            if isinstance(nlu_data, dict) and "avg_time_ms" in nlu_data:
                key_metrics["nlu_avg_response_ms"] = nlu_data["avg_time_ms"]

        if "适配器查询性能测试" in self.results and self.results["适配器查询性能测试"]["status"] == "PASSED":
            adapter_data = self.results["适配器查询性能测试"]["data"]
            if isinstance(adapter_data, dict) and "avg_query_time_ms" in adapter_data:
                key_metrics["adapter_avg_query_ms"] = adapter_data["avg_query_time_ms"]

        if "并发性能测试" in self.results and self.results["并发性能测试"]["status"] == "PASSED":
            concurrent_data = self.results["并发性能测试"]["data"]
            if isinstance(concurrent_data, dict) and "qps" in concurrent_data:
                key_metrics["concurrent_qps"] = concurrent_data["qps"]

        summary = {
            "total_tests": len(self.results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "success_rate": passed / len(self.results) if self.results else 0,
            "total_time_seconds": total_time,
            "key_metrics": key_metrics,
            "test_environment": {
                "mock_mode": self.mock_mode,
                "modules_loaded": self.modules_loaded,
                "python_version": sys.version,
                "platform": sys.platform
            }
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """打印测试摘要"""
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过: {summary['passed']}")
        print(f"失败: {summary['failed']}")
        print(f"跳过: {summary['skipped']}")
        print(f"成功率: {summary['success_rate']:.1%}")
        print(f"总用时: {summary['total_time_seconds']:.2f}秒")

        if summary['key_metrics']:
            print("\n关键性能指标:")
            for metric, value in summary['key_metrics'].items():
                if "ms" in metric:
                    print(f"  {metric}: {value:.1f}ms")
                elif "qps" in metric:
                    print(f"  {metric}: {value:.1f}")

        # 性能建议
        print("\n性能建议:")
        if summary['key_metrics'].get('nlu_avg_response_ms', 0) > 200:
            print("  ⚠️  NLU响应时间较高，考虑优化模型或使用缓存")
        else:
            print("  ✅ NLU响应时间良好")

        if summary['key_metrics'].get('adapter_avg_query_ms', 0) > 150:
            print("  ⚠️  适配器查询时间较高，考虑优化检索算法")
        else:
            print("  ✅ 适配器查询时间良好")

        if summary['key_metrics'].get('concurrent_qps', 0) < 50:
            print("  ⚠️  并发处理能力较低，考虑优化线程池或批处理")
        else:
            print("  ✅ 并发处理能力良好")

    def save_results(self, filename: str = "performance_test_results.json"):
        """保存测试结果到文件"""
        import json

        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": self.generate_summary(0),
            "detailed_results": self.results
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\n测试结果已保存到: {filename}")
        except Exception as e:
            print(f"\n保存测试结果失败: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Text2Loc Visionary 性能测试脚本")
    parser.add_argument("--mock", action="store_true", default=True,
                       help="使用模拟模式（默认：True）")
    parser.add_argument("--real", action="store_false", dest="mock",
                       help="使用真实模式（需要API服务）")
    parser.add_argument("--output", type=str, default="performance_test_results.json",
                       help="测试结果输出文件（默认：performance_test_results.json）")
    parser.add_argument("--quick", action="store_true",
                       help="快速测试模式（减少测试次数）")

    args = parser.parse_args()

    print("初始化性能测试器...")
    tester = PerformanceTester(mock_mode=args.mock)

    if args.quick:
        print("启用快速测试模式")
        # 在快速模式下，减少测试次数
        tester.test_concurrent_performance = lambda *args, **kwargs: tester.test_concurrent_performance(num_threads=5)

    # 运行测试
    results = tester.run_all_tests()

    # 保存结果
    tester.save_results(args.output)

    # 返回退出码
    if results["summary"]["success_rate"] < 0.7:
        print("\n⚠️  测试通过率低于70%，建议检查系统")
        return 1
    else:
        print("\n✅ 性能测试完成")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
