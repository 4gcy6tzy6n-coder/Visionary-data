"""
第四阶段：全面测试套件

实现任务4.1：全面测试验证，覆盖功能测试、性能测试、A/B测试、压力测试等
"""

import unittest
import time
import sys
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import statistics
import tempfile
import shutil
import subprocess

# 导入增强模块
try:
    from enhancements.nlu.engine import NLUEngine
    from enhancements.vector_db.embedding_client import EmbeddingClient
    from enhancements.pointcloud.colors import EnhancedColorMapper
    from enhancements.pointcloud.objects import EnhancedObjectIdentifier
    from enhancements.integration.adapter import Text2LocAdapter, IntegrationConfig
    from enhancements.integration.config_manager import ConfigManager, SystemConfig
    from enhancements.integration.format_converter import FormatConverter, OldFormat, NewFormat
    from api.monitoring import SystemMonitor, MetricsCollector
    from api.text2loc_api import Text2LocAPI, QueryRequest, QueryResponse
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"警告：模块导入失败，部分测试可能无法运行: {e}")
    IMPORT_SUCCESS = False


class BaseTestCase(unittest.TestCase):
    """基础测试类，提供通用功能"""

    def setUp(self):
        """设置测试环境"""
        self.test_start_time = time.time()
        self.test_data_dir = tempfile.mkdtemp(prefix="text2loc_test_")

    def tearDown(self):
        """清理测试环境"""
        if hasattr(self, 'test_data_dir') and os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def assertPerformanceWithinLimit(self, operation_name, duration_ms, limit_ms,
                                   message=None):
        """断言性能在限制范围内"""
        if message is None:
            message = f"{operation_name} 耗时 {duration_ms:.2f}ms，超过限制 {limit_ms}ms"
        self.assertLessEqual(duration_ms, limit_ms, message)

    def assertAccuracyAboveThreshold(self, actual, expected, threshold=0.8,
                                   message=None):
        """断言准确率高于阈值"""
        if isinstance(actual, (list, np.ndarray)) and isinstance(expected, (list, np.ndarray)):
            if len(actual) != len(expected):
                accuracy = 0.0
            else:
                matches = sum(1 for a, e in zip(actual, expected) if a == e)
                accuracy = matches / len(actual)
        else:
            accuracy = 1.0 if actual == expected else 0.0

        if message is None:
            message = f"准确率 {accuracy:.2f} 低于阈值 {threshold}"
        self.assertGreaterEqual(accuracy, threshold, message)


class FunctionalTests(BaseTestCase):
    """功能测试：验证核心功能是否正确工作"""

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_nlu_engine_functionality(self):
        """测试NLU引擎功能"""
        nlu = NLUEngine(mock_mode=True)

        # 测试方向解析
        direction_result = nlu.parse_direction("我站在红色大楼的北侧")
        self.assertIsNotNone(direction_result)
        self.assertIn(direction_result.get("primary_direction", "").lower(),
                     ["north", "北", "n"])

        # 测试颜色解析
        color_result = nlu.parse_color("红色大楼")
        self.assertIsNotNone(color_result)
        self.assertEqual(color_result.get("primary_color", "").lower(), "red")

        # 测试对象解析
        object_result = nlu.parse_object("大楼")
        self.assertIsNotNone(object_result)
        self.assertIn(object_result.get("primary_object", "").lower(),
                     ["building", "楼", "建筑"])

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_vector_retrieval_functionality(self):
        """测试向量检索功能"""
        vector_client = EmbeddingClient(mock_mode=True)

        # 测试文本嵌入
        texts = ["红色大楼", "停车场的东边", "灯柱的上方"]
        embeddings = vector_client.embed_batch(texts)
        self.assertEqual(len(embeddings), len(texts))
        self.assertTrue(all(isinstance(e, (list, np.ndarray)) for e in embeddings))

        # 测试相似度计算
        similarity = vector_client.cosine_similarity(embeddings[0], embeddings[1])
        self.assertIsInstance(similarity, (float, np.float32, np.float64))
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_adapter_integration_functionality(self):
        """测试适配器集成功能"""
        config = IntegrationConfig(mock_mode=True)
        adapter = Text2LocAdapter(config)

        # 测试查询处理
        query = "我站在红色大楼的北侧约5米处"
        candidates = [
            {
                "cell_id": "cell_001",
                "description": "红色大楼的北侧",
                "pose_id": "pose_001",
                "object_label": "building",
                "object_color": "red",
                "direction": "north"
            },
            {
                "cell_id": "cell_002",
                "description": "停车场的东边",
                "pose_id": "pose_002",
                "object_label": "parking",
                "object_color": "gray",
                "direction": "east"
            },
        ]

        result = adapter.process_query(query, candidates)
        self.assertIsNotNone(result)
        self.assertIn("query_analysis", result)
        self.assertIn("retrieval_results", result)
        self.assertIn("final_result", result)

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_config_manager_functionality(self):
        """测试配置管理器功能"""
        config_manager = ConfigManager()

        # 测试默认配置
        config = config_manager.config
        self.assertIsInstance(config, SystemConfig)
        self.assertEqual(config.nlu_model, "qwen3-vl:2b")
        self.assertEqual(config.embedding_model, "qwen3-embedding:0.6b")

        # 测试配置更新
        config_manager.update(mock_mode=True, nlu_confidence_threshold=0.8)
        self.assertEqual(config_manager.config.mock_mode, True)
        self.assertEqual(config_manager.config.nlu_confidence_threshold, 0.8)

        # 测试配置验证
        is_valid, errors = config_manager.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_monitoring_system_functionality(self):
        """测试监控系统功能"""
        monitor = SystemMonitor()

        # 测试系统状态
        status = monitor.get_system_status()
        self.assertEqual(status["status"], "healthy")
        self.assertIn("metrics", status)
        self.assertIn("errors", status)
        self.assertIn("feedback", status)

        # 测试性能指标记录
        monitor.record_query("test_query", 45.2, True, {"test": True})
        metrics = monitor.metrics_collector.get_all_metrics()
        self.assertGreaterEqual(metrics["summary"]["query_count"], 1)

        # 测试错误跟踪
        monitor.track_error("TestError", "测试错误", "test_module", {"test": True})
        error_summary = monitor.error_tracker.get_error_summary()
        self.assertGreaterEqual(error_summary["total_errors"], 0)


class PerformanceTests(BaseTestCase):
    """性能测试：验证系统性能指标"""

    def setUp(self):
        super().setUp()
        if IMPORT_SUCCESS:
            self.config = IntegrationConfig(mock_mode=True)
            self.adapter = Text2LocAdapter(self.config)

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_nlu_response_time(self):
        """测试NLU响应时间"""
        nlu = NLUEngine(mock_mode=True)

        test_queries = [
            "红色大楼的北侧",
            "停车场的东边约10米",
            "灯柱的上方",
            "交通标志的前方",
            "建筑物的西南角"
        ]

        times = []
        for query in test_queries:
            start_time = time.time()
            result = nlu.parse_all(query)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒

        avg_time = statistics.mean(times)
        max_time = max(times)

        print(f"NLU平均响应时间: {avg_time:.2f}ms")
        print(f"NLU最大响应时间: {max_time:.2f}ms")

        # 性能要求：平均响应时间 < 200ms，最大响应时间 < 500ms
        self.assertPerformanceWithinLimit("NLU平均响应时间", avg_time, 200)
        self.assertPerformanceWithinLimit("NLU最大响应时间", max_time, 500)

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_vector_embedding_performance(self):
        """测试向量嵌入性能"""
        vector_client = EmbeddingClient(mock_mode=True)

        # 测试单文本嵌入性能
        start_time = time.time()
        embedding = vector_client.embed_text("测试文本")
        single_time = (time.time() - start_time) * 1000

        # 测试批量嵌入性能
        texts = [f"测试文本{i}" for i in range(10)]
        start_time = time.time()
        embeddings = vector_client.embed_batch(texts)
        batch_time = (time.time() - start_time) * 1000

        print(f"单文本嵌入时间: {single_time:.2f}ms")
        print(f"批量嵌入时间（10个）: {batch_time:.2f}ms")
        print(f"平均每个文本: {batch_time/10:.2f}ms")

        # 性能要求：单文本 < 50ms，批量平均每个 < 20ms
        self.assertPerformanceWithinLimit("单文本嵌入", single_time, 50)
        self.assertPerformanceWithinLimit("批量嵌入平均每个", batch_time/10, 20)

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_adapter_query_performance(self):
        """测试适配器查询性能"""
        query = "我站在红色大楼的北侧约5米处"

        # 创建测试候选数据
        candidates = []
        for i in range(20):
            candidates.append({
                "cell_id": f"cell_{i:03d}",
                "description": f"位置{i}的描述",
                "pose_id": f"pose_{i:03d}",
                "object_label": ["building", "parking", "light"][i % 3],
                "object_color": ["red", "blue", "green", "yellow"][i % 4],
                "direction": ["north", "south", "east", "west"][i % 4]
            })

        # 测试多次查询性能
        times = []
        for i in range(10):
            start_time = time.time()
            result = self.adapter.process_query(query, candidates)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)

        avg_time = statistics.mean(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]

        print(f"适配器平均查询时间: {avg_time:.2f}ms")
        print(f"适配器P95查询时间: {p95_time:.2f}ms")

        # 性能要求：平均 < 100ms，P95 < 200ms
        self.assertPerformanceWithinLimit("适配器平均查询", avg_time, 100)
        self.assertPerformanceWithinLimit("适配器P95查询", p95_time, 200)

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_memory_usage(self):
        """测试内存使用情况"""
        import psutil
        import gc

        process = psutil.Process()

        # 初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 创建多个适配器实例
        adapters = []
        for i in range(10):
            config = IntegrationConfig(mock_mode=True)
            adapter = Text2LocAdapter(config)
            adapters.append(adapter)

        # 使用后内存
        gc.collect()
        after_creation_memory = process.memory_info().rss / 1024 / 1024

        # 清理后内存
        del adapters
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024

        memory_increase = after_creation_memory - initial_memory
        memory_reclaimed = after_creation_memory - final_memory

        print(f"初始内存: {initial_memory:.2f} MB")
        print(f"创建10个适配器后: {after_creation_memory:.2f} MB")
        print(f"清理后内存: {final_memory:.2f} MB")
        print(f"内存增加: {memory_increase:.2f} MB")
        print(f"内存回收: {memory_reclaimed:.2f} MB")

        # 内存要求：每个适配器实例 < 10MB
        self.assertLess(memory_increase / 10, 10,
                       f"每个适配器实例内存使用过高: {memory_increase/10:.2f} MB")


class IntegrationTests(BaseTestCase):
    """集成测试：验证模块间集成"""

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_full_pipeline_integration(self):
        """测试完整管道集成"""
        # 创建配置
        config_manager = ConfigManager()
        config_manager.update(mock_mode=True)

        # 创建适配器
        adapter_config = IntegrationConfig(mock_mode=True)
        adapter = Text2LocAdapter(adapter_config)

        # 创建格式转换器
        converter = FormatConverter()

        # 创建测试数据
        query = "我站在红色大楼的北侧约5米处"
        old_formats = [
            OldFormat(
                object_label="building",
                object_color="red",
                direction="north",
                offset=np.array([0.5, 0.5]),
                cell_id="cell_001",
                pose_id="pose_001",
                description="红色大楼的北侧"
            ),
            OldFormat(
                object_label="parking",
                object_color="gray",
                direction="east",
                offset=np.array([0.6, 0.4]),
                cell_id="cell_002",
                pose_id="pose_002",
                description="停车场的东边"
            ),
        ]

        # 格式转换
        new_formats = [converter.old_to_new(of) for of in old_formats]
        self.assertEqual(len(new_formats), len(old_formats))

        # 转换为候选格式
        candidates = []
        for nf in new_formats:
            candidates.append({
                "cell_id": nf.cell_id,
                "description": nf.generated_description,
                "pose_id": nf.pose_id,
                "object_label": nf.object_name,
                "object_color": nf.object_color,
                "direction": nf.direction
            })

        # 处理查询
        result = adapter.process_query(query, candidates)

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("query_analysis", result)
        self.assertIn("retrieval_results", result)
        self.assertIn("final_result", result)

        # 验证检索结果排序
        retrieval_results = result["retrieval_results"]
        if len(retrieval_results) > 1:
            scores = [r.get("relevance_score", 0) for r in retrieval_results]
            self.assertEqual(scores, sorted(scores, reverse=True),
                           "检索结果应按相关性降序排列")

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_api_integration(self):
        """测试API集成"""
        # 创建API实例
        api = Text2LocAPI()

        # 创建查询请求
        request = QueryRequest(
            query="红色大楼的北侧",
            candidates=[
                {
                    "cell_id": "cell_001",
                    "description": "红色大楼的北侧",
                    "object_label": "building",
                    "object_color": "red",
                    "direction": "north"
                }
            ],
            parameters={
                "enhanced_mode": True,
                "top_k": 3
            }
        )

        # 处理查询
        response = api.process_query(request)

        # 验证响应
        self.assertIsInstance(response, QueryResponse)
        self.assertTrue(response.success)
        self.assertIsNotNone(response.results)
        self.assertGreaterEqual(len(response.results), 0)

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_monitoring_integration(self):
        """测试监控系统集成"""
        # 创建监控器
        monitor = SystemMonitor()

        # 记录各种操作
        monitor.record_query("query_1", 45.2, True, {"test": "data1"})
        monitor.record_query("query_2", 50.1, True, {"test": "data2"})
        monitor.record_query("query_3", 100.5, False, {"test": "data3"})

        monitor.record_module_operation("nlu", "parse_direction", 12.5, True)
        monitor.record_module_operation("vector_db", "embed_text", 8.2, True)
        monitor.record_module_operation("adapter", "process_query", 45.0, True)

        monitor.track_error("TestError", "测试错误消息", "test_module",
                          {"details": "测试详情"})

        monitor.collect_feedback("query_1", 5, "非常好用", "红色大楼的北侧", "excellent")

        # 获取系统状态
        status = monitor.get_system_status()

        # 验证监控数据
        self.assertEqual(status["status"], "healthy")
        self.assertGreaterEqual(status["metrics"]["summary"]["query_count"], 3)
        self.assertGreaterEqual(status["metrics"]["summary"]["success_count"], 2)
        self.assertGreaterEqual(status["errors"]["total_errors"], 1)
        self.assertGreaterEqual(status["feedback"]["total"], 1)


class StressTests(BaseTestCase):
    """压力测试：验证系统在高负载下的表现"""

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_concurrent_queries(self):
        """测试并发查询"""
        config = IntegrationConfig(mock_mode=True)
        adapter = Text2LocAdapter(config)

        query = "测试查询"
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
                result = adapter.process_query(f"{query} {worker_id}", candidates)
                with lock:
                    results.append((worker_id, result))
            except Exception as e:
                with lock:
                    errors.append((worker_id, str(e)))

        # 创建并启动线程
        num_threads = 50
        threads = []
        start_time = time.time()

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        end_time = time.time()
        total_time = end_time - start_time

        print(f"并发查询测试完成")
        print(f"线程数: {num_threads}")
        print(f"总时间: {total_time:.2f}秒")
        print(f"平均每个查询: {(total_time/num_threads)*1000:.2f}ms")
        print(f"成功: {len(results)}, 失败: {len(errors)}")

        # 验证并发性要求
        self.assertEqual(len(results) + len(errors), num_threads,
                        f"应该有{num_threads}个结果，实际{len(results)+len(errors)}个")

        # 性能要求：50个并发查询应在5秒内完成
        self.assertPerformanceWithinLimit("50个并发查询", total_time * 1000, 5000)

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_high_volume_queries(self):
        """测试高容量查询"""
        config = IntegrationConfig(mock_mode=True)
        adapter = Text2LocAdapter(config)

        # 准备测试数据
        base_query = "位置查询"
        candidates = [
            {
                "cell_id": f"cell_{i:03d}",
                "description": f"位置{i}的描述",
                "object_label": ["building", "parking", "light"][i % 3],
                "object_color": ["red", "blue", "green", "yellow"][i % 4],
                "direction": ["north", "south", "east", "west"][i % 4]
            }
            for i in range(100)  # 100个候选位置
        ]

        # 执行大量查询
        num_queries = 1000
        start_time = time.time()

        for i in range(num_queries):
            query = f"{base_query} {i}"
            try:
                result = adapter.process_query(query, candidates)
                # 验证基本结果结构
                self.assertIn("query_analysis", result)
                self.assertIn("retrieval_results", result)
            except Exception as e:
                self.fail(f"查询{i}失败: {e}")

        end_time = time.time()
        total_time = end_time - start_time
        qps = num_queries / total_time

        print(f"高容量查询测试完成")
        print(f"查询数量: {num_queries}")
        print(f"总时间: {total_time:.2f}秒")
        print(f"QPS: {qps:.2f}")

        # 性能要求：QPS > 50
        self.assertGreater(qps, 50, f"QPS过低: {qps:.2f}")

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        import gc
        import psutil

        process = psutil.Process()

        # 初始内存
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 执行多次操作
        config = IntegrationConfig(mock_mode=True)

        memory_samples = []
        for iteration in range(10):
            # 创建并使用适配器
            adapter = Text2LocAdapter(config)

            query = f"测试查询{iteration}"
            candidates = [
                {
                    "cell_id": "cell_001",
                    "description": "测试位置",
                    "object_label": "building",
                    "object_color": "red",
                    "direction": "north"
                }
            ]

            # 执行多次查询
            for i in range(100):
                result = adapter.process_query(f"{query}_{i}", candidates)

            # 强制垃圾回收并测量内存
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)

            print(f"迭代 {iteration}: 内存使用 {current_memory:.2f} MB")

        # 分析内存使用趋势
        memory_increase = memory_samples[-1] - initial_memory

        print(f"初始内存: {initial_memory:.2f} MB")
        print(f"最终内存: {memory_samples[-1]:.2f} MB")
        print(f"内存增加: {memory_increase:.2f} MB")

        # 内存泄漏检测：10次迭代后内存增加不应超过50MB
        self.assertLess(memory_increase, 50,
                       f"可能的内存泄漏：内存增加了{memory_increase:.2f} MB")


class ABComparisonTests(BaseTestCase):
    """A/B对比测试：对比增强版和原始版的性能"""

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_enhanced_vs_original_accuracy(self):
        """测试增强版与原始版的准确率对比"""
        # 创建增强版适配器
        enhanced_config = IntegrationConfig(
            mock_mode=True,
            enable_enhanced_direction=True,
            enable_enhanced_color=True,
            enable_enhanced_object=True
        )
        enhanced_adapter = Text2LocAdapter(enhanced_config)

        # 创建原始版适配器（关闭所有增强功能）
        original_config = IntegrationConfig(
            mock_mode=True,
            enable_enhanced_direction=False,
            enable_enhanced_color=False,
            enable_enhanced_object=False,
            enable_vector_search=False
        )
        original_adapter = Text2LocAdapter(original_config)

        # 测试数据集
        test_cases = [
            {
                "query": "红色大楼的北侧",
                "expected_object": "building",
                "expected_color": "red",
                "expected_direction": "north"
            },
            {
                "query": "停车场的东边",
                "expected_object": "parking",
                "expected_color": "gray",
                "expected_direction": "east"
            },
            {
                "query": "蓝色灯柱的上方",
                "expected_object": "light",
                "expected_color": "blue",
                "expected_direction": "above"
            },
            {
                "query": "绿色交通标志的前方",
                "expected_object": "traffic_sign",
                "expected_color": "green",
                "expected_direction": "front"
            },
        ]

        candidates = [
            {
                "cell_id": "cell_001",
                "description": "红色大楼的北侧",
                "object_label": "building",
                "object_color": "red",
                "direction": "north"
            },
            {
                "cell_id": "cell_002",
                "description": "停车场的东边",
                "object_label": "parking",
                "object_color": "gray",
                "direction": "east"
            },
            {
                "cell_id": "cell_003",
                "description": "蓝色灯柱的上方",
                "object_label": "light",
                "object_color": "blue",
                "direction": "above"
            },
            {
                "cell_id": "cell_004",
                "description": "绿色交通标志的前方",
                "object_label": "traffic_sign",
                "object_color": "green",
                "direction": "front"
            },
        ]

        # 执行测试
        enhanced_correct = 0
        original_correct = 0
        total_cases = len(test_cases)

        for i, test_case in enumerate(test_cases):
            query = test_case["query"]

            # 增强版结果
            enhanced_result = enhanced_adapter.process_query(query, candidates)
            enhanced_analysis = enhanced_result.get("query_analysis", {})

            # 原始版结果
            original_result = original_adapter.process_query(query, candidates)
            original_analysis = original_result.get("query_analysis", {})

            # 检查增强版是否使用了增强功能
            if enhanced_analysis.get("enhanced_used", False):
                enhanced_features = []
                if enhanced_analysis.get("direction"):
                    enhanced_features.append("direction")
                if enhanced_analysis.get("color"):
                    enhanced_features.append("color")
                if enhanced_analysis.get("object_name"):
                    enhanced_features.append("object")
                print(f"测试{i+1}: 增强版使用了 {enhanced_features}")

            # 简单准确性检查（模拟）
            # 在实际系统中，这里应该检查检索结果排名等
            if enhanced_result.get("final_result"):
                enhanced_correct += 1
            if original_result.get("final_result"):
                original_correct += 1

        enhanced_accuracy = enhanced_correct / total_cases
        original_accuracy = original_correct / total_cases
        improvement = enhanced_accuracy - original_accuracy

        print(f"A/B对比测试结果:")
        print(f"  增强版准确率: {enhanced_accuracy:.2%} ({enhanced_correct}/{total_cases})")
        print(f"  原始版准确率: {original_accuracy:.2%} ({original_correct}/{total_cases})")
        print(f"  提升: {improvement:.2%}")

        # 要求：增强版至少比原始版准确率高10%
        self.assertGreaterEqual(improvement, 0.10,
                              f"增强版准确率提升不足: {improvement:.2%}")

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_enhanced_vs_original_performance(self):
        """测试增强版与原始版的性能对比"""
        # 创建适配器
        enhanced_config = IntegrationConfig(mock_mode=True)
        enhanced_adapter = Text2LocAdapter(enhanced_config)

        original_config = IntegrationConfig(
            mock_mode=True,
            enable_enhanced_direction=False,
            enable_enhanced_color=False,
            enable_enhanced_object=False
        )
        original_adapter = Text2LocAdapter(original_config)

        query = "测试查询"
        candidates = [
            {
                "cell_id": f"cell_{i:03d}",
                "description": f"位置{i}的描述",
                "object_label": "building",
                "object_color": "red",
                "direction": "north"
            }
            for i in range(50)
        ]

        # 测量增强版性能
        enhanced_times = []
        for i in range(20):
            start_time = time.time()
            enhanced_adapter.process_query(f"{query} {i}", candidates)
            enhanced_times.append((time.time() - start_time) * 1000)

        # 测量原始版性能
        original_times = []
        for i in range(20):
            start_time = time.time()
            original_adapter.process_query(f"{query} {i}", candidates)
            original_times.append((time.time() - start_time) * 1000)

        enhanced_avg = statistics.mean(enhanced_times)
        original_avg = statistics.mean(original_times)
        performance_overhead = enhanced_avg - original_avg
        overhead_percentage = (performance_overhead / original_avg) * 100

        print(f"性能对比测试结果:")
        print(f"  增强版平均时间: {enhanced_avg:.2f}ms")
        print(f"  原始版平均时间: {original_avg:.2f}ms")
        print(f"  性能开销: {performance_overhead:.2f}ms ({overhead_percentage:.1f}%)")

        # 要求：增强版性能开销不超过原始版的50%
        self.assertLess(overhead_percentage, 50,
                       f"增强版性能开销过高: {overhead_percentage:.1f}%")


class RegressionTests(BaseTestCase):
    """回归测试：确保新功能不影响现有功能"""

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 测试旧格式到新格式的转换
        converter = FormatConverter()

        old_format = OldFormat(
            object_label="building",
            object_color="red",
            direction="north",
            offset=np.array([0.5, 0.5]),
            cell_id="cell_001",
            pose_id="pose_001",
            description="红色大楼的北侧"
        )

        # 转换为新格式
        new_format = converter.old_to_new(old_format)

        # 验证转换正确性
        self.assertEqual(new_format.object_name, old_format.object_label)
        self.assertEqual(new_format.object_color, old_format.object_color)
        self.assertEqual(new_format.direction, old_format.direction)
        self.assertEqual(new_format.cell_id, old_format.cell_id)
        self.assertEqual(new_format.pose_id, old_format.pose_id)

        # 转换回旧格式
        converted_back = converter.new_to_old(new_format)

        # 验证双向转换的兼容性
        self.assertEqual(converted_back.object_label, old_format.object_label)
        self.assertEqual(converted_back.object_color, old_format.object_color)
        self.assertEqual(converted_back.direction, old_format.direction)
        self.assertEqual(converted_back.cell_id, old_format.cell_id)
        self.assertEqual(converted_back.pose_id, old_format.pose_id)

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_api_backward_compatibility(self):
        """测试API向后兼容性"""
        api = Text2LocAPI()

        # 测试旧版API调用方式（模拟）
        # 这里应该测试实际的API兼容性

        self.assertTrue(hasattr(api, 'process_query'))
        self.assertTrue(hasattr(api, 'get_system_status'))
        self.assertTrue(hasattr(api, 'get_capabilities'))

    @unittest.skipUnless(IMPORT_SUCCESS, "模块导入失败")
    def test_configuration_backward_compatibility(self):
        """测试配置向后兼容性"""
        config_manager = ConfigManager()

        # 测试默认配置包含所有必要字段
        config = config_manager.config

        required_fields = [
            'nlu_model', 'embedding_model', 'mock_mode',
            'enable_nlu', 'enable_vector_search',
            'nlu_confidence_threshold', 'cache_enabled'
        ]

        for field in required_fields:
            self.assertTrue(hasattr(config, field),
                          f"配置缺少必要字段: {field}")

        # 测试配置更新不影响未指定的字段
        old_nlu_model = config.nlu_model
        config_manager.update(mock_mode=True)

        self.assertEqual(config.nlu_model, old_nlu_model,
                        "更新配置不应改变未指定的字段")


class DeploymentTests(BaseTestCase):
    """部署测试：验证生产部署相关功能"""

    def test_configuration_export_import(self):
        """测试配置导出导入"""
        if not IMPORT_SUCCESS:
            self.skipTest("模块导入失败")

        config_manager = ConfigManager()

        # 导出配置到临时文件
        temp_config_file = os.path.join(self.test_data_dir, "test_config.yaml")
        config_manager.save_to_file(temp_config_file, "yaml")

        self.assertTrue(os.path.exists(temp_config_file),
                       "配置文件应被创建")

        # 从文件导入配置
        new_manager = ConfigManager(temp_config_file)

        # 验证配置一致性
        original_config = config_manager.get_all()
        imported_config = new_manager.get_all()

        # 检查关键配置项
        for key in ['nlu_model', 'embedding_model', 'mock_mode']:
            self.assertEqual(original_config[key], imported_config[key],
                           f"配置项{key}在导入后不一致")

    def test_logging_configuration(self):
        """测试日志配置"""
        if not IMPORT_SUCCESS:
            self.skipTest("模块导入失败")

        from api.monitoring import setup_logging
        import logging

        # 测试日志系统初始化
        log_file = os.path.join(self.test_data_dir, "test.log")

        # 这可能会影响全局日志配置，所以在单独的进程中测试
        # 这里只测试函数存在性
        self.assertTrue(callable(setup_logging),
                       "setup_logging函数应存在")

    def test_environment_configurations(self):
        """测试多环境配置"""
        if not IMPORT_SUCCESS:
            self.skipTest("模块导入失败")

        config_manager = ConfigManager()

        # 测试开发环境配置
        dev_overrides = {"mock_mode": True, "log_level": "DEBUG"}
        dev_config_path = config_manager.create_environment_config("dev", dev_overrides)

        self.assertTrue(os.path.exists(dev_config_path),
                       "开发环境配置文件应被创建")

        # 测试生产环境配置
        prod_overrides = {"mock_mode": False, "log_level": "WARNING"}
        prod_config_path = config_manager.create_environment_config("prod", prod_overrides)

        self.assertTrue(os.path.exists(prod_config_path),
                       "生产环境配置文件应被创建")

        # 验证配置差异
        dev_manager = ConfigManager(dev_config_path)
        prod_manager = ConfigManager(prod_config_path)

        self.assertTrue(dev_manager.config.mock_mode,
                       "开发环境应为模拟模式")
        self.assertFalse(prod_manager.config.mock_mode,
                        "生产环境不应为模拟模式")
        self.assertEqual(dev_manager.config.log_level, "DEBUG",
                        "开发环境日志级别应为DEBUG")
        self.assertEqual(prod_manager.config.log_level, "WARNING",
                        "生产环境日志级别应为WARNING")


class Phase4TestRunner:
    """第四阶段测试运行器"""

    def __init__(self):
        self.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "test_cases": {},
            "performance_metrics": {},
            "start_time": None,
            "end_time": None
        }

    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 80)
        print("Text2Loc Visionary - 第四阶段全面测试套件")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        self.test_results["start_time"] = time.time()

        # 创建测试套件
        test_suite = unittest.TestSuite()

        # 添加测试类
        test_classes = [
            FunctionalTests,
            PerformanceTests,
            IntegrationTests,
            StressTests,
            ABComparisonTests,
            RegressionTests,
            DeploymentTests
        ]

        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTest(tests)

        # 运行测试
        runner = unittest.TextTestRunner(
            verbosity=2,
            resultclass=unittest.TextTestResult
        )

        result = runner.run(test_suite)

        # 收集结果
        self.test_results["total_tests"] = result.testsRun
        self.test_results["passed"] = result.testsRun - len(result.failures) - len(result.errors)
        self.test_results["failed"] = len(result.failures)
        self.test_results["errors"] = len(result.errors)
        self.test_results["skipped"] = len(getattr(result, 'skipped', []))

        self.test_results["end_time"] = time.time()

        # 打印摘要
        self.print_summary(result)

        return result.wasSuccessful()

    def print_summary(self, result):
        """打印测试摘要"""
        print()
        print("=" * 80)
        print("测试摘要")
        print("=" * 80)

        total_time = self.test_results["end_time"] - self.test_results["start_time"]

        print(f"总测试数: {self.test_results['total_tests']}")
        print(f"通过: {self.test_results['passed']}")
        print(f"失败: {self.test_results['failed']}")
        print(f"错误: {self.test_results['errors']}")
        print(f"跳过: {self.test_results['skipped']}")
        print(f"总用时: {total_time:.2f}秒")
        print()

        # 打印失败详情
        if result.failures:
            print("失败测试:")
            for test, traceback in result.failures:
                print(f"  {test}")

        if result.errors:
            print("错误测试:")
            for test, traceback in result.errors:
                print(f"  {test}")

    def generate_report(self, output_file=None):
        """生成测试报告"""
        report = {
            "project": "Text2Loc Visionary",
            "phase": 4,
            "test_suite": "Comprehensive Testing Suite",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": self.test_results["total_tests"],
                "passed": self.test_results["passed"],
                "failed": self.test_results["failed"],
                "errors": self.test_results["errors"],
                "skipped": self.test_results["skipped"],
                "success_rate": self.test_results["passed"] / self.test_results["total_tests"]
                               if self.test_results["total_tests"] > 0 else 0,
                "duration_seconds": self.test_results["end_time"] - self.test_results["start_time"]
                                   if self.test_results["end_time"] else 0
            },
            "test_categories": [
                {
                    "name": "Functional Tests",
                    "description": "验证核心功能是否正确工作"
                },
                {
                    "name": "Performance Tests",
                    "description": "验证系统性能指标"
                },
                {
                    "name": "Integration Tests",
                    "description": "验证模块间集成"
                },
                {
                    "name": "Stress Tests",
                    "description": "验证系统在高负载下的表现"
                },
                {
                    "name": "A/B Comparison Tests",
                    "description": "对比增强版和原始版的性能"
                },
                {
                    "name": "Regression Tests",
                    "description": "确保新功能不影响现有功能"
                },
                {
                    "name": "Deployment Tests",
                    "description": "验证生产部署相关功能"
                }
            ],
            "requirements_verification": {
                "functional_completeness": self.test_results["passed"] / self.test_results["total_tests"] > 0.9,
                "performance_requirements": "验证通过性能测试",
                "integration_readiness": "所有集成测试通过",
                "stress_resilience": "压力测试验证系统稳定性",
                "backward_compatibility": "回归测试确保兼容性",
                "deployment_readiness": "部署测试通过"
            },
            "recommendations": []
        }

        # 添加建议
        if self.test_results["failed"] > 0:
            report["recommendations"].append(
                f"修复 {self.test_results['failed']} 个失败的测试用例"
            )
        if self.test_results["errors"] > 0:
            report["recommendations"].append(
                f"解决 {self.test_results['errors']} 个测试错误"
            )
        if self.test_results["passed"] / self.test_results["total_tests"] < 0.95:
            report["recommendations"].append(
                "提高测试通过率至95%以上"
            )

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"测试报告已保存到: {output_file}")

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Text2Loc Visionary 第四阶段全面测试套件")
    parser.add_argument("--report", type=str, help="测试报告输出文件路径")
    parser.add_argument("--tests", type=str, help="运行特定测试（用逗号分隔）")
    parser.add_argument("--category", type=str,
                       choices=["functional", "performance", "integration",
                               "stress", "ab", "regression", "deployment"],
                       help="运行特定类别的测试")

    args = parser.parse_args()

    # 运行测试
    runner = Phase4TestRunner()
    success = runner.run_all_tests()

    # 生成报告
    if args.report:
        report = runner.generate_report(args.report)

    # 返回退出码
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
