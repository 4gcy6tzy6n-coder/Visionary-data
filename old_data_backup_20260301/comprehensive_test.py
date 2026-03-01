#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text2Loc Visionary 综合测试脚本
测试内容：
1. 算法精度测试（NLU引擎、混合检索、适配器）
2. API接口测试（配置、查询、健康检查）
3. 端到端流程测试
"""

import sys
import json
import time
import requests
from typing import Dict, List, Tuple

# 配置
API_BASE_URL = "http://localhost:8080"
TIMEOUT = 30

class TestResults:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details = []
    
    def add_test(self, name: str, passed: bool, message: str = "", warning: bool = False):
        self.total += 1
        if warning:
            self.warnings += 1
            status = "⚠️  WARNING"
        elif passed:
            self.passed += 1
            status = "✅ PASS"
        else:
            self.failed += 1
            status = "❌ FAIL"
        
        self.details.append({
            "name": name,
            "status": status,
            "message": message
        })
        print(f"{status}: {name}")
        if message:
            print(f"    {message}")
    
    def print_summary(self):
        print("\n" + "="*70)
        print("测试总结")
        print("="*70)
        print(f"总测试数: {self.total}")
        print(f"通过: {self.passed} ✅")
        print(f"失败: {self.failed} ❌")
        print(f"警告: {self.warnings} ⚠️")
        print(f"通过率: {(self.passed/self.total*100):.1f}%")
        print("="*70)


def test_api_health(results: TestResults):
    """测试API健康检查"""
    print("\n" + "="*70)
    print("1. API 健康检查测试")
    print("="*70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            results.add_test(
                "API健康检查", 
                data.get("status") == "healthy",
                f"响应: {data}"
            )
        else:
            results.add_test("API健康检查", False, f"状态码: {response.status_code}")
    except Exception as e:
        results.add_test("API健康检查", False, f"错误: {str(e)}")


def test_model_config(results: TestResults):
    """测试模型配置接口"""
    print("\n" + "="*70)
    print("2. 模型配置测试")
    print("="*70)
    
    try:
        # 获取当前配置
        response = requests.get(f"{API_BASE_URL}/api/v1/config", timeout=5)
        if response.status_code == 200:
            data = response.json()
            is_configured = data.get("is_configured", False)
            
            results.add_test(
                "获取模型配置",
                data.get("status") == "success",
                f"提供商: {data.get('provider')}, 模型: {data.get('model')}, 配置状态: {'已配置' if is_configured else '未配置'}"
            )
            
            if not is_configured:
                results.add_test(
                    "模型配置状态",
                    False,
                    "模型未配置，可能影响NLU解析精度",
                    warning=True
                )
        else:
            results.add_test("获取模型配置", False, f"状态码: {response.status_code}")
    except Exception as e:
        results.add_test("获取模型配置", False, f"错误: {str(e)}")


def test_nlu_parsing(results: TestResults):
    """测试NLU引擎解析能力"""
    print("\n" + "="*70)
    print("3. NLU引擎精度测试")
    print("="*70)
    
    test_cases = [
        {
            "query": "找到红色的汽车",
            "expected": {"color": "红", "object": "car"},
            "name": "颜色+对象识别"
        },
        {
            "query": "在建筑物左侧的树",
            "expected": {"direction": "左", "object": "vegetation"},
            "name": "方向+对象识别"
        },
        {
            "query": "前方10米处的蓝色标志牌",
            "expected": {"direction": "前", "color": "蓝", "object": "sign"},
            "name": "方向+距离+颜色+对象"
        },
        {
            "query": "停车场入口",
            "expected": {"object": "parking"},
            "name": "复合对象识别"
        },
        {
            "query": "白色建筑物",
            "expected": {"color": "白", "object": "building"},
            "name": "颜色+建筑物"
        }
    ]
    
    for case in test_cases:
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/query",
                json={
                    "query": case["query"],
                    "top_k": 5,
                    "enable_enhanced": True
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get("query_analysis", {})
                
                # 检查解析结果
                matches = []
                mismatches = []
                
                if "color" in case["expected"]:
                    expected_color = case["expected"]["color"]
                    actual_color = analysis.get("color", "")
                    if expected_color in actual_color or actual_color in expected_color:
                        matches.append(f"颜色:{actual_color}")
                    else:
                        mismatches.append(f"颜色(期望:{expected_color}, 实际:{actual_color})")
                
                if "direction" in case["expected"]:
                    expected_dir = case["expected"]["direction"]
                    actual_dir = analysis.get("direction", "")
                    if expected_dir in actual_dir or actual_dir in expected_dir:
                        matches.append(f"方向:{actual_dir}")
                    else:
                        mismatches.append(f"方向(期望:{expected_dir}, 实际:{actual_dir})")
                
                if "object" in case["expected"]:
                    expected_obj = case["expected"]["object"]
                    actual_obj = analysis.get("object", "")
                    if expected_obj in actual_obj or actual_obj in expected_obj:
                        matches.append(f"对象:{actual_obj}")
                    else:
                        mismatches.append(f"对象(期望:{expected_obj}, 实际:{actual_obj})")
                
                passed = len(mismatches) == 0
                message = f"查询: '{case['query']}'\n"
                if matches:
                    message += f"    ✓ 匹配: {', '.join(matches)}\n"
                if mismatches:
                    message += f"    ✗ 不匹配: {', '.join(mismatches)}\n"
                message += f"    解析时间: {analysis.get('parse_time_ms', 0):.0f}ms"
                
                results.add_test(case["name"], passed, message)
            else:
                results.add_test(case["name"], False, f"状态码: {response.status_code}")
                
        except Exception as e:
            results.add_test(case["name"], False, f"错误: {str(e)}")


def test_retrieval_accuracy(results: TestResults):
    """测试混合检索准确性"""
    print("\n" + "="*70)
    print("4. 混合检索精度测试")
    print("="*70)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/query",
            json={
                "query": "找到红色的汽车",
                "top_k": 5,
                "enable_enhanced": True
            },
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            retrieval_results = data.get("retrieval_results", [])
            
            # 检查返回结果数量
            results.add_test(
                "检索结果数量",
                len(retrieval_results) > 0,
                f"返回 {len(retrieval_results)} 个结果"
            )
            
            # 检查结果置信度
            if retrieval_results:
                top_confidence = retrieval_results[0].get("confidence", 0)
                results.add_test(
                    "Top-1 置信度",
                    top_confidence > 0.3,
                    f"置信度: {top_confidence:.2f} (期望 > 0.3)"
                )
                
                # 检查置信度排序
                confidences = [r.get("confidence", 0) for r in retrieval_results]
                is_sorted = all(confidences[i] >= confidences[i+1] for i in range(len(confidences)-1))
                results.add_test(
                    "置信度排序",
                    is_sorted,
                    f"置信度序列: {[f'{c:.2f}' for c in confidences]}"
                )
        else:
            results.add_test("混合检索", False, f"状态码: {response.status_code}")
            
    except Exception as e:
        results.add_test("混合检索", False, f"错误: {str(e)}")


def test_adapter_matching(results: TestResults):
    """测试适配器匹配算法"""
    print("\n" + "="*70)
    print("5. 适配器匹配测试")
    print("="*70)
    
    test_queries = [
        ("在建筑物左侧", "direction", "左"),
        ("红色的标志", "color", "红"),
        ("停车场", "object", "parking")
    ]
    
    for query, match_type, expected_value in test_queries:
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/query",
                json={
                    "query": query,
                    "top_k": 3,
                    "enable_enhanced": True
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get("query_analysis", {})
                actual_value = analysis.get(match_type, "")
                
                matched = expected_value in actual_value or actual_value in expected_value
                results.add_test(
                    f"适配器匹配 - {match_type}",
                    matched,
                    f"查询: '{query}', 期望: {expected_value}, 实际: {actual_value}"
                )
            else:
                results.add_test(f"适配器匹配 - {match_type}", False, f"状态码: {response.status_code}")
                
        except Exception as e:
            results.add_test(f"适配器匹配 - {match_type}", False, f"错误: {str(e)}")


def test_end_to_end(results: TestResults):
    """端到端流程测试"""
    print("\n" + "="*70)
    print("6. 端到端流程测试")
    print("="*70)
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/query",
            json={
                "query": "找到前方的红色汽车",
                "top_k": 5,
                "enable_enhanced": True
            },
            timeout=TIMEOUT
        )
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            
            # 检查响应完整性
            has_query = "query" in data
            has_analysis = "query_analysis" in data
            has_results = "retrieval_results" in data
            has_time = "processing_time_ms" in data
            
            completeness = has_query and has_analysis and has_results and has_time
            results.add_test(
                "响应完整性",
                completeness,
                f"包含字段: query={has_query}, analysis={has_analysis}, results={has_results}, time={has_time}"
            )
            
            # 检查响应时间
            results.add_test(
                "响应时间",
                elapsed_ms < 10000,
                f"总耗时: {elapsed_ms:.0f}ms (目标 < 10s)"
            )
            
            # 检查定位精度
            retrieval_results = data.get("retrieval_results", [])
            if retrieval_results:
                top_result = retrieval_results[0]
                has_coords = "x" in top_result and "y" in top_result
                results.add_test(
                    "定位坐标",
                    has_coords,
                    f"坐标: ({top_result.get('x', 'N/A'):.2f}, {top_result.get('y', 'N/A'):.2f})" if has_coords else "缺少坐标"
                )
        else:
            results.add_test("端到端流程", False, f"状态码: {response.status_code}")
            
    except Exception as e:
        results.add_test("端到端流程", False, f"错误: {str(e)}")


def test_interactive_mode(results: TestResults):
    """测试交互式模式"""
    print("\n" + "="*70)
    print("7. 交互式模式测试")
    print("="*70)
    
    try:
        # 第一次查询（模糊）
        session_id = f"test_{int(time.time())}"
        response1 = requests.post(
            f"{API_BASE_URL}/api/v1/query",
            json={
                "query": "找车",
                "top_k": 3,
                "enable_enhanced": True,
                "interactive": True,
                "session_id": session_id
            },
            timeout=TIMEOUT
        )
        
        if response1.status_code == 200:
            data1 = response1.json()
            need_clarification = data1.get("need_clarification", False)
            
            results.add_test(
                "交互式模式 - 初次查询",
                data1.get("status") == "success",
                f"需要澄清: {need_clarification}, 会话ID: {data1.get('session_id', 'N/A')}"
            )
            
            # 第二次查询（补充信息）
            if need_clarification:
                response2 = requests.post(
                    f"{API_BASE_URL}/api/v1/query",
                    json={
                        "query": "红色的",
                        "top_k": 3,
                        "enable_enhanced": True,
                        "interactive": True,
                        "session_id": data1.get("session_id")
                    },
                    timeout=TIMEOUT
                )
                
                if response2.status_code == 200:
                    data2 = response2.json()
                    results.add_test(
                        "交互式模式 - 补充查询",
                        data2.get("status") == "success",
                        f"会话保持: {data2.get('session_id') == data1.get('session_id')}"
                    )
        else:
            results.add_test("交互式模式", False, f"状态码: {response1.status_code}")
            
    except Exception as e:
        results.add_test("交互式模式", False, f"错误: {str(e)}")


def main():
    print("\n" + "="*70)
    print("Text2Loc Visionary 综合测试")
    print("="*70)
    print(f"API 地址: {API_BASE_URL}")
    print(f"超时设置: {TIMEOUT}s")
    print("="*70)
    
    # 检查后端服务
    try:
        requests.get(f"{API_BASE_URL}/health", timeout=5)
        print("✅ 后端服务正常运行\n")
    except:
        print("❌ 后端服务未启动或无法访问")
        print("请先启动后端服务: python3 start_backend.py\n")
        sys.exit(1)
    
    # 初始化测试结果
    results = TestResults()
    
    # 执行测试
    test_api_health(results)
    test_model_config(results)
    test_nlu_parsing(results)
    test_retrieval_accuracy(results)
    test_adapter_matching(results)
    test_end_to_end(results)
    test_interactive_mode(results)
    
    # 打印总结
    results.print_summary()
    
    # 保存测试结果
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": results.total,
            "passed": results.passed,
            "failed": results.failed,
            "warnings": results.warnings,
            "pass_rate": round(results.passed / results.total * 100, 1)
        },
        "details": results.details
    }
    
    report_file = "test_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试报告已保存至: {report_file}")
    
    # 返回退出码
    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()
