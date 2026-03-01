"""
Text2Loc Visionary - 极简优化测试脚本

专注于核心优化效果的验证：
1. Ollama连接和响应时间
2. 优化配置加载
3. 基本查询处理性能

运行此脚本快速验证优化是否生效。
"""

import sys
import os
import json
import time
import requests
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ollama_connection():
    """测试Ollama连接"""
    print("=" * 60)
    print("🔌 测试Ollama连接")
    print("=" * 60)

    tests = [
        ("API连接", "http://localhost:11434/api/tags", "GET"),
        ("简单生成", "http://localhost:11434/api/generate", "POST")
    ]

    results = []

    for test_name, url, method in tests:
        print(f"\n🧪 {test_name}: {url}")

        try:
            start_time = time.time()

            if method == "GET":
                response = requests.get(url, timeout=10)
                payload = None
            else:
                payload = {
                    "model": "qwen3-vl:2b",
                    "prompt": "Quick test",
                    "stream": False,
                    "options": {
                        "num_predict": 10,
                        "temperature": 0.1,
                        "num_threads": 8
                    }
                }
                response = requests.post(url, json=payload, timeout=30)

            response_time = time.time() - start_time

            if response.status_code == 200:
                print(f"   ✅ 成功 ({response_time:.3f}s)")
                results.append((test_name, True, response_time))

                # 如果是生成测试，显示性能数据
                if method == "POST" and response.status_code == 200:
                    try:
                        data = response.json()
                        if "eval_duration" in data:
                            eval_time = data["eval_duration"] / 1e9
                            print(f"   推理时间: {eval_time:.3f}s")
                    except:
                        pass
            else:
                print(f"   ❌ 失败 (状态码: {response.status_code})")
                results.append((test_name, False, response_time))

        except requests.exceptions.Timeout:
            print(f"   ⏰ 超时")
            results.append((test_name, False, None))
        except Exception as e:
            print(f"   ❌ 异常: {e}")
            results.append((test_name, False, None))

    # 分析结果
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"\n📊 连接测试: {successful}/{total} 通过")

    if successful == total:
        print("✅ Ollama连接正常")
        return True
    else:
        print("❌ Ollama连接有问题")
        return False

def test_optimization_config():
    """测试优化配置"""
    print("\n" + "=" * 60)
    print("⚙️ 测试优化配置")
    print("=" * 60)

    config_files = [
        ("config.yaml", "主配置文件"),
        ("ollama_text2loc_config.json", "优化配置")
    ]

    loaded_configs = {}

    for file_path, description in config_files:
        path = Path(file_path)
        if path.exists():
            try:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    import yaml
                    with open(path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    print(f"✅ 加载 {description}: {file_path}")

                    # 检查关键配置
                    if config and "ollama" in config:
                        ollama_config = config["ollama"]
                        loaded_configs[file_path] = ollama_config

                        # 显示关键参数
                        print(f"   关键参数:")
                        if "timeout" in ollama_config:
                            print(f"     timeout: {ollama_config['timeout']}s")

                        if "options" in ollama_config:
                            options = ollama_config["options"]
                            key_params = ["num_predict", "temperature", "num_threads", "gpu_layers"]
                            for param in key_params:
                                if param in options:
                                    print(f"     {param}: {options[param]}")

                elif file_path.endswith('.json'):
                    with open(path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    print(f"✅ 加载 {description}: {file_path}")

                    if "ollama" in config:
                        loaded_configs[file_path] = config["ollama"]

            except Exception as e:
                print(f"❌ 加载 {file_path} 失败: {e}")
        else:
            print(f"❌ {description}不存在: {file_path}")

    # 检查优化参数是否设置正确
    print("\n🔍 检查优化参数:")

    expected_params = {
        "num_predict": 256,      # 减少输出长度
        "temperature": 0.1,      # 降低随机性
        "num_threads": 8,        # 合理CPU使用
        "gpu_layers": 15,        # 优化GPU使用
        "batch_size": 32,        # 减少内存压力
        "timeout": 30            # 合理超时
    }

    all_correct = True
    for config_path, config in loaded_configs.items():
        print(f"\n  配置文件: {config_path}")

        # 检查options中的参数
        options = config.get("options", {})

        for param, expected_value in expected_params.items():
            if param in options:
                actual_value = options[param]
                if actual_value == expected_value:
                    print(f"    ✅ {param}: {actual_value}")
                else:
                    print(f"    ⚠️ {param}: {actual_value} (期望: {expected_value})")
                    all_correct = False
            elif param in config:
                actual_value = config[param]
                if actual_value == expected_value:
                    print(f"    ✅ {param}: {actual_value}")
                else:
                    print(f"    ⚠️ {param}: {actual_value} (期望: {expected_value})")
                    all_correct = False
            else:
                print(f"    ❌ {param}: 未配置")
                all_correct = False

    if all_correct and loaded_configs:
        print("\n✅ 优化参数配置正确")
        return True
    else:
        print("\n⚠️ 优化参数需要调整")
        return False

def test_query_performance():
    """测试查询性能"""
    print("\n" + "=" * 60)
    print("⚡ 测试查询性能")
    print("=" * 60)

    # 测试查询
    test_queries = [
        "停车场北侧的红色汽车",
        "建筑物旁边的灯柱",
        "东边的红色建筑"
    ]

    print("📋 测试查询:")
    for i, query in enumerate(test_queries, 1):
        print(f"  {i}. {query}")

    # 测试优化配置
    print("\n🧪 测试优化配置响应:")

    optimization_config = {
        "num_predict": 256,
        "temperature": 0.1,
        "num_threads": 8,
        "batch_size": 32
    }

    default_config = {
        "num_predict": 512,
        "temperature": 0.7,
        "num_threads": 4,
        "batch_size": 512
    }

    configs = [
        ("优化配置", optimization_config),
        ("默认配置", default_config)
    ]

    results = {}

    for config_name, options in configs:
        print(f"\n⚙️ {config_name}:")

        query_times = []
        for query in test_queries:
            payload = {
                "model": "qwen3-vl:2b",
                "prompt": f"简要分析: {query}",
                "stream": False,
                "options": options
            }

            try:
                start_time = time.time()
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=30
                )
                response_time = time.time() - start_time

                if response.status_code == 200:
                    query_times.append(response_time)
                    print(f"   ✅ '{query[:10]}...': {response_time:.3f}s")
                else:
                    print(f"   ❌ '{query[:10]}...': 失败 ({response.status_code})")

            except Exception as e:
                print(f"   ❌ '{query[:10]}...': 异常 ({e})")

        # 计算平均时间
        if query_times:
            avg_time = sum(query_times) / len(query_times)
            results[config_name] = avg_time
            print(f"   📊 平均: {avg_time:.3f}s")

    # 性能对比
    if len(results) >= 2:
        print("\n📈 性能对比:")

        if "优化配置" in results and "默认配置" in results:
            optimized = results["优化配置"]
            default = results["默认配置"]

            print(f"   优化配置: {optimized:.3f}s")
            print(f"   默认配置: {default:.3f}s")

            if default > 0:
                improvement = ((default - optimized) / default) * 100
                print(f"\n🎯 性能提升: {improvement:.1f}%")

                if improvement > 0:
                    print("✅ 优化配置有效！")
                    return True
                else:
                    print("⚠️ 优化配置效果不明显")
                    return False

    return False

def test_instruction_optimizer():
    """测试InstructionOptimizer"""
    print("\n" + "=" * 60)
    print("🧠 测试InstructionOptimizer")
    print("=" * 60)

    try:
        # 尝试导入InstructionOptimizer
        from enhancements.nlu.instruction_optimizer import InstructionOptimizer

        print("✅ 成功导入InstructionOptimizer")

        # 测试模拟模式（快速测试）
        print("\n🧪 测试模拟模式:")
        optimizer = InstructionOptimizer(mock_mode=True)

        test_query = "停车场北侧的红色汽车"
        result = optimizer.optimize_query(test_query)

        print(f"   查询: {test_query}")
        print(f"   优化后: {result.optimized_input}")
        print(f"   置信度: {result.confidence_scores.get('overall', 0.0):.2f}")
        print(f"   需要澄清: {result.need_clarification}")

        print("✅ 模拟模式测试通过")

        # 测试真实模式（如果Ollama连接正常）
        print("\n🧪 测试真实模式:")
        try:
            optimizer_real = InstructionOptimizer(mock_mode=False, config_path="config.yaml")

            start_time = time.time()
            result = optimizer_real.optimize_query(test_query)
            processing_time = time.time() - start_time

            print(f"   处理时间: {processing_time:.3f}s")
            print(f"   优化后: {result.optimized_input}")

            if processing_time < 10:
                print("✅ 真实模式响应正常")
                return True
            else:
                print(f"⚠️ 响应时间较长: {processing_time:.3f}s")
                return False

        except Exception as e:
            print(f"❌ 真实模式测试失败: {e}")
            print("ℹ️ 建议检查Ollama服务状态")
            return False

    except ImportError as e:
        print(f"❌ 无法导入InstructionOptimizer: {e}")
        return False
    except Exception as e:
        print(f"❌ InstructionOptimizer测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Text2Loc Visionary - 极简优化测试")
    print("=" * 60)
    print("版本: 1.0.0")
    print("目标: 快速验证Ollama优化效果")
    print("=" * 60)

    try:
        # 测试1: Ollama连接
        connection_ok = test_ollama_connection()

        if not connection_ok:
            print("\n❌ Ollama连接失败，无法继续测试")
            print("   请检查:")
            print("   1. Ollama服务是否运行: ollama serve")
            print("   2. 端口11434是否可用")
            return 1

        # 测试2: 优化配置
        config_ok = test_optimization_config()

        if not config_ok:
            print("\n⚠️ 优化配置不完整，继续测试...")

        # 测试3: 查询性能
        print("\n" + "=" * 60)
        print("📊 性能测试开始")
        print("=" * 60)

        performance_ok = test_query_performance()

        # 测试4: InstructionOptimizer
        optimizer_ok = test_instruction_optimizer()

        # 总结
        print("\n" + "=" * 60)
        print("📋 测试总结")
        print("=" * 60)

        print(f"1. Ollama连接: {'✅ 正常' if connection_ok else '❌ 异常'}")
        print(f"2. 优化配置: {'✅ 正确' if config_ok else '⚠️ 需要调整'}")
        print(f"3. 查询性能: {'✅ 提升明显' if performance_ok else '⚠️ 效果有限'}")
        print(f"4. InstructionOptimizer: {'✅ 正常' if optimizer_ok else '❌ 异常'}")

        # 总体评估
        if connection_ok and optimizer_ok:
            if performance_ok:
                print("\n🎉 优化测试通过！Ollama性能已改善。")
                print("\n🚀 下一步:")
                print("   1. 启动Text2Loc服务: python start_server.py")
                print("   2. 运行完整演示: python quick_demo.py")
                print("   3. 监控性能: ollama ps")
            else:
                print("\n⚠️ 基础功能正常，但性能提升不明显")
                print("\n📋 建议:")
                print("   1. 检查Ollama内存使用: ollama ps")
                print("   2. 调整num_predict参数（当前256）")
                print("   3. 考虑使用更小模型")
        else:
            print("\n❌ 优化测试失败")
            print("\n🔧 需要修复:")
            if not connection_ok:
                print("   - Ollama连接问题")
            if not optimizer_ok:
                print("   - InstructionOptimizer问题")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断测试")
        return 1
    except Exception as e:
        print(f"\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
