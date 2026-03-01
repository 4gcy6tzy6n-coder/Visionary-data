#!/usr/bin/env python3
"""
Text2Loc Visionary - Ollama 简化优化脚本

专注于解决qwen3-vl:2b模型的超时和内存问题
执行参数优化和配置调整
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"📌 {text}")
    print("=" * 60)

def run_command(cmd, description, timeout=30):
    """执行命令并显示结果"""
    print(f"\n🔧 {description}")
    print(f"   命令: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8'
        )

        if result.returncode == 0:
            print(f"   ✅ 成功")
            if result.stdout and len(result.stdout.strip()) > 0:
                print(f"   输出: {result.stdout[:200]}")
        else:
            print(f"   ❌ 失败 (返回码: {result.returncode})")
            if result.stderr:
                print(f"   错误: {result.stderr[:200]}")

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"   ⏰ 超时")
        return False
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return False

def check_ollama_status():
    """检查Ollama状态"""
    print_header("检查Ollama状态")

    # 检查Ollama进程
    success = run_command("ollama ps", "检查Ollama进程")

    # 检查模型列表
    run_command("ollama list", "检查已安装模型")

    return success

def stop_ollama_services():
    """停止Ollama相关服务"""
    print_header("停止Ollama服务")

    print("🛑 停止所有正在运行的模型...")

    # 停止qwen3-vl:2b模型
    success = run_command("ollama stop qwen3-vl:2b", "停止qwen3-vl:2b模型")

    # 尝试停止可能运行的其他模型
    run_command("ollama ps", "检查停止后的状态")

    # 等待内存释放
    print("\n⏳ 等待5秒让内存释放...")
    time.sleep(5)

    return success

def create_optimized_modelfile():
    """创建优化后的Modelfile"""
    print_header("创建优化Modelfile")

    modelfile_content = """FROM qwen3-vl:2b

# 内存优化配置
PARAMETER num_predict 256
PARAMETER temperature 0.1
PARAMETER top_k 20
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

# 系统性能优化
PARAMETER num_threads 8
PARAMETER batch_size 32
PARAMETER gpu_layers 15

# Flash Attention加速
PARAMETER flash_attention true

# 内存管理优化
PARAMETER mirostat 0
PARAMETER mirostat_tau 5.0
PARAMETER mirostat_eta 0.1
PARAMETER tfs_z 1.0
PARAMETER typical_p 1.0
PARAMETER repeat_last_n 64
PARAMETER penalize_nl true

# Text2Loc专用模板
SYSTEM "Optimized for Text2Loc Visionary - 3D场景定位助手"
TEMPLATE \"\"\"[INST] <<SYS>>
你是一个专业的3D场景定位助手，用于Text2Loc系统。
请分析用户关于3D场景中物体的自然语言描述，提取关键信息：
1. 方向（north, south, east, west等）
2. 颜色（red, blue, green等）
3. 物体类型（building, car, tree等）
4. 位置关系（near, next to, beside等）

请以JSON格式返回，包含：
- "optimized_query": 标准化后的查询
- "entities": 识别的实体
- "confidence": 置信度(0-1)
<</SYS>>

{{ .Prompt }} [/INST]\"\"\"

# 元数据
LICENSE "MIT"
AUTHOR "Text2Loc Visionary Team"
DESCRIPTION "优化版qwen3-vl:2b模型，专为Text2Loc设计"
TAGS ["text2loc", "3d-localization", "optimized"]
"""

    try:
        modelfile_path = Path("modelfile.qwen3-vl-text2loc-optimized")
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)

        print(f"✅ Modelfile已创建: {modelfile_path}")
        print(f"📋 文件大小: {modelfile_path.stat().st_size} 字节")

        return modelfile_path
    except Exception as e:
        print(f"❌ 创建Modelfile失败: {e}")
        return None

def create_optimized_model(modelfile_path):
    """创建优化模型"""
    print_header("创建优化模型")

    if not modelfile_path or not modelfile_path.exists():
        print("❌ Modelfile不存在，无法创建模型")
        return False

    # 删除可能存在的旧模型
    print("🗑️  清理可能存在的旧模型...")
    run_command("ollama rm text2loc-optimized", "删除旧模型（如果存在）", timeout=10)

    # 创建新模型
    cmd = f'ollama create text2loc-optimized -f "{modelfile_path}"'
    success = run_command(cmd, "创建优化模型", timeout=120)

    if success:
        print("\n🎉 优化模型创建成功！")
        print("   模型名称: text2loc-optimized")
        print("   原始模型: qwen3-vl:2b")
        print("   优化参数: 内存优化 + 性能调整")

    return success

def test_optimized_model():
    """测试优化模型"""
    print_header("测试优化模型")

    test_queries = [
        "停车场北侧的红色汽车",
        "建筑物旁边的灯柱",
        "东边的红色建筑"
    ]

    print("🧪 测试查询:")
    for i, query in enumerate(test_queries, 1):
        print(f"  {i}. {query}")

    print("\n⚡ 执行测试...")

    # 测试1: 简单查询
    print("\n1. 测试简单查询:")
    test_cmd = 'ollama run text2loc-optimized "Hello, test response" --nowordwrap'
    success1 = run_command(test_cmd, "简单查询测试", timeout=15)

    # 测试2: Text2Loc查询
    print("\n2. 测试Text2Loc查询:")
    text2loc_query = "请分析：'停车场北侧的红色汽车'，以JSON格式返回"
    test_cmd = f'ollama run text2loc-optimized "{text2loc_query}" --nowordwrap'
    success2 = run_command(test_cmd, "Text2Loc查询测试", timeout=20)

    # 测试3: 性能检查
    print("\n3. 性能检查:")
    run_command("ollama ps", "检查模型运行状态")

    return success1 and success2

def update_text2loc_config():
    """更新Text2Loc配置文件"""
    print_header("更新Text2Loc配置")

    config_update = {
        "ollama": {
            "model": "text2loc-optimized",
            "timeout": 30,
            "options": {
                "num_predict": 256,
                "temperature": 0.1,
                "num_threads": 8,
                "gpu_layers": 15
            }
        }
    }

    try:
        # 更新config.yaml
        config_path = Path("config.yaml")
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            # 更新配置
            config.update(config_update)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            print(f"✅ 已更新 config.yaml")

        # 创建单独的JSON配置
        json_config_path = Path("optimized_ollama_config.json")
        with open(json_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_update, f, indent=2, ensure_ascii=False)

        print(f"✅ 已创建 optimized_ollama_config.json")

        return True
    except Exception as e:
        print(f"❌ 更新配置失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Text2Loc Visionary - Ollama 简化优化工具")
    print("=" * 60)
    print("版本: 1.0.0")
    print("目标: 解决qwen3-vl:2b超时和内存问题")
    print("=" * 60)

    try:
        # 步骤1: 检查状态
        if not check_ollama_status():
            print("⚠️ Ollama状态检查失败，继续执行...")

        # 步骤2: 停止服务
        if not stop_ollama_services():
            print("⚠️ 停止服务时遇到问题，继续执行...")

        # 步骤3: 创建优化配置
        modelfile_path = create_optimized_modelfile()
        if not modelfile_path:
            print("❌ 创建Modelfile失败，退出")
            return 1

        # 步骤4: 创建优化模型
        if not create_optimized_model(modelfile_path):
            print("❌ 创建优化模型失败，退出")
            return 1

        # 步骤5: 测试模型
        print("\n" + "=" * 60)
        print("🧪 是否测试优化模型？")
        print("=" * 60)
        test_model = input("运行测试？(Y/n): ").strip().lower()

        if test_model in ['y', 'yes', '']:
            if test_optimized_model():
                print("\n✅ 优化模型测试通过！")
            else:
                print("\n⚠️ 优化模型测试部分失败")
        else:
            print("ℹ️ 跳过测试")

        # 步骤6: 更新配置
        if update_text2loc_config():
            print("\n✅ Text2Loc配置已更新")

        # 步骤7: 总结
        print_header("优化完成总结")
        print("✅ 已完成以下优化:")
        print("   1. 创建了优化Modelfile")
        print("   2. 创建了text2loc-optimized模型")
        print("   3. 更新了Text2Loc配置")
        print("\n📋 优化参数:")
        print("   - num_predict: 256 (减少输出长度)")
        print("   - temperature: 0.1 (提高稳定性)")
        print("   - gpu_layers: 15 (优化GPU使用)")
        print("   - batch_size: 32 (降低内存压力)")
        print("\n🚀 使用方法:")
        print("   1. 使用新模型: ollama run text2loc-optimized")
        print("   2. 在Text2Loc中设置模型为: text2loc-optimized")
        print("   3. 启动服务: python start_server.py")

        print("\n🎉 Ollama优化完成！")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断操作")
        return 1
    except Exception as e:
        print(f"\n❌ 优化过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
