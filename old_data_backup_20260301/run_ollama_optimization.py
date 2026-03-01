#!/usr/bin/env python3
"""
Text2Loc Visionary Ollama 优化执行脚本

自动执行Ollama优化配置步骤
"""

import subprocess
import time
import sys
import os

def run_command(cmd, description):
    """执行命令并显示结果"""
    print(f"\n📌 {description}")
    print(f"   命令: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"   ✅ 成功")
            if result.stdout:
                print(f"   输出: {result.stdout[:200]}...")
        else:
            print(f"   ❌ 失败")
            if result.stderr:
                print(f"   错误: {result.stderr[:200]}...")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"   ⏰ 超时")
        return False
    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 Text2Loc Visionary Ollama 优化执行")
    print("=" * 60)

    # 步骤1: 停止当前模型
    if not run_command("ollama stop qwen3-vl:2b", "停止当前qwen3-vl:2b模型"):
        print("⚠️ 停止模型失败，继续执行...")

    time.sleep(2)

    # 步骤2: 删除旧模型（可选）
    delete_old = input("是否删除旧的text2loc-optimized模型？(y/N): ").strip().lower()
    if delete_old == 'y':
        run_command("ollama rm text2loc-optimized", "删除旧的text2loc-optimized模型")

    # 步骤3: 创建优化模型
    modelfile_path = os.path.join(os.path.dirname(__file__), "modelfile.qwen3-vl-text2loc")
    if os.path.exists(modelfile_path):
        success = run_command(
            f'ollama create text2loc-optimized -f "{modelfile_path}"',
            "创建优化模型"
        )

        if success:
            print("✅ 优化模型创建成功！")

            # 步骤4: 测试优化模型
            print("\n🧪 测试优化模型...")
            test_prompt = "请用JSON格式分析：'建筑物旁边的灯柱'"
            test_cmd = f'ollama run text2loc-optimized "{test_prompt}"'

            if run_command(test_cmd, "测试优化模型响应"):
                print("✅ 优化模型测试通过！")
            else:
                print("⚠️ 优化模型测试失败")

            # 步骤5: 更新Text2Loc配置
            print("\n🔧 更新Text2Loc配置...")
            import json
            try:
                with open("ollama_text2loc_config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)

                config["ollama"]["optimized_model"] = "text2loc-optimized"
                config["ollama"]["model"] = "text2loc-optimized"

                with open("ollama_text2loc_config.json", "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

                print("✅ Text2Loc配置已更新为使用优化模型")

            except Exception as e:
                print(f"❌ 更新配置失败: {e}")

        # 步骤6: 启动Text2Loc服务
        print("\n🚀 启动Text2Loc服务...")
        print("请在新的终端中执行以下命令:")
        print("  cd "D:\Text2Loc-main\Text2Loc visionary"")
        print("  python start_server.py")

    else:
        print(f"❌ 找不到Modelfile: {modelfile_path}")
        print("请先运行 optimize_ollama.py 生成配置文件")

if __name__ == "__main__":
    main()
