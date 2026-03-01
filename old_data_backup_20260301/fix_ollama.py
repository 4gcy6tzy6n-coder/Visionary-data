"""
修复Ollama和qwen3-vl:2b模型问题
"""
import subprocess
import time
import requests

OLLAMA_URL = "http://localhost:11434"

def step1_stop_model():
    """步骤1: 停止模型"""
    print("=" * 60)
    print("🛑 步骤1: 停止所有模型")
    print("=" * 60)
    
    try:
        result = subprocess.run(['ollama', 'stop', 'qwen3-vl:2b'], capture_output=True, text=True, timeout=30)
        print(f"停止 qwen3-vl:2b: {'✅ 成功' if result.returncode == 0 else '❌ 失败'}")
        if result.stderr:
            print(f"   输出: {result.stderr}")
        time.sleep(2)
    except Exception as e:
        print(f"   错误: {e}")

def step2_check_memory():
    """步骤2: 等待内存释放"""
    print("\n" + "=" * 60)
    print("⏳ 步骤2: 等待内存释放")
    print("=" * 60)
    
    print("等待 10 秒让系统释放内存...")
    time.sleep(10)
    
    try:
        result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=10)
        print(f"\n当前模型状态:")
        print(result.stdout if result.stdout else "(无模型运行)")
    except Exception as e:
        print(f"错误: {e}")

def step3_restart_ollama():
    """步骤3: 重启Ollama服务"""
    print("\n" + "=" * 60)
    print("🔄 步骤3: 重启Ollama服务")
    print("=" * 60)
    
    print("请手动执行以下命令:")
    print("  1. 打开任务管理器")
    print("  2. 结束 'ollama.exe' 进程")
    print("  3. 重新启动 Ollama")
    print()
    print("或者在终端执行:")
    print("  > taskkill /F /IM ollama.exe")
    print("  > ollama serve")

def step4_reload_model():
    """步骤4: 重新加载模型"""
    print("\n" + "=" * 60)
    print("📦 步骤4: 重新加载模型")
    print("=" * 60)
    
    print("在新的终端中执行:")
    print("  > ollama run qwen3-vl:2b \"hello\"")
    print()
    print("这将:")
    print("  - 加载模型到内存")
    print("  - 测试模型是否正常工作")
    print("  - 保持模型在运行状态")

def step5_verify():
    """步骤5: 验证模型"""
    print("\n" + "=" * 60)
    print("✅ 步骤5: 验证模型状态")
    print("=" * 60)
    
    try:
        result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=10)
        print("ollama ps 输出:")
        print(result.stdout)
        
        if "qwen3-vl:2b" in result.stdout and "Running" in result.stdout:
            print("\n✅ 模型正在运行!")
            
            # 测试生成
            print("\n测试生成...")
            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": "qwen3-vl:2b", "prompt": "hi", "stream": False},
                    timeout=10
                )
                if r.status_code == 200:
                    print("✅ 模型响应正常!")
                    return True
            except:
                pass
                
        elif "Stopping" in result.stdout:
            print("⚠️ 模型仍在停止中...")
        else:
            print("❌ 模型未运行")
            
    except Exception as e:
        print(f"错误: {e}")
    
    return False

def alternative_solution():
    """替代方案: 使用更小的模型"""
    print("\n" + "=" * 60)
    print("🔄 替代方案: 使用更小的模型")
    print("=" * 60)
    
    print("如果内存问题持续，可以:")
    print()
    print("选项1: 使用更小的视觉模型")
    print("  > ollama pull llava:7b")
    print("  然后修改代码中的模型名称")
    print()
    print("选项2: 使用纯文本模型 (不需要GPU内存)")
    print("  > ollama pull qwen2.5:1.5b")
    print("  修改代码使用 qwen2.5:1.5b")
    print()
    print("选项3: 清理内存后重试")
    print("  > 关闭不需要的程序")
    print("  > 重启 Ollama")
    print("  > 只运行 Text2Loc 系统")

def main():
    print("\n" + "=" * 60)
    print("🛠️ 修复 qwen3-vl:2b 超时问题")
    print("=" * 60)
    print()
    print("问题诊断:")
    print("  ❌ 模型状态: Stopping...")
    print("  ❌ 内存不足: 已使用 95.6%")
    print()
    
    step1_stop_model()
    step2_check_memory()
    step3_restart_ollama()
    step4_reload_model()
    
    verified = step5_verify()
    
    if not verified:
        alternative_solution()
    
    print("\n" + "=" * 60)
    print("📋 操作总结")
    print("=" * 60)
    print("1. 停止模型: ollama stop qwen3-vl:2b")
    print("2. 重启 Ollama 服务")
    print("3. 重新加载: ollama run qwen3-vl:2b \"hello\"")
    print("4. 验证: python check_ollama_status.py")
    print()

if __name__ == "__main__":
    main()
