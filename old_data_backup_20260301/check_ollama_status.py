"""
检查Ollama状态和内存
"""
import subprocess
import psutil
import requests
import time

OLLAMA_URL = "http://localhost:11434"

def check_ollama_processes():
    """检查Ollama相关进程"""
    print("=" * 60)
    print("🔍 检查Ollama进程")
    print("=" * 60)
    
    try:
        result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=10)
        print(f"命令: ollama ps")
        print(f"返回码: {result.returncode}")
        print(f"\n输出:")
        print(result.stdout if result.stdout else "(空)")
        if result.stderr:
            print(f"错误:\n{result.stderr}")
    except FileNotFoundFoundError:
        print("❌ ollama命令未找到")
    except subprocess.TimeoutExpired:
        print("⏰ 命令超时")
    except Exception as e:
        print(f"❌ 错误: {e}")

def check_system_memory():
    """检查系统内存"""
    print("\n" + "=" * 60)
    print("💾 检查系统内存")
    print("=" * 60)
    
    try:
        memory = psutil.virtual_memory()
        print(f"总内存: {memory.total / (1024**3):.1f} GB")
        print(f"可用内存: {memory.available / (1024**3):.1f} GB")
        print(f"已使用: {memory.percent}%")
        
        if memory.percent > 80:
            print("⚠️ 内存使用率较高，可能影响模型运行")
        else:
            print("✅ 内存充足")
            
        # 检查进程内存
        print("\n占用内存最多的进程:")
        for proc in psutil.process_iter(['name', 'memory_percent']):
            try:
                info = proc.info
                if info['memory_percent'] > 1.0:
                    print(f"   {info['name']}: {info['memory_percent']:.1f}%")
            except:
                pass
    except Exception as e:
        print(f"❌ 错误: {e}")

def check_model_status():
    """检查模型状态"""
    print("\n" + "=" * 60)
    print("🤖 检查模型状态")
    print("=" * 60)
    
    # 1. 获取模型列表
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = r.json().get("models", [])
        
        qwen = None
        for m in models:
            if "qwen3-vl" in m["name"]:
                qwen = m
                break
        
        if qwen:
            print(f"模型: {qwen['name']}")
            print(f"大小: {qwen.get('size', 0) / (1024**3):.1f} GB")
            print(f"已安装: ✅")
            
            # 2. 尝试获取模型详细信息
            print("\n尝试获取模型详细信息...")
            r = requests.post(
                f"{OLLAMA_URL}/api/show",
                json={"name": "qwen3-vl:2b"},
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                print(f"模型信息: ✅")
                print(f"   摘要: {data.get('short_description', 'N/A')}")
            else:
                print(f"获取模型信息失败: {r.status_code}")
                
    except Exception as e:
        print(f"❌ 错误: {e}")

def try_simple_generation():
    """尝试最简单的生成请求"""
    print("\n" + "=" * 60)
    print("🧪 尝试最简单的生成请求")
    print("=" * 60)
    
    # 最简单的请求
    payload = {
        "model": "qwen3-vl:2b",
        "prompt": "hi",
        "stream": False
    }
    
    print(f"请求: {payload}")
    print("超时: 5秒")
    
    try:
        start = time.time()
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=5
        )
        elapsed = time.time() - start
        
        print(f"\n✅ 成功! 耗时: {elapsed:.3f}秒")
        print(f"状态码: {r.status_code}")
        print(f"响应: {r.json().get('response', '')[:100]}")
        
    except requests.exceptions.Timeout:
        print(f"\n⏰ 5秒超时!")
        print("模型可能处于以下状态:")
        print("  1. 正在后台加载中")
        print("  2. Ollama服务阻塞")
        print("  3. 内存不足导致交换")
    except Exception as e:
        print(f"\n❌ 错误: {e}")

def main():
    """主诊断流程"""
    print("\n" + "🔍" * 20)
    print("Text2Loc - Qwen3-VL:2b 超时诊断")
    print("🔍" * 20 + "\n")
    
    check_ollama_processes()
    check_system_memory()
    check_model_status()
    try_simple_generation()
    
    print("\n" + "=" * 60)
    print("📋 诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
