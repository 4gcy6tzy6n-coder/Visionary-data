#!/usr/bin/env python3
"""
启动 Text2Loc 服务器（支持 DeepSeek 集成）
"""
import os
import sys
import subprocess
import time
import dotenv

def load_env():
    """加载环境变量"""
    # 尝试加载 .env 文件
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        dotenv.load_dotenv(env_file)
        print(f"✅ 已加载环境变量: {env_file}")
    
    # 检查关键配置
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key:
        print(f"✅ DeepSeek API Key: {api_key[:10]}...")
        print(f"   URL: {os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')}")
        print(f"   模型: {os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')}")
    else:
        print("⚠️ 未设置 DEEPSEEK_API_KEY，将使用规则解析模式")

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("🚀 启动 Text2Loc Visionary 服务器")
    print("=" * 60)
    
    # 加载环境变量
    load_env()
    
    print()
    print("📡 API: http://localhost:8080")
    print("🔧 配置: 设置 DEEPSEEK_API_KEY 启用 DeepSeek")
    print("🛑 按 Ctrl+C 停止")
    print()
    
    cmd = [sys.executable, "-m", "api.server"]
    
    try:
        print("启动服务器...")
        proc = subprocess.Popen(cmd)
        
        # 等待服务器初始化
        time.sleep(3)
        
        # 检查端口
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            result = sock.connect_ex(('127.0.0.1', 8080))
            if result == 0:
                print("\n✅ 服务器已就绪!")
            else:
                print("\n❌ 服务器未响应")
        except Exception as e:
            print(f"\n❌ 连接错误: {e}")
        finally:
            sock.close()
        
        # 等待进程
        proc.wait()
        
    except KeyboardInterrupt:
        print("\n👋 停止服务器...")
        proc.terminate()
    except Exception as e:
        print(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    main()
