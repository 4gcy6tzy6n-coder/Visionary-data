#!/usr/bin/env python3
"""
启动 Text2Loc API 服务器
"""
import subprocess
import sys
import time
import os

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("🚀 启动 Text2Loc Visionary 服务器")
    print("=" * 60)
    print()
    print("API: http://localhost:8080")
    print("按 Ctrl+C 停止")
    print()
    
    cmd = [sys.executable, "-m", "api.server"]
    
    try:
        proc = subprocess.Popen(cmd)
        print(f"进程已启动 (PID: {proc.pid})")
        print("等待服务器初始化...")
        time.sleep(3)
        
        # 检查端口
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            result = sock.connect_ex(('127.0.0.1', 8080))
            if result == 0:
                print("✅ 服务器已就绪!")
            else:
                print("❌ 服务器未响应")
        except:
            print("❌ 无法连接服务器")
        finally:
            sock.close()
        
        proc.wait()
    except KeyboardInterrupt:
        print("\n👋 停止服务器...")
        proc.terminate()
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    main()
