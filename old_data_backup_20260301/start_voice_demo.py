#!/usr/bin/env python3
"""
Text2Loc Visionary 语音交互演示启动脚本
使用Python启动前后端服务
"""

import os
import sys
import subprocess
import time
import signal
import webbrowser
from pathlib import Path

# 颜色定义
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'

def print_header():
    print(f"{Colors.GREEN}==========================================")
    print("Text2Loc Visionary 语音交互演示启动脚本")
    print("=========================================={Colors.END}\n")

def check_ollama():
    """检查Ollama服务"""
    print(f"{Colors.BLUE}[1/4] 检查Ollama服务...{Colors.END}")
    try:
        import urllib.request
        req = urllib.request.Request('http://localhost:11434/api/tags')
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print(f"{Colors.GREEN}✓ Ollama服务运行正常{Colors.END}\n")
                return True
    except Exception as e:
        print(f"{Colors.RED}✗ Ollama服务未运行{Colors.END}")
        print(f"请先在另一个终端运行: ollama serve")
        return False

def check_models():
    """检查模型是否已下载"""
    print(f"{Colors.BLUE}[2/4] 检查模型...{Colors.END}")
    try:
        import urllib.request
        import json
        req = urllib.request.Request('http://localhost:11434/api/tags')
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = [m['name'] for m in data.get('models', [])]
            
            has_vl = any('qwen3-vl:4b' in m for m in models)
            has_embedding = any('qwen3-embedding:0.6b' in m for m in models)
            
            if has_vl and has_embedding:
                print(f"{Colors.GREEN}✓ 模型检查完成{Colors.END}")
                print(f"  - qwen3-vl:4b ✓")
                print(f"  - qwen3-embedding:0.6b ✓\n")
                return True
            else:
                print(f"{Colors.YELLOW}⚠ 部分模型未找到{Colors.END}")
                if not has_vl:
                    print(f"  - qwen3-vl:4b ✗ (请运行: ollama pull qwen3-vl:4b)")
                if not has_embedding:
                    print(f"  - qwen3-embedding:0.6b ✗ (请运行: ollama pull qwen3-embedding:0.6b)")
                return False
    except Exception as e:
        print(f"{Colors.RED}✗ 检查模型失败: {e}{Colors.END}\n")
        return False

def start_backend():
    """启动后端服务"""
    print(f"{Colors.BLUE}[3/4] 启动后端服务...{Colors.END}")
    
    # 复制本机配置
    env_file = Path(__file__).parent / '.env.local'
    if env_file.exists():
        target = Path(__file__).parent / '.env'
        with open(env_file, 'r') as f:
            content = f.read()
        with open(target, 'w') as f:
            f.write(content)
        print(f"{Colors.GREEN}✓ 后端配置已激活{Colors.END}")
    
    # 启动后端
    backend_process = subprocess.Popen(
        [sys.executable, '-m', 'api.server', '--host', '127.0.0.1', '--port', '8080'],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待后端启动
    time.sleep(3)
    
    # 检查后端是否启动成功
    try:
        import urllib.request
        req = urllib.request.Request('http://localhost:8080/health')
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print(f"{Colors.GREEN}✓ 后端服务已启动 (端口: 8080){Colors.END}\n")
                return backend_process
    except Exception as e:
        print(f"{Colors.RED}✗ 后端服务启动失败{Colors.END}")
        backend_process.terminate()
        return None

def start_frontend():
    """启动前端服务"""
    print(f"{Colors.BLUE}[4/4] 启动前端服务...{Colors.END}")
    
    frontend_dir = Path(__file__).parent / 'frontend'
    
    # 复制本机配置
    env_file = frontend_dir / '.env.local'
    if env_file.exists():
        target = frontend_dir / '.env'
        with open(env_file, 'r') as f:
            content = f.read()
        with open(target, 'w') as f:
            f.write(content)
        print(f"{Colors.GREEN}✓ 前端配置已激活{Colors.END}")
    
    # 使用Python的http.server启动前端
    frontend_process = subprocess.Popen(
        [sys.executable, '-m', 'http.server', '5173'],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待前端启动
    time.sleep(2)
    
    print(f"{Colors.GREEN}✓ 前端服务已启动 (端口: 5173){Colors.END}\n")
    return frontend_process

def print_access_info():
    """打印访问信息"""
    print(f"{Colors.GREEN}==========================================")
    print("服务启动成功！")
    print("=========================================={Colors.END}\n")
    
    print(f"{Colors.GREEN}访问地址:{Colors.END}")
    print(f"  🎙️  语音版: {Colors.BLUE}http://localhost:5173/voice.html{Colors.END}")
    print(f"  📝 文本版: {Colors.BLUE}http://localhost:5173/{Colors.END}")
    print(f"  ⚙️  配置页: {Colors.BLUE}http://localhost:5173/config.html{Colors.END}")
    print(f"  🔌 后端API: {Colors.BLUE}http://localhost:8080{Colors.END}\n")
    
    print(f"{Colors.YELLOW}使用说明:{Colors.END}")
    print("  1. 在浏览器中打开语音版地址")
    print("  2. 点击麦克风按钮开始语音输入")
    print("  3. 用普通话描述位置，如：'找到红色的汽车'")
    print("  4. 系统会自动识别并显示结果\n")
    
    print(f"{Colors.YELLOW}提示:{Colors.END}")
    print("  - 请使用 Chrome、Edge 或 Safari 浏览器")
    print("  - 首次使用需要允许麦克风权限")
    print("  - 按 Ctrl+C 停止所有服务\n")

def open_browser():
    """自动打开浏览器"""
    try:
        webbrowser.open('http://localhost:5173/voice.html')
        print(f"{Colors.GREEN}✓ 已自动打开浏览器{Colors.END}\n")
    except Exception as e:
        print(f"{Colors.YELLOW}请手动打开浏览器访问上述地址{Colors.END}\n")

def main():
    print_header()
    
    # 检查Ollama
    if not check_ollama():
        print(f"{Colors.RED}请先启动Ollama服务后再运行此脚本{Colors.END}")
        sys.exit(1)
    
    # 检查模型
    check_models()
    
    # 启动后端
    backend = start_backend()
    if not backend:
        print(f"{Colors.RED}后端服务启动失败，请检查日志{Colors.END}")
        sys.exit(1)
    
    # 启动前端
    frontend = start_frontend()
    if not frontend:
        print(f"{Colors.RED}前端服务启动失败{Colors.END}")
        backend.terminate()
        sys.exit(1)
    
    # 打印访问信息
    print_access_info()
    
    # 自动打开浏览器
    open_browser()
    
    # 等待用户中断
    print(f"{Colors.YELLOW}服务运行中，按 Ctrl+C 停止...{Colors.END}\n")
    
    def signal_handler(sig, frame):
        print(f"\n{Colors.YELLOW}正在停止服务...{Colors.END}")
        backend.terminate()
        frontend.terminate()
        print(f"{Colors.GREEN}✓ 服务已停止{Colors.END}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == '__main__':
    main()
