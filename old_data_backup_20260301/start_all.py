"""
Text2Loc Visionary - 完整启动脚本
同时启动前后端服务
"""

import subprocess
import sys
import os
import time
import requests
import json

def run_command(cmd, title, cwd=None):
    """运行命令"""
    print(f"🚀 启动 {title}...")
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd or os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return proc
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return None

def wait_for_server(url, timeout=10):
    """等待服务器启动"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def main():
    print("=" * 60)
    print("   Text2Loc Visionary - 智能位置定位系统")
    print("=" * 60)
    print()
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(project_dir, "frontend")
    
    processes = []
    
    try:
        # 启动后端API服务
        backend_cmd = f'python -m api.server --port 8088'
        proc = run_command(backend_cmd, "后端API服务 (端口: 8088)", project_dir)
        if proc:
            processes.append(("后端API", proc))
        
        time.sleep(2)
        
        # 启动前端Web服务
        frontend_cmd = f'python -m http.server 6001 -d frontend'
        proc = run_command(frontend_cmd, "前端Web服务 (端口: 6001)", project_dir)
        if proc:
            processes.append(("前端Web", proc))
        
        time.sleep(2)
        
        # 验证服务
        print()
        print("=" * 60)
        print("   服务状态验证")
        print("=" * 60)
        print()
        
        # 检查后端健康
        try:
            resp = requests.get("http://localhost:8088/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"✅ 后端服务: 健康 (状态: {data.get('status')})")
            else:
                print(f"❌ 后端服务: HTTP {resp.status_code}")
        except Exception as e:
            print(f"❌ 后端服务: 连接失败 - {e}")
        
        # 测试查询
        try:
            resp = requests.post(
                "http://localhost:8088/api/v1/query",
                json={
                    "query": "我站在红色大楼的北侧约5米处",
                    "top_k": 3,
                    "enable_enhanced": True,
                    "return_debug_info": True
                },
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    analysis = data.get("query_analysis", {})
                    print(f"✅ 查询测试: 成功")
                    print(f"   - 方向: {analysis.get('direction', 'N/A')}")
                    print(f"   - 颜色: {analysis.get('color', 'N/A')}")
                    print(f"   - 对象: {analysis.get('object', 'N/A')}")
                    print(f"   - 置信度: {analysis.get('confidence', 0) * 100:.0f}%")
                    print(f"   - NLU模型: {analysis.get('nlu_model', 'N/A')}")
                else:
                    print(f"❌ 查询测试: 失败 - {data.get('error')}")
            else:
                print(f"❌ 查询测试: HTTP {resp.status_code}")
        except Exception as e:
            print(f"❌ 查询测试: 失败 - {e}")
        
        print()
        print("=" * 60)
        print("   ✅ 系统启动完成！")
        print("=" * 60)
        print()
        print("📱 访问地址:")
        print("   - 前端界面: http://localhost:6001")
        print("   - API健康检查: http://localhost:8088/health")
        print("   - API查询接口: http://localhost:8088/api/v1/query")
        print("   - 测试页面: http://localhost:6001/test.html")
        print()
        print("🧪 测试查询:")
        print('   curl -X POST http://localhost:8088/api/v1/query \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d "{\"query\":\"我站在红色大楼的北侧约5米处\",\"top_k\":3}"')
        print()
        print("⏸️  按 Ctrl+C 停止所有服务")
        print("=" * 60)
        
        # 等待用户中断
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 正在停止服务...")
    finally:
        for name, proc in processes:
            if proc and proc.poll() is None:
                proc.terminate()
                print(f"   已停止 {name}")

if __name__ == "__main__":
    main()
