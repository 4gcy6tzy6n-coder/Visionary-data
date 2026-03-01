"""
启动 Text2Loc Visionary 后端服务
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 现在可以导入 api 模块
from api.server import create_server

if __name__ == "__main__":
    print("=" * 60)
    print("Text2Loc Visionary 后端服务")
    print("=" * 60)
    print()
    print("配置信息:")
    print("  - 大语言模型: qwen3-vl:4b")
    print("  - 嵌入模型: qwen3-embedding:0.6b")
    print("  - API 地址: http://localhost:8080")
    print()

    server = create_server(host='0.0.0.0', port=8080, debug=False)
    server.run()
