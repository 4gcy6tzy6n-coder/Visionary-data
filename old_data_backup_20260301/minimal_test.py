"""
极简测试脚本
"""

print("开始测试...")

try:
    print("导入SystemMonitor...")
    from api.monitoring import SystemMonitor
    print("  导入成功")
    
    print("创建SystemMonitor实例...")
    monitor = SystemMonitor()
    print("  创建成功")
    
    print("调用get_system_status...")
    status = monitor.get_system_status()
    print(f"  成功: {status['status']}")
    
    print("\n✅ 所有测试通过!")
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
