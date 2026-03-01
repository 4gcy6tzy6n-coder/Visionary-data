"""
检查端口监听状态
"""
import socket

def check_port(port, host='127.0.0.1'):
    """检查端口是否在监听"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex((host, port))
        if result == 0:
            print(f"✅ 端口 {port} 正在监听")
            return True
        else:
            print(f"❌ 端口 {port} 未监听")
            return False
    except Exception as e:
        print(f"❌ 检查端口 {port} 失败: {e}")
        return False
    finally:
        sock.close()

def main():
    print("=" * 60)
    print("🔌 检查端口监听状态")
    print("=" * 60)
    
    ports = [8088, 6001, 8080]
    
    for port in ports:
        check_port(port)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
