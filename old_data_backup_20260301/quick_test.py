"""
测试Text2Loc API - 快速验证
"""
import requests
import time

API_URL = "http://127.0.0.1:8080/api/v1/query"

def test_query(query_name, query_text, enhanced=True):
    """测试单个查询"""
    print(f"\n📝 {query_name}")
    print(f"   查询: {query_text}")
    
    start = time.time()
    try:
        response = requests.post(
            API_URL,
            json={
                "query": query_text,
                "top_k": 3,
                "enable_enhanced": enhanced,
                "return_debug_info": True
            },
            timeout=30
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ 成功! 耗时: {elapsed:.2f}秒")
            
            if data.get('query_analysis'):
                qa = data['query_analysis']
                print(f"   📊 方向: {qa.get('direction') or '未识别'}")
                print(f"   📊 颜色: {qa.get('color') or '未识别'}")
                print(f"   📊 对象: {qa.get('object') or '未识别'}")
                print(f"   📊 置信度: {qa.get('confidence') or 0}")
                print(f"   🔧 方法: {qa.get('parse_method')}")
                print(f"   🤖 AI: {qa.get('real_model_used')}")
            
            if data.get('retrieval_results'):
                print(f"   🎯 结果数: {len(data['retrieval_results'])}")
                for r in data['retrieval_results'][:1]:
                    print(f"      - {r['cell_id']}: {r['score']:.1%}")
            
            return True
        else:
            print(f"   ❌ HTTP错误: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 Text2Loc API 快速测试")
    print("=" * 60)
    
    tests = [
        ("简单位置", "我在红色大楼的北边", False),
        ("中文增强", "我在红色大楼的北边", True),
        ("复杂描述", "我站在灰色建筑物的东南角附近", True),
        ("英文查询", "I am north of a red building", True),
        ("中文问答", "树林靠近山的位置", True),
    ]
    
    results = []
    for name, query, enhanced in tests:
        success = test_query(name, query, enhanced)
        results.append((name, success))
    
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {name}: {status}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过! 系统正常工作!")
    else:
        print("⚠️ 部分测试失败，需要检查")

if __name__ == "__main__":
    main()
