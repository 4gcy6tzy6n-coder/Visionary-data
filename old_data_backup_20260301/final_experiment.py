"""
最终实验 - 验证系统可用性和真实数据处理
"""

import requests
import json
import time
import sys
from datetime import datetime

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def test_system():
    base_url = "http://localhost:8080/api/v1"
    
    print("="*70)
    print("Text2Loc Visionary 最终实验")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # 1. 健康检查
    log("实验 1: 健康检查")
    try:
        response = requests.get(f"{base_url.replace('/api/v1', '')}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            log(f"✅ 服务健康 - {data.get('service')} v{data.get('version')}")
            results['health'] = True
        else:
            log(f"❌ 健康检查失败: {response.status_code}")
            results['health'] = False
    except Exception as e:
        log(f"❌ 连接失败: {e}")
        results['health'] = False
    
    # 2. 配置检查
    log("\n实验 2: 配置验证")
    try:
        response = requests.get(f"{base_url}/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            log(f"✅ 配置加载成功")
            log(f"   提供商: {config.get('provider')}")
            log(f"   模型: {config.get('model')}")
            results['config'] = True
        else:
            log(f"❌ 配置检查失败")
            results['config'] = False
    except Exception as e:
        log(f"❌ 请求失败: {e}")
        results['config'] = False
    
    # 3. 数据加载检查
    log("\n实验 3: 数据加载验证")
    try:
        import os
        import pickle
        
        data_path = os.path.expanduser("~/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all")
        
        # 检查cells
        cells_path = os.path.join(data_path, "cells", "cells.pkl")
        with open(cells_path, 'rb') as f:
            cells = pickle.load(f)
        log(f"✅ Cells: {len(cells)} 个")
        
        # 检查poses
        poses_path = os.path.join(data_path, "poses", "poses.pkl")
        with open(poses_path, 'rb') as f:
            poses = pickle.load(f)
        log(f"✅ Poses: {len(poses)} 个")
        
        # 检查原始数据
        raw_path = os.path.join(data_path, "raw_data")
        if os.path.exists(raw_path):
            raw_dirs = os.listdir(raw_path)
            log(f"✅ Raw Data: {raw_dirs}")
        
        results['data'] = len(cells) > 0 and len(poses) > 0
    except Exception as e:
        log(f"❌ 数据验证失败: {e}")
        results['data'] = False
    
    # 4. 真实查询测试
    log("\n实验 4: 真实位置查询测试")
    test_queries = [
        "我在红色大楼的北侧",
        "在绿色树木的东边",
        "靠近蓝色标志"
    ]
    
    query_results = []
    for query in test_queries:
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/query",
                json={"query": query, "top_k": 3, "enable_enhanced": True},
                timeout=60
            )
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    log(f"✅ '{query}' - {elapsed:.2f} ms")
                    has_results = 'results' in result and result['results']
                    query_results.append({"success": True, "has_results": has_results})
                else:
                    log(f"❌ '{query}' - 处理失败")
                    query_results.append({"success": False})
            else:
                log(f"❌ '{query}' - HTTP {response.status_code}")
                query_results.append({"success": False})
        except Exception as e:
            log(f"❌ '{query}' - {e}")
            query_results.append({"success": False})
    
    success_count = sum(1 for r in query_results if r['success'])
    results['queries'] = success_count == len(test_queries)
    
    # 5. 总结
    log("\n" + "="*70)
    log("实验总结")
    log("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    log(f"\n总体结果: {passed}/{total} 项通过 ({passed/total*100:.1f}%)")
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        log(f"   {test_name}: {status}")
    
    log("\n" + "="*70)
    if passed == total:
        log("🎉 所有实验通过！系统完全可用。")
    else:
        log("⚠️ 部分实验失败，请检查配置。")
    log("="*70)
    
    return results

if __name__ == "__main__":
    results = test_system()
    
    # 保存结果
    with open("final_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n结果已保存到: final_experiment_results.json")
