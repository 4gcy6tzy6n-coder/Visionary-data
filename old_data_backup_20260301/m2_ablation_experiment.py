#!/usr/bin/env python3
"""M2结构化NLU消融实验 - Text2Loc Visionary"""

import sys
import os
import time
import json
sys.path.insert(0, os.path.dirname(__file__))

from enhancements.nlu.ollama_engine import OllamaNLUEngine, OllamaConfig
from api.text2loc_adapter import Text2LocAdapter

def run_experiment(with_structured_nlu=True):
    """运行消融实验"""
    
    mode_str = "有M2（结构化NLU）" if with_structured_nlu else "无M2（结构化NLU）"
    print(f"\n{'='*80}")
    print(f"M2消融实验 - {mode_str}")
    print(f"{'='*80}")
    
    # 1. 初始化NLU引擎
    print("\n【初始化】NLU引擎...")
    nlu_config = OllamaConfig(
        base_url="http://localhost:11434",
        model="qwen3-vl:4b",
        temperature=0.3,
        timeout=30
    )
    nlu_engine = OllamaNLUEngine(nlu_config)
    
    conn_result = nlu_engine.test_connection()
    if not conn_result['success']:
        print(f"   ❌ NLU引擎连接失败")
        return None
    
    # 2. 初始化Text2LocAdapter
    print("【初始化】Text2LocAdapter...")
    adapter = Text2LocAdapter()
    
    # 3. 测试查询
    test_queries = [
        "在建筑物左侧的树",
        "红色的汽车",
        "前方的蓝色标志",
        "靠近绿色房子",
        "在北面的灰色建筑"
    ]
    
    print(f"\n【测试】执行{len(test_queries)}个查询...")
    print("-" * 80)
    
    total_success = 0
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询 {i}: \"{query}\"")
        
        # Step 1: NLU解析
        start_time = time.time()
        nlu_result = nlu_engine.parse(query)
        nlu_time = time.time() - start_time
        
        if nlu_result.error:
            print(f"   ❌ NLU解析失败")
            results.append({"query": query, "success": False})
            continue
        
        # 获取结构化字段
        direction = nlu_result.components.get('direction', 'none')
        color = nlu_result.components.get('color', 'none')
        obj = nlu_result.components.get('object', 'none')
        
        print(f"   NLU解析结果:")
        print(f"      方向: {direction}")
        print(f"      颜色: {color}")
        print(f"      对象: {obj}")
        
        # Step 2: 根据实验模式决定传入参数
        if with_structured_nlu:
            # 有M2：使用真实解析结果
            pass_params = {
                'direction': direction,
                'color': color,
                'obj': obj
            }
            print(f"   ✅ 使用结构化参数")
        else:
            # 无M2：全部设为'none'
            pass_params = {
                'direction': 'none',
                'color': 'none',
                'obj': 'none'
            }
            print(f"   ⚠️ 使用'none'参数（模拟无M2）")
        
        # Step 3: Text2Loc定位
        start_time = time.time()
        location_results = adapter.find_location(
            query=query,
            top_k=3,
            **pass_params
        )
        loc_time = time.time() - start_time
        
        if location_results:
            best = location_results[0]
            # 处理字典格式的返回结果
            if isinstance(best, dict):
                cell_id = best.get('cell_id', 'unknown')
                x = best.get('x', 0)
                y = best.get('y', 0)
                score = best.get('score', 0)
            else:
                cell_id = best.cell_id
                x = best.x
                y = best.y
                score = best.score
            
            print(f"   ✅ 定位成功 ({loc_time*1000:.1f}ms)")
            print(f"      Cell: {cell_id}")
            print(f"      坐标: ({x:.2f}, {y:.2f})")
            print(f"      得分: {score:.3f}")
            total_success += 1
            results.append({
                "query": query,
                "success": True,
                "cell_id": cell_id,
                "x": x,
                "y": y,
                "score": score
            })
        else:
            print(f"   ❌ 定位失败")
            results.append({"query": query, "success": False})
    
    # 统计结果
    success_rate = total_success / len(test_queries) * 100
    print(f"\n{'='*80}")
    print(f"实验结果统计 - {mode_str}")
    print(f"{'='*80}")
    print(f"总查询数: {len(test_queries)}")
    print(f"成功数: {total_success}")
    print(f"成功率: {success_rate:.1f}%")
    
    return {
        "mode": mode_str,
        "total": len(test_queries),
        "success": total_success,
        "success_rate": success_rate,
        "results": results
    }

def main():
    """主函数：分别运行有/无M2两种实验"""
    
    print("="*80)
    print("M2结构化NLU消融实验")
    print("Text2Loc Visionary")
    print("="*80)
    
    # 实验1：有M2（结构化NLU）
    result_with_m2 = run_experiment(with_structured_nlu=True)
    
    # 实验2：无M2（模拟）
    result_without_m2 = run_experiment(with_structured_nlu=False)
    
    # 对比结果
    if result_with_m2 and result_without_m2:
        print("\n" + "="*80)
        print("M2消融实验对比结果")
        print("="*80)
        print(f"\n{'模式':<20} {'成功数':<10} {'成功率':<15}")
        print("-"*50)
        print(f"{'有M2（结构化）':<20} {result_with_m2['success']:<10} {result_with_m2['success_rate']:.1f}%")
        print(f"{'无M2（全部none）':<20} {result_without_m2['success']:<10} {result_without_m2['success_rate']:.1f}%")
        
        improvement = result_with_m2['success_rate'] - result_without_m2['success_rate']
        print(f"\n提升: +{improvement:.1f}个百分点")
        
        # 保存结果
        experiment_result = {
            "experiment": "M2结构化NLU消融实验",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "with_m2": result_with_m2,
            "without_m2": result_without_m2,
            "improvement_percent": improvement
        }
        
        with open("m2_ablation_results.json", "w", encoding="utf-8") as f:
            json.dump(experiment_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存: m2_ablation_results.json")

if __name__ == "__main__":
    main()
