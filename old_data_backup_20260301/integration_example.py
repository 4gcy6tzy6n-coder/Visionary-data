# -*- coding: utf-8 -*-
"""
Text2Loc 智能查询集成示例

演示完整的流程：
1. 用户输入自然语言
2. Qwen模型智能分析
3. 转换为Text2Loc标准格式
4. 传递给原始Text2Loc系统
5. 返回真正的定位结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhancements.nlu.optimized_engine import OptimizedNLUEngine, NLUConfig


def simulate_original_text2loc(standard_format):
    """
    模拟原始Text2Loc系统的处理
    
    这里应该调用真实的Text2Loc系统（D:\Text2Loc-main\Text2Loc-main）
    """
    print(f"\n📤 传递给原始Text2Loc系统:")
    print(f"   对象类别: {standard_format.object_label}")
    print(f"   颜色: {standard_format.object_color}")
    print(f"   方向: {standard_format.direction}")
    print(f"   描述: {standard_format.description}")
    print()
    
    # 模拟Text2Loc系统的定位结果
    # 实际使用时，这里应该调用：
    # from Text2Loc_main.models import CellRetrievalNetwork
    # result = model.retrieve(standard_format)
    
    simulated_results = [
        {
            "cell_id": "cell_001",
            "score": 0.92,
            "description": f"{standard_format.description} - 定位成功",
            "pose_id": "pose_042"
        },
        {
            "cell_id": "cell_002",
            "score": 0.85,
            "description": f"{standard_format.description} - 备选位置",
            "pose_id": "pose_043"
        },
        {
            "cell_id": "cell_003",
            "score": 0.78,
            "description": f"{standard_format.description} - 远处位置",
            "pose_id": "pose_044"
        }
    ]
    
    print(f"✅ Text2Loc系统返回结果:")
    for i, result in enumerate(simulated_results, 1):
        print(f"   {i}. 单元格: {result['cell_id']}, 置信度: {result['score']:.2f}")
        print(f"      描述: {result['description']}")
    print()
    
    return simulated_results


def complete_workflow(natural_language_query):
    """
    完整的工作流程：自然语言 → 智能解析 → Text2Loc定位
    """
    print("=" * 70)
    print("完整工作流程演示")
    print("=" * 70)
    print()
    print(f"用户输入: {natural_language_query}")
    print()
    
    # 1. 创建NLU引擎
    config = NLUConfig(mock_mode=True, enable_dialog=False)
    engine = OptimizedNLUEngine(config)
    
    # 2. 智能解析到Text2Loc标准格式
    print("【步骤1: 智能解析】")
    print("-" * 70)
    standard_format, confidence, intent = engine.parse_to_standard_format(natural_language_query)
    
    print(f"✅ 解析完成:")
    print(f"   意图: {intent}")
    print(f"   置信度: {confidence:.2f}")
    print(f"   标准格式: {standard_format.to_dict()}")
    print()
    
    # 3. 传递给原始Text2Loc系统
    print("【步骤2: 传递给Text2Loc系统】")
    print("-" * 70)
    results = simulate_original_text2loc(standard_format)
    
    # 4. 返回最终结果
    print("【步骤3: 返回定位结果】")
    print("-" * 70)
    print(f"✅ 定位完成！")
    print(f"   找到 {len(results)} 个候选位置")
    print(f"   最佳位置: {results[0]['cell_id']} (置信度: {results[0]['score']:.2f})")
    print()
    
    return {
        "query": natural_language_query,
        "standard_format": standard_format.to_dict(),
        "confidence": confidence,
        "intent": intent,
        "results": results
    }


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("Text2Loc 智能查询集成示例")
    print("=" * 70)
    print()
    print("完整流程:")
    print("  1. 用户输入自然语言描述")
    print("  2. Qwen模型智能分析理解")
    print("  3. 转换为Text2Loc标准格式")
    print("  4. 传递给原始Text2Loc系统")
    print("  5. 返回真正的定位结果")
    print()
    
    # 测试用例
    test_queries = [
        "我在红色大楼的北侧",
        "停车场东边有一棵大树",
        "桥的左侧",
        "我在树林靠近山的位置",
        "交通灯的东边有一个停车区域"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"【测试用例 {i}】")
        print(f"{'=' * 70}")
        
        try:
            result = complete_workflow(query)
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ 所有测试完成！")
    print("=" * 70)
    print()
    print("🎯 关键点:")
    print("  ✓ 自然语言 → 智能解析 → 标准格式")
    print("  ✓ 无缝集成到原始Text2Loc系统")
    print("  ✓ 保持原有系统的定位能力")
    print("  ✓ 支持自由的自然语言输入")
    print()
    print("📝 集成到实际系统:")
    print("  1. 修改 simulate_original_text2loc() 函数")
    print("  2. 导入真实的Text2Loc模型")
    print("  3. 调用 model.retrieve(standard_format)")
    print("  4. 返回真实的定位结果")
    print()


if __name__ == "__main__":
    main()
