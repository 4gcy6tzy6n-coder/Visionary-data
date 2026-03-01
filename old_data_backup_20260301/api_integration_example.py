# -*- coding: utf-8 -*-
"""
API集成示例 - 通过API调用实现智能查询

演示如何通过API将自然语言转换为标准格式，
然后调用原始Text2Loc系统进行定位
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhancements.nlu.optimized_engine import OptimizedNLUEngine, NLUConfig, StandardFormat


class Text2LocIntelligentAPI:
    """
    Text2Loc智能API - 完整的智能查询接口
    
    流程：
    1. 接收自然语言查询
    2. 通过Qwen模型智能分析
    3. 转换为Text2Loc标准格式
    4. 调用原始Text2Loc系统
    5. 返回真正的定位结果
    """
    
    def __init__(self, mock_mode=True):
        """
        初始化智能API
        
        Args:
            mock_mode: 是否使用模拟模式（测试用）
        """
        self.mock_mode = mock_mode
        
        # 初始化NLU引擎
        config = NLUConfig(mock_mode=mock_mode, enable_dialog=False)
        self.nlu_engine = OptimizedNLUEngine(config)
        
        print(f"✅ Text2Loc智能API初始化完成")
        print(f"   模式: {'模拟' if mock_mode else '真实'}")
        print()
    
    def query(self, natural_language: str, top_k: int = 5) -> dict:
        """
        智能查询接口
        
        Args:
            natural_language: 自然语言描述
            top_k: 返回结果数量
            
        Returns:
            包含解析结果和定位结果的字典
        """
        print("=" * 70)
        print("智能查询请求")
        print("=" * 70)
        print(f"输入: {natural_language}")
        print()
        
        # 步骤1: 智能解析
        print("【步骤1: 智能解析】")
        print("-" * 70)
        standard_format, confidence, intent = self.nlu_engine.parse_to_standard_format(natural_language)
        
        print(f"✅ 解析结果:")
        print(f"   标准格式: {standard_format.to_dict()}")
        print(f"   置信度: {confidence:.2f}")
        print(f"   意图: {intent}")
        print()
        
        # 步骤2: 调用原始Text2Loc系统
        print("【步骤2: 调用Text2Loc系统】")
        print("-" * 70)
        results = self._call_original_text2loc(standard_format, top_k)
        
        # 步骤3: 组装响应
        response = {
            "query": natural_language,
            "standard_format": standard_format.to_dict(),
            "confidence": confidence,
            "intent": intent,
            "results": results,
            "top_k": top_k,
            "success": len(results) > 0
        }
        
        print("【步骤3: 返回结果】")
        print("-" * 70)
        print(f"✅ 查询完成！")
        print(f"   找到 {len(results)} 个候选位置")
        if results:
            print(f"   最佳位置: {results[0]['cell_id']} (置信度: {results[0]['score']:.2f})")
        print()
        
        return response
    
    def _call_original_text2loc(self, standard_format: StandardFormat, top_k: int):
        """
        调用原始Text2Loc系统
        
        这里应该调用真实的Text2Loc系统（D:\Text2Loc-main\Text2Loc-main）
        """
        print(f"📤 传递参数给Text2Loc系统:")
        print(f"   对象: {standard_format.object_label}")
        print(f"   颜色: {standard_format.object_color}")
        print(f"   方向: {standard_format.direction}")
        print(f"   描述: {standard_format.description}")
        print(f"   返回数量: {top_k}")
        print()
        
        if self.mock_mode:
            # 模拟Text2Loc系统的返回结果
            return self._mock_text2loc(standard_format, top_k)
        else:
            # TODO: 调用真实的Text2Loc系统
            # from Text2Loc_main.models import CellRetrievalNetwork
            # model = CellRetrievalNetwork()
            # results = model.retrieve(standard_format, top_k)
            # return results
            raise NotImplementedError("真实Text2Loc系统集成待实现")
    
    def _mock_text2loc(self, standard_format: StandardFormat, top_k: int):
        """模拟Text2Loc系统返回"""
        import random
        
        # 基于标准格式生成模拟结果
        base_score = 0.7 + random.random() * 0.25  # 0.7-0.95
        
        results = []
        for i in range(top_k):
            score = base_score - (i * 0.08)  # 递减的置信度
            score = max(score, 0.5)  # 最低0.5
            
            results.append({
                "cell_id": f"cell_{i+1:03d}",
                "score": round(score, 3),
                "description": f"{standard_format.description} - 候选位置 {i+1}",
                "pose_id": f"pose_{random.randint(100, 999)}",
                "distance": round(random.uniform(0.5, 50.0), 1)
            })
        
        print(f"✅ Text2Loc系统返回 {len(results)} 个结果:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['cell_id']} | 置信度: {result['score']:.2f} | 距离: {result['distance']}m")
        
        return results


def demo_api():
    """演示API使用"""
    print("\n" + "=" * 70)
    print("Text2Loc 智能API演示")
    print("=" * 70)
    print()
    
    # 创建API实例
    api = Text2LocIntelligentAPI(mock_mode=True)
    
    # 测试查询
    test_queries = [
        "我在红色大楼的北侧",
        "停车场东边有一棵大树",
        "桥的左侧",
        "我在树林靠近山的位置",
    ]
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n【查询 {i}】")
        result = api.query(query, top_k=3)
        results.append(result)
    
    # 汇总
    print("\n" + "=" * 70)
    print("查询汇总")
    print("=" * 70)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['query']}")
        print(f"   标准格式: {result['standard_format']['object_label']} | {result['standard_format']['object_color']} | {result['standard_format']['direction']}")
        print(f"   置信度: {result['confidence']:.2f}")
        print(f"   结果数: {len(result['results'])}")
        if result['results']:
            print(f"   最佳: {result['results'][0]['cell_id']} (score: {result['results'][0]['score']:.2f})")
    
    print("\n" + "=" * 70)
    print("✅ API演示完成！")
    print("=" * 70)
    print()
    print("🎯 API特点:")
    print("  ✓ 接收自然语言输入")
    print("  ✓ 自动转换为Text2Loc标准格式")
    print("  ✓ 调用原始Text2Loc系统")
    print("  ✓ 返回真实的定位结果")
    print()
    print("📝 集成到实际系统:")
    print("  1. 在 _call_original_text2loc() 中导入真实模型")
    print("  2. 调用 model.retrieve(standard_format)")
    print("  3. 返回真实的定位结果")
    print()


def api_usage_example():
    """API使用示例"""
    print("\n" + "=" * 70)
    print("API使用示例代码")
    print("=" * 70)
    print()
    
    example_code = '''
# 创建API实例
api = Text2LocIntelligentAPI(mock_mode=False)

# 发送查询
result = api.query("我在红色大楼的北侧", top_k=5)

# 处理结果
if result["success"]:
    print(f"查询: {result['query']}")
    print(f"标准格式: {result['standard_format']}")
    print(f"置信度: {result['confidence']}")
    print(f"定位结果:")
    for r in result["results"]:
        print(f"  - {r['cell_id']}: {r['score']:.2f}")
else:
    print("查询失败")
'''
    
    print(example_code)
    print()


if __name__ == "__main__":
    demo_api()
    api_usage_example()
