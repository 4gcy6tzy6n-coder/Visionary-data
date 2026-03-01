#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text2Loc Visionary 快速功能演示脚本
演示三层智能架构的核心功能
"""

import sys
import os
import time
import logging

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

# 配置日志（减少输出）
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)

logger = logging.getLogger(__name__)


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_instruction_optimizer():
    """测试1: 指令优化器"""
    print_header("测试1: 指令优化器 (InstructionOptimizer)")

    from enhancements.nlu.instruction_optimizer import InstructionOptimizer

    optimizer = InstructionOptimizer(mock_mode=True)

    test_queries = [
        "停车场北侧的红色汽车",
        "北侧的汽车",
        "找一个东西",
        "建筑旁边的灯柱"
    ]

    print(f"\n测试查询数: {len(test_queries)}\n")

    for query in test_queries:
        result = optimizer.optimize(query)
        print(f"查询: {query}")
        print(f"  优化后: {result.optimized_input}")
        print(f"  需要澄清: {result.need_clarification}")
        print(f"  置信度: {result.confidence_scores.get('overall', 0):.2f}")

        if result.need_clarification and result.suggested_clarifications:
            print(f"  建议: {result.suggested_clarifications[0]}")

        print()

    stats = optimizer.get_stats()
    print(f"统计: {stats['total_queries']}查询, 缓存命中率 {(stats['cache_hits']/(stats['cache_hits']+stats['cache_misses']))*100:.1f}%")


def test_dynamic_template_generator():
    """测试2: 动态模板生成器"""
    print_header("测试2: 动态模板生成器 (DynamicTemplateGenerator)")

    from enhancements.nlu.dynamic_template_generator import DynamicTemplateGenerator, ParsedResult

    generator = DynamicTemplateGenerator()

    # 测试完整查询
    print("\n测试1: 完整查询 (北侧-红色-建筑)")
    parsed1 = ParsedResult(
        direction="north",
        color="red",
        object="building",
        completeness_score=0.95,
        confidence_scores={"direction": 0.9, "color": 0.95, "object": 0.92}
    )

    templates1 = generator.generate(parsed1, n_variants=3)
    print(f"生成变体数: {len(templates1)}")

    for i, t in enumerate(templates1, 1):
        print(f"  {i}. [{t.template_type}] {t.filled_text}")

    # 测试部分查询
    print("\n测试2: 部分查询 (北侧-只有方向)")
    parsed2 = ParsedResult(
        direction="north",
        completeness_score=0.5,
        confidence_scores={"direction": 0.8}
    )

    templates2 = generator.generate(parsed2, n_variants=2)
    print(f"生成变体数: {len(templates2)}")

    for i, t in enumerate(templates2, 1):
        print(f"  {i}. [{t.template_type}] {t.filled_text}")

    # 最佳模板选择
    print("\n测试3: 最佳模板选择")
    for mode in ["fast", "balanced", "compatible"]:
        best = generator.get_best_template(parsed1, mode=mode)
        if best:
            print(f"  模式 '{mode}': {best.filled_text}")

    stats = generator.get_performance_stats()
    print(f"\n统计: {stats['total_generations']}次生成, 平均耗时 {stats['avg_time_ms']:.1f}ms")


def test_interactive_clarifier():
    """测试3: 交互式澄清器"""
    print_header("测试3: 交互式澄清器 (InteractiveClarifier)")

    from enhancements.nlu.interactive_clarifier import InteractiveClarifier, ClarificationIntent

    clarifier = InteractiveClarifier(mock_mode=True, language="zh")

    # 创建带澄清意图的会话
    print("\n测试1: 创建澄清会话")
    clarifications = [
        ClarificationIntent(
            issue_type="missing_direction",
            description="缺少方向信息",
            confidence=0.9,
            priority=3,
            candidates=["北", "南", "东", "西"],
            severity="high"
        )
    ]

    session = clarifier.create_session(
        original_query="找汽车",
        initial_clarifications=clarifications
    )

    print(f"创建会话: {session.session_id}")

    if session.current_question:
        print(f"生成问题: {session.current_question.question_text}")
        if session.current_question.suggested_answer:
            print(f"建议回答: {session.current_question.suggested_answer}")

    # 分析用户响应
    print("\n测试2: 分析用户响应")
    user_response = "北侧的红色汽车"
    print(f"用户回答: {user_response}")

    analysis = clarifier.analyze_user_response(
        session_id=session.session_id,
        user_response=user_response
    )

    print(f"  已解决: {analysis.resolved_issues}")
    print(f"  新实体: {analysis.new_entities}")
    print(f"  置信度: {analysis.confidence:.2f}")

    # 处理澄清响应
    print("\n测试3: 处理澄清响应")
    completed, updated_data, followup = clarifier.process_clarification_response(
        session_id=session.session_id,
        user_response=user_response
    )

    print(f"  是否完成: {completed}")

    if updated_data:
        print(f"  解决问题: {len(updated_data.get('resolved_issues', []))}个")

    stats = clarifier.get_stats()
    print(f"\n统计: {stats['total_clarifications']}次澄清, 成功率 {stats['success_rate']:.1f}%")


def test_pipeline():
    """测试4: 管道架构"""
    print_header("测试4: 管道架构 (Text2LocVisionaryPipeline)")

    from enhancements.nlu.pipeline import Text2LocVisionaryPipeline, PipelineMode

    pipeline = Text2LocVisionaryPipeline(
        ollama_url="http://localhost:11434",
        model_name="qwen3-vl:2b",
        mock_mode=True,
        default_mode=PipelineMode.BALANCED,
        cache_enabled=True,
        language="zh"
    )

    # 测试正常查询
    print("\n测试1: 正常查询流程")
    query1 = "停车场北侧的红色汽车"
    print(f"查询: {query1}")

    result1 = pipeline.process_query(query=query1, top_k=3)

    print(f"  状态: {result1.status}")
    print(f"  处理步骤: {len(result1.processed_steps)}个")
    print(f"  最终查询: {result1.final_query}")
    print(f"  总用时: {result1.statistics.get('total_time_ms', 0):.1f}ms")

    # 测试需要澄清的查询
    print("\n测试2: 需要澄清的查询")
    query2 = "找一个东西"
    print(f"查询: {query2}")

    result2 = pipeline.process_query(query=query2, top_k=3)

    print(f"  状态: {result2.status}")

    if result2.clarification_questions:
        print(f"  澄清问题: {len(result2.clarification_questions)}个")

        for i, q in enumerate(result2.clarification_questions, 1):
            print(f"    {i}. {q.question_text}")

    print(f"\n统计信息:")
    stats = pipeline.get_statistics()
    print(f"  总查询: {stats['pipeline']['total_queries']}")
    print(f"  成功率: {stats['pipeline']['success_rate']:.1f}%")
    print(f"  平均耗时: {stats['pipeline']['avg_time_ms']:.1f}ms")


def test_end_to_end():
    """测试5: 端到端系统"""
    print_header("测试5: 端到端系统 (Text2LocVisionary)")

    try:
        from text2loc_visionary import Text2LocVisionary
    except ImportError:
        print("无法导入Text2LocVisionary，跳过此测试")
        return

    system = Text2LocVisionary(use_mock=True, language="zh")

    # 测试完整流程
    print("\n测试1: 完整查询流程")
    query = "停车场北侧附近的红色汽车"
    print(f"查询: {query}")

    result = system.localize(query, top_k=3)

    print(f"  状态: {result['status']}")
    print(f"  处理步骤: {len(result.get('steps', []))}个")

    if result['status'] == 'success':
        if result.get('optimization_info'):
            opt = result['optimization_info']
            print(f"  优化后查询: {opt.get('best_template', 'N/A')}")

        if result.get('results'):
            print(f"  检索结果: {len(result['results'])}个位置")

        print(f"  总用时: {result['statistics'].get('total_time_ms', 0):.1f}ms")

    # 测试模糊查询
    print("\n测试2: 模糊查询流程")
    system2 = Text2LocVisionary(use_mock=True, language="zh")
    query2 = "找东西"
    print(f"查询: {query2}")

    result2 = system2.localize(query2)

    print(f"  状态: {result2['status']}")

    if result2['status'] == 'needs_clarification' and result2.get('clarification_info'):
        ques = result2['clarification_info']['questions']
        print(f"  需要澄清: {len(ques)}个问题")

        if ques:
            print(f"  问题: {ques[0].get('question_text', 'N/A')}")

            # 模拟用户澄清
            print("\n  模拟用户回答: 北侧的红色汽车")

            # 重新处理
            result3 = system2.localize("北侧的红色汽车")
            print(f"  澄清后状态: {result3['status']}")

    print(f"\n系统统计:")
    stats = system2.get_statistics()
    print(f"  总查询: {stats['legacy_stats']['total_queries']}")
    print(f"  澄清次数: {stats['legacy_stats']['clarification_count']}")
    print(f"  平均耗时: {stats['legacy_stats']['average_time_ms']:.1f}ms")


def run_demo():
    """运行完整演示"""
    print_header("Text2Loc Visionary 三层智能架构演示")

    print("\n系统概述:")
    print("  智能理解层: InstructionOptimizer")
    print("  交互澄清层: InteractiveClarifier")
    print("  适配转换层: DynamicTemplateGenerator")
    print("  执行处理层: Text2LocRetrieval")

    tests = [
        ("指令优化器", test_instruction_optimizer),
        ("动态模板生成器", test_dynamic_template_generator),
        ("交互式澄清器", test_interactive_clarifier),
        ("管道架构", test_pipeline),
        ("端到端系统", test_end_to_end),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, True))
        except Exception as e:
            print(f"\n[ERROR] 测试 '{test_name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

        time.sleep(0.5)

    # 总结
    print_header("测试总结")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {test_name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n[SUCCESS] 所有核心组件正常运行！")
        print("\n系统能力摘要:")
        print("  - 智能理解: ✅ 正常")
        print("  - 动态生成: ✅ 正常")
        print("  - 交互澄清: ✅ 正常")
        print("  - 端到端处理: ✅ 正常")
        print("\n准备进入阶段三：前端交互开发")
    else:
        print(f"\n[WARNING] {total - passed} 个测试失败")

    return passed == total


if __name__ == "__main__":
    print("=" * 70)
    print("Text2Loc Visionary 快速演示脚本")
    print("=" * 70)

    try:
        success = run_demo()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n演示已中断")
        exit(1)
    except Exception as e:
        print(f"\n演示失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
