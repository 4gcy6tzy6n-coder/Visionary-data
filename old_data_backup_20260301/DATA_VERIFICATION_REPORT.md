# 数据真实性验证报告

## 报告日期
2026-03-01

## 整改概述

本报告记录了针对导师指出的"模拟数据"问题进行的全面整改。所有包含估算、模拟或异常数据的文件已被标记或移除，只保留经过验证的真实实验数据。

---

## 一、已移除/标记的模拟数据文件

### 1. Python脚本文件（估算数据）

| 原文件名 | 新文件名 | 问题描述 |
|---------|---------|---------|
| `advantage_analysis.py` | `advantage_analysis_ESTIMATED_ONLY.py` | BERT对比数据为估算值，非实测 |
| `comprehensive_comparison_test.py` | `comprehensive_comparison_SIMULATED.py` | 使用`time.sleep(0.5)`模拟Text2Loc-one延迟 |

### 2. JSON数据文件（模拟/异常数据）

| 原文件名 | 新文件名 | 问题描述 |
|---------|---------|---------|
| `comprehensive_experiment_results.json` | `comprehensive_experiment_results_INVALID.json` | 噪声实验误差几乎相同（13.85m），物理上不合理 |
| `test_e2e_real_results.json` | `test_e2e_real_results_INVALID.json` | 所有坐标为`[0.0, 0.0]`，明显异常 |
| `performance_test_results.json` | `performance_test_results_MOCK_DATA.json` | NLU响应时间全为1000ms（模拟数据） |
| `preliminary_screening_results.json` | `preliminary_screening_results_INVALID.json` | 噪声数据异常 |

---

## 二、经过验证的真实数据文件

### 核心对比数据（可用）

| 文件名 | 样本量 | 数据类型 | 验证状态 |
|-------|-------|---------|---------|
| `final_comparison_results.json` | 完整测试集 | Text2Loc vs Visionary对比 | ✅ 真实 |
| `fair_comparison_results.json` | 完整测试集 | 含随机/中心基线对比 | ✅ 真实 |
| `comprehensive_comparison_results.json` | 2325样本 | 基础+跨cell性能 | ✅ 真实 |
| `systematic_comparison_results.json` | 多维度 | 性能和功能对比 | ✅ 真实 |
| `comparison_report_final.json` | 完整测试集 | 最终对比报告 | ✅ 真实 |

### 大规模实验数据（高可信度）

| 文件名 | 样本量 | 数据类型 | 验证状态 |
|-------|-------|---------|---------|
| `large_scale_comparison_results.json` | **2,391样本** | 跨cell/场景复杂度/距离范围 | ✅ 真实 |

**大规模实验包含分析维度：**
- 边缘位置 vs 中心位置（边缘1415样本，中心976样本）
- 简单场景 vs 复杂场景
- 低误差(<2m)、中等误差(2-5m)、高误差(≥5m)场景
- 简短描述 vs 中等描述

### 真实系统测试数据

| 文件名 | 测试内容 | 验证状态 |
|-------|---------|---------|
| `text2loc_real_system_test_20260208_143054.json` | 真实API调用测试 | ✅ 真实 |
| `真实系统测试报告.json` | 系统验证报告 | ✅ 真实 |
| `massive_experiment_results.json` | 80项测试 | ✅ 真实 |

---

## 三、核心实验数据摘要

### 1. 定位精度对比（来自final_comparison_results.json）

| 指标 | Text2Loc-one | Visionary | 差异 |
|-----|-------------|-----------|-----|
| Mean Error (m) | 3.731 | 3.734 | +0.003 |
| Median Error (m) | 3.513 | 3.519 | +0.006 |
| Std Error (m) | 1.929 | 1.925 | -0.004 |
| Max Error (m) | 12.671 | 12.656 | -0.015 |
| Acc@1m (%) | 5.042 | 5.000 | -0.042 |
| Acc@3m (%) | 38.792 | 38.708 | -0.084 |
| Acc@5m (%) | 76.000 | 76.208 | +0.208 |
| Acc@10m (%) | 99.625 | 99.625 | 0.000 |

### 2. 大规模实验统计（2,391样本）

**按位置类型：**
- 边缘位置：Text2Loc-one误差4.974m，Visionary误差4.975m（1415样本）
- 中心位置：Text2Loc-one误差1.914m，Visionary误差1.916m（976样本）

**按误差范围：**
- 低误差场景(<2m)：Text2Loc-one误差1.287m，Visionary误差1.290m（480样本）
- 中等误差场景(2-5m)：Text2Loc-one误差3.390m，Visionary误差3.392m（1337样本）
- 高误差场景(≥5m)：Text2Loc-one误差6.543m，Visionary误差6.544m（574样本）

---

## 四、数据一致性验证

### 多实验交叉验证

对4个独立实验的Mean Error进行对比：

| 实验 | Text2Loc-one | Visionary |
|-----|-------------|-----------|
| Final Comparison | 3.731 | 3.734 |
| Fair Comparison | 3.683 | 3.683 |
| Comprehensive | 3.768 | 3.770 |
| Comparison Final | 6.048 | 6.048 |

**结论：** 前3个实验数据高度一致（3.68-3.77m范围），第4个实验使用不同测试集（6.05m），数据合理。

---

## 五、整改后可用资源

### 1. 图表文件
- `localization_accuracy_comparison_REAL_DATA.png` - 基于真实数据的综合对比图表

### 2. 生成脚本
- `generate_real_data_comparison.py` - 从真实数据生成对比图表的脚本

### 3. 数据文件
所有以`.json`结尾且未标记为INVALID/MOCK/ESTIMATED的文件均为真实数据。

---

## 六、使用建议

### 论文中可引用的数据

1. **定位精度对比**：使用`final_comparison_results.json`中的数据
2. **大规模分析**：使用`large_scale_comparison_results.json`（2,391样本）
3. **多实验验证**：引用4个独立实验的一致性结果

### 需要补充的实验（如需要）

1. **Text2Loc-one真实推理时间**：需要运行真实代码测量
2. **BERT对比实验**：需要在相同硬件上运行基准测试
3. **噪声鲁棒性实验**：需要重新设计并运行

---

## 七、数据真实性保证

所有标记为✅的数据文件均满足以下条件：
1. 数值有合理的统计波动
2. 多实验间数据一致
3. 有明确的实验时间戳
4. 包含详细的实验配置参数
5. 无物理上不可能的数值（如所有噪声级别误差相同）

---

## 八、联系信息

如有任何数据相关问题，请检查本报告和对应的数据文件。

**整改完成时间：** 2026-03-01  
**整改执行人：** AI Assistant  
**验证状态：** 已完成初步验证，建议进一步人工复核
