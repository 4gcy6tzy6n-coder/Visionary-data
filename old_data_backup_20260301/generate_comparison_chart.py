#!/usr/bin/env python3
"""
生成Text2Loc-one与Visionary定位精度对比图表
包含完整的统计指标对比
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 对比数据 - 从多个实验结果文件中汇总
data = {
    # 实验1: final_comparison_results.json
    "exp1": {
        "text2loc_one": {
            "mean_error": 3.731,
            "median_error": 3.513,
            "acc_1m": 5.042,
            "acc_3m": 38.792,
            "acc_5m": 76.000,
            "acc_10m": 99.625,
            "max_error": 12.671,
            "std_error": 1.929,
            "rmse": 4.198,
            "min_error": 2.011,
        },
        "visionary": {
            "mean_error": 3.734,
            "median_error": 3.519,
            "acc_1m": 5.000,
            "acc_3m": 38.708,
            "acc_5m": 76.208,
            "acc_10m": 99.625,
            "max_error": 12.656,
            "std_error": 1.925,
            "rmse": 4.194,
            "min_error": 1.987,
        }
    },
    # 实验2: fair_comparison_results.json
    "exp2": {
        "text2loc_one": {
            "mean_error": 3.683,
            "median_error": 3.477,
            "acc_1m": 5.934,
            "acc_3m": 41.623,
            "acc_5m": 76.483,
            "acc_10m": 99.695,
            "max_error": 12.132,
            "std_error": 1.962,
        },
        "visionary": {
            "mean_error": 3.683,
            "median_error": 3.466,
            "acc_1m": 5.672,
            "acc_3m": 41.318,
            "acc_5m": 76.483,
            "acc_10m": 99.695,
            "max_error": 12.014,
            "std_error": 1.962,
        }
    },
    # 实验3: comprehensive_comparison_results.json
    "exp3": {
        "text2loc_one": {
            "mean_error": 3.768,
            "median_error": 3.566,
            "acc_1m": 5.591,
            "acc_3m": 38.323,
            "acc_5m": 74.925,
            "acc_10m": 99.570,
            "max_error": 12.110,
            "std_error": 1.969,
        },
        "visionary": {
            "mean_error": 3.770,
            "median_error": 3.561,
            "acc_1m": 5.849,
            "acc_3m": 38.108,
            "acc_5m": 75.011,
            "acc_10m": 99.570,
            "max_error": 12.026,
            "std_error": 1.970,
        }
    },
    # 实验4: comparison_report_final.json
    "exp4": {
        "text2loc_one": {
            "mean_error": 6.048,
            "median_error": 6.041,
            "acc_1m": 0.000,
            "acc_2m": 0.000,
            "acc_5m": 36.050,
            "acc_10m": 99.570,
            "max_error": 9.999,
            "std_error": 2.295,
            "rmse": 6.469,
            "min_error": 2.011,
            "p25": 4.106,
            "p50": 6.041,
            "p75": 8.048,
            "p90": 9.222,
            "p95": 9.610,
            "p99": 9.918,
        },
        "visionary": {
            "mean_error": 6.048,
            "median_error": 6.041,
            "acc_1m": 0.000,
            "acc_2m": 0.050,
            "acc_5m": 36.000,
            "acc_10m": 99.570,
            "max_error": 10.027,
            "std_error": 2.295,
            "rmse": 6.468,
            "min_error": 1.987,
            "p25": 4.115,
            "p50": 6.041,
            "p75": 8.050,
            "p90": 9.220,
            "p95": 9.599,
            "p99": 9.912,
        }
    }
}

# 创建大图
fig = plt.figure(figsize=(24, 32))
fig.suptitle('Text2Loc-one vs Visionary: Comprehensive Localization Accuracy Comparison', 
             fontsize=20, fontweight='bold', y=0.995)

# 颜色定义
color_one = '#E74C3C'  # 红色
color_visionary = '#3498DB'  # 蓝色
color_diff = '#2ECC71'  # 绿色

# ========== 1. 误差指标对比 (实验1-3平均值) ==========
ax1 = plt.subplot(5, 2, 1)
metrics = ['Mean Error\n(m)', 'Median Error\n(m)', 'Max Error\n(m)', 'Std Error\n(m)', 'RMSE\n(m)']
one_values = [
    np.mean([data['exp1']['text2loc_one']['mean_error'], data['exp2']['text2loc_one']['mean_error'], data['exp3']['text2loc_one']['mean_error']]),
    np.mean([data['exp1']['text2loc_one']['median_error'], data['exp2']['text2loc_one']['median_error'], data['exp3']['text2loc_one']['median_error']]),
    np.mean([data['exp1']['text2loc_one']['max_error'], data['exp2']['text2loc_one']['max_error'], data['exp3']['text2loc_one']['max_error']]),
    np.mean([data['exp1']['text2loc_one']['std_error'], data['exp2']['text2loc_one']['std_error'], data['exp3']['text2loc_one']['std_error']]),
    data['exp1']['text2loc_one']['rmse']
]
visionary_values = [
    np.mean([data['exp1']['visionary']['mean_error'], data['exp2']['visionary']['mean_error'], data['exp3']['visionary']['mean_error']]),
    np.mean([data['exp1']['visionary']['median_error'], data['exp2']['visionary']['median_error'], data['exp3']['visionary']['median_error']]),
    np.mean([data['exp1']['visionary']['max_error'], data['exp2']['visionary']['max_error'], data['exp3']['visionary']['max_error']]),
    np.mean([data['exp1']['visionary']['std_error'], data['exp2']['visionary']['std_error'], data['exp3']['visionary']['std_error']]),
    data['exp1']['visionary']['rmse']
]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax1.bar(x - width/2, one_values, width, label='Text2Loc-one', color=color_one, alpha=0.8)
bars2 = ax1.bar(x + width/2, visionary_values, width, label='Visionary', color=color_visionary, alpha=0.8)
ax1.set_ylabel('Value (m)', fontsize=11)
ax1.set_title('Error Metrics Comparison (Mean of Exp 1-3)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=9)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

# ========== 2. 精度指标对比 ==========
ax2 = plt.subplot(5, 2, 2)
acc_metrics = ['Acc@1m (%)', 'Acc@3m (%)', 'Acc@5m (%)', 'Acc@10m (%)']
one_acc = [
    np.mean([data['exp1']['text2loc_one']['acc_1m'], data['exp2']['text2loc_one']['acc_1m'], data['exp3']['text2loc_one']['acc_1m']]),
    np.mean([data['exp1']['text2loc_one']['acc_3m'], data['exp2']['text2loc_one']['acc_3m'], data['exp3']['text2loc_one']['acc_3m']]),
    np.mean([data['exp1']['text2loc_one']['acc_5m'], data['exp2']['text2loc_one']['acc_5m'], data['exp3']['text2loc_one']['acc_5m']]),
    np.mean([data['exp1']['text2loc_one']['acc_10m'], data['exp2']['text2loc_one']['acc_10m'], data['exp3']['text2loc_one']['acc_10m']])
]
visionary_acc = [
    np.mean([data['exp1']['visionary']['acc_1m'], data['exp2']['visionary']['acc_1m'], data['exp3']['visionary']['acc_1m']]),
    np.mean([data['exp1']['visionary']['acc_3m'], data['exp2']['visionary']['acc_3m'], data['exp3']['visionary']['acc_3m']]),
    np.mean([data['exp1']['visionary']['acc_5m'], data['exp2']['visionary']['acc_5m'], data['exp3']['visionary']['acc_5m']]),
    np.mean([data['exp1']['visionary']['acc_10m'], data['exp2']['visionary']['acc_10m'], data['exp3']['visionary']['acc_10m']])
]

x = np.arange(len(acc_metrics))
bars1 = ax2.bar(x - width/2, one_acc, width, label='Text2Loc-one', color=color_one, alpha=0.8)
bars2 = ax2.bar(x + width/2, visionary_acc, width, label='Visionary', color=color_visionary, alpha=0.8)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_title('Accuracy Metrics Comparison (Mean of Exp 1-3)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(acc_metrics, fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 105)

for bar in bars1:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

# ========== 3. 实验4详细误差分布 ==========
ax3 = plt.subplot(5, 2, 3)
percentiles = ['P25', 'P50', 'P75', 'P90', 'P95', 'P99']
one_p = [data['exp4']['text2loc_one']['p25'], data['exp4']['text2loc_one']['p50'], 
         data['exp4']['text2loc_one']['p75'], data['exp4']['text2loc_one']['p90'],
         data['exp4']['text2loc_one']['p95'], data['exp4']['text2loc_one']['p99']]
visionary_p = [data['exp4']['visionary']['p25'], data['exp4']['visionary']['p50'],
               data['exp4']['visionary']['p75'], data['exp4']['visionary']['p90'],
               data['exp4']['visionary']['p95'], data['exp4']['visionary']['p99']]

x = np.arange(len(percentiles))
ax3.plot(x, one_p, 'o-', color=color_one, linewidth=2, markersize=8, label='Text2Loc-one')
ax3.plot(x, visionary_p, 's-', color=color_visionary, linewidth=2, markersize=8, label='Visionary')
ax3.set_ylabel('Error (m)', fontsize=11)
ax3.set_title('Error Distribution by Percentile (Experiment 4)', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(percentiles)
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 添加数值标签
for i, (v1, v2) in enumerate(zip(one_p, visionary_p)):
    ax3.annotate(f'{v1:.3f}', (i, v1), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color=color_one)
    ax3.annotate(f'{v2:.3f}', (i, v2), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8, color=color_visionary)

# ========== 4. 各实验平均误差对比 ==========
ax4 = plt.subplot(5, 2, 4)
experiments = ['Exp 1', 'Exp 2', 'Exp 3', 'Exp 4']
one_means = [data['exp1']['text2loc_one']['mean_error'], 
             data['exp2']['text2loc_one']['mean_error'],
             data['exp3']['text2loc_one']['mean_error'],
             data['exp4']['text2loc_one']['mean_error']]
visionary_means = [data['exp1']['visionary']['mean_error'],
                   data['exp2']['visionary']['mean_error'],
                   data['exp3']['visionary']['mean_error'],
                   data['exp4']['visionary']['mean_error']]

x = np.arange(len(experiments))
bars1 = ax4.bar(x - width/2, one_means, width, label='Text2Loc-one', color=color_one, alpha=0.8)
bars2 = ax4.bar(x + width/2, visionary_means, width, label='Visionary', color=color_visionary, alpha=0.8)
ax4.set_ylabel('Mean Error (m)', fontsize=11)
ax4.set_title('Mean Error Across All Experiments', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(experiments)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax4.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax4.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# ========== 5. 标准差对比 ==========
ax5 = plt.subplot(5, 2, 5)
std_metrics = ['Exp 1', 'Exp 2', 'Exp 3', 'Exp 4']
one_std = [data['exp1']['text2loc_one']['std_error'],
           data['exp2']['text2loc_one']['std_error'],
           data['exp3']['text2loc_one']['std_error'],
           data['exp4']['text2loc_one']['std_error']]
visionary_std = [data['exp1']['visionary']['std_error'],
                 data['exp2']['visionary']['std_error'],
                 data['exp3']['visionary']['std_error'],
                 data['exp4']['visionary']['std_error']]

x = np.arange(len(std_metrics))
bars1 = ax5.bar(x - width/2, one_std, width, label='Text2Loc-one', color=color_one, alpha=0.8)
bars2 = ax5.bar(x + width/2, visionary_std, width, label='Visionary', color=color_visionary, alpha=0.8)
ax5.set_ylabel('Standard Deviation (m)', fontsize=11)
ax5.set_title('Standard Deviation Comparison', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(std_metrics)
ax5.legend(fontsize=10)
ax5.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax5.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax5.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# ========== 6. 极值对比 ==========
ax6 = plt.subplot(5, 2, 6)
min_max_metrics = ['Min Error', 'Max Error']
exp4_data = {
    'text2loc_one': [data['exp4']['text2loc_one']['min_error'], data['exp4']['text2loc_one']['max_error']],
    'visionary': [data['exp4']['visionary']['min_error'], data['exp4']['visionary']['max_error']]
}

x = np.arange(len(min_max_metrics))
bars1 = ax6.bar(x - width/2, exp4_data['text2loc_one'], width, label='Text2Loc-one', color=color_one, alpha=0.8)
bars2 = ax6.bar(x + width/2, exp4_data['visionary'], width, label='Visionary', color=color_visionary, alpha=0.8)
ax6.set_ylabel('Error (m)', fontsize=11)
ax6.set_title('Min/Max Error Comparison (Experiment 4)', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(min_max_metrics)
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax6.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax6.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# ========== 7. 详细数据表格 ==========
ax7 = plt.subplot(5, 1, 4)
ax7.axis('off')

# 准备表格数据
table_data = [
    ['Metric', 'Text2Loc-one', 'Visionary', 'Difference', 'Unit'],
    ['Mean Error (Exp 1)', f"{data['exp1']['text2loc_one']['mean_error']:.3f}", f"{data['exp1']['visionary']['mean_error']:.3f}", 
     f"{data['exp1']['visionary']['mean_error'] - data['exp1']['text2loc_one']['mean_error']:.3f}", 'm'],
    ['Mean Error (Exp 2)', f"{data['exp2']['text2loc_one']['mean_error']:.3f}", f"{data['exp2']['visionary']['mean_error']:.3f}", 
     f"{data['exp2']['visionary']['mean_error'] - data['exp2']['text2loc_one']['mean_error']:.3f}", 'm'],
    ['Mean Error (Exp 3)', f"{data['exp3']['text2loc_one']['mean_error']:.3f}", f"{data['exp3']['visionary']['mean_error']:.3f}", 
     f"{data['exp3']['visionary']['mean_error'] - data['exp3']['text2loc_one']['mean_error']:.3f}", 'm'],
    ['Mean Error (Exp 4)', f"{data['exp4']['text2loc_one']['mean_error']:.3f}", f"{data['exp4']['visionary']['mean_error']:.3f}", 
     f"{data['exp4']['visionary']['mean_error'] - data['exp4']['text2loc_one']['mean_error']:.3f}", 'm'],
    ['Median Error (Exp 1)', f"{data['exp1']['text2loc_one']['median_error']:.3f}", f"{data['exp1']['visionary']['median_error']:.3f}", 
     f"{data['exp1']['visionary']['median_error'] - data['exp1']['text2loc_one']['median_error']:.3f}", 'm'],
    ['Median Error (Exp 2)', f"{data['exp2']['text2loc_one']['median_error']:.3f}", f"{data['exp2']['visionary']['median_error']:.3f}", 
     f"{data['exp2']['visionary']['median_error'] - data['exp2']['text2loc_one']['median_error']:.3f}", 'm'],
    ['Median Error (Exp 3)', f"{data['exp3']['text2loc_one']['median_error']:.3f}", f"{data['exp3']['visionary']['median_error']:.3f}", 
     f"{data['exp3']['visionary']['median_error'] - data['exp3']['text2loc_one']['median_error']:.3f}", 'm'],
    ['Median Error (Exp 4)', f"{data['exp4']['text2loc_one']['median_error']:.3f}", f"{data['exp4']['visionary']['median_error']:.3f}", 
     f"{data['exp4']['visionary']['median_error'] - data['exp4']['text2loc_one']['median_error']:.3f}", 'm'],
    ['Std Error (Exp 1)', f"{data['exp1']['text2loc_one']['std_error']:.3f}", f"{data['exp1']['visionary']['std_error']:.3f}", 
     f"{data['exp1']['visionary']['std_error'] - data['exp1']['text2loc_one']['std_error']:.3f}", 'm'],
    ['Std Error (Exp 2)', f"{data['exp2']['text2loc_one']['std_error']:.3f}", f"{data['exp2']['visionary']['std_error']:.3f}", 
     f"{data['exp2']['visionary']['std_error'] - data['exp2']['text2loc_one']['std_error']:.3f}", 'm'],
    ['Std Error (Exp 3)', f"{data['exp3']['text2loc_one']['std_error']:.3f}", f"{data['exp3']['visionary']['std_error']:.3f}", 
     f"{data['exp3']['visionary']['std_error'] - data['exp3']['text2loc_one']['std_error']:.3f}", 'm'],
    ['Std Error (Exp 4)', f"{data['exp4']['text2loc_one']['std_error']:.3f}", f"{data['exp4']['visionary']['std_error']:.3f}", 
     f"{data['exp4']['visionary']['std_error'] - data['exp4']['text2loc_one']['std_error']:.3f}", 'm'],
    ['RMSE (Exp 1)', f"{data['exp1']['text2loc_one']['rmse']:.3f}", f"{data['exp1']['visionary']['rmse']:.3f}", 
     f"{data['exp1']['visionary']['rmse'] - data['exp1']['text2loc_one']['rmse']:.3f}", 'm'],
    ['RMSE (Exp 4)', f"{data['exp4']['text2loc_one']['rmse']:.3f}", f"{data['exp4']['visionary']['rmse']:.3f}", 
     f"{data['exp4']['visionary']['rmse'] - data['exp4']['text2loc_one']['rmse']:.3f}", 'm'],
    ['Max Error (Exp 1)', f"{data['exp1']['text2loc_one']['max_error']:.3f}", f"{data['exp1']['visionary']['max_error']:.3f}", 
     f"{data['exp1']['visionary']['max_error'] - data['exp1']['text2loc_one']['max_error']:.3f}", 'm'],
    ['Max Error (Exp 2)', f"{data['exp2']['text2loc_one']['max_error']:.3f}", f"{data['exp2']['visionary']['max_error']:.3f}", 
     f"{data['exp2']['visionary']['max_error'] - data['exp2']['text2loc_one']['max_error']:.3f}", 'm'],
    ['Max Error (Exp 3)', f"{data['exp3']['text2loc_one']['max_error']:.3f}", f"{data['exp3']['visionary']['max_error']:.3f}", 
     f"{data['exp3']['visionary']['max_error'] - data['exp3']['text2loc_one']['max_error']:.3f}", 'm'],
    ['Max Error (Exp 4)', f"{data['exp4']['text2loc_one']['max_error']:.3f}", f"{data['exp4']['visionary']['max_error']:.3f}", 
     f"{data['exp4']['visionary']['max_error'] - data['exp4']['text2loc_one']['max_error']:.3f}", 'm'],
    ['Min Error (Exp 4)', f"{data['exp4']['text2loc_one']['min_error']:.3f}", f"{data['exp4']['visionary']['min_error']:.3f}", 
     f"{data['exp4']['visionary']['min_error'] - data['exp4']['text2loc_one']['min_error']:.3f}", 'm'],
]

table = ax7.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center', loc='center',
                  colWidths=[0.25, 0.2, 0.2, 0.15, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

# 设置表头样式
for i in range(5):
    table[(0, i)].set_facecolor('#3498DB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置交替行颜色
for i in range(1, len(table_data)):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')

ax7.set_title('Detailed Comparison Table (All Metrics)', fontsize=14, fontweight='bold', pad=20)

# ========== 8. 精度指标详细表格 ==========
ax8 = plt.subplot(5, 1, 5)
ax8.axis('off')

acc_table_data = [
    ['Metric', 'Text2Loc-one', 'Visionary', 'Difference', 'Unit'],
    ['Acc@1m (Exp 1)', f"{data['exp1']['text2loc_one']['acc_1m']:.3f}", f"{data['exp1']['visionary']['acc_1m']:.3f}", 
     f"{data['exp1']['visionary']['acc_1m'] - data['exp1']['text2loc_one']['acc_1m']:.3f}", '%'],
    ['Acc@1m (Exp 2)', f"{data['exp2']['text2loc_one']['acc_1m']:.3f}", f"{data['exp2']['visionary']['acc_1m']:.3f}", 
     f"{data['exp2']['visionary']['acc_1m'] - data['exp2']['text2loc_one']['acc_1m']:.3f}", '%'],
    ['Acc@1m (Exp 3)', f"{data['exp3']['text2loc_one']['acc_1m']:.3f}", f"{data['exp3']['visionary']['acc_1m']:.3f}", 
     f"{data['exp3']['visionary']['acc_1m'] - data['exp3']['text2loc_one']['acc_1m']:.3f}", '%'],
    ['Acc@1m (Exp 4)', f"{data['exp4']['text2loc_one']['acc_1m']:.3f}", f"{data['exp4']['visionary']['acc_1m']:.3f}", 
     f"{data['exp4']['visionary']['acc_1m'] - data['exp4']['text2loc_one']['acc_1m']:.3f}", '%'],
    ['Acc@3m (Exp 1)', f"{data['exp1']['text2loc_one']['acc_3m']:.3f}", f"{data['exp1']['visionary']['acc_3m']:.3f}", 
     f"{data['exp1']['visionary']['acc_3m'] - data['exp1']['text2loc_one']['acc_3m']:.3f}", '%'],
    ['Acc@3m (Exp 2)', f"{data['exp2']['text2loc_one']['acc_3m']:.3f}", f"{data['exp2']['visionary']['acc_3m']:.3f}", 
     f"{data['exp2']['visionary']['acc_3m'] - data['exp2']['text2loc_one']['acc_3m']:.3f}", '%'],
    ['Acc@3m (Exp 3)', f"{data['exp3']['text2loc_one']['acc_3m']:.3f}", f"{data['exp3']['visionary']['acc_3m']:.3f}", 
     f"{data['exp3']['visionary']['acc_3m'] - data['exp3']['text2loc_one']['acc_3m']:.3f}", '%'],
    ['Acc@5m (Exp 1)', f"{data['exp1']['text2loc_one']['acc_5m']:.3f}", f"{data['exp1']['visionary']['acc_5m']:.3f}", 
     f"{data['exp1']['visionary']['acc_5m'] - data['exp1']['text2loc_one']['acc_5m']:.3f}", '%'],
    ['Acc@5m (Exp 2)', f"{data['exp2']['text2loc_one']['acc_5m']:.3f}", f"{data['exp2']['visionary']['acc_5m']:.3f}", 
     f"{data['exp2']['visionary']['acc_5m'] - data['exp2']['text2loc_one']['acc_5m']:.3f}", '%'],
    ['Acc@5m (Exp 3)', f"{data['exp3']['text2loc_one']['acc_5m']:.3f}", f"{data['exp3']['visionary']['acc_5m']:.3f}", 
     f"{data['exp3']['visionary']['acc_5m'] - data['exp3']['text2loc_one']['acc_5m']:.3f}", '%'],
    ['Acc@5m (Exp 4)', f"{data['exp4']['text2loc_one']['acc_5m']:.3f}", f"{data['exp4']['visionary']['acc_5m']:.3f}", 
     f"{data['exp4']['visionary']['acc_5m'] - data['exp4']['text2loc_one']['acc_5m']:.3f}", '%'],
    ['Acc@10m (Exp 1)', f"{data['exp1']['text2loc_one']['acc_10m']:.3f}", f"{data['exp1']['visionary']['acc_10m']:.3f}", 
     f"{data['exp1']['visionary']['acc_10m'] - data['exp1']['text2loc_one']['acc_10m']:.3f}", '%'],
    ['Acc@10m (Exp 2)', f"{data['exp2']['text2loc_one']['acc_10m']:.3f}", f"{data['exp2']['visionary']['acc_10m']:.3f}", 
     f"{data['exp2']['visionary']['acc_10m'] - data['exp2']['text2loc_one']['acc_10m']:.3f}", '%'],
    ['Acc@10m (Exp 3)', f"{data['exp3']['text2loc_one']['acc_10m']:.3f}", f"{data['exp3']['visionary']['acc_10m']:.3f}", 
     f"{data['exp3']['visionary']['acc_10m'] - data['exp3']['text2loc_one']['acc_10m']:.3f}", '%'],
    ['Acc@10m (Exp 4)', f"{data['exp4']['text2loc_one']['acc_10m']:.3f}", f"{data['exp4']['visionary']['acc_10m']:.3f}", 
     f"{data['exp4']['visionary']['acc_10m'] - data['exp4']['text2loc_one']['acc_10m']:.3f}", '%'],
]

table2 = ax8.table(cellText=acc_table_data[1:], colLabels=acc_table_data[0], cellLoc='center', loc='center',
                   colWidths=[0.25, 0.2, 0.2, 0.15, 0.1])
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 1.8)

# 设置表头样式
for i in range(5):
    table2[(0, i)].set_facecolor('#E74C3C')
    table2[(0, i)].set_text_props(weight='bold', color='white')

# 设置交替行颜色
for i in range(1, len(acc_table_data)):
    for j in range(5):
        if i % 2 == 0:
            table2[(i, j)].set_facecolor('#FADBD8')

ax8.set_title('Accuracy Metrics Comparison Table', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.995])
plt.savefig('/Users/yaoyingliang/visionary/Visionary-data-main/localization_accuracy_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("图表已保存: localization_accuracy_comparison.png")
