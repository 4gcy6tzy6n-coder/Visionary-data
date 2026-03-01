#!/usr/bin/env python3
"""
深度分析：为什么Visionary和Text2Loc-one表现相似
"""

import torch
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_model_predictions():
    """分析模型预测行为"""
    print("="*80)
    print("深度分析：模型预测行为")
    print("="*80)
    
    # 加载数据
    data_dir = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_2d_semantics_full")
    with open(data_dir / "cells.pkl", 'rb') as f:
        cells = pickle.load(f)
    with open(data_dir / "poses.pkl", 'rb') as f:
        poses = pickle.load(f)
    
    cells_dict = {c['id']: c for c in cells}
    
    # 修复cell_center
    for pose in poses:
        cell_id = pose['cell_id']
        if cell_id in cells_dict:
            pose['cell_center'] = cells_dict[cell_id]['center']
    
    # 计算真实偏移分布
    offsets = []
    for pose in poses[:5000]:
        location = np.array(pose['location'])
        center = np.array(pose['cell_center'])
        offset = location - center
        offsets.append(offset[:2])
    
    offsets = np.array(offsets)
    
    print("\n1. 真实偏移分布分析")
    print("-"*80)
    print(f"   X偏移: mean={offsets[:,0].mean():.3f}m, std={offsets[:,0].std():.3f}m")
    print(f"   Y偏移: mean={offsets[:,1].mean():.3f}m, std={offsets[:,1].std():.3f}m")
    print(f"   X范围: [{offsets[:,0].min():.2f}, {offsets[:,0].max():.2f}]")
    print(f"   Y范围: [{offsets[:,1].min():.2f}, {offsets[:,1].max():.2f}]")
    
    # 加载模型检查点
    checkpoint_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/checkpoints/k360_2d_semantics_best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    offset_mean = checkpoint['offset_mean'].numpy()
    offset_std = checkpoint['offset_std'].numpy()
    
    print(f"\n2. 模型归一化参数")
    print("-"*80)
    print(f"   Offset均值: {offset_mean}")
    print(f"   Offset标准差: {offset_std}")
    
    # 分析训练历史
    history_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/checkpoints/training_history_2d_semantics.json")
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print(f"\n3. 训练历史分析")
    print("-"*80)
    print(f"   最终训练损失: {history['train_losses'][-1]:.6f}")
    print(f"   最终验证损失: {history['val_losses'][-1]:.6f}")
    print(f"   最佳验证损失: {history['best_val_loss']:.6f}")
    print(f"   训练轮数: {history['epochs']}")
    
    # 检查损失是否收敛
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    
    # 计算损失变化
    train_change = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    val_change = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
    
    print(f"\n4. 收敛性分析")
    print("-"*80)
    print(f"   训练损失变化: {train_change:.2f}% (从{train_losses[0]:.4f}到{train_losses[-1]:.4f})")
    print(f"   验证损失变化: {val_change:.2f}% (从{val_losses[0]:.4f}到{val_losses[-1]:.4f})")
    
    if train_change < 5:
        print(f"   ⚠️ 警告: 训练损失几乎无变化，模型可能没有学到有效特征")
    
    # 分析预测偏移分布
    print(f"\n5. 预测偏移分析")
    print("-"*80)
    print(f"   如果模型总是预测接近0的偏移，则等同于Text2Loc-one")
    print(f"   真实偏移标准差: X={offsets[:,0].std():.2f}m, Y={offsets[:,1].std():.2f}m")
    print(f"   如果预测偏移标准差 < 1m，说明模型倾向于预测接近0")
    
    # 计算cell center预测的理论误差
    center_errors = np.linalg.norm(offsets, axis=1)
    print(f"\n6. Cell Center预测的理论性能")
    print("-"*80)
    print(f"   平均误差: {center_errors.mean():.3f}m")
    print(f"   中位数误差: {np.median(center_errors):.3f}m")
    print(f"   RMSE: {np.sqrt(np.mean(center_errors**2)):.3f}m")
    print(f"   1m准确率: {np.mean(center_errors <= 1.0)*100:.1f}%")
    print(f"   2m准确率: {np.mean(center_errors <= 2.0)*100:.1f}%")
    print(f"   5m准确率: {np.mean(center_errors <= 5.0)*100:.1f}%")
    
    # 问题诊断
    print(f"\n7. 问题诊断")
    print("-"*80)
    
    issues = []
    
    # 检查1: 损失值过高
    if history['best_val_loss'] > 0.9:
        issues.append("验证损失过高(>0.9)，模型可能没有收敛")
    
    # 检查2: 训练损失变化小
    if train_change < 10:
        issues.append("训练损失变化小于10%，模型可能没有学到有效特征")
    
    # 检查3: 偏移分布
    if offsets[:,0].std() < 2.0 and offsets[:,1].std() < 2.0:
        issues.append("偏移分布标准差较小，cell center预测可能已经足够好")
    
    # 检查4: 数据规模
    if history['train_samples'] < 50000:
        issues.append(f"训练样本数较少({history['train_samples']})，可能需要更多数据")
    
    if not issues:
        print("   ✓ 未发现明显问题")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    # 建议
    print(f"\n8. 改进建议")
    print("-"*80)
    
    suggestions = []
    
    if history['best_val_loss'] > 0.9:
        suggestions.append("增加模型复杂度或训练轮数")
        suggestions.append("调整学习率调度策略")
        suggestions.append("使用预训练文本嵌入而非随机初始化")
    
    if offsets[:,0].std() < 3.0:
        suggestions.append("增大cell size，使cell center预测更具挑战性")
        suggestions.append("生成更多边缘位置的poses")
    
    suggestions.append("使用真实文本嵌入模型(如BERT)而非随机哈希")
    suggestions.append("增加模型监督信号，如辅助任务")
    suggestions.append("尝试不同的损失函数，如Huber Loss")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
    
    return {
        'offset_stats': {
            'mean_x': float(offsets[:,0].mean()),
            'std_x': float(offsets[:,0].std()),
            'mean_y': float(offsets[:,1].mean()),
            'std_y': float(offsets[:,1].std())
        },
        'center_prediction_performance': {
            'mean_error': float(center_errors.mean()),
            'median_error': float(np.median(center_errors)),
            'rmse': float(np.sqrt(np.mean(center_errors**2))),
            'accuracy_1m': float(np.mean(center_errors <= 1.0) * 100),
            'accuracy_2m': float(np.mean(center_errors <= 2.0) * 100),
            'accuracy_5m': float(np.mean(center_errors <= 5.0) * 100)
        },
        'training_analysis': {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': history['best_val_loss'],
            'train_loss_change_pct': train_change,
            'converged': train_change > 10
        },
        'issues': issues,
        'suggestions': suggestions
    }


def main():
    results = analyze_model_predictions()
    
    # 保存分析结果
    output_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/deep_analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n分析结果已保存: {output_path}")


if __name__ == '__main__':
    main()
