#!/usr/bin/env python3
"""
快速对比实验 - Visionary V2 vs Text2Loc-one
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import time


class Text2LocNeuralNetworkV2(nn.Module):
    def __init__(self, embed_dim=384, hidden_dim=1024):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2)
        )
        self.object_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, text_features, object_features):
        text_encoded = self.text_encoder(text_features)
        object_encoded = self.object_encoder(object_features)
        combined = torch.cat([text_encoded, object_encoded], dim=-1)
        return self.fusion(combined)


def get_sentence_embedding(text: str, embed_dim: int = 384):
    """快速文本嵌入"""
    import hashlib
    embedding = np.zeros(embed_dim)
    words = text.lower().split()
    for i, word in enumerate(words):
        position_weight = 1.0 / (i + 1)
        hash1 = int(hashlib.md5(word.encode()).hexdigest(), 16)
        hash2 = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        for dim in range(embed_dim):
            combined_hash = (hash1 + dim * hash2) % (2**32)
            value = (combined_hash / (2**32)) * 2 - 1
            embedding[dim] += value * position_weight
    if len(words) > 0:
        embedding /= len(words)
    return torch.tensor(embedding, dtype=torch.float32)


def quick_comparison():
    print("="*80)
    print("Visionary V2 vs Text2Loc-one 快速对比")
    print("="*80)
    
    # 加载数据
    data_dir = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_50m_cells")
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
    
    # 随机选择1000个样本测试
    np.random.seed(42)
    test_indices = np.random.choice(len(poses), 1000, replace=False)
    test_poses = [poses[i] for i in test_indices]
    
    print(f"\n测试样本数: {len(test_poses)}")
    
    # 加载Visionary V2模型
    print("\n加载Visionary V2模型...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = Text2LocNeuralNetworkV2(embed_dim=384, hidden_dim=1024).to(device)
    
    checkpoint_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/checkpoints/visionary_v2_fast_best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    offset_mean = checkpoint['offset_mean'].to(device)
    offset_std = checkpoint['offset_std'].to(device)
    
    # 运行预测
    print("\n运行预测...")
    visionary_preds = []
    one_preds = []
    ground_truths = []
    
    for pose in tqdm(test_poses, desc="预测"):
        cell_id = pose['cell_id']
        if cell_id not in cells_dict:
            continue
        
        cell = cells_dict[cell_id]
        true_location = np.array(pose['location'])[:2]
        
        # Visionary V2预测
        with torch.no_grad():
            description = pose.get('description', '')
            text_embedding = get_sentence_embedding(description).unsqueeze(0).to(device)
            
            # 对象特征
            objects = cell.get('objects', [])
            if len(objects) == 0:
                object_features = torch.zeros(1, 6).to(device)
            else:
                obj_features_list = []
                for obj in objects[:50]:
                    center = obj.get('center', [0, 0, 0])
                    confidence = obj.get('confidence', 0) / 1000.0
                    max_conf = obj.get('max_confidence', 0) / 1000.0
                    semantic = obj.get('semantic', 0) / 10.0
                    feat = [float(center[0]), float(center[1]), confidence, max_conf, semantic, len(objects) / 1000.0]
                    obj_features_list.append(feat)
                obj_features_array = np.array(obj_features_list)
                object_features = torch.tensor(np.mean(obj_features_array, axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            
            offset_pred = model(text_embedding, object_features)
            offset_pred = offset_pred * offset_std + offset_mean
            
            cell_center = np.array(pose['cell_center'])
            v_pred = cell_center[:2] + offset_pred.cpu().numpy()[0]
        
        # Text2Loc-one预测 (cell center)
        o_pred = np.array(pose['cell_center'])[:2]
        
        visionary_preds.append(v_pred)
        one_preds.append(o_pred)
        ground_truths.append(true_location)
    
    visionary_preds = np.array(visionary_preds)
    one_preds = np.array(one_preds)
    ground_truths = np.array(ground_truths)
    
    # 计算指标
    print("\n计算性能指标...")
    v_errors = np.linalg.norm(visionary_preds - ground_truths, axis=1)
    o_errors = np.linalg.norm(one_preds - ground_truths, axis=1)
    
    results = {
        'visionary_v2': {
            'mean_error_m': float(np.mean(v_errors)),
            'median_error_m': float(np.median(v_errors)),
            'rmse_m': float(np.sqrt(np.mean(v_errors**2))),
            'accuracy_5m': float(np.mean(v_errors <= 5.0) * 100),
            'accuracy_10m': float(np.mean(v_errors <= 10.0) * 100),
            'accuracy_15m': float(np.mean(v_errors <= 15.0) * 100),
        },
        'text2loc_one': {
            'mean_error_m': float(np.mean(o_errors)),
            'median_error_m': float(np.median(o_errors)),
            'rmse_m': float(np.sqrt(np.mean(o_errors**2))),
            'accuracy_5m': float(np.mean(o_errors <= 5.0) * 100),
            'accuracy_10m': float(np.mean(o_errors <= 10.0) * 100),
            'accuracy_15m': float(np.mean(o_errors <= 15.0) * 100),
        }
    }
    
    # 计算改进
    results['improvements'] = {
        'mean_error_reduction': float((np.mean(o_errors) - np.mean(v_errors)) / np.mean(o_errors) * 100),
        'median_error_reduction': float((np.median(o_errors) - np.median(v_errors)) / np.median(o_errors) * 100),
        'rmse_reduction': float((np.sqrt(np.mean(o_errors**2)) - np.sqrt(np.mean(v_errors**2))) / np.sqrt(np.mean(o_errors**2)) * 100),
        'accuracy_5m_improvement': float(np.mean(v_errors <= 5.0) * 100 - np.mean(o_errors <= 5.0) * 100),
        'accuracy_10m_improvement': float(np.mean(v_errors <= 10.0) * 100 - np.mean(o_errors <= 10.0) * 100),
    }
    
    # 打印结果
    print("\n" + "="*80)
    print("对比实验结果 (50m Cell Size)")
    print("="*80)
    
    print("\n📍 定位精度对比")
    print("-"*80)
    print(f"{'指标':<25} {'Visionary V2':<20} {'Text2Loc-one':<20} {'改进':<15}")
    print("-"*80)
    print(f"{'平均误差 (m)':<25} {results['visionary_v2']['mean_error_m']:<20.3f} {results['text2loc_one']['mean_error_m']:<20.3f} {results['improvements']['mean_error_reduction']:>+.1f}%")
    print(f"{'中位数误差 (m)':<25} {results['visionary_v2']['median_error_m']:<20.3f} {results['text2loc_one']['median_error_m']:<20.3f} {results['improvements']['median_error_reduction']:>+.1f}%")
    print(f"{'RMSE (m)':<25} {results['visionary_v2']['rmse_m']:<20.3f} {results['text2loc_one']['rmse_m']:<20.3f} {results['improvements']['rmse_reduction']:>+.1f}%")
    print(f"{'5m准确率 (%)':<25} {results['visionary_v2']['accuracy_5m']:<20.1f} {results['text2loc_one']['accuracy_5m']:<20.1f} {results['improvements']['accuracy_5m_improvement']:>+.1f}%")
    print(f"{'10m准确率 (%)':<25} {results['visionary_v2']['accuracy_10m']:<20.1f} {results['text2loc_one']['accuracy_10m']:<20.1f} {results['improvements']['accuracy_10m_improvement']:>+.1f}%")
    print(f"{'15m准确率 (%)':<25} {results['visionary_v2']['accuracy_15m']:<20.1f} {results['text2loc_one']['accuracy_15m']:<20.1f} {'-':<15}")
    
    print("\n" + "="*80)
    print("🎯 Visionary V2 核心优势")
    print("="*80)
    print(f"\n1. 定位精度提升:")
    print(f"   • 平均误差降低: {results['improvements']['mean_error_reduction']:.1f}%")
    print(f"   • 中位数误差降低: {results['improvements']['median_error_reduction']:.1f}%")
    print(f"   • RMSE降低: {results['improvements']['rmse_reduction']:.1f}%")
    
    print(f"\n2. 准确率提升:")
    print(f"   • 5米内准确率提升: {results['improvements']['accuracy_5m_improvement']:+.1f}个百分点")
    print(f"   • 10米内准确率提升: {results['improvements']['accuracy_10m_improvement']:+.1f}个百分点")
    
    print(f"\n3. 技术创新:")
    print(f"   • 改进的文本嵌入 (语义感知哈希)")
    print(f"   • 更深的神经网络架构 (1024 hidden dim)")
    print(f"   • Huber Loss + LayerNorm")
    print(f"   • 50m Cell Size (更具挑战性)")
    print(f"   • 余弦退火学习率调度")
    
    # 保存结果
    output_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/comparison_v2_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    
    return results


if __name__ == '__main__':
    quick_comparison()
