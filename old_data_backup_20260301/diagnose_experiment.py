#!/usr/bin/env python3
"""
诊断实验问题
深入分析为什么神经网络没有比基线更好
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

class Text2LocNeuralNetwork(nn.Module):
    """与训练相同的模型架构"""
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.obj_conv1 = nn.Conv1d(6, 64, 1)
        self.obj_bn1 = nn.BatchNorm1d(64)
        self.obj_conv2 = nn.Conv1d(64, 128, 1)
        self.obj_bn2 = nn.BatchNorm1d(128)
        self.obj_conv3 = nn.Conv1d(128, embed_dim, 1)
        self.obj_bn3 = nn.BatchNorm1d(embed_dim)
        self.obj_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        self.char_embed = nn.Embedding(256, 128)
        self.text_conv1 = nn.Conv1d(128, 256, 3, padding=1)
        self.text_bn1 = nn.BatchNorm1d(256)
        self.text_conv2 = nn.Conv1d(256, embed_dim, 3, padding=1)
        self.text_bn2 = nn.BatchNorm1d(embed_dim)
        self.text_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.text_fc = nn.Linear(hidden_dim * 2, embed_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.location_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )
    
    def encode_objects(self, object_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, _ = object_features.shape
        x = object_features.transpose(1, 2)
        x = torch.relu(self.obj_bn1(self.obj_conv1(x)))
        x = torch.relu(self.obj_bn2(self.obj_conv2(x)))
        x = torch.relu(self.obj_bn3(self.obj_conv3(x)))
        x = x.transpose(1, 2)
        x, _ = self.obj_attention(x, x, x)
        x = torch.max(x, dim=1)[0]
        return x
    
    def encode_text(self, descriptions: List[str]) -> torch.Tensor:
        batch_size = len(descriptions)
        max_len = max(len(d) for d in descriptions) if descriptions else 1
        char_indices = []
        for desc in descriptions:
            indices = [min(ord(c), 255) for c in desc[:max_len]]
            indices += [0] * (max_len - len(indices))
            char_indices.append(indices)
        x = torch.tensor(char_indices, dtype=torch.long).to(next(self.parameters()).device)
        x = self.char_embed(x)
        x = x.transpose(1, 2)
        x = torch.relu(self.text_bn1(self.text_conv1(x)))
        x = torch.relu(self.text_bn2(self.text_conv2(x)))
        x = x.transpose(1, 2)
        x, _ = self.text_lstm(x)
        x = x[:, -1, :]
        x = self.text_fc(x)
        return x
    
    def forward(self, descriptions: List[str], object_features: torch.Tensor) -> torch.Tensor:
        text_enc = self.encode_text(descriptions)
        obj_enc = self.encode_objects(object_features)
        fused = torch.cat([text_enc, obj_enc], dim=1)
        fused = self.fusion(fused)
        offset = self.location_head(fused)
        return offset


def collate_fn(batch):
    descriptions = [item['description'] for item in batch]
    object_features = torch.stack([item['object_features'] for item in batch])
    gt_offsets = torch.stack([item['gt_offset'] for item in batch])
    cell_centers = torch.stack([item['cell_center'] for item in batch])
    cell_ids = [item['cell_id'] for item in batch]
    global_xys = [item['global_xy'] for item in batch]
    return {
        'descriptions': descriptions,
        'object_features': object_features,
        'gt_offset': gt_offsets,
        'cell_center': cell_centers,
        'cell_ids': cell_ids,
        'global_xys': global_xys
    }


class SimpleDataset:
    def __init__(self, cells: List[Dict], poses: List[Dict]):
        self.cells_dict = {cell['id']: cell for cell in cells if isinstance(cell, dict)}
        self.poses = [p for p in poses if isinstance(p, dict) and p.get('cell_id') in self.cells_dict]
        
        all_centers = []
        for cell in self.cells_dict.values():
            center = cell.get('center', [0, 0, 0])
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                all_centers.append([float(center[0]), float(center[1])])
        
        if all_centers:
            all_centers = np.array(all_centers)
            self.global_mean = np.mean(all_centers, axis=0)
            self.global_std = np.std(all_centers, axis=0) + 1e-8
        else:
            self.global_mean = np.array([0.0, 0.0])
            self.global_std = np.array([1.0, 1.0])
    
    def _normalize(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self.global_mean) / self.global_std
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        pose = self.poses[idx]
        cell = self.cells_dict[pose['cell_id']]
        description = pose.get('description', f"Location near {pose['cell_id']}")
        cell_center = cell.get('center', [0, 0, 0])
        cell_center_xy = np.array([float(cell_center[0]), float(cell_center[1])])
        location = pose.get('location', [0, 0])
        if isinstance(location, (list, tuple)) and len(location) >= 2:
            global_xy = np.array([float(location[0]), float(location[1])])
        else:
            global_xy = cell_center_xy.copy()
        
        offset_xy = global_xy - cell_center_xy
        normalized_offset = self._normalize(offset_xy)
        
        objects = cell.get('objects', [])
        object_features = []
        for obj in objects:
            if isinstance(obj, dict):
                center = obj.get('center', [0, 0, 0])
                obj_xy = np.array([float(center[0]) if len(center) > 0 else 0,
                                  float(center[1]) if len(center) > 1 else 0])
                obj_offset = obj_xy - cell_center_xy
                feat = [obj_offset[0], obj_offset[1], float(center[2]) if len(center) > 2 else 0, 0.5, 0.5, 0.5]
                object_features.append(feat)
        
        if not object_features:
            object_features = [[0.0] * 6]
        
        max_objects = 50
        if len(object_features) < max_objects:
            object_features.extend([[0.0] * 6] * (max_objects - len(object_features)))
        else:
            object_features = object_features[:max_objects]
        
        return {
            'description': description,
            'object_features': torch.tensor(object_features, dtype=torch.float32),
            'gt_offset': torch.tensor(normalized_offset, dtype=torch.float32),
            'cell_center': torch.tensor(cell_center_xy, dtype=torch.float32),
            'cell_id': pose['cell_id'],
            'global_xy': global_xy
        }


def analyze_predictions(model, dataset, device, num_samples=100):
    """详细分析预测结果"""
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    model.eval()
    
    all_pred_offsets = []
    all_gt_offsets = []
    all_pred_global = []
    all_gt_global = []
    all_cell_centers = []
    
    with torch.no_grad():
        for batch in loader:
            descriptions = batch['descriptions']
            object_features = batch['object_features'].to(device)
            gt_offsets = batch['gt_offset'].cpu().numpy()
            cell_centers = batch['cell_center'].cpu().numpy()
            gt_global = np.array(batch['global_xys'])
            
            pred_offsets = model(descriptions, object_features).cpu().numpy()
            pred_offsets_denorm = pred_offsets * dataset.global_std + dataset.global_mean
            pred_global = cell_centers + pred_offsets_denorm
            
            all_pred_offsets.extend(pred_offsets)
            all_gt_offsets.extend(gt_offsets)
            all_pred_global.extend(pred_global)
            all_gt_global.extend(gt_global)
            all_cell_centers.extend(cell_centers)
            
            if len(all_pred_offsets) >= num_samples:
                break
    
    return {
        'pred_offsets': np.array(all_pred_offsets[:num_samples]),
        'gt_offsets': np.array(all_gt_offsets[:num_samples]),
        'pred_global': np.array(all_pred_global[:num_samples]),
        'gt_global': np.array(all_gt_global[:num_samples]),
        'cell_centers': np.array(all_cell_centers[:num_samples])
    }


def main():
    print("="*80)
    print("实验问题诊断")
    print("="*80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_semantics_full")
    
    with open(data_path / "cells" / "cells.pkl", 'rb') as f:
        cells = pickle.load(f)
    with open(data_path / "poses" / "poses.pkl", 'rb') as f:
        poses = pickle.load(f)
    
    cells_dict = {cell['id']: cell for cell in cells if isinstance(cell, dict)}
    
    # 2. 创建测试集
    print("\n2. 创建测试集...")
    np.random.seed(42)
    cell_ids = list(set(p['cell_id'] for p in poses if isinstance(p, dict) and 'cell_id' in p))
    num_test = int(len(cell_ids) * 0.2)
    test_cell_ids = set(np.random.choice(cell_ids, size=num_test, replace=False))
    test_poses = [p for p in poses if isinstance(p, dict) and p.get('cell_id') in test_cell_ids]
    test_dataset = SimpleDataset(cells, test_poses)
    
    print(f"   测试集: {len(test_poses)} poses")
    
    # 3. 加载模型
    print("\n3. 加载模型...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = Text2LocNeuralNetwork(embed_dim=256, hidden_dim=512).to(device)
    
    model_path = Path("checkpoints/semantics_full_best_model.pth")
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ 模型加载成功")
    else:
        print(f"   ❌ 模型不存在")
        return
    
    # 4. 分析预测
    print("\n4. 分析神经网络预测...")
    analysis = analyze_predictions(model, test_dataset, device, num_samples=100)
    
    pred_offsets = analysis['pred_offsets']
    gt_offsets = analysis['gt_offsets']
    pred_global = analysis['pred_global']
    gt_global = analysis['gt_global']
    cell_centers = analysis['cell_centers']
    
    # 5. 关键诊断
    print("\n" + "="*80)
    print("诊断结果")
    print("="*80)
    
    # 5.1 检查预测偏移量
    print("\n【1. 预测偏移量分析】")
    print(f"   预测偏移量均值: [{np.mean(pred_offsets[:, 0]):.4f}, {np.mean(pred_offsets[:, 1]):.4f}]")
    print(f"   预测偏移量标准差: [{np.std(pred_offsets[:, 0]):.4f}, {np.std(pred_offsets[:, 1]):.4f}]")
    print(f"   真实偏移量均值: [{np.mean(gt_offsets[:, 0]):.4f}, {np.mean(gt_offsets[:, 1]):.4f}]")
    print(f"   真实偏移量标准差: [{np.std(gt_offsets[:, 0]):.4f}, {np.std(gt_offsets[:, 1]):.4f}]")
    
    # 5.2 检查预测是否接近0（即接近cell中心）
    pred_offset_magnitudes = np.linalg.norm(pred_offsets, axis=1)
    print(f"\n   预测偏移量大小均值: {np.mean(pred_offset_magnitudes):.4f}")
    print(f"   预测偏移量大小中位数: {np.median(pred_offset_magnitudes):.4f}")
    print(f"   预测偏移量接近0的比例 (<0.1): {np.mean(pred_offset_magnitudes < 0.1)*100:.1f}%")
    
    # 5.3 对比基线（cell中心）
    baseline_errors = np.linalg.norm(cell_centers - gt_global, axis=1)
    nn_errors = np.linalg.norm(pred_global - gt_global, axis=1)
    
    print("\n【2. 误差对比】")
    print(f"   基线误差 (cell中心): {np.mean(baseline_errors):.2f}m")
    print(f"   神经网络误差: {np.mean(nn_errors):.2f}m")
    print(f"   改进: {np.mean(baseline_errors) - np.mean(nn_errors):.2f}m")
    
    # 5.4 检查是否有过拟合
    print("\n【3. 训练验证曲线分析】")
    if 'train_losses' in checkpoint and 'val_errors' in checkpoint:
        train_losses = checkpoint['train_losses']
        val_errors = checkpoint['val_errors']
        print(f"   最终训练损失: {train_losses[-1]:.6f}")
        print(f"   最终验证误差: {val_errors[-1]:.2f}m")
        print(f"   最佳验证误差: {min(val_errors):.2f}m")
    else:
        print("   训练历史不可用")
    
    # 5.5 问题诊断
    print("\n【4. 问题诊断】")
    
    if np.mean(pred_offset_magnitudes) < 0.1:
        print("   ⚠️ 问题1: 神经网络预测的偏移量几乎为0")
        print("      → 模型可能只是学习了预测cell中心")
        print("      → 可能原因: 训练时偏移量归一化问题或模型容量不足")
    
    if np.mean(baseline_errors) - np.mean(nn_errors) < 0.5:
        print("   ⚠️ 问题2: 神经网络相比基线改进很小")
        print("      → 可能原因: 任务本身很简单，cell中心已经是很好的预测")
        print("      → 或者模型没有充分学习")
    
    # 6. 建议
    print("\n" + "="*80)
    print("改进建议")
    print("="*80)
    print("\n1. 增加cell大小（如20m或50m），让任务更具挑战性")
    print("2. 检查训练时的归一化参数是否正确")
    print("3. 增加模型复杂度或训练更多epoch")
    print("4. 使用更大的数据集")
    print("5. 检查损失函数是否设计合理")

if __name__ == '__main__':
    main()
