#!/usr/bin/env python3
"""
公平对比实验 - 修正版本
对比：
1. Text2Loc-one基线：使用cell中心
2. Text2Loc Visionary：使用神经网络预测
3. 随机基线：随机选择位置
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import List, Dict
from torch.utils.data import DataLoader

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
    
    def _denormalize(self, normalized_xy: np.ndarray) -> np.ndarray:
        return normalized_xy * self.global_std + self.global_mean
    
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


def evaluate_visionary(model, dataset, device, batch_size=32):
    """评估Text2Loc Visionary"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for batch in loader:
            descriptions = batch['descriptions']
            object_features = batch['object_features'].to(device)
            cell_centers = batch['cell_center'].cpu().numpy()
            gt_global = np.array(batch['global_xys'])
            
            pred_offset = model(descriptions, object_features).cpu().numpy()
            pred_offset_denorm = dataset._denormalize(pred_offset)
            pred_global = cell_centers + pred_offset_denorm
            
            errors = np.linalg.norm(pred_global - gt_global, axis=1)
            all_errors.extend(errors)
    
    errors = np.array(all_errors)
    
    return {
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'acc_1m': float(np.mean(errors <= 1.0) * 100),
        'acc_3m': float(np.mean(errors <= 3.0) * 100),
        'acc_5m': float(np.mean(errors <= 5.0) * 100),
        'acc_10m': float(np.mean(errors <= 10.0) * 100),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors))
    }


def evaluate_baseline_cell_center(test_poses, cells_dict):
    """基线：使用cell中心"""
    errors = []
    
    for pose in test_poses:
        cell_id = pose.get('cell_id')
        cell = cells_dict.get(cell_id, {})
        pred_global = np.array(cell.get('center', [0, 0, 0])[:2])
        
        location = pose.get('location', [0, 0])
        gt_global = np.array([float(location[0]), float(location[1])])
        
        error = np.linalg.norm(pred_global - gt_global)
        errors.append(error)
    
    errors = np.array(errors)
    
    return {
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'acc_1m': float(np.mean(errors <= 1.0) * 100),
        'acc_3m': float(np.mean(errors <= 3.0) * 100),
        'acc_5m': float(np.mean(errors <= 5.0) * 100),
        'acc_10m': float(np.mean(errors <= 10.0) * 100),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors))
    }


def evaluate_baseline_random(test_poses, cells_dict):
    """基线：随机选择cell中心"""
    errors = []
    all_centers = [c.get('center', [0, 0, 0])[:2] for c in cells_dict.values()]
    
    for pose in test_poses:
        location = pose.get('location', [0, 0])
        gt_global = np.array([float(location[0]), float(location[1])])
        
        # 随机选择一个cell中心
        pred_global = np.array(all_centers[np.random.randint(len(all_centers))])
        
        error = np.linalg.norm(pred_global - gt_global)
        errors.append(error)
    
    errors = np.array(errors)
    
    return {
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'acc_1m': float(np.mean(errors <= 1.0) * 100),
        'acc_3m': float(np.mean(errors <= 3.0) * 100),
        'acc_5m': float(np.mean(errors <= 5.0) * 100),
        'acc_10m': float(np.mean(errors <= 10.0) * 100),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors))
    }


def main():
    print("="*80)
    print("公平对比实验 - 修正版本")
    print("="*80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_semantics_full")
    
    with open(data_path / "cells" / "cells.pkl", 'rb') as f:
        cells = pickle.load(f)
    with open(data_path / "poses" / "poses.pkl", 'rb') as f:
        poses = pickle.load(f)
    
    cells_dict = {cell['id']: cell for cell in cells if isinstance(cell, dict)}
    
    print(f"   Cells: {len(cells)}, Poses: {len(poses)}")
    
    # 2. 创建测试集
    print("\n2. 创建测试集...")
    np.random.seed(42)
    cell_ids = list(set(p['cell_id'] for p in poses if isinstance(p, dict) and 'cell_id' in p))
    num_test = int(len(cell_ids) * 0.2)
    test_cell_ids = set(np.random.choice(cell_ids, size=num_test, replace=False))
    test_poses = [p for p in poses if isinstance(p, dict) and p.get('cell_id') in test_cell_ids]
    test_dataset = SimpleDataset(cells, test_poses)
    
    print(f"   测试集: {len(test_poses)} poses, {len(test_cell_ids)} cells")
    
    # 3. 评估随机基线
    print("\n3. 评估随机基线...")
    random_results = evaluate_baseline_random(test_poses, cells_dict)
    
    # 4. 评估cell中心基线
    print("\n4. 评估Cell中心基线...")
    center_results = evaluate_baseline_cell_center(test_poses, cells_dict)
    
    # 5. 加载并评估神经网络
    print("\n5. 加载Text2Loc Visionary模型...")
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
    
    print("\n6. 评估Text2Loc Visionary...")
    visionary_results = evaluate_visionary(model, test_dataset, device)
    
    # 7. 打印对比结果
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    
    print(f"\n{'指标':<25} {'随机基线':>15} {'Cell中心':>15} {'Visionary':>15} {'提升(vs中心)':>15}")
    print("-"*85)
    
    metrics = [
        ('平均误差 (m)', 'mean_error'),
        ('中位数误差 (m)', 'median_error'),
        ('Acc@1m (%)', 'acc_1m'),
        ('Acc@3m (%)', 'acc_3m'),
        ('Acc@5m (%)', 'acc_5m'),
        ('Acc@10m (%)', 'acc_10m'),
        ('最大误差 (m)', 'max_error')
    ]
    
    for name, key in metrics:
        rand_val = random_results[key]
        center_val = center_results[key]
        vis_val = visionary_results[key]
        
        if '误差' in name:
            improvement = ((center_val - vis_val) / center_val * 100) if center_val > 0 else 0
        else:
            improvement = ((vis_val - center_val) / center_val * 100) if center_val > 0 else 0
        
        print(f"{name:<25} {rand_val:>14.2f} {center_val:>14.2f} {vis_val:>14.2f} {improvement:>14.1f}%")
    
    # 8. 关键结论
    print("\n" + "="*80)
    print("关键结论")
    print("="*80)
    
    error_reduction = center_results['mean_error'] - visionary_results['mean_error']
    acc_improvement_5m = visionary_results['acc_5m'] - center_results['acc_5m']
    
    print(f"\n相比Cell中心基线：")
    print(f"  误差降低: {error_reduction:.2f}m ({error_reduction/center_results['mean_error']*100:.1f}%)")
    print(f"  5米准确率提升: {acc_improvement_5m:.1f}个百分点")
    
    print(f"\n相比随机基线：")
    error_reduction_rand = random_results['mean_error'] - visionary_results['mean_error']
    print(f"  误差降低: {error_reduction_rand:.2f}m ({error_reduction_rand/random_results['mean_error']*100:.1f}%)")
    
    # 9. 保存结果
    results = {
        'random_baseline': random_results,
        'center_baseline': center_results,
        'visionary': visionary_results,
        'improvements_vs_center': {
            'error_reduction_m': float(error_reduction),
            'acc_5m_improvement': float(acc_improvement_5m),
            'relative_improvement_percent': float(error_reduction / center_results['mean_error'] * 100)
        }
    }
    
    with open('fair_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n详细结果已保存到: fair_comparison_results.json")

if __name__ == '__main__':
    main()
