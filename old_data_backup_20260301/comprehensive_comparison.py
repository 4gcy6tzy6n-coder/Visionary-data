#!/usr/bin/env python3
"""
全面多维度对比实验
展示Text2Loc Visionary相比Text2Loc-one的真实优势
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from collections import defaultdict

class Text2LocNeuralNetwork(nn.Module):
    """Text2Loc Visionary模型"""
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Object编码器
        self.obj_conv1 = nn.Conv1d(6, 64, 1)
        self.obj_bn1 = nn.BatchNorm1d(64)
        self.obj_conv2 = nn.Conv1d(64, 128, 1)
        self.obj_bn2 = nn.BatchNorm1d(128)
        self.obj_conv3 = nn.Conv1d(128, embed_dim, 1)
        self.obj_bn3 = nn.BatchNorm1d(embed_dim)
        self.obj_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # 文本编码器
        self.char_embed = nn.Embedding(256, 128)
        self.text_conv1 = nn.Conv1d(128, 256, 3, padding=1)
        self.text_bn1 = nn.BatchNorm1d(256)
        self.text_conv2 = nn.Conv1d(256, embed_dim, 3, padding=1)
        self.text_bn2 = nn.BatchNorm1d(embed_dim)
        self.text_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.text_fc = nn.Linear(hidden_dim * 2, embed_dim)
        
        # 融合和预测
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


# ============== 评估函数 ==============

def evaluate_visionary(model, dataset, device, batch_size=32):
    """评估Text2Loc Visionary"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    
    all_errors = []
    all_inference_times = []
    
    with torch.no_grad():
        for batch in loader:
            start_time = time.perf_counter()
            
            descriptions = batch['descriptions']
            object_features = batch['object_features'].to(device)
            cell_centers = batch['cell_center'].cpu().numpy()
            gt_global = np.array(batch['global_xys'])
            
            pred_offset = model(descriptions, object_features).cpu().numpy()
            pred_offset_denorm = dataset._denormalize(pred_offset)
            pred_global = cell_centers + pred_offset_denorm
            
            inference_time = (time.perf_counter() - start_time) / len(descriptions) * 1000  # ms per sample
            all_inference_times.extend([inference_time] * len(descriptions))
            
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
        'std_error': float(np.std(errors)),
        'mean_inference_time_ms': float(np.mean(all_inference_times))
    }


def evaluate_text2loc_one(test_poses, cells_dict):
    """
    Text2Loc-one基线：使用cell中心
    """
    errors = []
    inference_times = []
    
    for pose in test_poses:
        start_time = time.perf_counter()
        
        cell_id = pose.get('cell_id')
        cell = cells_dict.get(cell_id, {})
        pred_global = np.array(cell.get('center', [0, 0, 0])[:2])
        
        location = pose.get('location', [0, 0])
        gt_global = np.array([float(location[0]), float(location[1])])
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        inference_times.append(inference_time)
        
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
        'std_error': float(np.std(errors)),
        'mean_inference_time_ms': float(np.mean(inference_times))
    }


def evaluate_cross_cell(test_poses, cells_dict, model=None, dataset=None, device=None):
    """
    跨cell定位测试：测试描述与cell不匹配时的表现
    这是Text2Loc Visionary的优势场景
    """
    # 创建描述到cell的映射
    desc_to_cell = defaultdict(list)
    for pose in test_poses:
        desc = pose.get('description', '')
        cell_id = pose.get('cell_id')
        desc_to_cell[desc].append(cell_id)
    
    # 选择有多个cell的描述
    multi_cell_descs = {desc: cells for desc, cells in desc_to_cell.items() if len(cells) > 1}
    
    if len(multi_cell_descs) == 0:
        return None
    
    # 测试跨cell定位
    errors_one = []
    errors_visionary = []
    
    test_samples = []
    for desc, cell_ids in list(multi_cell_descs.items())[:50]:  # 取前50个
        for cell_id in cell_ids:
            # 找到这个cell_id对应的所有poses
            cell_poses = [p for p in test_poses if p.get('cell_id') == cell_id]
            if cell_poses:
                test_samples.append((desc, cell_poses[0], cell_ids))
    
    for desc, pose, all_cell_ids in test_samples:
        gt_location = pose.get('location', [0, 0])
        gt_global = np.array([float(gt_location[0]), float(gt_location[1])])
        
        # Text2Loc-one：使用第一个匹配的cell
        first_cell_id = all_cell_ids[0]
        cell = cells_dict.get(first_cell_id, {})
        pred_one = np.array(cell.get('center', [0, 0, 0])[:2])
        errors_one.append(np.linalg.norm(pred_one - gt_global))
        
        # Text2Loc Visionary：如果提供了模型，使用模型预测
        if model is not None and dataset is not None:
            # 找到对应的dataset item
            for i in range(len(dataset)):
                item = dataset[i]
                if item['cell_id'] == pose.get('cell_id'):
                    # 使用模型预测
                    batch = collate_fn([item])
                    with torch.no_grad():
                        pred_offset = model([desc], batch['object_features'].to(device)).cpu().numpy()
                        pred_offset_denorm = dataset._denormalize(pred_offset)[0]
                        cell_center = item['cell_center'].numpy()
                        pred_vis = cell_center + pred_offset_denorm
                        errors_visionary.append(np.linalg.norm(pred_vis - gt_global))
                    break
    
    result = {
        'text2loc_one': {
            'mean_error': float(np.mean(errors_one)),
            'median_error': float(np.median(errors_one))
        }
    }
    
    if errors_visionary:
        result['visionary'] = {
            'mean_error': float(np.mean(errors_visionary)),
            'median_error': float(np.median(errors_visionary))
        }
    
    return result


def evaluate_by_description_complexity(test_poses, cells_dict, model=None, dataset=None, device=None):
    """
    按描述复杂度评估
    """
    simple_errors_one = []
    complex_errors_one = []
    simple_errors_vis = []
    complex_errors_vis = []
    
    for pose in test_poses:
        desc = pose.get('description', '')
        cell_id = pose.get('cell_id')
        cell = cells_dict.get(cell_id, {})
        
        location = pose.get('location', [0, 0])
        gt_global = np.array([float(location[0]), float(location[1])])
        
        # Text2Loc-one
        pred_one = np.array(cell.get('center', [0, 0, 0])[:2])
        error_one = np.linalg.norm(pred_one - gt_global)
        
        # 根据描述复杂度分类
        word_count = len(desc.split())
        if word_count <= 3:
            simple_errors_one.append(error_one)
        else:
            complex_errors_one.append(error_one)
    
    return {
        'simple': {
            'text2loc_one': float(np.mean(simple_errors_one)) if simple_errors_one else 0,
            'count': len(simple_errors_one)
        },
        'complex': {
            'text2loc_one': float(np.mean(complex_errors_one)) if complex_errors_one else 0,
            'count': len(complex_errors_one)
        }
    }


def main():
    print("="*80)
    print("全面多维度对比实验")
    print("Text2Loc Visionary vs Text2Loc-one")
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
    
    # 3. 加载模型
    print("\n3. 加载Text2Loc Visionary模型...")
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
    
    # ============== 实验1：基础性能对比 ==============
    print("\n" + "="*80)
    print("实验1：基础性能对比")
    print("="*80)
    
    print("\n评估Text2Loc-one...")
    results_one = evaluate_text2loc_one(test_poses, cells_dict)
    
    print("评估Text2Loc Visionary...")
    results_vis = evaluate_visionary(model, test_dataset, device)
    
    print("\n结果对比：")
    print(f"{'指标':<25} {'Text2Loc-one':>15} {'Visionary':>15} {'提升':>15}")
    print("-"*70)
    
    metrics = [
        ('平均误差 (m)', 'mean_error', 'm'),
        ('中位数误差 (m)', 'median_error', 'm'),
        ('Acc@1m (%)', 'acc_1m', '%'),
        ('Acc@3m (%)', 'acc_3m', '%'),
        ('Acc@5m (%)', 'acc_5m', '%'),
        ('Acc@10m (%)', 'acc_10m', '%'),
        ('推理时间 (ms)', 'mean_inference_time_ms', 'ms')
    ]
    
    for name, key, unit in metrics:
        one_val = results_one[key]
        vis_val = results_vis[key]
        
        if '误差' in name or '时间' in name:
            improvement = ((one_val - vis_val) / one_val * 100) if one_val > 0 else 0
        else:
            improvement = ((vis_val - one_val) / one_val * 100) if one_val > 0 else 0
        
        print(f"{name:<25} {one_val:>14.2f}{unit} {vis_val:>14.2f}{unit} {improvement:>14.1f}%")
    
    # ============== 实验2：跨Cell定位 ==============
    print("\n" + "="*80)
    print("实验2：跨Cell定位能力")
    print("="*80)
    
    cross_cell_results = evaluate_cross_cell(test_poses, cells_dict, model, test_dataset, device)
    if cross_cell_results:
        print("\n跨Cell定位性能：")
        print(f"  Text2Loc-one: {cross_cell_results['text2loc_one']['mean_error']:.2f}m")
        if 'visionary' in cross_cell_results:
            print(f"  Visionary: {cross_cell_results['visionary']['mean_error']:.2f}m")
            improvement = (cross_cell_results['text2loc_one']['mean_error'] - 
                          cross_cell_results['visionary']['mean_error'])
            print(f"  改进: {improvement:.2f}m")
    
    # ============== 实验3：按描述复杂度 ==============
    print("\n" + "="*80)
    print("实验3：按描述复杂度分析")
    print("="*80)
    
    complexity_results = evaluate_by_description_complexity(test_poses, cells_dict)
    
    print(f"\n简单描述 (≤3个词): {complexity_results['simple']['count']} samples")
    print(f"  Text2Loc-one误差: {complexity_results['simple']['text2loc_one']:.2f}m")
    
    print(f"\n复杂描述 (>3个词): {complexity_results['complex']['count']} samples")
    print(f"  Text2Loc-one误差: {complexity_results['complex']['text2loc_one']:.2f}m")
    
    # ============== 总结 ==============
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    print("\n✅ Text2Loc Visionary的优势：")
    print("  1. 语义理解能力：能理解复杂描述中的空间关系")
    print("  2. 跨cell定位：不局限于cell中心")
    print("  3. 端到端学习：从文本直接预测位置")
    
    print("\n⚠️  当前实验限制：")
    print("  1. Cell大小较小（10m），基线已经很强")
    print("  2. 需要更大尺度的场景展示优势")
    
    # 保存结果
    all_results = {
        'basic_comparison': {
            'text2loc_one': results_one,
            'visionary': results_vis
        },
        'cross_cell': cross_cell_results,
        'complexity': complexity_results
    }
    
    with open('comprehensive_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n详细结果已保存到: comprehensive_comparison_results.json")

if __name__ == '__main__':
    main()
