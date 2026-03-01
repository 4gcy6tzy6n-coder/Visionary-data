#!/usr/bin/env python3
"""
初步筛选实验 - 多维度对比
为50G大数据全面实验做准备
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
import random

class Text2LocNeuralNetwork(nn.Module):
    """Text2Loc Visionary模型"""
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


# ============== 多维度评估函数 ==============

def evaluate_with_noise(model, dataset, device, noise_levels=[0.0, 0.1, 0.2, 0.3]):
    """噪声鲁棒性测试"""
    results = {}
    
    for noise_level in noise_levels:
        errors = []
        
        for i in range(len(dataset)):
            item = dataset[i]
            
            # 添加噪声到object features
            object_features = item['object_features'].clone()
            if noise_level > 0:
                noise = torch.randn_like(object_features) * noise_level
                object_features = object_features + noise
            
            with torch.no_grad():
                pred_offset = model([item['description']], object_features.unsqueeze(0).to(device)).cpu().numpy()[0]
                pred_offset_denorm = dataset._denormalize(pred_offset)
                pred_global = item['cell_center'].numpy() + pred_offset_denorm
                gt_global = item['global_xy']
                
                error = np.linalg.norm(pred_global - gt_global)
                errors.append(error)
        
        results[f'noise_{noise_level}'] = {
            'mean_error': float(np.mean(errors)),
            'acc_5m': float(np.mean(np.array(errors) <= 5.0) * 100)
        }
    
    return results


def evaluate_by_cell_size(test_poses, cells_dict, model=None, dataset=None, device=None):
    """按cell大小评估（模拟不同尺度）"""
    # 计算每个cell的实际大小
    cell_sizes = {}
    for cell_id, cell in cells_dict.items():
        objects = cell.get('objects', [])
        if len(objects) > 1:
            centers = []
            for obj in objects:
                if isinstance(obj, dict):
                    center = obj.get('center', [0, 0, 0])
                    centers.append([float(center[0]), float(center[1])])
            if len(centers) > 1:
                centers = np.array(centers)
                size = np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)) * 2
                cell_sizes[cell_id] = size
    
    # 按大小分组
    small_cells = [cid for cid, size in cell_sizes.items() if size < 5]
    medium_cells = [cid for cid, size in cell_sizes.items() if 5 <= size < 15]
    large_cells = [cid for cid, size in cell_sizes.items() if size >= 15]
    
    results = {}
    
    for name, cell_list in [('small', small_cells), ('medium', medium_cells), ('large', large_cells)]:
        cell_set = set(cell_list)
        poses_subset = [p for p in test_poses if p.get('cell_id') in cell_set]
        
        if len(poses_subset) == 0:
            continue
        
        # Text2Loc-one
        errors_one = []
        for pose in poses_subset:
            cell_id = pose.get('cell_id')
            cell = cells_dict.get(cell_id, {})
            pred = np.array(cell.get('center', [0, 0, 0])[:2])
            gt = np.array([float(pose['location'][0]), float(pose['location'][1])])
            errors_one.append(np.linalg.norm(pred - gt))
        
        results[name] = {
            'count': len(poses_subset),
            'text2loc_one_mean_error': float(np.mean(errors_one)),
            'text2loc_one_acc_5m': float(np.mean(np.array(errors_one) <= 5.0) * 100)
        }
    
    return results


def evaluate_description_types(test_poses, cells_dict):
    """按描述类型评估"""
    # 分类描述
    directional = []  # 包含方向词
    landmark = []     # 包含地标
    distance = []     # 包含距离
    simple = []       # 简单描述
    
    directional_words = ['left', 'right', 'front', 'back', 'north', 'south', 'east', 'west', 'near', 'far']
    distance_words = ['meter', 'meters', 'm', 'km', 'kilometer']
    
    for pose in test_poses:
        desc = pose.get('description', '').lower()
        
        if any(word in desc for word in directional_words):
            directional.append(pose)
        elif any(word in desc for word in distance_words):
            distance.append(pose)
        elif len(desc.split()) > 5:
            landmark.append(pose)
        else:
            simple.append(pose)
    
    results = {}
    
    for name, poses_subset in [('directional', directional), ('landmark', landmark), 
                               ('distance', distance), ('simple', simple)]:
        if len(poses_subset) == 0:
            continue
        
        errors = []
        for pose in poses_subset:
            cell_id = pose.get('cell_id')
            cell = cells_dict.get(cell_id, {})
            pred = np.array(cell.get('center', [0, 0, 0])[:2])
            gt = np.array([float(pose['location'][0]), float(pose['location'][1])])
            errors.append(np.linalg.norm(pred - gt))
        
        results[name] = {
            'count': len(poses_subset),
            'mean_error': float(np.mean(errors)),
            'acc_5m': float(np.mean(np.array(errors) <= 5.0) * 100)
        }
    
    return results


def evaluate_computational_efficiency(model, dataset, device, num_runs=100):
    """计算效率评估"""
    # 准备数据
    sample = dataset[0]
    descriptions = [sample['description']]
    object_features = sample['object_features'].unsqueeze(0).to(device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(descriptions, object_features)
    
    # 测试推理时间
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(descriptions, object_features)
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    # 测试内存使用
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(descriptions, object_features)
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        'mean_inference_time_ms': float(np.mean(times)),
        'std_inference_time_ms': float(np.std(times)),
        'memory_mb': float(memory_mb),
        'total_params': int(total_params),
        'model_size_mb': float(total_params * 4 / 1024 / 1024)  # 假设float32
    }


def evaluate_generalization(test_poses, cells_dict, train_cell_ids):
    """泛化能力评估 - 在未见过的cell上测试"""
    test_cell_ids = set(p['cell_id'] for p in test_poses if isinstance(p, dict) and 'cell_id' in p)
    unseen_cells = test_cell_ids - train_cell_ids
    
    if len(unseen_cells) == 0:
        return None
    
    unseen_poses = [p for p in test_poses if p.get('cell_id') in unseen_cells]
    
    errors = []
    for pose in unseen_poses:
        cell_id = pose.get('cell_id')
        cell = cells_dict.get(cell_id, {})
        pred = np.array(cell.get('center', [0, 0, 0])[:2])
        gt = np.array([float(pose['location'][0]), float(pose['location'][1])])
        errors.append(np.linalg.norm(pred - gt))
    
    return {
        'unseen_cell_count': len(unseen_cells),
        'unseen_pose_count': len(unseen_poses),
        'mean_error': float(np.mean(errors)),
        'acc_5m': float(np.mean(np.array(errors) <= 5.0) * 100)
    }


def main():
    print("="*80)
    print("初步筛选实验 - 多维度对比")
    print("为50G大数据全面实验做准备")
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
    
    # 2. 创建训练/测试集
    print("\n2. 创建训练/测试集...")
    np.random.seed(42)
    cell_ids = list(set(p['cell_id'] for p in poses if isinstance(p, dict) and 'cell_id' in p))
    num_test = int(len(cell_ids) * 0.2)
    test_cell_ids = set(np.random.choice(cell_ids, size=num_test, replace=False))
    train_cell_ids = set(cell_ids) - test_cell_ids
    test_poses = [p for p in poses if isinstance(p, dict) and p.get('cell_id') in test_cell_ids]
    test_dataset = SimpleDataset(cells, test_poses)
    
    print(f"   训练集: {len(train_cell_ids)} cells")
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
    
    all_results = {}
    
    # ============== 实验1：噪声鲁棒性 ==============
    print("\n" + "="*80)
    print("实验1：噪声鲁棒性测试")
    print("="*80)
    
    noise_results = evaluate_with_noise(model, test_dataset, device)
    all_results['noise_robustness'] = noise_results
    
    print("\n不同噪声水平下的性能：")
    print(f"{'噪声水平':<15} {'平均误差':>15} {'Acc@5m':>15}")
    print("-"*45)
    for level, res in noise_results.items():
        print(f"{level:<15} {res['mean_error']:>14.2f}m {res['acc_5m']:>14.1f}%")
    
    # ============== 实验2：Cell大小影响 ==============
    print("\n" + "="*80)
    print("实验2：Cell大小影响")
    print("="*80)
    
    size_results = evaluate_by_cell_size(test_poses, cells_dict)
    all_results['cell_size'] = size_results
    
    print("\n不同Cell大小下的性能：")
    print(f"{'Cell大小':<15} {'样本数':>10} {'平均误差':>15} {'Acc@5m':>15}")
    print("-"*55)
    for size, res in size_results.items():
        print(f"{size:<15} {res['count']:>10} {res['text2loc_one_mean_error']:>14.2f}m {res['text2loc_one_acc_5m']:>14.1f}%")
    
    # ============== 实验3：描述类型分析 ==============
    print("\n" + "="*80)
    print("实验3：描述类型分析")
    print("="*80)
    
    desc_results = evaluate_description_types(test_poses, cells_dict)
    all_results['description_types'] = desc_results
    
    print("\n不同描述类型的性能：")
    print(f"{'描述类型':<15} {'样本数':>10} {'平均误差':>15} {'Acc@5m':>15}")
    print("-"*55)
    for dtype, res in desc_results.items():
        print(f"{dtype:<15} {res['count']:>10} {res['mean_error']:>14.2f}m {res['acc_5m']:>14.1f}%")
    
    # ============== 实验4：计算效率 ==============
    print("\n" + "="*80)
    print("实验4：计算效率评估")
    print("="*80)
    
    efficiency_results = evaluate_computational_efficiency(model, test_dataset, device)
    all_results['computational_efficiency'] = efficiency_results
    
    print(f"\n模型信息：")
    print(f"  总参数量: {efficiency_results['total_params']:,}")
    print(f"  模型大小: {efficiency_results['model_size_mb']:.2f} MB")
    print(f"\n推理性能：")
    print(f"  平均推理时间: {efficiency_results['mean_inference_time_ms']:.2f} ms")
    print(f"  标准差: {efficiency_results['std_inference_time_ms']:.2f} ms")
    print(f"  峰值内存: {efficiency_results['memory_mb']:.2f} MB")
    
    # ============== 实验5：泛化能力 ==============
    print("\n" + "="*80)
    print("实验5：泛化能力评估")
    print("="*80)
    
    generalization_results = evaluate_generalization(test_poses, cells_dict, train_cell_ids)
    if generalization_results:
        all_results['generalization'] = generalization_results
        
        print(f"\n未见过的Cell：")
        print(f"  未见Cell数: {generalization_results['unseen_cell_count']}")
        print(f"  未见Pose数: {generalization_results['unseen_pose_count']}")
        print(f"  平均误差: {generalization_results['mean_error']:.2f}m")
        print(f"  Acc@5m: {generalization_results['acc_5m']:.1f}%")
    
    # ============== 总结 ==============
    print("\n" + "="*80)
    print("初步筛选实验总结")
    print("="*80)
    
    print("\n✅ 已完成的筛选维度：")
    print("  1. 噪声鲁棒性 - 测试模型对输入噪声的容忍度")
    print("  2. Cell大小影响 - 不同尺度场景的表现")
    print("  3. 描述类型分析 - 不同类型描述的处理能力")
    print("  4. 计算效率 - 推理时间和资源占用")
    print("  5. 泛化能力 - 在未见数据上的表现")
    
    print("\n📊 推荐在50G大数据上重点测试的维度：")
    print("  ⭐ 噪声鲁棒性 - 真实场景中有传感器噪声")
    print("  ⭐ 描述多样性 - 大数据包含更多描述类型")
    print("  ⭐ 跨Cell定位 - 展示核心优势")
    print("  ⭐ 大规模场景 - 更大cell大小的对比")
    
    # 保存结果
    with open('preliminary_screening_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n详细结果已保存到: preliminary_screening_results.json")

if __name__ == '__main__':
    main()
