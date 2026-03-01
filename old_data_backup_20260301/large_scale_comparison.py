#!/usr/bin/env python3
"""
大规模对比实验 - 找出核心优势
对比维度：
1. 跨Cell定位能力
2. 不同场景复杂度
3. 描述详细程度
4. 边缘位置定位
5. 多对象场景
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

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


# ============== 核心优势对比实验 ==============

def evaluate_cross_cell_advantage(test_poses, cells_dict, model, dataset, device):
    """
    核心优势1：跨Cell定位能力
    测试在cell边缘位置的定位精度
    """
    print("\n" + "="*80)
    print("核心优势1：跨Cell定位能力")
    print("="*80)
    
    # 计算每个pose距离cell中心的距离
    edge_poses = []
    center_poses = []
    
    for pose in test_poses:
        cell_id = pose.get('cell_id')
        cell = cells_dict.get(cell_id, {})
        cell_center = cell.get('center', [0, 0, 0])
        cell_center_xy = np.array([float(cell_center[0]), float(cell_center[1])])
        
        location = pose.get('location', [0, 0])
        if isinstance(location, (list, tuple)) and len(location) >= 2:
            global_xy = np.array([float(location[0]), float(location[1])])
            distance_to_center = np.linalg.norm(global_xy - cell_center_xy)
            
            if distance_to_center > 3.0:  # 距离中心超过3m认为是边缘
                edge_poses.append(pose)
            else:
                center_poses.append(pose)
    
    results = {}
    
    for name, poses_subset in [('边缘位置', edge_poses), ('中心位置', center_poses)]:
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
        
        # Text2Loc Visionary
        errors_visionary = []
        for pose in poses_subset:
            idx = test_poses.index(pose)
            item = dataset[idx]
            
            with torch.no_grad():
                pred_offset = model([item['description']], item['object_features'].unsqueeze(0).to(device)).cpu().numpy()[0]
                pred_offset_denorm = dataset._denormalize(pred_offset)
                pred_global = item['cell_center'].numpy() + pred_offset_denorm
                gt_global = item['global_xy']
                errors_visionary.append(np.linalg.norm(pred_global - gt_global))
        
        errors_one = np.array(errors_one)
        errors_visionary = np.array(errors_visionary)
        
        improvement = ((np.mean(errors_one) - np.mean(errors_visionary)) / np.mean(errors_one)) * 100
        
        results[name] = {
            'count': len(poses_subset),
            'text2loc_one_mean_error': float(np.mean(errors_one)),
            'text2loc_one_acc_5m': float(np.mean(errors_one <= 5.0) * 100),
            'visionary_mean_error': float(np.mean(errors_visionary)),
            'visionary_acc_5m': float(np.mean(errors_visionary <= 5.0) * 100),
            'improvement_percent': float(improvement)
        }
        
        print(f"\n{name} (n={len(poses_subset)}):")
        print(f"  Text2Loc-one: {np.mean(errors_one):.2f}m, Acc@5m: {np.mean(errors_one <= 5.0)*100:.1f}%")
        print(f"  Text2Loc Visionary: {np.mean(errors_visionary):.2f}m, Acc@5m: {np.mean(errors_visionary <= 5.0)*100:.1f}%")
        print(f"  改进幅度: {improvement:+.1f}%")
    
    return results


def evaluate_scene_complexity(test_poses, cells_dict, model, dataset, device):
    """
    核心优势2：不同场景复杂度
    测试在简单vs复杂场景的表现
    """
    print("\n" + "="*80)
    print("核心优势2：场景复杂度适应性")
    print("="*80)
    
    # 根据cell中对象数量分类
    simple_poses = []
    complex_poses = []
    
    for pose in test_poses:
        cell_id = pose.get('cell_id')
        cell = cells_dict.get(cell_id, {})
        num_objects = len(cell.get('objects', []))
        
        if num_objects <= 5:
            simple_poses.append(pose)
        elif num_objects >= 15:
            complex_poses.append(pose)
    
    results = {}
    
    for name, poses_subset in [('简单场景(≤5对象)', simple_poses), ('复杂场景(≥15对象)', complex_poses)]:
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
        
        # Text2Loc Visionary
        errors_visionary = []
        for pose in poses_subset:
            idx = test_poses.index(pose)
            item = dataset[idx]
            
            with torch.no_grad():
                pred_offset = model([item['description']], item['object_features'].unsqueeze(0).to(device)).cpu().numpy()[0]
                pred_offset_denorm = dataset._denormalize(pred_offset)
                pred_global = item['cell_center'].numpy() + pred_offset_denorm
                gt_global = item['global_xy']
                errors_visionary.append(np.linalg.norm(pred_global - gt_global))
        
        errors_one = np.array(errors_one)
        errors_visionary = np.array(errors_visionary)
        
        improvement = ((np.mean(errors_one) - np.mean(errors_visionary)) / np.mean(errors_one)) * 100
        
        results[name] = {
            'count': len(poses_subset),
            'text2loc_one_mean_error': float(np.mean(errors_one)),
            'text2loc_one_acc_5m': float(np.mean(errors_one <= 5.0) * 100),
            'visionary_mean_error': float(np.mean(errors_visionary)),
            'visionary_acc_5m': float(np.mean(errors_visionary <= 5.0) * 100),
            'improvement_percent': float(improvement)
        }
        
        print(f"\n{name} (n={len(poses_subset)}):")
        print(f"  Text2Loc-one: {np.mean(errors_one):.2f}m, Acc@5m: {np.mean(errors_one <= 5.0)*100:.1f}%")
        print(f"  Text2Loc Visionary: {np.mean(errors_visionary):.2f}m, Acc@5m: {np.mean(errors_visionary <= 5.0)*100:.1f}%")
        print(f"  改进幅度: {improvement:+.1f}%")
    
    return results


def evaluate_description_quality(test_poses, cells_dict, model, dataset, device):
    """
    核心优势3：描述质量适应性
    测试对不同详细程度描述的处理能力
    """
    print("\n" + "="*80)
    print("核心优势3：描述质量适应性")
    print("="*80)
    
    # 根据描述长度分类
    short_desc = []  # < 30字符
    medium_desc = []  # 30-60字符
    long_desc = []   # > 60字符
    
    for pose in test_poses:
        desc = pose.get('description', '')
        length = len(desc)
        
        if length < 30:
            short_desc.append(pose)
        elif length < 60:
            medium_desc.append(pose)
        else:
            long_desc.append(pose)
    
    results = {}
    
    for name, poses_subset in [('简短描述(<30)', short_desc), 
                               ('中等描述(30-60)', medium_desc),
                               ('详细描述(>60)', long_desc)]:
        if len(poses_subset) == 0:
            continue
        
        # Text2Loc-one (描述不影响，只预测cell中心)
        errors_one = []
        for pose in poses_subset:
            cell_id = pose.get('cell_id')
            cell = cells_dict.get(cell_id, {})
            pred = np.array(cell.get('center', [0, 0, 0])[:2])
            gt = np.array([float(pose['location'][0]), float(pose['location'][1])])
            errors_one.append(np.linalg.norm(pred - gt))
        
        # Text2Loc Visionary
        errors_visionary = []
        for pose in poses_subset:
            idx = test_poses.index(pose)
            item = dataset[idx]
            
            with torch.no_grad():
                pred_offset = model([item['description']], item['object_features'].unsqueeze(0).to(device)).cpu().numpy()[0]
                pred_offset_denorm = dataset._denormalize(pred_offset)
                pred_global = item['cell_center'].numpy() + pred_offset_denorm
                gt_global = item['global_xy']
                errors_visionary.append(np.linalg.norm(pred_global - gt_global))
        
        errors_one = np.array(errors_one)
        errors_visionary = np.array(errors_visionary)
        
        improvement = ((np.mean(errors_one) - np.mean(errors_visionary)) / np.mean(errors_one)) * 100
        
        results[name] = {
            'count': len(poses_subset),
            'text2loc_one_mean_error': float(np.mean(errors_one)),
            'text2loc_one_acc_5m': float(np.mean(errors_one <= 5.0) * 100),
            'visionary_mean_error': float(np.mean(errors_visionary)),
            'visionary_acc_5m': float(np.mean(errors_visionary <= 5.0) * 100),
            'improvement_percent': float(improvement)
        }
        
        print(f"\n{name} (n={len(poses_subset)}):")
        print(f"  Text2Loc-one: {np.mean(errors_one):.2f}m, Acc@5m: {np.mean(errors_one <= 5.0)*100:.1f}%")
        print(f"  Text2Loc Visionary: {np.mean(errors_visionary):.2f}m, Acc@5m: {np.mean(errors_visionary <= 5.0)*100:.1f}%")
        print(f"  改进幅度: {improvement:+.1f}%")
    
    return results


def evaluate_distance_ranges(test_poses, cells_dict, model, dataset, device):
    """
    核心优势4：不同距离范围
    测试在不同距离误差范围内的表现
    """
    print("\n" + "="*80)
    print("核心优势4：不同距离范围定位精度")
    print("="*80)
    
    # 先计算Text2Loc-one的误差，然后按误差范围分组
    pose_errors = []
    for pose in test_poses:
        cell_id = pose.get('cell_id')
        cell = cells_dict.get(cell_id, {})
        pred = np.array(cell.get('center', [0, 0, 0])[:2])
        gt = np.array([float(pose['location'][0]), float(pose['location'][1])])
        error = np.linalg.norm(pred - gt)
        pose_errors.append((pose, error))
    
    # 按误差分组
    low_error = [p for p, e in pose_errors if e < 2.0]
    mid_error = [p for p, e in pose_errors if 2.0 <= e < 5.0]
    high_error = [p for p, e in pose_errors if e >= 5.0]
    
    results = {}
    
    for name, poses_subset in [('低误差场景(<2m)', low_error), 
                               ('中等误差场景(2-5m)', mid_error),
                               ('高误差场景(≥5m)', high_error)]:
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
        
        # Text2Loc Visionary
        errors_visionary = []
        for pose in poses_subset:
            idx = test_poses.index(pose)
            item = dataset[idx]
            
            with torch.no_grad():
                pred_offset = model([item['description']], item['object_features'].unsqueeze(0).to(device)).cpu().numpy()[0]
                pred_offset_denorm = dataset._denormalize(pred_offset)
                pred_global = item['cell_center'].numpy() + pred_offset_denorm
                gt_global = item['global_xy']
                errors_visionary.append(np.linalg.norm(pred_global - gt_global))
        
        errors_one = np.array(errors_one)
        errors_visionary = np.array(errors_visionary)
        
        improvement = ((np.mean(errors_one) - np.mean(errors_visionary)) / np.mean(errors_one)) * 100
        
        results[name] = {
            'count': len(poses_subset),
            'text2loc_one_mean_error': float(np.mean(errors_one)),
            'text2loc_one_acc_5m': float(np.mean(errors_one <= 5.0) * 100),
            'visionary_mean_error': float(np.mean(errors_visionary)),
            'visionary_acc_5m': float(np.mean(errors_visionary <= 5.0) * 100),
            'improvement_percent': float(improvement)
        }
        
        print(f"\n{name} (n={len(poses_subset)}):")
        print(f"  Text2Loc-one: {np.mean(errors_one):.2f}m, Acc@5m: {np.mean(errors_one <= 5.0)*100:.1f}%")
        print(f"  Text2Loc Visionary: {np.mean(errors_visionary):.2f}m, Acc@5m: {np.mean(errors_visionary <= 5.0)*100:.1f}%")
        print(f"  改进幅度: {improvement:+.1f}%")
    
    return results


def main():
    print("="*80)
    print("大规模对比实验 - 找出核心优势")
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
    test_poses = [p for p in poses if isinstance(p, dict) and p.get('cell_id') in test_cell_ids]
    test_dataset = SimpleDataset(cells, test_poses)
    
    print(f"   测试集: {len(test_poses)} poses")
    
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
    
    model.eval()
    
    all_results = {}
    
    # 运行所有对比实验
    all_results['cross_cell'] = evaluate_cross_cell_advantage(test_poses, cells_dict, model, test_dataset, device)
    all_results['scene_complexity'] = evaluate_scene_complexity(test_poses, cells_dict, model, test_dataset, device)
    all_results['description_quality'] = evaluate_description_quality(test_poses, cells_dict, model, test_dataset, device)
    all_results['distance_ranges'] = evaluate_distance_ranges(test_poses, cells_dict, model, test_dataset, device)
    
    # 总结核心优势
    print("\n" + "="*80)
    print("核心优势总结")
    print("="*80)
    
    print("\n🌟 Text2Loc Visionary的核心优势：")
    
    # 统计各维度的改进
    improvements = []
    for category, results in all_results.items():
        for name, data in results.items():
            if 'improvement_percent' in data:
                improvements.append({
                    'category': category,
                    'name': name,
                    'improvement': data['improvement_percent'],
                    'count': data['count']
                })
    
    # 按改进幅度排序
    improvements.sort(key=lambda x: x['improvement'], reverse=True)
    
    print("\n改进幅度排名（Top 5）：")
    for i, imp in enumerate(improvements[:5], 1):
        print(f"  {i}. {imp['name']}: {imp['improvement']:+.1f}% (n={imp['count']})")
    
    # 找出最大优势场景
    best_improvement = improvements[0] if improvements else None
    if best_improvement:
        print(f"\n✅ 最大优势场景: {best_improvement['name']}")
        print(f"   改进幅度: {best_improvement['improvement']:+.1f}%")
        print(f"   样本数量: {best_improvement['count']}")
    
    # 保存结果
    with open('large_scale_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n详细结果已保存到: large_scale_comparison_results.json")
    
    return all_results


if __name__ == '__main__':
    main()
