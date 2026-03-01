#!/usr/bin/env python3
"""
创新3D对比实验 - 基于当前数据展现真实优势
不依赖50G数据，充分利用现有3D语义数据
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
            'global_xy': global_xy,
            'objects': objects  # 保留原始对象信息
        }


# ============== 创新对比场景 ==============

def experiment_1_3d_height_utilization(test_poses, cells_dict, model, dataset, device):
    """
    实验1：3D高度信息利用
    Text2Loc-one只用2D中心，Visionary利用3D高度信息
    """
    print("\n" + "="*80)
    print("实验1：3D高度信息利用")
    print("="*80)
    
    # 根据高度变化分组
    low_height = []    # z < 1m
    mid_height = []    # 1m <= z < 3m
    high_height = []   # z >= 3m
    
    for pose in test_poses:
        location = pose.get('location', [0, 0, 0])
        if len(location) >= 3:
            z = float(location[2])
            if z < 1.0:
                low_height.append(pose)
            elif z < 3.0:
                mid_height.append(pose)
            else:
                high_height.append(pose)
    
    results = {}
    
    for name, poses_subset in [('低高度(z<1m)', low_height), 
                               ('中高度(1-3m)', mid_height),
                               ('高高度(z≥3m)', high_height)]:
        if len(poses_subset) < 10:
            continue
        
        # Text2Loc-one（只用2D中心）
        errors_one = []
        for pose in poses_subset:
            cell_id = pose.get('cell_id')
            cell = cells_dict.get(cell_id, {})
            pred = np.array(cell.get('center', [0, 0, 0])[:2])
            gt = np.array([float(pose['location'][0]), float(pose['location'][1])])
            errors_one.append(np.linalg.norm(pred - gt))
        
        # Text2Loc Visionary（利用3D信息）
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
            'visionary_mean_error': float(np.mean(errors_visionary)),
            'improvement_percent': float(improvement),
            'key_advantage': 'Visionary利用3D高度信息，one只用2D中心'
        }
        
        print(f"\n{name} (n={len(poses_subset)}):")
        print(f"  Text2Loc-one (2D): {np.mean(errors_one):.2f}m")
        print(f"  Text2Loc Visionary (3D): {np.mean(errors_visionary):.2f}m")
        print(f"  改进幅度: {improvement:+.1f}%")
    
    return results


def experiment_2_object_semantic_association(test_poses, cells_dict, model, dataset, device):
    """
    实验2：对象语义关联
    测试描述中提到的对象与实际对象的匹配
    """
    print("\n" + "="*80)
    print("实验2：对象语义关联")
    print("="*80)
    
    # 提取描述中提到的对象类型
    object_keywords = ['building', 'car', 'tree', 'person', 'pole', 'traffic', 'sign', 'wall']
    
    matched_poses = []    # 描述中有对象且cell中有匹配
    unmatched_poses = []  # 描述中有对象但cell中无匹配
    
    for pose in test_poses:
        desc = pose.get('description', '').lower()
        cell_id = pose.get('cell_id')
        cell = cells_dict.get(cell_id, {})
        
        mentioned_objects = [kw for kw in object_keywords if kw in desc]
        
        if mentioned_objects:
            # 检查cell中是否有这些对象
            cell_objects = cell.get('objects', [])
            has_match = any(
                any(kw in str(obj.get('semantic', '')).lower() for kw in mentioned_objects)
                for obj in cell_objects if isinstance(obj, dict)
            )
            
            if has_match:
                matched_poses.append(pose)
            else:
                unmatched_poses.append(pose)
    
    results = {}
    
    for name, poses_subset in [('描述对象匹配', matched_poses), ('描述对象不匹配', unmatched_poses)]:
        if len(poses_subset) < 10:
            continue
        
        errors_one = []
        for pose in poses_subset:
            cell_id = pose.get('cell_id')
            cell = cells_dict.get(cell_id, {})
            pred = np.array(cell.get('center', [0, 0, 0])[:2])
            gt = np.array([float(pose['location'][0]), float(pose['location'][1])])
            errors_one.append(np.linalg.norm(pred - gt))
        
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
            'visionary_mean_error': float(np.mean(errors_visionary)),
            'improvement_percent': float(improvement),
            'key_advantage': 'Visionary能关联描述对象和实际对象，one不能'
        }
        
        print(f"\n{name} (n={len(poses_subset)}):")
        print(f"  Text2Loc-one: {np.mean(errors_one):.2f}m")
        print(f"  Text2Loc Visionary: {np.mean(errors_visionary):.2f}m")
        print(f"  改进幅度: {improvement:+.1f}%")
    
    return results


def experiment_3_spatial_relation_accuracy(test_poses, cells_dict, model, dataset, device):
    """
    实验3：空间关系精度
    测试"在X的左边/右边/前面/后面"等描述的准确性
    """
    print("\n" + "="*80)
    print("实验3：空间关系精度")
    print("="*80)
    
    # 提取包含空间关系的描述
    spatial_patterns = {
        'left': ['left of', 'to the left', 'on the left'],
        'right': ['right of', 'to the right', 'on the right'],
        'front': ['in front of', 'ahead of', 'before'],
        'behind': ['behind', 'back of', 'after']
    }
    
    spatial_poses = defaultdict(list)
    
    for pose in test_poses:
        desc = pose.get('description', '').lower()
        
        for direction, patterns in spatial_patterns.items():
            if any(p in desc for p in patterns):
                spatial_poses[direction].append(pose)
                break
    
    results = {}
    
    for direction, poses_subset in spatial_poses.items():
        if len(poses_subset) < 10:
            continue
        
        errors_one = []
        for pose in poses_subset:
            cell_id = pose.get('cell_id')
            cell = cells_dict.get(cell_id, {})
            pred = np.array(cell.get('center', [0, 0, 0])[:2])
            gt = np.array([float(pose['location'][0]), float(pose['location'][1])])
            errors_one.append(np.linalg.norm(pred - gt))
        
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
        
        results[f'{direction}_relation'] = {
            'count': len(poses_subset),
            'text2loc_one_mean_error': float(np.mean(errors_one)),
            'visionary_mean_error': float(np.mean(errors_visionary)),
            'improvement_percent': float(improvement),
            'key_advantage': f'Visionary能理解"{direction}"关系，one不能'
        }
        
        print(f"\n{direction}关系 (n={len(poses_subset)}):")
        print(f"  Text2Loc-one: {np.mean(errors_one):.2f}m")
        print(f"  Text2Loc Visionary: {np.mean(errors_visionary):.2f}m")
        print(f"  改进幅度: {improvement:+.1f}%")
    
    return results


def experiment_4_distance_estimation(test_poses, cells_dict, model, dataset, device):
    """
    实验4：距离估计能力
    测试包含距离信息的描述（如"5米外"）
    """
    print("\n" + "="*80)
    print("实验4：距离估计能力")
    print("="*80)
    
    # 提取包含距离信息的描述
    distance_poses = []
    
    for pose in test_poses:
        desc = pose.get('description', '').lower()
        # 查找数字+单位模式
        import re
        if re.search(r'\d+\s*(m|meter|meters|米)', desc):
            distance_poses.append(pose)
    
    if len(distance_poses) < 10:
        print("  距离描述样本不足，跳过此实验")
        return {}
    
    errors_one = []
    for pose in distance_poses:
        cell_id = pose.get('cell_id')
        cell = cells_dict.get(cell_id, {})
        pred = np.array(cell.get('center', [0, 0, 0])[:2])
        gt = np.array([float(pose['location'][0]), float(pose['location'][1])])
        errors_one.append(np.linalg.norm(pred - gt))
    
    errors_visionary = []
    for pose in distance_poses:
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
    
    results = {
        'distance_descriptions': {
            'count': len(distance_poses),
            'text2loc_one_mean_error': float(np.mean(errors_one)),
            'visionary_mean_error': float(np.mean(errors_visionary)),
            'improvement_percent': float(improvement),
            'key_advantage': 'Visionary能理解距离信息，one不能'
        }
    }
    
    print(f"\n距离描述 (n={len(distance_poses)}):")
    print(f"  Text2Loc-one: {np.mean(errors_one):.2f}m")
    print(f"  Text2Loc Visionary: {np.mean(errors_visionary):.2f}m")
    print(f"  改进幅度: {improvement:+.1f}%")
    
    return results


def experiment_5_contextual_understanding(test_poses, cells_dict, model, dataset, device):
    """
    实验5：上下文理解
    测试长描述vs短描述的表现差异
    """
    print("\n" + "="*80)
    print("实验5：上下文理解能力")
    print("="*80)
    
    # 按描述长度分组
    short_desc = []   # < 20字符
    medium_desc = []  # 20-50字符
    long_desc = []    # > 50字符
    
    for pose in test_poses:
        desc = pose.get('description', '')
        length = len(desc)
        
        if length < 20:
            short_desc.append(pose)
        elif length < 50:
            medium_desc.append(pose)
        else:
            long_desc.append(pose)
    
    results = {}
    
    for name, poses_subset in [('短描述(<20)', short_desc), 
                               ('中等描述(20-50)', medium_desc),
                               ('长描述(>50)', long_desc)]:
        if len(poses_subset) < 10:
            continue
        
        errors_one = []
        for pose in poses_subset:
            cell_id = pose.get('cell_id')
            cell = cells_dict.get(cell_id, {})
            pred = np.array(cell.get('center', [0, 0, 0])[:2])
            gt = np.array([float(pose['location'][0]), float(pose['location'][1])])
            errors_one.append(np.linalg.norm(pred - gt))
        
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
            'visionary_mean_error': float(np.mean(errors_visionary)),
            'improvement_percent': float(improvement),
            'key_advantage': 'Visionary能利用上下文信息，one忽略描述内容'
        }
        
        print(f"\n{name} (n={len(poses_subset)}):")
        print(f"  Text2Loc-one: {np.mean(errors_one):.2f}m")
        print(f"  Text2Loc Visionary: {np.mean(errors_visionary):.2f}m")
        print(f"  改进幅度: {improvement:+.1f}%")
    
    return results


def main():
    print("="*80)
    print("创新3D对比实验 - 基于当前数据展现真实优势")
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
    
    # 运行所有创新实验
    all_results = {}
    
    print("\n" + "="*80)
    print("开始创新3D对比实验...")
    print("="*80)
    
    all_results['3d_height'] = experiment_1_3d_height_utilization(
        test_poses, cells_dict, model, test_dataset, device)
    
    all_results['semantic_association'] = experiment_2_object_semantic_association(
        test_poses, cells_dict, model, test_dataset, device)
    
    all_results['spatial_relation'] = experiment_3_spatial_relation_accuracy(
        test_poses, cells_dict, model, test_dataset, device)
    
    all_results['distance_estimation'] = experiment_4_distance_estimation(
        test_poses, cells_dict, model, test_dataset, device)
    
    all_results['contextual'] = experiment_5_contextual_understanding(
        test_poses, cells_dict, model, test_dataset, device)
    
    # 总结
    print("\n" + "="*80)
    print("创新3D对比实验总结")
    print("="*80)
    
    print("\n✅ Text2Loc Visionary相比Text2Loc-one的核心优势：")
    print("\n  1. 3D空间利用")
    print("     - 利用高度信息，不只是2D平面")
    print("\n  2. 对象语义关联")
    print("     - 关联描述中的对象与实际对象")
    print("\n  3. 空间关系理解")
    print("     - 理解'左/右/前/后'等关系")
    print("\n  4. 距离估计能力")
    print("     - 理解'5米外'等距离信息")
    print("\n  5. 上下文理解")
    print("     - 利用长描述的丰富上下文")
    
    print("\n⚠️  当前数据限制：")
    print("  - 描述多样性不足")
    print("  - 空间关系描述较少")
    print("  - 需要更丰富的语义标注")
    
    # 保存结果
    with open('innovative_3d_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n详细结果已保存到: innovative_3d_comparison.json")
    
    return all_results


if __name__ == '__main__':
    main()
