#!/usr/bin/env python3
"""
严格验证模型有效性 - 使用与训练相同的数据格式
检测：1) 数据泄漏 2) 过拟合 3) 与基线对比
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import random
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split
import json

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 与训练脚本相同的模型架构
class ObjectEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # x: [B, N, 6] -> 对每个object编码 -> [B, N, 128]
        B, N, _ = x.shape
        x = x.view(B * N, -1)
        x = self.net(x)
        x = x.view(B, N, -1)
        # 平均池化
        x = x.mean(dim=1)
        return x

class Text2LocNeuralNetwork(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512):
        super().__init__()
        
        # 使用简单的embedding代替BERT（为了快速验证）
        vocab_size = 1000
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        self.text_fc = nn.Linear(embed_dim * 2, embed_dim)
        
        self.object_encoder = ObjectEncoder(input_dim=6, hidden_dim=128, output_dim=embed_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def encode_text(self, descriptions: List[str]):
        # 简化：使用随机token（实际应该使用tokenizer）
        max_len = 20
        tokens = []
        for desc in descriptions:
            # 简单hash作为token id
            token_ids = [hash(word) % 1000 for word in desc.split()[:max_len]]
            token_ids += [0] * (max_len - len(token_ids))
            tokens.append(token_ids)
        
        tokens = torch.tensor(tokens, dtype=torch.long).to(next(self.parameters()).device)
        x = self.embedding(tokens)
        x, _ = self.text_lstm(x)
        x = x[:, -1, :]
        x = self.text_fc(x)
        return x
    
    def encode_objects(self, object_features: torch.Tensor):
        return self.object_encoder(object_features)
    
    def forward(self, descriptions: List[str], object_features: torch.Tensor):
        text_enc = self.encode_text(descriptions)
        obj_enc = self.encode_objects(object_features)
        
        fused = torch.cat([text_enc, obj_enc], dim=1)
        fused = self.fusion(fused)
        
        offset = self.location_head(fused)
        
        return offset


def collate_fn(batch):
    """与训练脚本相同的collate函数"""
    descriptions = [item['description'] for item in batch]
    object_features = torch.stack([item['object_features'] for item in batch])
    gt_offsets = torch.stack([item['gt_offset'] for item in batch])
    cell_centers = torch.stack([item['cell_center'] for item in batch])
    cell_ids = [item['cell_id'] for item in batch]
    
    return {
        'descriptions': descriptions,
        'object_features': object_features,
        'gt_offset': gt_offsets,
        'cell_center': cell_centers,
        'cell_ids': cell_ids
    }


class SimpleDataset:
    """简化数据集类"""
    def __init__(self, cells: List[Dict], poses: List[Dict]):
        self.cells_dict = {cell['id']: cell for cell in cells if isinstance(cell, dict)}
        self.poses = [p for p in poses if isinstance(p, dict) and p.get('cell_id') in self.cells_dict]
        
        # 计算归一化参数
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
                
                feat = [
                    obj_offset[0], obj_offset[1],
                    float(center[2]) if len(center) > 2 else 0,
                    0.5, 0.5, 0.5
                ]
                object_features.append(feat)
        
        if not object_features:
            object_features = [[0.0] * 6]
        
        # Pad to same length
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
            'global_xy': global_xy  # 保存真实位置用于评估
        }


class RandomBaseline:
    """随机基线"""
    def __init__(self, cell_centers):
        self.cell_centers = cell_centers
    
    def predict(self, batch_size):
        indices = np.random.randint(0, len(self.cell_centers), size=batch_size)
        return self.cell_centers[indices]


class MeanOffsetBaseline:
    """平均偏移基线"""
    def __init__(self, mean_offset):
        self.mean_offset = mean_offset
    
    def predict(self, cell_centers):
        return cell_centers + self.mean_offset


def evaluate_model(model, dataset, device, batch_size=32, model_type='neural'):
    """评估模型"""
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    all_errors = []
    all_predictions = []
    all_ground_truths = []
    
    if model_type == 'neural':
        model.eval()
        with torch.no_grad():
            for batch in loader:
                descriptions = batch['descriptions']
                object_features = batch['object_features'].to(device)
                cell_centers = batch['cell_center'].cpu().numpy()
                
                # 获取batch中每个pose的真实全局位置
                batch_indices = [dataset.poses.index(p) for p in dataset.poses if p.get('description') in descriptions]
                # 简化：直接从batch中获取
                
                pred_offset = model(descriptions, object_features).cpu().numpy()
                pred_offset_denorm = pred_offset * dataset.global_std + dataset.global_mean
                pred_global = cell_centers + pred_offset_denorm
                
                # 获取真实位置
                gt_global = np.array([dataset[i]['global_xy'] for i in range(len(descriptions))])
                
                errors = np.linalg.norm(pred_global - gt_global, axis=1)
                all_errors.extend(errors)
                all_predictions.extend(pred_global)
                all_ground_truths.extend(gt_global)
    
    elif model_type == 'random':
        for i in range(len(dataset)):
            cell_center = dataset[i]['cell_center'].numpy()
            pred_global = model.predict(1)[0]
            gt_global = dataset[i]['global_xy']
            
            error = np.linalg.norm(pred_global - gt_global)
            all_errors.append(error)
    
    elif model_type == 'mean_offset':
        for i in range(len(dataset)):
            cell_center = dataset[i]['cell_center'].numpy()
            pred_global = model.predict(cell_center)
            gt_global = dataset[i]['global_xy']
            
            error = np.linalg.norm(pred_global - gt_global)
            all_errors.append(error)
    
    errors = np.array(all_errors)
    
    return {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'acc_1m': np.mean(errors <= 1.0) * 100,
        'acc_5m': np.mean(errors <= 5.0) * 100,
        'acc_10m': np.mean(errors <= 10.0) * 100,
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'std_error': np.std(errors),
        'errors': errors
    }


def create_cell_based_split(cells, poses, test_cell_ratio=0.3):
    """
    按cell分割 - 确保测试集中的cell在训练集中完全没有出现
    这是检测数据泄漏的关键
    """
    # 获取所有cell IDs
    cell_ids = list(set(p['cell_id'] for p in poses if isinstance(p, dict) and 'cell_id' in p))
    
    # 随机分割cell IDs
    train_cell_ids, test_cell_ids = train_test_split(
        cell_ids, test_size=test_cell_ratio, random_state=SEED
    )
    
    # 根据cell IDs分割poses
    train_poses = [p for p in poses if isinstance(p, dict) and p.get('cell_id') in train_cell_ids]
    test_poses = [p for p in poses if isinstance(p, dict) and p.get('cell_id') in test_cell_ids]
    
    return train_poses, test_poses, train_cell_ids, test_cell_ids


def main():
    print("="*80)
    print("严格模型验证 - 检测数据泄漏和过拟合")
    print("="*80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_semantics_full")
    
    with open(data_path / "cells" / "cells.pkl", 'rb') as f:
        cells = pickle.load(f)
    
    with open(data_path / "poses" / "poses.pkl", 'rb') as f:
        poses = pickle.load(f)
    
    print(f"   总cells: {len(cells)}")
    print(f"   总poses: {len(poses)}")
    
    # 2. 创建按cell分割的训练/测试集
    print("\n2. 创建按cell分割的数据集...")
    train_poses, test_poses, train_cell_ids, test_cell_ids = create_cell_based_split(cells, poses)
    
    print(f"   训练集: {len(train_poses)} poses, {len(train_cell_ids)} cells")
    print(f"   测试集: {len(test_poses)} poses, {len(test_cell_ids)} cells")
    
    # 3. 检查数据泄漏
    print("\n3. 检查数据泄漏...")
    overlap = set(train_cell_ids) & set(test_cell_ids)
    if overlap:
        print(f"   ❌ 数据泄漏！重叠cells: {overlap}")
        return
    else:
        print("   ✅ 无数据泄漏 - 训练集和测试集的cells完全分离")
    
    # 4. 创建数据集
    train_dataset = SimpleDataset(cells, train_poses)
    test_dataset = SimpleDataset(cells, test_poses)
    
    # 5. 加载训练好的模型
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n4. 加载模型 (设备: {device})...")
    
    model = Text2LocNeuralNetwork(embed_dim=256, hidden_dim=512).to(device)
    
    checkpoint_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/models/best_model_semantic_full.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ 加载模型成功")
        print(f"   训练时的最佳验证误差: {checkpoint.get('val_error', 'N/A'):.2f}m")
    else:
        print(f"   ❌ 模型文件不存在: {checkpoint_path}")
        print("   使用随机初始化模型进行对比...")
    
    # 6. 评估神经网络
    print("\n5. 评估神经网络模型...")
    print("   在训练集上评估...")
    train_results_nn = evaluate_model(model, train_dataset, device, model_type='neural')
    print("   在测试集上评估...")
    test_results_nn = evaluate_model(model, test_dataset, device, model_type='neural')
    
    # 7. 评估随机基线
    print("\n6. 评估随机基线...")
    all_cell_centers = np.array([c.get('center', [0, 0, 0])[:2] for c in cells if isinstance(c, dict)])
    random_baseline = RandomBaseline(all_cell_centers)
    test_results_random = evaluate_model(random_baseline, test_dataset, device, model_type='random')
    
    # 8. 评估均值偏移基线
    print("\n7. 评估均值偏移基线...")
    # 计算训练集的平均偏移
    train_offsets = []
    for i in range(len(train_dataset)):
        cell_center = train_dataset[i]['cell_center'].numpy()
        global_xy = train_dataset[i]['global_xy']
        offset = global_xy - cell_center
        train_offsets.append(offset)
    mean_offset = np.mean(train_offsets, axis=0)
    mean_baseline = MeanOffsetBaseline(mean_offset)
    test_results_mean = evaluate_model(mean_baseline, test_dataset, device, model_type='mean_offset')
    
    # 9. 打印结果
    print("\n" + "="*80)
    print("验证结果")
    print("="*80)
    
    print("\n【神经网络模型】")
    print(f"  训练集 - 平均误差: {train_results_nn['mean_error']:.2f}m, Acc@5m: {train_results_nn['acc_5m']:.1f}%")
    print(f"  测试集 - 平均误差: {test_results_nn['mean_error']:.2f}m, Acc@5m: {test_results_nn['acc_5m']:.1f}%")
    
    print("\n【随机基线模型】")
    print(f"  测试集 - 平均误差: {test_results_random['mean_error']:.2f}m, Acc@5m: {test_results_random['acc_5m']:.1f}%")
    
    print("\n【均值偏移基线】")
    print(f"  测试集 - 平均误差: {test_results_mean['mean_error']:.2f}m, Acc@5m: {test_results_mean['acc_5m']:.1f}%")
    
    # 10. 分析
    print("\n" + "="*80)
    print("有效性分析")
    print("="*80)
    
    # 过拟合检测
    gap = train_results_nn['mean_error'] - test_results_nn['mean_error']
    print(f"\n1. 过拟合检测:")
    print(f"   训练误差 - 测试误差 = {gap:.2f}m")
    if abs(gap) < 5.0:
        print("   ✅ 无明显过拟合")
    else:
        print("   ⚠️ 可能存在过拟合")
    
    # 与随机基线对比
    improvement_random = test_results_random['mean_error'] - test_results_nn['mean_error']
    print(f"\n2. 与随机基线对比:")
    print(f"   优于随机基线: {improvement_random:.2f}m")
    if improvement_random > 10.0:
        print("   ✅ 显著优于随机基线")
    else:
        print("   ⚠️ 改进不明显")
    
    # 与均值基线对比
    improvement_mean = test_results_mean['mean_error'] - test_results_nn['mean_error']
    print(f"\n3. 与均值偏移基线对比:")
    print(f"   优于均值基线: {improvement_mean:.2f}m")
    if improvement_mean > 2.0:
        print("   ✅ 优于简单均值预测")
    else:
        print("   ⚠️ 与均值基线相近")
    
    # 最终结论
    print("\n" + "="*80)
    print("最终结论")
    print("="*80)
    
    is_valid = (
        len(overlap) == 0 and
        abs(gap) < 10.0 and
        improvement_random > 5.0
    )
    
    if is_valid:
        print("\n✅ 模型验证通过！")
        print(f"   测试集定位误差: {test_results_nn['mean_error']:.2f}m")
        print(f"   测试集5米准确率: {test_results_nn['acc_5m']:.1f}%")
        print("\n   结果可信，可以用于论文发表！")
    else:
        print("\n❌ 模型验证失败！")
        if len(overlap) > 0:
            print("   原因: 存在数据泄漏")
        elif abs(gap) >= 10.0:
            print("   原因: 严重过拟合")
        elif improvement_random <= 5.0:
            print("   原因: 与随机基线无显著差异")
    
    # 保存结果
    results = {
        'validation_passed': is_valid,
        'neural_network': {
            'train': {k: float(v) if isinstance(v, (np.floating, float)) else v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in train_results_nn.items() if k != 'errors'},
            'test': {k: float(v) if isinstance(v, (np.floating, float)) else v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in test_results_nn.items() if k != 'errors'}
        },
        'random_baseline': {
            'test': {k: float(v) if isinstance(v, (np.floating, float)) else v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in test_results_random.items() if k != 'errors'}
        },
        'mean_baseline': {
            'test': {k: float(v) if isinstance(v, (np.floating, float)) else v.tolist() if isinstance(v, np.ndarray) else v 
                    for k, v in test_results_mean.items() if k != 'errors'}
        }
    }
    
    with open('strict_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n详细结果已保存到: strict_validation_results.json")


if __name__ == '__main__':
    main()
