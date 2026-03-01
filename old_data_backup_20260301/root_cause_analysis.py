#!/usr/bin/env python3
"""
根本原因分析 - 为什么Visionary没有比one更好
彻底检查实验设计、数据生成、模型预测机制
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


def analyze_data_distribution():
    """
    分析数据分布，找出问题根源
    """
    print("="*80)
    print("根本原因分析 - 数据分布检查")
    print("="*80)
    
    # 加载数据
    data_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_semantics_full")
    
    with open(data_path / "cells" / "cells.pkl", 'rb') as f:
        cells = pickle.load(f)
    with open(data_path / "poses" / "poses.pkl", 'rb') as f:
        poses = pickle.load(f)
    
    cells_dict = {cell['id']: cell for cell in cells if isinstance(cell, dict)}
    
    print(f"\n1. 数据规模")
    print(f"   Cells: {len(cells)}")
    print(f"   Poses: {len(poses)}")
    
    # 2. 分析cell大小分布
    print(f"\n2. Cell大小分析")
    cell_sizes = []
    for cell in cells_dict.values():
        objects = cell.get('objects', [])
        if len(objects) > 1:
            centers = []
            for obj in objects:
                if isinstance(obj, dict):
                    center = obj.get('center', [0, 0, 0])
                    if len(center) >= 2:
                        centers.append([float(center[0]), float(center[1])])
            if len(centers) > 1:
                centers = np.array(centers)
                size = np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)) * 2
                cell_sizes.append(size)
    
    if cell_sizes:
        cell_sizes = np.array(cell_sizes)
        print(f"   Cell大小范围: {cell_sizes.min():.2f}m - {cell_sizes.max():.2f}m")
        print(f"   Cell大小均值: {cell_sizes.mean():.2f}m")
        print(f"   Cell大小中位数: {np.median(cell_sizes):.2f}m")
    
    # 3. 分析pose与cell中心的关系
    print(f"\n3. Pose与cell中心关系")
    distances_to_center = []
    for pose in poses:
        if isinstance(pose, dict) and 'cell_id' in pose:
            cell_id = pose['cell_id']
            cell = cells_dict.get(cell_id)
            if cell:
                cell_center = cell.get('center', [0, 0, 0])
                cell_center_xy = np.array([float(cell_center[0]), float(cell_center[1])])
                
                location = pose.get('location', [0, 0])
                if isinstance(location, (list, tuple)) and len(location) >= 2:
                    pose_xy = np.array([float(location[0]), float(location[1])])
                    distance = np.linalg.norm(pose_xy - cell_center_xy)
                    distances_to_center.append(distance)
    
    if distances_to_center:
        distances_to_center = np.array(distances_to_center)
        print(f"   Pose到cell中心距离范围: {distances_to_center.min():.2f}m - {distances_to_center.max():.2f}m")
        print(f"   均值: {distances_to_center.mean():.2f}m")
        print(f"   中位数: {np.median(distances_to_center):.2f}m")
        print(f"   标准差: {distances_to_center.std():.2f}m")
        
        # 统计分布
        print(f"\n   距离分布:")
        print(f"   < 1m: {np.mean(distances_to_center < 1.0)*100:.1f}%")
        print(f"   1-3m: {np.mean((distances_to_center >= 1.0) & (distances_to_center < 3.0))*100:.1f}%")
        print(f"   3-5m: {np.mean((distances_to_center >= 3.0) & (distances_to_center < 5.0))*100:.1f}%")
        print(f"   > 5m: {np.mean(distances_to_center >= 5.0)*100:.1f}%")
    
    # 4. 分析描述特征
    print(f"\n4. 描述特征分析")
    desc_lengths = []
    for pose in poses:
        if isinstance(pose, dict):
            desc = pose.get('description', '')
            desc_lengths.append(len(desc))
    
    if desc_lengths:
        desc_lengths = np.array(desc_lengths)
        print(f"   描述长度范围: {desc_lengths.min()} - {desc_lengths.max()} 字符")
        print(f"   均值: {desc_lengths.mean():.1f} 字符")
        print(f"   中位数: {np.median(desc_lengths):.1f} 字符")
        
        # 检查描述多样性
        unique_descs = set()
        for pose in poses:
            if isinstance(pose, dict):
                unique_descs.add(pose.get('description', ''))
        print(f"   唯一描述数: {len(unique_descs)} / {len(poses)} ({len(unique_descs)/len(poses)*100:.1f}%)")
    
    return {
        'num_cells': len(cells),
        'num_poses': len(poses),
        'cell_size_mean': float(np.mean(cell_sizes)) if len(cell_sizes) > 0 else 0,
        'distance_to_center_mean': float(np.mean(distances_to_center)) if len(distances_to_center) > 0 else 0,
        'desc_length_mean': float(np.mean(desc_lengths)) if len(desc_lengths) > 0 else 0
    }


def analyze_model_predictions():
    """
    深度分析模型预测机制
    """
    print("\n" + "="*80)
    print("根本原因分析 - 模型预测机制检查")
    print("="*80)
    
    # 加载数据
    data_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_semantics_full")
    
    with open(data_path / "cells" / "cells.pkl", 'rb') as f:
        cells = pickle.load(f)
    with open(data_path / "poses" / "poses.pkl", 'rb') as f:
        poses = pickle.load(f)
    
    cells_dict = {cell['id']: cell for cell in cells if isinstance(cell, dict)}
    
    # 加载模型
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = Text2LocNeuralNetwork(embed_dim=256, hidden_dim=512).to(device)
    
    model_path = Path("checkpoints/semantics_full_best_model.pth")
    if not model_path.exists():
        print("模型不存在")
        return None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n1. 模型结构分析")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")
    
    # 2. 分析预测行为
    print("\n2. 预测行为分析")
    
    # 准备数据
    all_centers = []
    for cell in cells_dict.values():
        center = cell.get('center', [0, 0, 0])
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            all_centers.append([float(center[0]), float(center[1])])
    
    all_centers = np.array(all_centers)
    global_mean = np.mean(all_centers, axis=0)
    global_std = np.std(all_centers, axis=0) + 1e-8
    
    # 随机选择100个样本分析
    np.random.seed(42)
    sample_indices = np.random.choice(len(poses), min(100, len(poses)), replace=False)
    
    pred_offsets = []
    gt_offsets = []
    
    for idx in sample_indices:
        pose = poses[idx]
        if not isinstance(pose, dict) or 'cell_id' not in pose:
            continue
        
        cell_id = pose['cell_id']
        cell = cells_dict.get(cell_id)
        if not cell:
            continue
        
        # 准备输入
        description = pose.get('description', f"Location near {cell_id}")
        cell_center = cell.get('center', [0, 0, 0])
        cell_center_xy = np.array([float(cell_center[0]), float(cell_center[1])])
        
        location = pose.get('location', [0, 0])
        if isinstance(location, (list, tuple)) and len(location) >= 2:
            global_xy = np.array([float(location[0]), float(location[1])])
        else:
            global_xy = cell_center_xy.copy()
        
        # 计算gt offset
        gt_offset_xy = global_xy - cell_center_xy
        gt_offset_normalized = (gt_offset_xy - global_mean) / global_std
        
        # 准备object features
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
        
        object_features_tensor = torch.tensor([object_features], dtype=torch.float32).to(device)
        
        # 预测
        with torch.no_grad():
            pred_offset = model([description], object_features_tensor).cpu().numpy()[0]
        
        pred_offsets.append(pred_offset)
        gt_offsets.append(gt_offset_normalized)
    
    pred_offsets = np.array(pred_offsets)
    gt_offsets = np.array(gt_offsets)
    
    print(f"   分析样本数: {len(pred_offsets)}")
    print(f"\n   预测偏移量统计:")
    print(f"   X方向 - 均值: {pred_offsets[:, 0].mean():.4f}, 标准差: {pred_offsets[:, 0].std():.4f}")
    print(f"   Y方向 - 均值: {pred_offsets[:, 1].mean():.4f}, 标准差: {pred_offsets[:, 1].std():.4f}")
    
    print(f"\n   真实偏移量统计:")
    print(f"   X方向 - 均值: {gt_offsets[:, 0].mean():.4f}, 标准差: {gt_offsets[:, 0].std():.4f}")
    print(f"   Y方向 - 均值: {gt_offsets[:, 1].mean():.4f}, 标准差: {gt_offsets[:, 1].std():.4f}")
    
    # 检查预测是否接近0
    pred_magnitudes = np.linalg.norm(pred_offsets, axis=1)
    print(f"\n   预测偏移量大小:")
    print(f"   均值: {pred_magnitudes.mean():.4f}")
    print(f"   接近0的比例 (<0.1): {np.mean(pred_magnitudes < 0.1)*100:.1f}%")
    print(f"   接近0的比例 (<0.5): {np.mean(pred_magnitudes < 0.5)*100:.1f}%")
    
    # 3. 分析预测与真实值的相关性
    correlation_x = np.corrcoef(pred_offsets[:, 0], gt_offsets[:, 0])[0, 1]
    correlation_y = np.corrcoef(pred_offsets[:, 1], gt_offsets[:, 1])[0, 1]
    print(f"\n3. 预测与真实值相关性")
    print(f"   X方向相关性: {correlation_x:.4f}")
    print(f"   Y方向相关性: {correlation_y:.4f}")
    
    return {
        'pred_offset_mean_x': float(pred_offsets[:, 0].mean()),
        'pred_offset_std_x': float(pred_offsets[:, 0].std()),
        'pred_offset_mean_y': float(pred_offsets[:, 1].mean()),
        'pred_offset_std_y': float(pred_offsets[:, 1].std()),
        'correlation_x': float(correlation_x),
        'correlation_y': float(correlation_y),
        'near_zero_ratio': float(np.mean(pred_magnitudes < 0.1))
    }


def identify_root_causes(data_analysis, model_analysis):
    """
    识别根本原因
    """
    print("\n" + "="*80)
    print("根本原因识别")
    print("="*80)
    
    root_causes = []
    
    # 原因1: 数据规模
    if data_analysis['num_cells'] < 5000:
        root_causes.append({
            'issue': '数据规模不足',
            'severity': '高',
            'details': f"只有{data_analysis['num_cells']}个cell，不足以训练复杂神经网络",
            'solution': '需要更多数据或数据增强'
        })
    
    # 原因2: Cell大小
    if data_analysis['cell_size_mean'] < 15:
        root_causes.append({
            'issue': 'Cell划分过细',
            'severity': '高',
            'details': f"平均cell大小只有{data_analysis['cell_size_mean']:.1f}m，中心点预测已足够好",
            'solution': '增大cell size或测试跨cell定位'
        })
    
    # 原因3: Pose分布
    if data_analysis['distance_to_center_mean'] < 3.0:
        root_causes.append({
            'issue': 'Pose过于集中在cell中心',
            'severity': '中',
            'details': f"平均距离中心只有{data_analysis['distance_to_center_mean']:.1f}m",
            'solution': '生成更多边缘位置的pose'
        })
    
    # 原因4: 描述多样性
    if data_analysis['desc_length_mean'] < 30:
        root_causes.append({
            'issue': '描述过于简单',
            'severity': '高',
            'details': f"平均描述长度只有{data_analysis['desc_length_mean']:.1f}字符",
            'solution': '生成更丰富的描述'
        })
    
    # 原因5: 模型预测
    if model_analysis and model_analysis['near_zero_ratio'] > 0.5:
        root_causes.append({
            'issue': '模型倾向于预测接近0的偏移',
            'severity': '中',
            'details': f"{model_analysis['near_zero_ratio']*100:.1f}%的预测接近0",
            'solution': '检查损失函数或增加训练难度'
        })
    
    # 原因6: 相关性低
    if model_analysis and abs(model_analysis['correlation_x']) < 0.3:
        root_causes.append({
            'issue': '模型预测与真实值相关性低',
            'severity': '高',
            'details': f"X方向相关性只有{model_analysis['correlation_x']:.3f}",
            'solution': '重新设计模型架构或训练策略'
        })
    
    print("\n识别的根本原因：")
    for i, cause in enumerate(root_causes, 1):
        print(f"\n{i}. {cause['issue']} [严重程度: {cause['severity']}]")
        print(f"   问题: {cause['details']}")
        print(f"   解决方案: {cause['solution']}")
    
    return root_causes


def main():
    print("="*80)
    print("根本原因分析 - 彻底解决实验问题")
    print("="*80)
    
    # 1. 分析数据分布
    data_analysis = analyze_data_distribution()
    
    # 2. 分析模型预测
    model_analysis = analyze_model_predictions()
    
    # 3. 识别根本原因
    root_causes = identify_root_causes(data_analysis, model_analysis)
    
    # 4. 生成解决方案
    print("\n" + "="*80)
    print("推荐解决方案（按优先级排序）")
    print("="*80)
    
    solutions = [
        {
            'priority': 1,
            'action': '重新生成训练数据',
            'details': '生成更大cell size（20m+）和更丰富描述的数据',
            'effort': '中等',
            'impact': '高'
        },
        {
            'priority': 2,
            'action': '修改损失函数',
            'details': '增加对边缘位置的惩罚，避免预测总是接近0',
            'effort': '低',
            'impact': '中'
        },
        {
            'priority': 3,
            'action': '设计跨cell测试',
            'details': '测试描述与cell不匹配时的定位能力',
            'effort': '低',
            'impact': '高'
        },
        {
            'priority': 4,
            'action': '增加模型复杂度',
            'details': '使用更大的模型或更复杂的架构',
            'effort': '中等',
            'impact': '中'
        },
        {
            'priority': 5,
            'action': '数据增强',
            'details': '对现有数据进行旋转、缩放等增强',
            'effort': '低',
            'impact': '中'
        }
    ]
    
    for sol in solutions:
        print(f"\n{sol['priority']}. {sol['action']}")
        print(f"   详情: {sol['details']}")
        print(f"   工作量: {sol['effort']}, 预期效果: {sol['impact']}")
    
    # 保存分析结果
    analysis_result = {
        'data_analysis': data_analysis,
        'model_analysis': model_analysis,
        'root_causes': root_causes,
        'solutions': solutions
    }
    
    with open('root_cause_analysis.json', 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"\n分析结果已保存到: root_cause_analysis.json")
    
    return analysis_result


if __name__ == '__main__':
    main()
