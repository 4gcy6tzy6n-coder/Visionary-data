"""
使用完整3D语义数据训练神经网络
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Text2LocDataset(Dataset):
    """Text2Loc数据集"""
    
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
        
        logger.info(f"📊 数据集: {len(self.cells_dict)} cells, {len(self.poses)} poses")
    
    def _normalize(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self.global_mean) / self.global_std
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        pose = self.poses[idx]
        cell = self.cells_dict[pose['cell_id']]
        
        description = pose.get('description', f"Location near {pose['cell_id']}")
        
        # Cell中心
        cell_center = cell.get('center', [0, 0, 0])
        cell_center_xy = np.array([float(cell_center[0]), float(cell_center[1])])
        
        # 全局位置
        location = pose.get('location', [0, 0])
        if isinstance(location, (list, tuple)) and len(location) >= 2:
            global_xy = np.array([float(location[0]), float(location[1])])
        else:
            global_xy = cell_center_xy.copy()
        
        # 计算偏移并归一化
        offset_xy = global_xy - cell_center_xy
        normalized_offset = self._normalize(offset_xy)
        
        # Object features
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
        
        return {
            'description': description,
            'object_features': torch.tensor(object_features, dtype=torch.float32),
            'gt_offset': torch.tensor(normalized_offset, dtype=torch.float32),
            'cell_center': torch.tensor(cell_center_xy, dtype=torch.float32),
            'cell_id': pose['cell_id']
        }


def collate_fn(batch):
    """自定义collate函数"""
    descriptions = [item['description'] for item in batch]
    
    max_objects = max(item['object_features'].shape[0] for item in batch)
    object_features = []
    
    for item in batch:
        feat = item['object_features']
        if feat.shape[0] < max_objects:
            padding = torch.zeros(max_objects - feat.shape[0], feat.shape[1])
            feat = torch.cat([feat, padding], dim=0)
        object_features.append(feat)
    
    return {
        'descriptions': descriptions,
        'object_features': torch.stack(object_features),
        'gt_offset': torch.stack([item['gt_offset'] for item in batch]),
        'cell_centers': torch.stack([item['cell_center'] for item in batch]),
        'cell_ids': [item['cell_id'] for item in batch]
    }


class Text2LocNeuralNetwork(nn.Module):
    """Text2Loc神经网络"""
    
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


def train_model(data_path: str, num_epochs: int = 100, batch_size: int = 32, lr: float = 0.001):
    """训练模型"""
    
    data_path = Path(data_path)
    
    logger.info("\n" + "="*60)
    logger.info("加载完整3D语义数据集")
    logger.info("="*60)
    
    with open(data_path / "cells" / "cells.pkl", 'rb') as f:
        cells = pickle.load(f)
    
    with open(data_path / "poses" / "poses.pkl", 'rb') as f:
        poses = pickle.load(f)
    
    logger.info(f"✅ 加载 {len(cells)} cells, {len(poses)} poses")
    
    # 划分数据集
    np.random.seed(42)
    indices = np.random.permutation(len(poses))
    split = int(len(poses) * 0.8)
    
    train_poses = [poses[i] for i in indices[:split]]
    val_poses = [poses[i] for i in indices[split:]]
    
    train_dataset = Text2LocDataset(cells, train_poses)
    val_dataset = Text2LocDataset(cells, val_poses)
    
    global_mean = train_dataset.global_mean
    global_std = train_dataset.global_std
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"\n🖥️ 使用设备: {device}")
    
    model = Text2LocNeuralNetwork(embed_dim=256, hidden_dim=512).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    logger.info("\n" + "="*60)
    logger.info("开始训练")
    logger.info("="*60)
    
    best_val_error = float('inf')
    
    save_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/checkpoints")
    save_path.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            descriptions = batch['descriptions']
            object_features = batch['object_features'].to(device)
            gt_offset = batch['gt_offset'].to(device)
            
            pred_offset = model(descriptions, object_features)
            
            loss = criterion(pred_offset, gt_offset)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_train_loss = np.mean(epoch_losses)
        
        # 验证
        model.eval()
        val_distances = []
        
        with torch.no_grad():
            for batch in val_loader:
                descriptions = batch['descriptions']
                object_features = batch['object_features'].to(device)
                gt_offset = batch['gt_offset'].to(device)
                cell_centers = batch['cell_centers'].to(device)
                
                pred_offset = model(descriptions, object_features)
                
                # 反归一化
                pred_offset_denorm = pred_offset * torch.tensor(global_std, dtype=torch.float32).to(device) + \
                                     torch.tensor(global_mean, dtype=torch.float32).to(device)
                gt_offset_denorm = gt_offset * torch.tensor(global_std, dtype=torch.float32).to(device) + \
                                   torch.tensor(global_mean, dtype=torch.float32).to(device)
                
                pred_global = cell_centers + pred_offset_denorm
                gt_global = cell_centers + gt_offset_denorm
                
                distances = torch.sqrt(torch.sum((pred_global - gt_global) ** 2, dim=1))
                val_distances.extend(distances.cpu().numpy())
        
        val_distances = np.array(val_distances)
        avg_val_error = np.mean(val_distances)
        
        acc_1m = np.mean(val_distances <= 1.0) * 100
        acc_5m = np.mean(val_distances <= 5.0) * 100
        acc_10m = np.mean(val_distances <= 10.0) * 100
        
        logger.info(f"Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.4f}, Val Error={avg_val_error:.2f}m, "
                   f"Acc@1m={acc_1m:.1f}%, Acc@5m={acc_5m:.1f}%, Acc@10m={acc_10m:.1f}%")
        
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': avg_val_error,
                'acc_1m': acc_1m,
                'acc_5m': acc_5m,
                'acc_10m': acc_10m,
                'global_mean': global_mean.tolist(),
                'global_std': global_std.tolist()
            }, save_path / 'semantics_full_best_model.pth')
            logger.info(f"  💾 保存最佳模型 (Val Error: {avg_val_error:.2f}m)")
        
        scheduler.step(avg_val_error)
    
    logger.info("\n" + "="*60)
    logger.info("训练完成")
    logger.info(f"最佳验证误差: {best_val_error:.2f}m")
    logger.info("="*60)
    
    return model, best_val_error


if __name__ == "__main__":
    model, best_error = train_model(
        data_path="/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_semantics_full",
        num_epochs=100,
        batch_size=32,
        lr=0.001
    )
