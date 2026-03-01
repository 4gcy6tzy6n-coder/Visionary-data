"""
使用大规模数据集训练神经网络
基于KITTI360 3D BBox数据 (4955 cells, 15888 poses)
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
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Text2LocNeuralNetwork(nn.Module):
    """Text2Loc神经网络模型"""
    
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Object编码器（PointNet风格）
        self.obj_conv1 = nn.Conv1d(6, 64, 1)
        self.obj_bn1 = nn.BatchNorm1d(64)
        self.obj_conv2 = nn.Conv1d(64, 128, 1)
        self.obj_bn2 = nn.BatchNorm1d(128)
        self.obj_conv3 = nn.Conv1d(128, embed_dim, 1)
        self.obj_bn3 = nn.BatchNorm1d(embed_dim)
        
        # Object间注意力
        self.obj_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # 文本编码器（字符级CNN + LSTM）
        self.char_embed = nn.Embedding(256, 128)  # ASCII字符
        self.text_conv1 = nn.Conv1d(128, 256, 3, padding=1)
        self.text_bn1 = nn.BatchNorm1d(256)
        self.text_conv2 = nn.Conv1d(256, embed_dim, 3, padding=1)
        self.text_bn2 = nn.BatchNorm1d(embed_dim)
        
        # 文本LSTM
        self.text_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.text_fc = nn.Linear(hidden_dim * 2, embed_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 位置预测头
        self.location_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )
    
    def encode_objects(self, object_features: torch.Tensor) -> torch.Tensor:
        """编码Objects"""
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
        """编码文本描述"""
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
        """前向传播"""
        text_enc = self.encode_text(descriptions)
        obj_enc = self.encode_objects(object_features)
        
        fused = torch.cat([text_enc, obj_enc], dim=1)
        fused = self.fusion(fused)
        
        location = self.location_head(fused)
        
        return location


class Text2LocDataset(Dataset):
    """Text2Loc数据集"""
    
    def __init__(self, cells: List[Dict], poses: List[Dict]):
        self.cells_dict = {}
        for cell in cells:
            if isinstance(cell, dict):
                cell_id = cell.get('id', 'unknown')
                self.cells_dict[cell_id] = cell
        
        self.poses = [p for p in poses if isinstance(p, dict) and p.get('cell_id') in self.cells_dict]
        
        logger.info(f"📊 数据集: {len(self.cells_dict)} cells, {len(self.poses)} poses")
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        pose = self.poses[idx]
        cell = self.cells_dict[pose['cell_id']]
        
        description = pose.get('description', '')
        if not description:
            description = f"Location near {pose['cell_id']}"
        
        objects = cell.get('objects', [])
        object_features = []
        
        for obj in objects:
            if isinstance(obj, dict):
                center = obj.get('center', [0, 0, 0])
                
                feat = [
                    float(center[0]) if len(center) > 0 else 0,
                    float(center[1]) if len(center) > 1 else 0,
                    float(center[2]) if len(center) > 2 else 0,
                    0.5, 0.5, 0.5  # 默认颜色
                ]
                object_features.append(feat)
        
        if not object_features:
            object_features = [[0.0] * 6]
        
        location = pose.get('location', [0, 0])
        if isinstance(location, (list, tuple)) and len(location) >= 2:
            gt_xy = [float(location[0]), float(location[1])]
        else:
            gt_xy = [0.0, 0.0]
        
        return {
            'description': description,
            'object_features': torch.tensor(object_features, dtype=torch.float32),
            'gt_xy': torch.tensor(gt_xy, dtype=torch.float32),
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
        'gt_xy': torch.stack([item['gt_xy'] for item in batch]),
        'cell_ids': [item['cell_id'] for item in batch]
    }


def train_model(data_path: str = None, num_epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
    """训练模型"""
    
    if data_path is None:
        data_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_bboxes"
    
    data_path = Path(data_path)
    
    logger.info("\n" + "="*60)
    logger.info("加载大规模数据集")
    logger.info("="*60)
    
    with open(data_path / "cells" / "cells.pkl", 'rb') as f:
        cells = pickle.load(f)
    
    with open(data_path / "poses" / "poses.pkl", 'rb') as f:
        poses = pickle.load(f)
    
    logger.info(f"✅ 加载 {len(cells)} cells, {len(poses)} poses")
    
    np.random.seed(42)
    indices = np.random.permutation(len(poses))
    split = int(len(poses) * 0.8)
    
    train_poses = [poses[i] for i in indices[:split]]
    val_poses = [poses[i] for i in indices[split:]]
    
    train_dataset = Text2LocDataset(cells, train_poses)
    val_dataset = Text2LocDataset(cells, val_poses)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # 检测最佳可用设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"\n🖥️ 使用设备: {device}")
    
    model = Text2LocNeuralNetwork(embed_dim=256, hidden_dim=512).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    logger.info("\n" + "="*60)
    logger.info("开始训练神经网络")
    logger.info("="*60)
    
    best_val_error = float('inf')
    train_losses = []
    val_errors = []
    
    save_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/checkpoints")
    save_path.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            descriptions = batch['descriptions']
            object_features = batch['object_features'].to(device)
            gt_xy = batch['gt_xy'].to(device)
            
            pred_xy = model(descriptions, object_features)
            
            loss = criterion(pred_xy, gt_xy)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_distances = []
        
        with torch.no_grad():
            for batch in val_loader:
                descriptions = batch['descriptions']
                object_features = batch['object_features'].to(device)
                gt_xy = batch['gt_xy'].to(device)
                
                pred_xy = model(descriptions, object_features)
                
                distances = torch.sqrt(torch.sum((pred_xy - gt_xy) ** 2, dim=1))
                val_distances.extend(distances.cpu().numpy())
        
        val_distances = np.array(val_distances)
        avg_val_error = np.mean(val_distances)
        val_errors.append(avg_val_error)
        
        acc_1m = np.mean(val_distances <= 1.0) * 100
        acc_3m = np.mean(val_distances <= 3.0) * 100
        acc_5m = np.mean(val_distances <= 5.0) * 100
        acc_10m = np.mean(val_distances <= 10.0) * 100
        acc_20m = np.mean(val_distances <= 20.0) * 100
        
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
                'acc_3m': acc_3m,
                'acc_5m': acc_5m,
                'acc_10m': acc_10m,
                'acc_20m': acc_20m
            }, save_path / 'large_dataset_best_model.pth')
            logger.info(f"  💾 保存最佳模型 (Val Error: {avg_val_error:.2f}m)")
        
        scheduler.step(avg_val_error)
    
    logger.info("\n" + "="*60)
    logger.info("训练完成")
    logger.info("="*60)
    logger.info(f"最佳验证误差: {best_val_error:.2f}m")
    
    checkpoint = torch.load(save_path / 'large_dataset_best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    final_distances = []
    
    with torch.no_grad():
        for batch in val_loader:
            descriptions = batch['descriptions']
            object_features = batch['object_features'].to(device)
            gt_xy = batch['gt_xy'].to(device)
            
            pred_xy = model(descriptions, object_features)
            
            distances = torch.sqrt(torch.sum((pred_xy - gt_xy) ** 2, dim=1))
            final_distances.extend(distances.cpu().numpy())
    
    final_distances = np.array(final_distances)
    
    logger.info(f"\n📊 最终评估结果:")
    logger.info(f"  样本数: {len(final_distances)}")
    logger.info(f"  平均误差: {np.mean(final_distances):.2f}m")
    logger.info(f"  中位数误差: {np.median(final_distances):.2f}m")
    logger.info(f"  最小/最大误差: {np.min(final_distances):.2f}m / {np.max(final_distances):.2f}m")
    logger.info(f"  标准差: {np.std(final_distances):.2f}m")
    logger.info(f"\n📈 准确率:")
    logger.info(f"  1m内: {np.mean(final_distances <= 1.0) * 100:.1f}%")
    logger.info(f"  3m内: {np.mean(final_distances <= 3.0) * 100:.1f}%")
    logger.info(f"  5m内: {np.mean(final_distances <= 5.0) * 100:.1f}%")
    logger.info(f"  10m内: {np.mean(final_distances <= 10.0) * 100:.1f}%")
    logger.info(f"  20m内: {np.mean(final_distances <= 20.0) * 100:.1f}%")
    
    results = {
        'avg_error': float(np.mean(final_distances)),
        'median_error': float(np.median(final_distances)),
        'min_error': float(np.min(final_distances)),
        'max_error': float(np.max(final_distances)),
        'std_error': float(np.std(final_distances)),
        'acc_1m': float(np.mean(final_distances <= 1.0) * 100),
        'acc_3m': float(np.mean(final_distances <= 3.0) * 100),
        'acc_5m': float(np.mean(final_distances <= 5.0) * 100),
        'acc_10m': float(np.mean(final_distances <= 10.0) * 100),
        'acc_20m': float(np.mean(final_distances <= 20.0) * 100),
        'total_samples': int(len(final_distances))
    }
    
    with open(save_path / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, train_losses, val_errors, results


if __name__ == "__main__":
    model, train_losses, val_errors, results = train_model(
        num_epochs=50,
        batch_size=32,
        lr=0.001
    )
