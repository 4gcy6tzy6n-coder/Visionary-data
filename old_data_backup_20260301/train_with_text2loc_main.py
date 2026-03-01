"""
使用 Text2Loc-main 的数据结构训练神经网络
直接复用 Text2Loc-main 的数据加载和评估代码
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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 添加 Text2Loc-main 到路径
TEXT2LOC_MAIN_PATH = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc-main")
if str(TEXT2LOC_MAIN_PATH) not in sys.path:
    sys.path.insert(0, str(TEXT2LOC_MAIN_PATH))


class Text2LocNeuralNetwork(nn.Module):
    """
    简化的 Text2Loc 神经网络
    基于 Text2Loc-main 架构但简化实现
    """
    
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Object编码器（PointNet风格）
        self.obj_conv1 = nn.Conv1d(6, 64, 1)
        self.obj_conv2 = nn.Conv1d(64, 128, 1)
        self.obj_conv3 = nn.Conv1d(128, embed_dim, 1)
        
        # Object间注意力
        self.obj_attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        # 文本编码器（字符级CNN）
        self.char_embed = nn.Embedding(256, 64)  # ASCII字符
        self.text_conv1 = nn.Conv1d(64, 128, 3, padding=1)
        self.text_conv2 = nn.Conv1d(128, embed_dim, 3, padding=1)
        
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
        """
        编码Objects
        object_features: (batch_size, num_objects, 6) - [x, y, z, r, g, b]
        """
        batch_size, num_objects, _ = object_features.shape
        
        # 转置为 (batch_size, 6, num_objects)
        x = object_features.transpose(1, 2)
        
        # 1D卷积
        x = torch.relu(self.obj_conv1(x))
        x = torch.relu(self.obj_conv2(x))
        x = torch.relu(self.obj_conv3(x))
        
        # 转置回 (batch_size, num_objects, embed_dim)
        x = x.transpose(1, 2)
        
        # 自注意力
        x, _ = self.obj_attention(x, x, x)
        
        # Max pooling
        x = torch.max(x, dim=1)[0]
        
        return x
    
    def encode_text(self, descriptions: List[str]) -> torch.Tensor:
        """
        编码文本描述
        """
        batch_size = len(descriptions)
        max_len = max(len(d) for d in descriptions) if descriptions else 1
        
        # 将文本转换为字符索引
        char_indices = []
        for desc in descriptions:
            indices = [min(ord(c), 255) for c in desc[:max_len]]
            indices += [0] * (max_len - len(indices))
            char_indices.append(indices)
        
        # (batch_size, max_len)
        x = torch.tensor(char_indices, dtype=torch.long).to(next(self.parameters()).device)
        
        # Embedding: (batch_size, max_len, 64)
        x = self.char_embed(x)
        
        # 转置为 (batch_size, 64, max_len)
        x = x.transpose(1, 2)
        
        # 1D卷积
        x = torch.relu(self.text_conv1(x))
        x = torch.relu(self.text_conv2(x))
        
        # Max pooling
        x = torch.max(x, dim=2)[0]
        
        return x
    
    def forward(self, descriptions: List[str], object_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 编码
        text_enc = self.encode_text(descriptions)
        obj_enc = self.encode_objects(object_features)
        
        # 融合
        fused = torch.cat([text_enc, obj_enc], dim=1)
        fused = self.fusion(fused)
        
        # 预测位置
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
        
        # 描述
        description = pose.get('description', '')
        if not description:
            description = f"Location near {pose['cell_id']}"
        
        # Object特征
        objects = cell.get('objects', [])
        object_features = []
        
        for obj in objects:
            if isinstance(obj, dict):
                center = obj.get('center', [0, 0, 0])
                color = obj.get('color', [0.5, 0.5, 0.5])
                
                feat = [
                    float(center[0]) if len(center) > 0 else 0,
                    float(center[1]) if len(center) > 1 else 0,
                    float(center[2]) if len(center) > 2 else 0,
                    float(color[0]) if isinstance(color, (list, tuple)) and len(color) > 0 else 0.5,
                    float(color[1]) if isinstance(color, (list, tuple)) and len(color) > 1 else 0.5,
                    float(color[2]) if isinstance(color, (list, tuple)) and len(color) > 2 else 0.5,
                ]
                object_features.append(feat)
        
        if not object_features:
            object_features = [[0.0] * 6]
        
        # 真实位置
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
    
    # 填充object_features
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


def train_model(data_path: str = None, num_epochs: int = 100, batch_size: int = 4, lr: float = 0.001):
    """训练模型"""
    
    if data_path is None:
        data_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all"
    
    data_path = Path(data_path)
    
    # 加载数据
    logger.info("\n" + "="*60)
    logger.info("加载数据")
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
    
    # 创建数据集
    train_dataset = Text2LocDataset(cells, train_poses)
    val_dataset = Text2LocDataset(cells, val_poses)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🖥️ 使用设备: {device}")
    
    model = Text2LocNeuralNetwork(embed_dim=128, hidden_dim=256).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 训练循环
    logger.info("\n" + "="*60)
    logger.info("开始训练神经网络")
    logger.info("="*60)
    
    best_val_error = float('inf')
    train_losses = []
    val_errors = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            descriptions = batch['descriptions']
            object_features = batch['object_features'].to(device)
            gt_xy = batch['gt_xy'].to(device)
            
            pred_xy = model(descriptions, object_features)
            
            loss = criterion(pred_xy, gt_xy)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # 验证
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
        
        # 计算准确率
        acc_1m = np.mean(val_distances <= 1.0) * 100
        acc_3m = np.mean(val_distances <= 3.0) * 100
        acc_5m = np.mean(val_distances <= 5.0) * 100
        acc_10m = np.mean(val_distances <= 10.0) * 100
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, Val Error={avg_val_error:.2f}m, "
                       f"Acc@1m={acc_1m:.1f}%, Acc@5m={acc_5m:.1f}%, Acc@10m={acc_10m:.1f}%")
        
        # 保存最佳模型
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            save_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/checkpoints")
            save_path.mkdir(exist_ok=True, parents=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': avg_val_error,
                'acc_1m': acc_1m,
                'acc_5m': acc_5m,
                'acc_10m': acc_10m
            }, save_path / 'text2loc_best_model.pth')
        
        # 学习率调整
        scheduler.step(avg_val_error)
    
    # 最终结果
    logger.info("\n" + "="*60)
    logger.info("训练完成")
    logger.info("="*60)
    logger.info(f"最佳验证误差: {best_val_error:.2f}m")
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load(save_path / 'text2loc_best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最终评估
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
    logger.info(f"  平均误差: {np.mean(final_distances):.2f}m")
    logger.info(f"  中位数误差: {np.median(final_distances):.2f}m")
    logger.info(f"  1m准确率: {np.mean(final_distances <= 1.0) * 100:.1f}%")
    logger.info(f"  3m准确率: {np.mean(final_distances <= 3.0) * 100:.1f}%")
    logger.info(f"  5m准确率: {np.mean(final_distances <= 5.0) * 100:.1f}%")
    logger.info(f"  10m准确率: {np.mean(final_distances <= 10.0) * 100:.1f}%")
    
    return model, train_losses, val_errors


if __name__ == "__main__":
    model, train_losses, val_errors = train_model(num_epochs=100, batch_size=4, lr=0.001)
