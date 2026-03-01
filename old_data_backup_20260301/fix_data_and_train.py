#!/usr/bin/env python3
"""
修复数据并训练 Text2Loc Visionary 模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import json
import time


class Text2LocNeuralNetwork(nn.Module):
    """Text2Loc Visionary 神经网络模型"""
    
    def __init__(self, embed_dim=256, hidden_dim=512):
        super().__init__()
        
        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 对象特征编码器
        self.object_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)  # 输出 x, y 偏移
        )
    
    def forward(self, text_features, object_features):
        text_encoded = self.text_encoder(text_features)
        object_encoded = self.object_encoder(object_features)
        combined = torch.cat([text_encoded, object_encoded], dim=-1)
        output = self.fusion(combined)
        return output


class Text2LocDataset(Dataset):
    """Text2Loc 数据集"""
    
    def __init__(self, data_dir: str, embed_dim: int = 256):
        self.data_dir = Path(data_dir)
        self.embed_dim = embed_dim
        
        # 加载数据
        with open(self.data_dir / "cells.pkl", 'rb') as f:
            self.cells = pickle.load(f)
        with open(self.data_dir / "poses.pkl", 'rb') as f:
            self.poses = pickle.load(f)
        
        # 创建cell字典
        self.cells_dict = {cell['id']: cell for cell in self.cells if isinstance(cell, dict)}
        
        # 验证并修复poses
        self.valid_poses = []
        for pose in self.poses:
            cell_id = pose.get('cell_id')
            if cell_id in self.cells_dict:
                cell = self.cells_dict[cell_id]
                # 确保cell_center正确
                pose['cell_center'] = cell.get('center', [0, 0, 0])
                self.valid_poses.append(pose)
        
        print(f"数据集加载完成:")
        print(f"  Cells: {len(self.cells)}")
        print(f"  Valid Poses: {len(self.valid_poses)}")
        
        # 计算offset统计
        offsets = []
        for pose in self.valid_poses[:1000]:
            location = np.array(pose['location'])
            cell_center = np.array(pose['cell_center'])
            offset = location - cell_center
            offsets.append(offset[:2])
        offsets = np.array(offsets)
        
        self.offset_mean = torch.tensor(np.mean(offsets, axis=0), dtype=torch.float32)
        self.offset_std = torch.tensor(np.std(offsets, axis=0) + 1e-8, dtype=torch.float32)
        
        print(f"  Offset均值: {self.offset_mean.numpy()}")
        print(f"  Offset标准差: {self.offset_std.numpy()}")
    
    def __len__(self):
        return len(self.valid_poses)
    
    def __getitem__(self, idx):
        pose = self.valid_poses[idx]
        cell_id = pose['cell_id']
        cell = self.cells_dict[cell_id]
        
        # 文本嵌入 (使用随机初始化，实际应该使用预训练模型)
        description = pose.get('description', '')
        text_hash = hash(description) % (2**31)
        np.random.seed(text_hash)
        text_embedding = torch.randn(self.embed_dim)
        
        # 对象特征
        objects = cell.get('objects', [])
        if len(objects) == 0:
            object_features = torch.zeros(6)
        else:
            # 聚合对象特征
            obj_features_list = []
            for obj in objects[:50]:  # 限制最多50个对象
                center = obj.get('center', [0, 0, 0])
                confidence = obj.get('confidence', 0) / 1000.0  # 归一化
                max_conf = obj.get('max_confidence', 0) / 1000.0
                semantic = obj.get('semantic', 0) / 10.0
                
                feat = [
                    float(center[0]), float(center[1]),
                    confidence, max_conf, semantic,
                    len(objects) / 1000.0  # 对象数量特征
                ]
                obj_features_list.append(feat)
            
            # 平均池化
            obj_features_array = np.array(obj_features_list)
            object_features = torch.tensor(np.mean(obj_features_array, axis=0), dtype=torch.float32)
        
        # 目标偏移 (归一化)
        location = np.array(pose['location'])
        cell_center = np.array(pose['cell_center'])
        offset = location - cell_center
        offset_xy = torch.tensor(offset[:2], dtype=torch.float32)
        offset_normalized = (offset_xy - self.offset_mean) / self.offset_std
        
        return {
            'text_embedding': text_embedding,
            'object_features': object_features,
            'offset': offset_normalized,
            'offset_raw': offset_xy,
            'cell_center': torch.tensor(cell_center[:2], dtype=torch.float32),
            'description': description
        }


def train_model(data_dir: str, output_dir: str, epochs: int = 100, batch_size: int = 64):
    """训练模型"""
    
    print("="*80)
    print("训练 Text2Loc Visionary 模型 (44GB 2D语义数据)")
    print("="*80)
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建数据集
    dataset = Text2LocDataset(data_dir)
    
    # 划分训练集和验证集 (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    model = Text2LocNeuralNetwork(embed_dim=256, hidden_dim=512).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            text_emb = batch['text_embedding'].to(device)
            obj_feat = batch['object_features'].to(device)
            target = batch['offset'].to(device)
            
            optimizer.zero_grad()
            output = model(text_emb, obj_feat)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                text_emb = batch['text_embedding'].to(device)
                obj_feat = batch['object_features'].to(device)
                target = batch['offset'].to(device)
                target_raw = batch['offset_raw']
                
                output = model(text_emb, obj_feat)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                # 计算MAE (米)
                output_denorm = output.cpu() * dataset.offset_std + dataset.offset_mean
                mae = torch.mean(torch.abs(output_denorm - target_raw)).item()
                val_mae += mae
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"Val MAE: {val_mae:.3f}m, "
                  f"LR: {current_lr:.6f}, "
                  f"Time: {epoch_time:.1f}s")
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'offset_mean': dataset.offset_mean,
                'offset_std': dataset.offset_std,
                'best_val_loss': best_val_loss
            }, output_path / "k360_2d_semantics_best_model.pth")
            
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.6f}, MAE: {val_mae:.3f}m)")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"训练完成!")
    print(f"{'='*80}")
    print(f"总时间: {total_time/60:.1f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'epochs': epochs,
        'batch_size': batch_size,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }
    
    with open(output_path / "training_history_2d_semantics.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


def main():
    data_dir = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_2d_semantics_full"
    output_dir = "/Users/yaoyingliang/Desktop/Text2Loc-main/checkpoints"
    
    model, history = train_model(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=100,
        batch_size=64
    )
    
    print("\n" + "="*80)
    print("所有任务完成!")
    print("="*80)


if __name__ == '__main__':
    main()
