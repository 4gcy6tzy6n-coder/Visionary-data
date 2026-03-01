#!/usr/bin/env python3
"""
在完整的 test_slam 数据上训练 Text2Loc Visionary 模型
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
        # 编码文本
        text_encoded = self.text_encoder(text_features)
        
        # 编码对象特征
        object_encoded = self.object_encoder(object_features)
        
        # 融合
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
        
        # 计算全局统计信息用于归一化
        all_centers = []
        for cell in self.cells_dict.values():
            center = cell.get('center', [0, 0, 0])
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                all_centers.append([float(center[0]), float(center[1])])
        
        all_centers = np.array(all_centers)
        self.global_mean = torch.tensor(np.mean(all_centers, axis=0), dtype=torch.float32)
        self.global_std = torch.tensor(np.std(all_centers, axis=0) + 1e-8, dtype=torch.float32)
        
        print(f"数据集加载完成:")
        print(f"  Cells: {len(self.cells)}")
        print(f"  Poses: {len(self.poses)}")
        print(f"  全局均值: {self.global_mean.numpy()}")
        print(f"  全局标准差: {self.global_std.numpy()}")
    
    def _get_text_embedding(self, description: str):
        """简单的文本嵌入（使用哈希）"""
        # 创建一个简单的基于词频的嵌入
        words = description.lower().split()
        embedding = np.zeros(self.embed_dim)
        
        for i, word in enumerate(words):
            # 使用哈希生成词向量
            hash_val = hash(word) % self.embed_dim
            embedding[hash_val] += 1.0
        
        # 归一化
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return torch.tensor(embedding, dtype=torch.float32)
    
    def _get_object_features(self, cell_id: str):
        """获取对象特征"""
        cell = self.cells_dict.get(cell_id)
        if not cell:
            return torch.zeros(6, dtype=torch.float32)
        
        objects = cell.get('objects', [])
        if not objects:
            return torch.zeros(6, dtype=torch.float32)
        
        # 选择最多10个对象
        selected_objects = objects[:10]
        
        features = []
        for obj in selected_objects:
            if isinstance(obj, dict):
                center = obj.get('center', [0, 0, 0])
                cell_center = cell.get('center', [0, 0, 0])
                
                # 相对位置
                rel_x = float(center[0]) - float(cell_center[0]) if len(center) > 0 else 0
                rel_y = float(center[1]) - float(cell_center[1]) if len(center) > 1 else 0
                rel_z = float(center[2]) if len(center) > 2 else 0
                
                # 语义标签
                semantic = obj.get('semantic', 0)
                if isinstance(semantic, int):
                    semantic_feat = semantic / 10.0
                else:
                    semantic_feat = 0.0
                
                # 对象大小
                bbox = obj.get('bbox', [0, 0, 0, 0, 0, 0])
                if len(bbox) >= 6:
                    size_x = (bbox[1] - bbox[0]) / 10.0
                    size_y = (bbox[3] - bbox[2]) / 10.0
                else:
                    size_x, size_y = 0.5, 0.5
                
                feat = [rel_x, rel_y, rel_z, semantic_feat, size_x, size_y]
                features.append(feat)
        
        if not features:
            return torch.zeros(6, dtype=torch.float32)
        
        # 平均所有对象的特征
        avg_features = np.mean(features, axis=0)
        return torch.tensor(avg_features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        pose = self.poses[idx]
        
        # 获取描述
        description = pose.get('description', '')
        text_embedding = self._get_text_embedding(description)
        
        # 获取cell信息
        cell_id = pose.get('cell_id', '')
        cell = self.cells_dict.get(cell_id, {})
        cell_center = cell.get('center', [0, 0, 0])
        cell_center_xy = torch.tensor([float(cell_center[0]), float(cell_center[1])], dtype=torch.float32)
        
        # 获取对象特征
        object_features = self._get_object_features(cell_id)
        
        # 计算目标偏移量
        location = pose.get('location', [0, 0, 0])
        if isinstance(location, (list, tuple)) and len(location) >= 2:
            global_xy = torch.tensor([float(location[0]), float(location[1])], dtype=torch.float32)
        else:
            global_xy = cell_center_xy.clone()
        
        # 归一化偏移量
        offset_xy = global_xy - cell_center_xy
        offset_normalized = (offset_xy - self.global_mean) / self.global_std
        
        return {
            'text_embedding': text_embedding,
            'object_features': object_features,
            'offset': offset_normalized,
            'cell_center': cell_center_xy,
            'global_xy': global_xy,
            'description': description
        }


def train_model(data_dir: str, output_dir: str, epochs: int = 150, batch_size: int = 32):
    """训练模型"""
    
    print("="*80)
    print("训练 Text2Loc Visionary 模型 (完整数据集)")
    print("="*80)
    
    # 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建数据集
    dataset = Text2LocDataset(data_dir)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    model = Text2LocNeuralNetwork(embed_dim=256, hidden_dim=512).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\n开始训练...")
    for epoch in range(epochs):
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
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                text_emb = batch['text_embedding'].to(device)
                obj_feat = batch['object_features'].to(device)
                target = batch['offset'].to(device)
                
                output = model(text_emb, obj_feat)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
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
                'global_mean': dataset.global_mean,
                'global_std': dataset.global_std
            }, output_path / "test_slam_full_best_model.pth")
    
    print(f"\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    
    with open(output_path / "training_history_full.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


def main():
    data_dir = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_test_slam_full"
    output_dir = "/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/checkpoints"
    
    model, history = train_model(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=150,
        batch_size=32
    )
    
    print("\n" + "="*80)
    print("模型训练完成!")
    print("="*80)
    print(f"模型保存位置: {output_dir}/test_slam_full_best_model.pth")


if __name__ == '__main__':
    main()
