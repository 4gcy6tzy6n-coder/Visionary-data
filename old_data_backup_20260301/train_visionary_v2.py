#!/usr/bin/env python3
"""
Text2Loc Visionary V2 - 全面改进版
- BERT文本嵌入
- 50m cell size
- 优化训练策略
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
from transformers import BertTokenizer, BertModel


class Text2LocNeuralNetworkV2(nn.Module):
    """改进版神经网络 - 更深的架构"""
    
    def __init__(self, embed_dim=768, hidden_dim=1024):
        super().__init__()
        
        # 文本编码器 - 更深的网络
        self.text_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 对象特征编码器
        self.object_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # 输出 x, y 偏移
        )
    
    def forward(self, text_features, object_features):
        text_encoded = self.text_encoder(text_features)
        object_encoded = self.object_encoder(object_features)
        combined = torch.cat([text_encoded, object_encoded], dim=-1)
        output = self.fusion(combined)
        return output


class BERTTextEmbedder:
    """BERT文本嵌入器"""
    
    def __init__(self):
        print("加载BERT模型...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        
        # 使用MPS加速
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"BERT模型已加载到 {self.device}")
    
    def embed(self, text: str) -> torch.Tensor:
        """生成文本嵌入"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # 使用[CLS] token的嵌入
            embedding = outputs.last_hidden_state[:, 0, :]
            return embedding.cpu().squeeze(0)


class Text2LocDatasetV2(Dataset):
    """改进版数据集 - 使用BERT嵌入"""
    
    def __init__(self, data_dir: str, text_embedder: BERTTextEmbedder, 
                 embed_cache_path: str = None):
        self.data_dir = Path(data_dir)
        self.text_embedder = text_embedder
        
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
                pose['cell_center'] = cell.get('center', [0, 0, 0])
                self.valid_poses.append(pose)
        
        print(f"数据集加载完成: {len(self.valid_poses)} poses")
        
        # 预计算文本嵌入
        self.embed_cache_path = embed_cache_path
        if embed_cache_path and Path(embed_cache_path).exists():
            print("加载预计算的文本嵌入...")
            with open(embed_cache_path, 'rb') as f:
                self.text_embeddings = pickle.load(f)
        else:
            print("预计算文本嵌入...")
            self.text_embeddings = self._precompute_embeddings()
            if embed_cache_path:
                with open(embed_cache_path, 'wb') as f:
                    pickle.dump(self.text_embeddings, f)
        
        # 计算offset统计
        offsets = []
        for pose in self.valid_poses[:1000]:
            location = np.array(pose['location'])
            center = np.array(pose['cell_center'])
            offset = location - center
            offsets.append(offset[:2])
        
        offsets = np.array(offsets)
        self.offset_mean = torch.tensor(np.mean(offsets, axis=0), dtype=torch.float32)
        self.offset_std = torch.tensor(np.std(offsets, axis=0) + 1e-8, dtype=torch.float32)
        
        print(f"Offset均值: {self.offset_mean.numpy()}")
        print(f"Offset标准差: {self.offset_std.numpy()}")
    
    def _precompute_embeddings(self):
        """预计算所有文本嵌入"""
        embeddings = []
        for pose in tqdm(self.valid_poses, desc="生成BERT嵌入"):
            description = pose.get('description', '')
            embedding = self.text_embedder.embed(description)
            embeddings.append(embedding)
        return embeddings
    
    def __len__(self):
        return len(self.valid_poses)
    
    def __getitem__(self, idx):
        pose = self.valid_poses[idx]
        cell_id = pose['cell_id']
        cell = self.cells_dict[cell_id]
        
        # 使用预计算的BERT嵌入
        text_embedding = self.text_embeddings[idx]
        
        # 对象特征
        objects = cell.get('objects', [])
        if len(objects) == 0:
            object_features = torch.zeros(6)
        else:
            obj_features_list = []
            for obj in objects[:50]:
                center = obj.get('center', [0, 0, 0])
                confidence = obj.get('confidence', 0) / 1000.0
                max_conf = obj.get('max_confidence', 0) / 1000.0
                semantic = obj.get('semantic', 0) / 10.0
                feat = [
                    float(center[0]), float(center[1]),
                    confidence, max_conf, semantic,
                    len(objects) / 1000.0
                ]
                obj_features_list.append(feat)
            
            obj_features_array = np.array(obj_features_list)
            object_features = torch.tensor(np.mean(obj_features_array, axis=0), dtype=torch.float32)
        
        # 目标偏移
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
            'cell_center': torch.tensor(cell_center[:2], dtype=torch.float32)
        }


def train_model_v2(data_dir: str, output_dir: str, epochs: int = 50, batch_size: int = 128):
    """改进版训练 - 使用BERT和Huber Loss"""
    
    print("="*80)
    print("Text2Loc Visionary V2 - 改进版训练")
    print("="*80)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建BERT嵌入器
    text_embedder = BERTTextEmbedder()
    
    # 创建数据集
    embed_cache = Path(output_dir) / "bert_embeddings.pkl"
    dataset = Text2LocDatasetV2(data_dir, text_embedder, str(embed_cache))
    
    # 划分数据集
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
    model = Text2LocNeuralNetworkV2(embed_dim=768, hidden_dim=1024).to(device)
    
    # 使用Huber Loss
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # 学习率调度 - 预热 + 余弦退火
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    train_losses = []
    val_losses = []
    
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
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
                
                # 计算MAE
                output_denorm = output.cpu() * dataset.offset_std + dataset.offset_mean
                mae = torch.mean(torch.abs(output_denorm - target_raw)).item()
                val_mae += mae
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"Val MAE: {val_mae:.3f}m, "
                  f"LR: {current_lr:.6f}, "
                  f"Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
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
                'best_val_loss': best_val_loss,
                'best_val_mae': best_val_mae
            }, output_path / "visionary_v2_best_model.pth")
            
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.6f}, MAE: {val_mae:.3f}m)")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"训练完成!")
    print(f"{'='*80}")
    print(f"总时间: {total_time/60:.1f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最佳验证MAE: {best_val_mae:.3f}m")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae,
        'total_time': total_time,
        'epochs': epochs,
        'batch_size': batch_size
    }
    
    with open(output_path / "training_history_v2.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


def main():
    data_dir = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_2d_semantics_full"
    output_dir = "/Users/yaoyingliang/Desktop/Text2Loc-main/checkpoints"
    
    model, history = train_model_v2(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=50,
        batch_size=128
    )
    
    print("\n" + "="*80)
    print("Visionary V2 训练完成!")
    print("="*80)


if __name__ == '__main__':
    main()
