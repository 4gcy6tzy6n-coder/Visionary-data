"""
训练神经网络模型
基于 Text2Loc-main 的架构，使用当前数据进行训练
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

# 尝试导入 Text2Loc-main 的模型
try:
    from models.cell_retrieval import CellRetrievalNetwork
    from models.language_encoder import LanguageEncoder
    from models.object_encoder import ObjectEncoder
    TEXT2LOC_MODELS_AVAILABLE = True
    logger.info("✅ Text2Loc-main 模型导入成功")
except ImportError as e:
    logger.warning(f"⚠️ Text2Loc-main 模型导入失败: {e}")
    TEXT2LOC_MODELS_AVAILABLE = False


class SimpleCellDataset(Dataset):
    """简化的Cell数据集"""
    
    def __init__(self, cells: List[Dict], poses: List[Dict]):
        self.cells = cells
        self.poses = poses
        
        # 构建cell查找表
        self.cell_dict = {}
        for cell in cells:
            if isinstance(cell, dict):
                cell_id = cell.get('id', 'unknown')
                self.cell_dict[cell_id] = cell
        
        # 过滤有效的poses
        self.valid_poses = []
        for pose in poses:
            if isinstance(pose, dict):
                cell_id = pose.get('cell_id', '')
                if cell_id in self.cell_dict:
                    self.valid_poses.append(pose)
        
        logger.info(f"📊 数据集: {len(self.cells)} cells, {len(self.valid_poses)} valid poses")
    
    def __len__(self):
        return len(self.valid_poses)
    
    def __getitem__(self, idx):
        pose = self.valid_poses[idx]
        cell_id = pose.get('cell_id', '')
        cell = self.cell_dict[cell_id]
        
        # 获取描述
        description = pose.get('description', '')
        if not description:
            description = f"Location in {cell_id}"
        
        # 获取cell特征（从objects）
        objects = cell.get('objects', [])
        object_features = []
        
        for obj in objects:
            if isinstance(obj, dict):
                center = obj.get('center', [0, 0, 0])
                color = obj.get('color', [0.5, 0.5, 0.5])
                
                # 简单的特征：位置和颜色
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
            gt_x, gt_y = float(location[0]), float(location[1])
        else:
            gt_x, gt_y = 0.0, 0.0
        
        # Cell中心（从objects计算）
        cell_center = self._calculate_cell_center(cell)
        
        return {
            'description': description,
            'cell_id': cell_id,
            'object_features': torch.tensor(object_features, dtype=torch.float32),
            'gt_location': torch.tensor([gt_x, gt_y], dtype=torch.float32),
            'cell_center': torch.tensor(cell_center, dtype=torch.float32),
            'num_objects': len(object_features)
        }
    
    def _calculate_cell_center(self, cell: Dict) -> List[float]:
        """计算cell中心"""
        objects = cell.get('objects', [])
        if not objects:
            return [0.0, 0.0]
        
        centers = []
        for obj in objects:
            if isinstance(obj, dict):
                center = obj.get('center', [])
                if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                    centers.append([float(center[0]), float(center[1])])
        
        if centers:
            centers = np.array(centers)
            avg = np.mean(centers, axis=0)
            return [avg[0], avg[1]]
        
        return [0.0, 0.0]


class SimpleCellRetrievalModel(nn.Module):
    """简化的Cell检索模型"""
    
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 文本编码器（简单的Embedding + LSTM）
        vocab_size = 10000
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.text_fc = nn.Linear(hidden_dim * 2, embed_dim)
        
        # Object编码器（PointNet风格）
        self.object_conv1 = nn.Conv1d(6, 64, 1)
        self.object_conv2 = nn.Conv1d(64, 128, 1)
        self.object_conv3 = nn.Conv1d(128, embed_dim, 1)
        self.object_fc = nn.Linear(embed_dim, embed_dim)
        
        # 融合层
        self.fusion_fc = nn.Linear(embed_dim * 2, embed_dim)
        
        # 位置预测头
        self.location_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # 输出 (x, y)
        )
    
    def encode_text(self, descriptions: List[str]) -> torch.Tensor:
        """编码文本描述"""
        # 简化的文本编码：使用字符级别的编码
        batch_size = len(descriptions)
        max_len = max(len(d) for d in descriptions) if descriptions else 1
        
        # 将文本转换为索引
        text_indices = []
        for desc in descriptions:
            indices = [ord(c) % 10000 for c in desc[:max_len]]
            indices += [0] * (max_len - len(indices))
            text_indices.append(indices)
        
        text_tensor = torch.tensor(text_indices, dtype=torch.long).to(next(self.parameters()).device)
        
        # Embedding + LSTM
        x = self.text_embedding(text_tensor)
        x, _ = self.text_lstm(x)
        x = x[:, -1, :]  # 取最后一个时间步
        x = self.text_fc(x)
        
        return x
    
    def encode_objects(self, object_features: torch.Tensor) -> torch.Tensor:
        """编码Object特征"""
        # object_features: (batch_size, num_objects, 6)
        batch_size, num_objects, _ = object_features.shape
        
        # 转置为 (batch_size, 6, num_objects)
        x = object_features.transpose(1, 2)
        
        # 1D卷积
        x = torch.relu(self.object_conv1(x))
        x = torch.relu(self.object_conv2(x))
        x = torch.relu(self.object_conv3(x))
        
        # Max pooling
        x = torch.max(x, 2)[0]  # (batch_size, embed_dim)
        
        x = self.object_fc(x)
        
        return x
    
    def forward(self, descriptions: List[str], object_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 编码文本和Object
        text_enc = self.encode_text(descriptions)
        object_enc = self.encode_objects(object_features)
        
        # 融合
        fused = torch.cat([text_enc, object_enc], dim=1)
        fused = self.fusion_fc(fused)
        
        # 预测位置
        location = self.location_head(fused)
        
        return location, fused


def collate_fn(batch):
    """自定义collate函数"""
    descriptions = [item['description'] for item in batch]
    cell_ids = [item['cell_id'] for item in batch]
    
    # 处理object_features（填充到相同长度）
    max_objects = max(item['num_objects'] for item in batch)
    object_features = []
    
    for item in batch:
        feat = item['object_features']
        if feat.shape[0] < max_objects:
            padding = torch.zeros(max_objects - feat.shape[0], feat.shape[1])
            feat = torch.cat([feat, padding], dim=0)
        object_features.append(feat)
    
    object_features = torch.stack(object_features)
    
    gt_locations = torch.stack([item['gt_location'] for item in batch])
    cell_centers = torch.stack([item['cell_center'] for item in batch])
    
    return {
        'descriptions': descriptions,
        'cell_ids': cell_ids,
        'object_features': object_features,
        'gt_locations': gt_locations,
        'cell_centers': cell_centers
    }


def train_model(data_path: str = None, num_epochs: int = 50, batch_size: int = 4, lr: float = 0.001):
    """训练模型"""
    
    if data_path is None:
        data_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all"
    
    data_path = Path(data_path)
    
    # 加载数据
    logger.info("\n" + "="*60)
    logger.info("加载数据")
    logger.info("="*60)
    
    cells_file = data_path / "cells" / "cells.pkl"
    poses_file = data_path / "poses" / "poses.pkl"
    
    with open(cells_file, 'rb') as f:
        cells = pickle.load(f)
    
    with open(poses_file, 'rb') as f:
        poses = pickle.load(f)
    
    logger.info(f"✅ 加载 {len(cells)} cells, {len(poses)} poses")
    
    # 划分训练集和验证集
    np.random.seed(42)
    indices = np.random.permutation(len(poses))
    split = int(len(poses) * 0.8)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_poses = [poses[i] for i in train_indices]
    val_poses = [poses[i] for i in val_indices]
    
    logger.info(f"📊 训练集: {len(train_poses)}, 验证集: {len(val_poses)}")
    
    # 创建数据集
    train_dataset = SimpleCellDataset(cells, train_poses)
    val_dataset = SimpleCellDataset(cells, val_poses)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🖥️ 使用设备: {device}")
    
    model = SimpleCellRetrievalModel(embed_dim=128, hidden_dim=256).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 训练循环
    logger.info("\n" + "="*60)
    logger.info("开始训练")
    logger.info("="*60)
    
    best_val_error = float('inf')
    train_losses = []
    val_errors = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        epoch_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            descriptions = batch['descriptions']
            object_features = batch['object_features'].to(device)
            gt_locations = batch['gt_locations'].to(device)
            
            pred_locations, _ = model(descriptions, object_features)
            
            loss = criterion(pred_locations, gt_locations)
            loss.backward()
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
                gt_locations = batch['gt_locations'].to(device)
                
                pred_locations, _ = model(descriptions, object_features)
                
                # 计算距离误差
                distances = torch.sqrt(torch.sum((pred_locations - gt_locations) ** 2, dim=1))
                val_distances.extend(distances.cpu().numpy())
        
        val_distances = np.array(val_distances)
        avg_val_error = np.mean(val_distances)
        val_errors.append(avg_val_error)
        
        # 计算准确率
        acc_5m = np.mean(val_distances <= 5.0) * 100
        acc_10m = np.mean(val_distances <= 10.0) * 100
        
        logger.info(f"Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.4f}, Val Error={avg_val_error:.2f}m, Acc@5m={acc_5m:.1f}%, Acc@10m={acc_10m:.1f}%")
        
        # 保存最佳模型
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            save_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/checkpoints")
            save_path.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': avg_val_error,
                'acc_5m': acc_5m,
                'acc_10m': acc_10m
            }, save_path / 'best_model.pth')
            logger.info(f"  💾 保存最佳模型 (Val Error: {avg_val_error:.2f}m)")
        
        scheduler.step()
    
    # 最终结果
    logger.info("\n" + "="*60)
    logger.info("训练完成")
    logger.info("="*60)
    logger.info(f"最佳验证误差: {best_val_error:.2f}m")
    
    return model, train_losses, val_errors


if __name__ == "__main__":
    model, train_losses, val_errors = train_model(num_epochs=50, batch_size=4, lr=0.001)
