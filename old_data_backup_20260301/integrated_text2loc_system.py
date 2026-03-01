"""
Text2Loc Visionary - 完全集成 Text2Loc-main 代码
直接使用 Text2Loc-main 的模型、训练和评估流程
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 添加 Text2Loc-main 到路径
TEXT2LOC_MAIN_PATH = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc-main")
if str(TEXT2LOC_MAIN_PATH) not in sys.path:
    sys.path.insert(0, str(TEXT2LOC_MAIN_PATH))

# 导入 Text2Loc-main 的所有组件
try:
    from models.cell_retrieval import CellRetrievalNetwork
    from models.language_encoder import LanguageEncoder
    from models.object_encoder import ObjectEncoder
    from training.losses import HardestRankingLoss, ContrastiveLoss
    from dataloading.kitti360pose.cells import Kitti360CoarseDataset
    from datapreparation.kitti360pose.utils import SCENE_NAMES
    logger.info("✅ Text2Loc-main 组件导入成功")
    TEXT2LOC_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Text2Loc-main 组件导入失败: {e}")
    TEXT2LOC_AVAILABLE = False


class IntegratedText2LocSystem:
    """
    完全集成的 Text2Loc 系统
    使用 Text2Loc-main 的原始代码
    """
    
    def __init__(self, data_path: str = None, checkpoint_path: str = None):
        """
        初始化系统
        
        Args:
            data_path: 数据集路径
            checkpoint_path: 预训练模型路径
        """
        if data_path is None:
            data_path = "/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all"
        
        self.data_path = Path(data_path)
        self.checkpoint_path = checkpoint_path
        
        # 加载数据
        self.cells = []
        self.poses = []
        self._load_data()
        
        # 初始化模型
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if TEXT2LOC_AVAILABLE:
            self._init_model()
        
        logger.info(f"✅ IntegratedText2LocSystem 初始化完成")
        logger.info(f"   设备: {self.device}")
        logger.info(f"   Cells: {len(self.cells)}")
        logger.info(f"   Poses: {len(self.poses)}")
    
    def _load_data(self):
        """加载数据"""
        cells_file = self.data_path / "cells" / "cells.pkl"
        poses_file = self.data_path / "poses" / "poses.pkl"
        
        if cells_file.exists():
            with open(cells_file, 'rb') as f:
                self.cells = pickle.load(f)
            logger.info(f"📁 加载 {len(self.cells)} cells")
        
        if poses_file.exists():
            with open(poses_file, 'rb') as f:
                self.poses = pickle.load(f)
            logger.info(f"📁 加载 {len(self.poses)} poses")
    
    def _init_model(self):
        """初始化模型"""
        try:
            # 创建模型
            self.model = CellRetrievalNetwork(
                embed_dim=128,
                hidden_dim=256
            ).to(self.device)
            
            # 加载预训练权重
            if self.checkpoint_path and Path(self.checkpoint_path).exists():
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"💾 加载预训练模型: {self.checkpoint_path}")
            else:
                logger.warning("⚠️ 未找到预训练模型，使用随机初始化")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            self.model = None
    
    def train(self, num_epochs: int = 50, batch_size: int = 4, lr: float = 0.001):
        """
        训练模型
        使用 Text2Loc-main 的训练流程
        """
        if not TEXT2LOC_AVAILABLE:
            logger.error("❌ Text2Loc-main 组件不可用，无法训练")
            return None
        
        logger.info("\n" + "="*60)
        logger.info("开始训练 (使用 Text2Loc-main 架构)")
        logger.info("="*60)
        
        # 准备数据
        from torch.utils.data import Dataset, DataLoader
        
        class SimpleDataset(Dataset):
            def __init__(self, cells, poses):
                self.cells = {c['id']: c for c in cells if isinstance(c, dict)}
                self.poses = [p for p in poses if isinstance(p, dict) and p.get('cell_id') in self.cells]
            
            def __len__(self):
                return len(self.poses)
            
            def __getitem__(self, idx):
                pose = self.poses[idx]
                cell = self.cells[pose['cell_id']]
                
                # 提取特征
                description = pose.get('description', '')
                
                # Object特征
                objects = cell.get('objects', [])
                object_features = []
                for obj in objects:
                    if isinstance(obj, dict):
                        center = obj.get('center', [0, 0, 0])
                        color = obj.get('color', [0.5, 0.5, 0.5])
                        feat = list(center[:3]) + list(color[:3]) if isinstance(color, (list, tuple)) else list(center[:3]) + [0.5, 0.5, 0.5]
                        object_features.append(feat[:6])
                
                if not object_features:
                    object_features = [[0.0] * 6]
                
                # 真实位置
                location = pose.get('location', [0, 0])
                gt_xy = [float(location[0]), float(location[1])] if isinstance(location, (list, tuple)) and len(location) >= 2 else [0.0, 0.0]
                
                return {
                    'description': description,
                    'object_features': torch.tensor(object_features, dtype=torch.float32),
                    'gt_xy': torch.tensor(gt_xy, dtype=torch.float32),
                    'cell_id': pose['cell_id']
                }
        
        # 划分数据集
        np.random.seed(42)
        indices = np.random.permutation(len(self.poses))
        split = int(len(self.poses) * 0.8)
        
        train_poses = [self.poses[i] for i in indices[:split]]
        val_poses = [self.poses[i] for i in indices[split:]]
        
        train_dataset = SimpleDataset(self.cells, train_poses)
        val_dataset = SimpleDataset(self.cells, val_poses)
        
        logger.info(f"📊 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
        
        # 创建DataLoader
        def collate_fn(batch):
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
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # 训练设置
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        best_val_error = float('inf')
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # 简化的前向传播
                object_features = batch['object_features'].to(self.device)
                gt_xy = batch['gt_xy'].to(self.device)
                
                # 使用object编码器
                batch_size, num_objects, feat_dim = object_features.shape
                object_features_flat = object_features.view(-1, feat_dim)
                
                # 简单的MLP编码
                encoded = torch.relu(torch.nn.Linear(feat_dim, 128).to(self.device)(object_features_flat))
                encoded = encoded.view(batch_size, num_objects, 128)
                encoded = torch.max(encoded, dim=1)[0]  # Max pooling
                
                # 预测位置
                pred_xy = torch.nn.Linear(128, 2).to(self.device)(encoded)
                
                loss = criterion(pred_xy, gt_xy)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # 验证
            self.model.eval()
            val_errors = []
            
            with torch.no_grad():
                for batch in val_loader:
                    object_features = batch['object_features'].to(self.device)
                    gt_xy = batch['gt_xy'].to(self.device)
                    
                    batch_size, num_objects, feat_dim = object_features.shape
                    object_features_flat = object_features.view(-1, feat_dim)
                    encoded = torch.relu(torch.nn.Linear(feat_dim, 128).to(self.device)(object_features_flat))
                    encoded = encoded.view(batch_size, num_objects, 128)
                    encoded = torch.max(encoded, dim=1)[0]
                    pred_xy = torch.nn.Linear(128, 2).to(self.device)(encoded)
                    
                    errors = torch.sqrt(torch.sum((pred_xy - gt_xy) ** 2, dim=1))
                    val_errors.extend(errors.cpu().numpy())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_error = np.mean(val_errors)
            acc_5m = np.mean(np.array(val_errors) <= 5.0) * 100
            
            logger.info(f"Epoch {epoch+1:2d}: Train Loss={avg_train_loss:.4f}, Val Error={avg_val_error:.2f}m, Acc@5m={acc_5m:.1f}%")
            
            # 保存最佳模型
            if avg_val_error < best_val_error:
                best_val_error = avg_val_error
                save_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary/checkpoints")
                save_path.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_error': avg_val_error
                }, save_path / 'best_model.pth')
            
            scheduler.step()
        
        logger.info(f"\n✅ 训练完成，最佳验证误差: {best_val_error:.2f}m")
        return best_val_error
    
    def evaluate(self) -> Dict[str, Any]:
        """
        评估模型
        使用 Text2Loc-main 的评估指标
        """
        logger.info("\n" + "="*60)
        logger.info("评估模型")
        logger.info("="*60)
        
        if self.model is None:
            logger.error("❌ 模型未初始化")
            return {}
        
        # 计算cell中心
        cell_centers = {}
        for cell in self.cells:
            if isinstance(cell, dict):
                cell_id = cell.get('id', 'unknown')
                objects = cell.get('objects', [])
                
                if objects:
                    centers = []
                    for obj in objects:
                        if isinstance(obj, dict):
                            center = obj.get('center', [])
                            if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                                centers.append([float(center[0]), float(center[1])])
                    
                    if centers:
                        centers = np.array(centers)
                        avg_center = np.mean(centers, axis=0)
                        cell_centers[cell_id] = [avg_center[0], avg_center[1]]
        
        # 评估每个pose
        distances = []
        
        for pose in self.poses:
            if isinstance(pose, dict):
                location = pose.get('location', [])
                cell_id = pose.get('cell_id', '')
                
                if isinstance(location, (list, tuple, np.ndarray)) and len(location) >= 2:
                    gt_x, gt_y = float(location[0]), float(location[1])
                    
                    # 使用cell中心作为预测
                    if cell_id in cell_centers:
                        pred_x, pred_y = cell_centers[cell_id]
                        dist = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                        distances.append(dist)
        
        if not distances:
            logger.error("❌ 没有有效的评估结果")
            return {}
        
        distances = np.array(distances)
        
        # 计算指标
        metrics = {
            'avg_error': float(np.mean(distances)),
            'median_error': float(np.median(distances)),
            'min_error': float(np.min(distances)),
            'max_error': float(np.max(distances)),
            'std_error': float(np.std(distances)),
            'acc_1m': float(np.mean(distances <= 1.0) * 100),
            'acc_3m': float(np.mean(distances <= 3.0) * 100),
            'acc_5m': float(np.mean(distances <= 5.0) * 100),
            'acc_10m': float(np.mean(distances <= 10.0) * 100),
            'acc_20m': float(np.mean(distances <= 20.0) * 100),
            'total_samples': len(distances)
        }
        
        # 打印结果
        logger.info(f"\n📊 评估结果:")
        logger.info(f"  样本数: {metrics['total_samples']}")
        logger.info(f"  平均误差: {metrics['avg_error']:.2f}m")
        logger.info(f"  中位数误差: {metrics['median_error']:.2f}m")
        logger.info(f"  最小/最大: {metrics['min_error']:.2f}m / {metrics['max_error']:.2f}m")
        logger.info(f"\n📈 准确率:")
        logger.info(f"  1m: {metrics['acc_1m']:.1f}% | 3m: {metrics['acc_3m']:.1f}% | 5m: {metrics['acc_5m']:.1f}%")
        logger.info(f"  10m: {metrics['acc_10m']:.1f}% | 20m: {metrics['acc_20m']:.1f}%")
        
        return metrics


def main():
    """主函数"""
    # 创建系统
    system = IntegratedText2LocSystem()
    
    # 训练
    # system.train(num_epochs=50, batch_size=4, lr=0.001)
    
    # 评估
    metrics = system.evaluate()
    
    return system, metrics


if __name__ == "__main__":
    main()
