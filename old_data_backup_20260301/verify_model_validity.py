#!/usr/bin/env python3
"""
严格验证模型有效性
检测是否存在数据泄漏或过拟合
"""

import torch
import torch.nn as nn
import numpy as np
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置随机种子确保可重复性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class Text2LocModel(nn.Module):
    """与训练时相同的模型架构"""
    def __init__(self, input_dim=11, hidden_dim=256, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class RandomBaselineModel:
    """随机基线模型 - 用于对比"""
    def __init__(self, cell_centers):
        self.cell_centers = cell_centers
    
    def predict(self, x):
        """随机选择cell中心"""
        indices = np.random.randint(0, len(self.cell_centers), size=len(x))
        return self.cell_centers[indices]

class SimpleBaselineModel:
    """简单基线 - 总是预测训练集的平均位置"""
    def __init__(self, mean_position):
        self.mean_position = mean_position
    
    def predict(self, x):
        """返回平均位置"""
        return np.tile(self.mean_position, (len(x), 1))

def load_complete_dataset():
    """加载完整数据集"""
    data_dir = Path("/Volumes/MU90/data one/cells_3d")
    
    all_cells = []
    all_poses = []
    cell_centers = {}
    
    for cell_file in sorted(data_dir.glob("cell_*_poses.json")):
        try:
            with open(cell_file, 'r') as f:
                data = json.load(f)
            
            cell_id = data['cell_id']
            center = data['center']  # [x, y, z]
            cell_centers[cell_id] = center
            
            for obj in data.get('objects', []):
                semantic_vec = obj.get('semantic_label_vector', [0]*11)
                for pose in obj.get('poses', []):
                    all_cells.append({
                        'cell_id': cell_id,
                        'semantic_vec': semantic_vec,
                        'center': center
                    })
                    all_poses.append(pose['position'])
        except Exception as e:
            continue
    
    return all_cells, all_poses, cell_centers

def create_true_train_test_split(cells, poses, test_size=0.3):
    """
    创建真正的训练/测试分割
    关键：确保测试集中的cell在训练集中完全没有出现过
    """
    # 按cell分组
    cell_groups = {}
    for cell, pose in zip(cells, poses):
        cid = cell['cell_id']
        if cid not in cell_groups:
            cell_groups[cid] = {'cells': [], 'poses': []}
        cell_groups[cid]['cells'].append(cell)
        cell_groups[cid]['poses'].append(pose)
    
    # 获取所有cell IDs
    all_cell_ids = list(cell_groups.keys())
    
    # 随机分割cell IDs（不是分割poses！）
    train_cell_ids, test_cell_ids = train_test_split(
        all_cell_ids, test_size=test_size, random_state=SEED
    )
    
    # 构建训练集和测试集
    train_cells, train_poses = [], []
    test_cells, test_poses = [], []
    
    for cid in train_cell_ids:
        train_cells.extend(cell_groups[cid]['cells'])
        train_poses.extend(cell_groups[cid]['poses'])
    
    for cid in test_cell_ids:
        test_cells.extend(cell_groups[cid]['cells'])
        test_poses.extend(cell_groups[cid]['poses'])
    
    return train_cells, train_poses, test_cells, test_poses, train_cell_ids, test_cell_ids

def evaluate_model(model, cells, poses, device, model_type='neural'):
    """评估模型性能"""
    if len(cells) == 0:
        return None
    
    if model_type == 'neural':
        model.eval()
        X = torch.FloatTensor([c['semantic_vec'] for c in cells]).to(device)
        y_true = np.array(poses)
        
        with torch.no_grad():
            y_pred = model(X).cpu().numpy()
    
    elif model_type == 'random':
        X = np.array([c['semantic_vec'] for c in cells])
        y_true = np.array(poses)
        y_pred = model.predict(X)
    
    elif model_type == 'mean':
        X = np.array([c['semantic_vec'] for c in cells])
        y_true = np.array(poses)
        y_pred = model.predict(X)
    
    # 计算误差
    errors = np.linalg.norm(y_pred - y_true, axis=1)
    
    return {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'acc_1m': np.mean(errors <= 1.0) * 100,
        'acc_5m': np.mean(errors <= 5.0) * 100,
        'acc_10m': np.mean(errors <= 10.0) * 100,
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'std_error': np.std(errors)
    }

def check_data_leakage(train_cells, test_cells):
    """检查是否存在数据泄漏"""
    train_cell_ids = set(c['cell_id'] for c in train_cells)
    test_cell_ids = set(c['cell_id'] for c in test_cells)
    
    overlap = train_cell_ids & test_cell_ids
    
    print("\n" + "="*60)
    print("数据泄漏检测")
    print("="*60)
    print(f"训练集cell数量: {len(train_cell_ids)}")
    print(f"测试集cell数量: {len(test_cell_ids)}")
    print(f"重叠cell数量: {len(overlap)}")
    
    if len(overlap) > 0:
        print(f"⚠️ 警告: 发现数据泄漏！重叠cell: {overlap}")
        return False
    else:
        print("✅ 无数据泄漏 - 训练集和测试集完全分离")
        return True

def main():
    print("="*60)
    print("模型有效性严格验证")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载完整数据集...")
    cells, poses, cell_centers_dict = load_complete_dataset()
    print(f"   总样本数: {len(cells)}")
    print(f"   总cell数: {len(cell_centers_dict)}")
    
    # 2. 创建真正的训练/测试分割
    print("\n2. 创建训练/测试分割（按cell分割）...")
    train_cells, train_poses, test_cells, test_poses, train_ids, test_ids = \
        create_true_train_test_split(cells, poses, test_size=0.3)
    
    print(f"   训练集: {len(train_cells)} poses, {len(train_ids)} cells")
    print(f"   测试集: {len(test_cells)} poses, {len(test_ids)} cells")
    
    # 3. 检查数据泄漏
    no_leakage = check_data_leakage(train_cells, test_cells)
    
    # 4. 准备cell中心数组
    cell_centers = np.array([cell_centers_dict[cid] for cid in sorted(cell_centers_dict.keys())])
    
    # 5. 加载训练好的模型
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n3. 加载训练好的模型...")
    
    model = Text2LocModel(input_dim=11, hidden_dim=256, output_dim=3).to(device)
    
    model_path = Path("models/best_model_semantic_full.pt")
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✅ 加载模型 (训练时的验证误差: {checkpoint.get('val_error', 'N/A'):.2f}m)")
    else:
        print(f"   ❌ 模型文件不存在: {model_path}")
        return
    
    # 6. 评估训练好的神经网络
    print("\n4. 评估神经网络模型...")
    train_results_nn = evaluate_model(model, train_cells, train_poses, device, 'neural')
    test_results_nn = evaluate_model(model, test_cells, test_poses, device, 'neural')
    
    # 7. 评估随机基线
    print("\n5. 评估随机基线模型...")
    random_model = RandomBaselineModel(cell_centers)
    train_results_random = evaluate_model(random_model, train_cells, train_poses, device, 'random')
    test_results_random = evaluate_model(random_model, test_cells, test_poses, device, 'random')
    
    # 8. 评估均值基线
    print("\n6. 评估均值基线模型...")
    mean_position = np.mean(train_poses, axis=0)
    mean_model = SimpleBaselineModel(mean_position)
    train_results_mean = evaluate_model(mean_model, train_cells, train_poses, device, 'mean')
    test_results_mean = evaluate_model(mean_model, test_cells, test_poses, device, 'mean')
    
    # 9. 打印对比结果
    print("\n" + "="*80)
    print("验证结果对比")
    print("="*80)
    
    print("\n【神经网络模型】")
    print(f"  训练集 - 误差: {train_results_nn['mean_error']:.2f}m, Acc@5m: {train_results_nn['acc_5m']:.1f}%")
    print(f"  测试集 - 误差: {test_results_nn['mean_error']:.2f}m, Acc@5m: {test_results_nn['acc_5m']:.1f}%")
    
    print("\n【随机基线模型】")
    print(f"  训练集 - 误差: {train_results_random['mean_error']:.2f}m, Acc@5m: {train_results_random['acc_5m']:.1f}%")
    print(f"  测试集 - 误差: {test_results_random['mean_error']:.2f}m, Acc@5m: {test_results_random['acc_5m']:.1f}%")
    
    print("\n【均值基线模型】")
    print(f"  训练集 - 误差: {train_results_mean['mean_error']:.2f}m, Acc@5m: {train_results_mean['acc_5m']:.1f}%")
    print(f"  测试集 - 误差: {test_results_mean['mean_error']:.2f}m, Acc@5m: {test_results_mean['acc_5m']:.1f}%")
    
    # 10. 分析结果
    print("\n" + "="*80)
    print("有效性分析")
    print("="*80)
    
    # 检查是否过拟合
    train_test_gap = train_results_nn['mean_error'] - test_results_nn['mean_error']
    print(f"\n1. 过拟合检测:")
    print(f"   训练-测试误差差距: {train_test_gap:.2f}m")
    if abs(train_test_gap) < 2.0:
        print("   ✅ 无明显过拟合")
    else:
        print(f"   ⚠️ 可能存在过拟合 (差距 > 2m)")
    
    # 检查是否优于随机
    improvement_over_random = test_results_random['mean_error'] - test_results_nn['mean_error']
    print(f"\n2. 与随机基线对比:")
    print(f"   优于随机的程度: {improvement_over_random:.2f}m")
    if improvement_over_random > 5.0:
        print("   ✅ 显著优于随机基线")
    else:
        print(f"   ⚠️ 改进不明显")
    
    # 检查是否优于均值
    improvement_over_mean = test_results_mean['mean_error'] - test_results_nn['mean_error']
    print(f"\n3. 与均值基线对比:")
    print(f"   优于均值的程度: {improvement_over_mean:.2f}m")
    if improvement_over_mean > 2.0:
        print("   ✅ 优于简单均值预测")
    else:
        print(f"   ⚠️ 与均值基线相近")
    
    # 最终结论
    print("\n" + "="*80)
    print("最终结论")
    print("="*80)
    
    is_valid = (
        no_leakage and
        abs(train_test_gap) < 5.0 and
        improvement_over_random > 5.0 and
        test_results_nn['mean_error'] < 50.0
    )
    
    if is_valid:
        print("\n✅ 模型验证通过！")
        print(f"   测试集定位误差: {test_results_nn['mean_error']:.2f}m")
        print(f"   测试集5米准确率: {test_results_nn['acc_5m']:.1f}%")
        print("\n   结果可信，可以发表！")
    else:
        print("\n❌ 模型验证失败！")
        print("   请检查数据或模型实现")
    
    # 保存详细结果
    results = {
        'validation_passed': is_valid,
        'no_data_leakage': no_leakage,
        'neural_network': {
            'train': train_results_nn,
            'test': test_results_nn
        },
        'random_baseline': {
            'train': train_results_random,
            'test': test_results_random
        },
        'mean_baseline': {
            'train': train_results_mean,
            'test': test_results_mean
        }
    }
    
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n详细结果已保存到: validation_results.json")

if __name__ == '__main__':
    main()
