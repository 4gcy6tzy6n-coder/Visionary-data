#!/usr/bin/env python3
"""
检查训练数据的实际内容，确认是否使用了语义数据
"""

import pickle
import json
from pathlib import Path

def check_processed_data():
    """检查处理后的数据"""
    data_path = Path("/Users/yaoyingliang/Desktop/Text2Loc-main/data/k360_from_semantics_full")
    
    print("="*80)
    print("检查处理后的训练数据")
    print("="*80)
    
    # 加载cells
    with open(data_path / "cells" / "cells.pkl", 'rb') as f:
        cells = pickle.load(f)
    
    # 加载poses
    with open(data_path / "poses" / "poses.pkl", 'rb') as f:
        poses = pickle.load(f)
    
    print(f"\n总cells: {len(cells)}")
    print(f"总poses: {len(poses)}")
    
    # 检查第一个cell的内容
    print("\n" + "="*80)
    print("检查第一个cell的内容")
    print("="*80)
    
    if cells:
        first_cell = cells[0]
        print(f"\nCell ID: {first_cell.get('id')}")
        print(f"Scene: {first_cell.get('scene')}")
        print(f"Center: {first_cell.get('center')}")
        
        objects = first_cell.get('objects', [])
        print(f"\nObjects数量: {len(objects)}")
        
        if objects:
            print("\n前3个objects:")
            for i, obj in enumerate(objects[:3]):
                print(f"  {i+1}. Label: {obj.get('label')}")
                print(f"     Center: {obj.get('center')}")
                print(f"     Semantic ID: {obj.get('semantic_id')}")
                print(f"     Keys: {list(obj.keys())}")
    
    # 检查第一个pose
    print("\n" + "="*80)
    print("检查第一个pose的内容")
    print("="*80)
    
    if poses:
        first_pose = poses[0]
        print(f"\nCell ID: {first_pose.get('cell_id')}")
        print(f"Description: {first_pose.get('description')}")
        print(f"Location: {first_pose.get('location')}")
        print(f"Keys: {list(first_pose.keys())}")
    
    # 统计语义标签分布
    print("\n" + "="*80)
    print("统计语义标签分布")
    print("="*80)
    
    from collections import Counter
    all_labels = []
    
    for cell in cells:
        for obj in cell.get('objects', []):
            label = obj.get('label', 'unknown')
            all_labels.append(label)
    
    label_counts = Counter(all_labels)
    print(f"\n总objects: {len(all_labels)}")
    print(f"唯一labels: {len(label_counts)}")
    print("\nTop 10 labels:")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count}")
    
    # 检查数据来源
    print("\n" + "="*80)
    print("验证数据来源")
    print("="*80)
    
    # 检查是否有semantic_id字段
    has_semantic = any(
        'semantic_id' in obj 
        for cell in cells 
        for obj in cell.get('objects', [])
    )
    
    print(f"\n是否包含semantic_id字段: {has_semantic}")
    
    if has_semantic:
        print("✅ 数据包含语义信息（来自3D语义点云）")
    else:
        print("⚠️ 数据可能不包含语义信息")
    
    # 检查stats.json
    stats_path = data_path / "stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"\nStats: {json.dumps(stats, indent=2)}")

if __name__ == '__main__':
    check_processed_data()
