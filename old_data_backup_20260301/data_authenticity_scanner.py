#!/usr/bin/env python3
"""
数据真实性全面扫描工具
检测模拟数据、异常值、不合理模式
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

class DataAuthenticityScanner:
    """数据真实性扫描器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.results = {
            'verified_real': [],
            'suspicious': [],
            'simulated': [],
            'errors': []
        }
        
    def scan_all_json_files(self):
        """扫描所有JSON文件"""
        json_files = list(self.data_dir.glob('*.json'))
        print(f"🔍 发现 {len(json_files)} 个JSON文件\n")
        
        for json_file in sorted(json_files):
            # 跳过已标记的文件
            if any(x in json_file.name for x in ['INVALID', 'MOCK', 'ESTIMATED', 'SIMULATED']):
                continue
                
            print(f"📄 检查: {json_file.name}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                status, reasons = self.analyze_data_authenticity(data, json_file.name)
                
                if status == 'REAL':
                    self.results['verified_real'].append({
                        'file': json_file.name,
                        'reasons': reasons
                    })
                    print(f"   ✅ 真实数据 - {reasons[0] if reasons else '通过所有检查'}")
                elif status == 'SUSPICIOUS':
                    self.results['suspicious'].append({
                        'file': json_file.name,
                        'reasons': reasons
                    })
                    print(f"   ⚠️  存疑 - {'; '.join(reasons)}")
                elif status == 'SIMULATED':
                    self.results['simulated'].append({
                        'file': json_file.name,
                        'reasons': reasons
                    })
                    print(f"   ❌ 模拟数据 - {'; '.join(reasons)}")
                    
            except Exception as e:
                self.results['errors'].append({
                    'file': json_file.name,
                    'error': str(e)
                })
                print(f"   💥 错误: {e}")
        
        print()
        
    def analyze_data_authenticity(self, data: Any, filename: str) -> Tuple[str, List[str]]:
        """分析数据真实性
        
        Returns:
            (status, reasons): status可以是 'REAL', 'SUSPICIOUS', 'SIMULATED'
        """
        reasons = []
        
        # 1. 检查是否包含模拟关键词
        data_str = json.dumps(data, ensure_ascii=False).lower()
        simulated_keywords = ['simulate', 'mock', 'fake', 'artificial', 'synthetic', 'generated', 'demo', 'test_data']
        if any(kw in data_str for kw in simulated_keywords):
            return 'SIMULATED', [f'包含模拟关键词: {kw}' for kw in simulated_keywords if kw in data_str]
        
        # 2. 检查数值异常
        numeric_values = self.extract_numeric_values(data)
        
        if numeric_values:
            # 检查是否所有值都相同（异常）
            if len(set([round(v, 6) for v in numeric_values])) == 1:
                return 'SUSPICIOUS', ['所有数值完全相同，可能是模拟数据']
            
            # 检查是否有过多整数（可能是人为构造）
            integers = [v for v in numeric_values if v == int(v)]
            if len(integers) / len(numeric_values) > 0.8 and len(numeric_values) > 10:
                return 'SUSPICIOUS', [f'{len(integers)}/{len(numeric_values)} 是整数，比例过高']
            
            # 检查是否有过多的0或100（可能是理论值）
            extreme_values = [v for v in numeric_values if v in [0, 100, 0.0, 100.0]]
            if len(extreme_values) / len(numeric_values) > 0.5 and len(numeric_values) > 10:
                return 'SUSPICIOUS', [f'{len(extreme_values)}/{len(numeric_values)} 是0或100，可能是理论值']
        
        # 3. 检查样本量
        sample_count = self.estimate_sample_count(data)
        if sample_count is not None:
            if sample_count < 10:
                return 'SUSPICIOUS', [f'样本量过小: {sample_count}']
        
        # 4. 检查时间戳
        if 'timestamp' in data_str or 'time' in data_str:
            reasons.append('包含时间戳')
        
        # 5. 检查统计指标的合理性
        if self.has_reasonable_statistics(data):
            reasons.append('统计指标合理')
        
        # 6. 检查是否有合理的波动
        if numeric_values and len(numeric_values) > 5:
            std = np.std(numeric_values)
            if std > 0.001:  # 有合理的波动
                reasons.append(f'数值有合理波动 (std={std:.4f})')
        
        return 'REAL', reasons if reasons else ['通过真实性检查']
    
    def extract_numeric_values(self, data: Any, values: List[float] = None) -> List[float]:
        """递归提取所有数值"""
        if values is None:
            values = []
            
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            values.append(float(data))
        elif isinstance(data, dict):
            for v in data.values():
                self.extract_numeric_values(v, values)
        elif isinstance(data, list):
            for item in data:
                self.extract_numeric_values(item, values)
                
        return values
    
    def estimate_sample_count(self, data: Any) -> int:
        """估计样本数量"""
        if isinstance(data, dict):
            # 检查常见的样本数量字段
            for key in ['count', 'total', 'samples', 'n_samples', 'sample_count']:
                if key in data and isinstance(data[key], (int, float)):
                    return int(data[key])
            
            # 检查results数组长度
            if 'results' in data and isinstance(data['results'], list):
                return len(data['results'])
            
            # 递归检查
            for v in data.values():
                count = self.estimate_sample_count(v)
                if count is not None:
                    return count
        elif isinstance(data, list):
            return len(data)
        
        return None
    
    def has_reasonable_statistics(self, data: Any) -> bool:
        """检查是否有合理的统计指标"""
        data_str = json.dumps(data, ensure_ascii=False).lower()
        
        # 检查是否包含常见的统计指标
        stat_indicators = ['mean', 'median', 'std', 'accuracy', 'error', 'success_rate']
        return any(ind in data_str for ind in stat_indicators)
    
    def generate_report(self):
        """生成扫描报告"""
        print("="*80)
        print("📊 数据真实性扫描报告")
        print("="*80)
        
        print(f"\n✅ 已验证的真实数据 ({len(self.results['verified_real'])} 个):")
        for item in self.results['verified_real']:
            print(f"   • {item['file']}")
            for reason in item['reasons'][:2]:
                print(f"     - {reason}")
        
        print(f"\n⚠️  存疑数据 ({len(self.results['suspicious'])} 个):")
        for item in self.results['suspicious']:
            print(f"   • {item['file']}")
            for reason in item['reasons']:
                print(f"     - {reason}")
        
        print(f"\n❌ 模拟数据 ({len(self.results['simulated'])} 个):")
        for item in self.results['simulated']:
            print(f"   • {item['file']}")
            for reason in item['reasons']:
                print(f"     - {reason}")
        
        print(f"\n💥 读取错误 ({len(self.results['errors'])} 个):")
        for item in self.results['errors']:
            print(f"   • {item['file']}: {item['error']}")
        
        print("\n" + "="*80)
        
        # 保存详细报告
        report_file = self.data_dir / 'DATA_AUTHENTICITY_SCAN_REPORT.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n📄 详细报告已保存: {report_file.name}")

if __name__ == '__main__':
    scanner = DataAuthenticityScanner('/Users/yaoyingliang/visionary/Visionary-data-main')
    scanner.scan_all_json_files()
    scanner.generate_report()
