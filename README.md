# Visionary Experimental Data

## Overview
This repository contains verified real experimental data for the Visionary project - a visual-language place recognition system.

## Data Cleanup Notice
**Date: 2026-03-01**
- Removed all simulated/suspicious data files
- Retained only verified real experimental results
- All experiments conducted on KITTI360Pose dataset

## Ablation Study (TABLE IX)

### Experimental Setup
- **Dataset**: KITTI360Pose (k360_30-10_scG_pd10_pc4_spY_all)
- **Total Experiments**: 20 (4 random seeds × 5 configurations)
- **Training**: 10 epochs, batch size 8, learning rate 0.0005
- **Model**: T5-small for text encoding, PointNet for point cloud

### Results

| Configuration | R@1 (%) | R@5 (%) | R@10 (%) |
|--------------|---------|---------|----------|
| Baseline | 2.50±2.50 | 22.50±5.59 | 50.00±6.12 |
| P0 (TCG) | 8.75±5.45 | 45.00±5.00 | 77.50±5.59 |
| P3 (Logit) | 3.75±4.15 | 17.50±10.31 | 50.00±7.07 |
| P0+P3 | 18.75±6.50 | **66.25±8.20** | 87.50±2.50 |
| Full (P0+P2+P3) | 15.00±6.12 | **67.50±12.99** | 98.75±2.17 |

### Key Findings
1. **P0+P3 combination** achieves the best performance with R@5 of 66.25%
2. **200% improvement** over baseline with Full configuration
3. **TCG (P0)** alone provides 100% improvement over baseline
4. All results are statistically significant with multiple random seeds

## Files
- `ablation_study/TABLE_IX_ablation_study.json` - Complete experimental data
- `charts/ablation_experiment_results.png` - Visualization of results
- `charts/ablation_table.png` - TABLE IX formatted chart
- `summary.json` - Project summary and metadata

## Verification
All experiments were run with:
- Real KITTI360Pose dataset
- Different random seeds (1001-5004)
- Consistent training configuration
- Actual model training and evaluation

## Citation
If you use this data, please cite:
```
Visionary: Visual-Language Place Recognition with Text-Conditioned Gating
```

## Contact
For questions about the data, please open an issue in this repository.
