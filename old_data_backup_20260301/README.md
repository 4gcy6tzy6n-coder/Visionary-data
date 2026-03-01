# Visionary Data - Experimental Data and Scripts

## Overview

This repository contains experimental data, evaluation scripts, and analysis tools for the Visionary 3D point cloud localization project. It complements the main [Visionary core repository](https://github.com/4gcy6tzy6n-coder/Visionary) by providing comprehensive experimental workflows and result analysis tools.

## Contents

### Training Scripts

- `train_visionary_enhanced.py` - Enhanced model training
- `train_visionary_v2.py` - V2 model training
- `train_on_semantics.py` - Semantic-aware training
- `train_on_large_dataset.py` - Large-scale dataset training

### Evaluation & Comparison

- `comprehensive_experiments.py` - Complete experimental pipeline
- `ablation_study.py` - Component ablation analysis
- `robustness_experiment.py` - Robustness testing
- `fair_comparison.py` - Fair comparison with baselines
- `large_scale_comparison.py` - Large-scale evaluation

### Data Processing

- `process_2d_semantics_incremental.py` - 2D semantic data processing
- `process_3d_semantics.py` - 3D semantic data processing
- `reprocess_50m_cells.py` - Cell reprocessing with 50m granularity
- `repair_dataset.py` - Dataset repair and validation

### Analysis & Visualization

- `root_cause_analysis.py` - Root cause analysis tools
- `deep_analysis.py` - Deep performance analysis
- `plot_comparison_charts.py` - Comparison chart generation
- `plot_english_charts.py` - English visualization charts
- `advantage_analysis.py` - Advantage analysis

## Usage

### Running Experiments

```bash
# Comprehensive experiments
python comprehensive_experiments.py

# Ablation study
python ablation_study.py

# Robustness testing
python robustness_experiment.py
```

### Generating Reports

```bash
# Generate comparison charts
python plot_comparison_charts.py

# Generate English charts
python plot_english_charts.py
```

### Data Processing

```bash
# Process 2D semantic data
python process_2d_semantics_incremental.py

# Reprocess cells
python reprocess_50m_cells.py
```

## Configuration

- `config.yaml` - Main configuration file
- `requirements.txt` - Python dependencies

## Charts and Results

The `charts/` directory contains:
- Ablation study results
- Component contribution analysis
- Robustness test results
- Efficiency comparisons
- Summary radar charts

## Related Repositories

- [Visionary Core](https://github.com/4gcy6tzy6n-coder/Visionary) - Main algorithm implementation
- [Text2Loc Original](https://github.com/Yan-Xia/Text2Loc) - Original CVPR 2024 implementation

## License

For research purposes only.

## Notes

- Large training data files (datasets, checkpoints) are not included in this repository
- Use Git LFS for version control of large files if needed
- All experimental results are documented in JSON format for reproducibility
