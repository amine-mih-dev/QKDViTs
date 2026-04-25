# Report Scripts

This directory contains phase-2 report entry scripts.

## Scripts

- `aucviz.py`: Aggregates micro-average AUC results across datasets, writes merged CSV outputs, and generates teacher/student consistency plots when per-dataset final reports are available.
- `result_final.py`: Merges experiment and quantization metrics and generates comparison plots.

## Usage

```bash
python report/result_final.py --dataset zpdd --date 2025-06-30-15-05
python report/aucviz.py --datasets taiwan:2025-06-23-12-54,plantvillage:2025-06-12-22-27,zpdd:2025-06-30-15-05 --final-results-dir final_results --output-dir results
```

`aucviz.py` outputs:

- `merged_datasets_micro_auc.csv`
- `micro_auc_long.csv`
- `micro_auc_boxplot.png`
- `merged_datasets_final_results.csv` (when final reports are found)
- `teacher_consistency.png` (when final reports are found)
- `student_architecture_consistency.png` (when final reports are found)
