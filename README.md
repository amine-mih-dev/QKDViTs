# QKDT: Code Repository for Journal Reproducibility

This repository contains the implementation used for model training, knowledge distillation, quantization, and evaluation in our plant disease image classification study.

It is prepared as a code-availability and reproducibility companion for journal editorial review and public GitHub release.

## Manuscript Information

- Manuscript title: Knowledge-Distilled and Quantized Vision Transformers for Data-Efficient Tomato Disease Analysis
- Journal: The Visual Computer
<!-- - Submission/revision ID: [insert manuscript ID] -->
- Corresponding author: mihammed.mihoubi@univ-bejaia.dz

## Code Availability Statement

All source code required to reproduce the reported computational workflow is provided in this repository.

## Repository Scope

- End-to-end training pipeline for teacher and student models.
- Distillation pipeline across multiple teacher-student combinations.
- Post-training dynamic quantization evaluation (FP32 vs INT8).
- Reporting scripts that merge training and quantization outputs into publication-ready summaries and figures.

## Repository Structure

```text
.
|-- src/
|   |-- analysis/     # fold-level analysis scripts 
|   |-- core/         # shared metrics, plotting, merge, and data-loading utilities
|   |-- models/       # architecture inspection helpers
|   `-- train/        # training, distillation, and quantization pipelines
|-- report/           # report-generation scripts (tables/figures)
|-- utils/            # shared path/config utilities
|-- requirements.txt
`-- README.md
```

## Installation

### 1) Create environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```



### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Reproducibility Workflow

The intended execution order is:

1. Run training and distillation.
2. Run quantization on a completed timestamp.
3. Merge FP32 and INT8 outputs for final tables/figures.
4. Optionally aggregate AUC outputs across datasets.

## Data Layout Contract

The training and quantization entrypoints accept either explicit split paths or a dataset root.

Supported root layouts:

1. `output/train`, `output/val`, `output/test`
2. `train`, `val`, `test`
3. Taiwan-specific order in root: `train`, `test`, `val`


## Reproduce the Main Pipeline

### Step A: Training + Distillation

```bash
python -m src.train.pl_distill_datasets \
	--dataset-name zpdd \
	--dataset-root <path_to_dataset_root>
```

Alternative with explicit split directories:

```bash
python -m src.train.pl_distill_datasets \
	--dataset-name zpdd \
	--train-dir <train_dir> \
	--val-dir <val_dir> \
	--test-dir <test_dir>
```

### Step B: Quantization Sweep

```bash
python -m src.train.quant \
	--dataset-name zpdd \
	--date <run_timestamp> \
	--dataset-root <path_to_dataset_root>
```

### Step C: Merge Final Results

```bash
python report/result_final.py --dataset zpdd --date <run_timestamp>
```

### Step D: Multi-dataset AUC + Consistency Aggregation

```bash
python report/aucviz.py \
	--datasets taiwan:<run_timestamp>,plantvillage:<run_timestamp>,zpdd:<run_timestamp> \
	--final-results-dir final_results \
	--output-dir results
```

Note: To generate teacher/student consistency plots, run Step C for each dataset/date first so
`final_results/<dataset>/<timestamp>/final_results.csv` exists.

## Key Generated Artifacts

For each dataset and run timestamp, the main outputs are:

- Experiment metrics:
	`results/<dataset>/<timestamp>/experiment_results_get_model_by_name<timestamp>.csv`
- Quantization comparison:
	`qres/<dataset>/<timestamp>/results_quantization_comparison.csv`
- AUC summary:
	`aucs/<dataset>/<timestamp>/results_micro_average_auc.csv`
	`aucs/<dataset>/<timestamp>/results_micro_average_auc.pkl`
- Final merged report:
	`final_results/<dataset>/<timestamp>/final_results.csv`
- ROC/summary visualizations:
	`visualizations/<dataset>/<timestamp>/AUCs/`

Cross-dataset aggregation outputs (from `report/aucviz.py`) include:

- Merged micro-AUC table:
	`results/merged_datasets_micro_auc.csv`
- Long-format micro-AUC table:
	`results/micro_auc_long.csv`
- Micro-AUC distribution plot:
	`results/micro_auc_boxplot.png`
- Merged final-results table:
	`results/merged_datasets_final_results.csv`
- Teacher strategy consistency plot:
	`results/teacher_consistency.png`
- Student architecture consistency plot:
	`results/student_architecture_consistency.png`

## Additional Entrypoints

- Fold analysis (CSV):

```bash
python src/analysis/fold_exp.py --input-dir ./resnet50fr/ --output-dir report/outputs/analysis
```

- Fold analysis (PKL):

```bash
python src/analysis/fold_exp_pkl.py --input-dir ./hybridpkl05lr/ --output-dir report/outputs/analysis
```

- Model architecture inspection:

```bash
python src/models/model_arch.py --show-summary
```

## Reproducibility Notes

- Random seed is set through the training CLI (`--seed`, default `42`).
- Hardware-dependent performance may vary between CPU and GPU runs.
- Dataset paths in legacy defaults are machine-specific; override with CLI flags for portability.


## Data Availability

This repository contains code and experiment orchestration.
Dataset files are publically available.

## Citation

If you use this code, please cite the associated manuscript:

<!-- ```bibtex
@article{qkdt2026,
	title   = {<insert paper title>},
	author  = {<insert authors>},
	journal = {<insert journal>},
	year    = {2026},
	doi     = {<insert doi>}
}
``` -->


## Contact

- Correspondence: mohammed.mihoubi@univ-bejaia.dz
