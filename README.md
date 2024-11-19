# POET: Prompt Offset Tuning for Continual Human Action Adaptation

# ECCV 2024 (Oral Presentation)

### Authors: Prachi Garg, K J Joseph, Vineeth N Balasubramanian, Necati Cihan Camgoz, Chengde Wan, Kenrick Kin, Weiguang Si, Shugao Ma, and Fernando De La Torre

## Installation

1. Create a new conda environment:
```bash
conda create -n poet python=3.8
conda activate poet
```

2. Install PyTorch and CUDA toolkit:
```bash 
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

4. Clone the repository:
```bash
git clone <repository-url>
cd Cleaned_POET_final
```

## Dataset Preparation

1. Download the NTU RGB+D 60 dataset and preprocess it following the instructions in the original CTR-GCN repository.

2. The preprocessed data should be organized as:
```
data/
├── prachi_data/
│   └── ntu/
│       └── NTU60_CS.npz
└── few_shot_files/
    └── NTU60_5shots_set*.npz  # Few-shot data files
```

## Training

The `POET_final_10run.sh` script performs incremental learning over 4 steps:
- Base training: 40 classes
- Step 1: Classes 40-45  
- Step 2: Classes 45-50
- Step 3: Classes 50-55
- Step 4: Classes 55-60

To run training:

```bash
# Run for a specific few-shot data file
bash POET_final_10run.sh 1  # For set1

# Run for multiple sets
for i in {1..10}; do
  bash POET_final_10run.sh $i
done
```

The script will:
1. Train on each incremental step
2. Evaluate performance on old and new classes
3. Save model checkpoints and evaluation metrics
4. Generate visualization plots

## Key Parameters

- `--k_shot`: Number of samples per class for few-shot learning (default: 5)
- `--prompt_layer`: Which layer to add prompts (default: 1) 
- `--device`: GPU device ID to use
- `--save_name_args`: Experiment name for saving results
- `--prompt_sim_reg`: Enable prompt similarity regularization
- `--classifier_average_init`: Initialize new classifier weights as average of old ones

## Output Files

The training will generate:
- Model checkpoints (.pt files)
- Training logs
- Evaluation metrics in CSV files
- Visualization plots for:
  - Prompt selection frequencies
  - Logit distributions
  - Prompt gradients

Results are saved in:
```
work_dir/ntu60/csub/ctrgcn_prompt/
├── checkpoints/
├── logs/
├── plots/
└── results.csv
```

## Citation

If you use this code, please cite:
```
@inproceedings{garg2024poet,
  title={POET: Prompt Offset Tuning for Continual Human Action Adaptation},
  author={Garg, Prachi and Joseph, KJ and Balasubramanian, Vineeth N and Camgoz, Necati Cihan and Wan, Chengde and Kin, Kenrick and Si, Weiguang and Ma, Shugao and De La Torre, Fernando},
  booktitle={European Conference on Computer Vision},
  pages={436--455},
  year={2024},
  organization={Springer}
}
```