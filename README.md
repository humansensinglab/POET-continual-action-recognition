# POET: Prompt Offset Tuning for Continual Human Action Adaptation

### ECCV 2024, Oral Presentation | [Project Page](https://humansensinglab.github.io/POET-continual-action-recognition/)

*Authors: Prachi Garg, K J Joseph, Vineeth N Balasubramanian, Necati Cihan Camgoz, Chengde Wan, Kenrick Kin, Weiguang Si, Shugao Ma, and Fernando De La Torre*

![method_poet](https://github.com/user-attachments/assets/41c9716f-ee8e-47c2-9cb0-08afc1231f5e)

## Abstract
POET enables users to personalize their experience by adding new action classes efficiently and continually whenever they want.

We demonstrate the efficacy of prompt tuning a significantly lightweight backbone, pretrained exclusively on the base class data. We propose a novel spatio-temporal learnable prompt offset tuning approach, and are the first to apply such prompt tuning to Graph Neural Networks.

We contribute two new benchmarks for our new problem setting in human action recognition: (i) NTU RGB+D dataset for activity recognition, and (ii) SHREC-2017 dataset for hand gesture recognition. 

## :rocket: **Release Overview and Updates**
:white_large_square: Code for Gesture Recognition benchmark on SHREC 2017, DG-STA graph transformer backbone. 

:white_large_square: Plan to release all 10+1 sets of few-shots for full reproducibility soon.

:white_check_mark: Released POET training and evaluation code for our Activity Recognition benchmark on NTU RGB+D dataset. We use the CTR-GCN backbone. 

:white_check_mark: Additionally, this release includes (i) the base step model checkpoints, (ii) a few-shot data file. 

:pushpin: Note, additional code for adaptation of various baselines and ablations can be made available upon request. 

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
```

## Dataset Preparation

1. We had downloaded the NTU RGB+D 60 dataset and preprocessed it following the instructions in the original CTR-GCN repository. Sample few-shot data file is [here](https://uillinoisedu-my.sharepoint.com/:u:/g/personal/prachig3_illinois_edu/ERT-y01R2YFGtkzfWiC5jxUBJtBgaffAzBVm0ntH2fNpLQ?e=bXEoKs).

2. Provide path to data files inside [`temp_24nov.yaml`](https://github.com/humansensinglab/POET-continual-action-recognition/blob/main/config/nturgbd-cross-subject/temp_24nov.yaml) -> feeder -> `data_path` and `few_shot_data_file` variables. The preprocessed data should be organized as:

## Training

The `POET_final_10run.sh` script performs incremental learning over 4 steps:
- Step 1: Classes 40-45
- Step 2: Classes 45-50
- Step 3: Classes 50-55
- Step 4: Classes 55-60

To run training:

```bash
# Run for a specific few-shot data file
./POET_train.sh 1  # For set1

# Run for multiple sets
./POET_train.sh 1 2 3 4 5 6 7 8 9 10
```

This script will:
1. Train on each incremental step
2. Evaluate performance: (A) average of all classes; (B) old-only average class accuracy; (C) new-only average class accuracy; (D) HM of Old and New. 
3. Save model checkpoints and evaluation metrics

To run only evaluation:

```bash
# Run for a specific few-shot data file
./POET_eval.sh 1  # For set1

# Run for multiple sets
./POET_eval.sh 1 2 3 4 5 6 7 8 9 10
```
The per-run performance is in [this file](https://github.com/humansensinglab/POET-continual-action-recognition/blob/main/POET_NTU_CTRGCN_activity_results.pdf) for reproducibility and comparison. 

## Key Parameters

- `--k_shot`: Number of samples per class for few-shot learning (default: 5)
- `--prompt_layer`: Which layer to add prompts (default: 1) 
- `--device`: GPU device ID to use
- `--save_name_args`: Experiment name for saving results
- `--prompt_sim_reg`: Enable prompt similarity regularization
- `--classifier_average_init`: Initialize new classifier weights as average of old ones

Results are saved in:
```
work_dir/ntu60/csub/ctrgcn_prompt/
├── checkpoints/
├── logs/
├── plots/
└── results.csv
```

## Acknowledgements 
We thank authors of [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) and [Learning to Prompt for Continual Learning](https://github.com/JH-LEE-KR/l2p-pytorch](https://github.com/google-research/l2p ) and their [Pytorch reimplementation](https://github.com/JH-LEE-KR/l2p-pytorch), their code was useful starting point for our project. 

## Citation

If you find our work useful for your project, please consider citing our work:
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
