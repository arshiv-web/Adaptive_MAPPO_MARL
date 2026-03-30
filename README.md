# Multi-Agent RL Reproducibility

This repository contains our **Phase-1 reproducibility study** for Multi-Agent Reinforcement Learning using PPO-based methods.

We build on top of the official BenchMARL repository and reproduce results for:

- Algorithms: MAPPO, IPPO  
- Environments:
  - VMAS: Navigation, Balance  
  - PettingZoo: Multiwalker, Simple World Comm  
- Seeds: 0, 1, 2  
- Training Budget: 2M frames per run  

---

## Setup

### 1. Create environment

conda create -n benchmarl python=3.10  
conda activate benchmarl  

---

### 2. Clone BenchMARL

git clone https://github.com/facebookresearch/BenchMARL.git  

You could also just use are repository as a starting point.

Then,

pip install -e BenchMARL  

---

### 3. Install Environments

pip install vmas  
pip install "pettingzoo[all]"  

---

### 4. If required, install exact requirements 

pip install -r 567_phase1_requirements.txt  

---

### 5. (Optional) WandB login

wandb login  

Runs are configured in offline mode by default.

---

## Running Experiments

bash submit_2M.sh  

This launches:
4 tasks × 2 algorithms × 3 seeds = 24 runs


To sync results online to wandb, 

find /runs -type d -name "offline-run-*" -exec wandb sync -e $WANDB_ENTITY -p benchmarl_phase1 {} +

---

## Evaluation

python evaluate_results.py  

This generates:
- summary.csv
- aggregate.csv
- rollout visualizations (Fig 2 in report)

For plots (Fig 1 in report) we use wandb visualisations.

Results can be check in /eval_results for each run.

---

## Experimental Setup

- Learning rate: 3e-4  
- Entropy coefficient: 0.01  
- Frames: 2M  
- Evaluation: every 120k frames  

---

## Notes

- No environment-specific tuning was applied  
- Shared configuration across all tasks  

---

## Contact

For any issues reach out to,

arshiv@umich.edu  
mseopark@umich.edu  
zhuoyuc@umich.edu  

---

## Acknowledgements

BenchMARL (Facebook Research): https://github.com/facebookresearch/BenchMARL
