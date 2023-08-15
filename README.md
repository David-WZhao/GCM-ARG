# GCM-ARG

This is the code of the gating-controlled mechanism network for ARG (GCM-ARG) properties prediction in our manuscript (Subtask-aware Representation Learning for Predicting Antibiotic Resistance Gene Properties via Gating-controlled Mechanism)

### Requirement
- python 3.8
- torch == 1.12.1
- scikit-learn == 1.2.2
- numpy == 1.24.2
- pandas == 1.3.4

### Data
The process of collecting data is described in our manuscript and metadata can be accessed upon request.
You need first unzip all compressed files under "data", and put the files in the same directory.

### How to run the code?
1. Data preprocessing: "arg_v5.fasta" file is the original data set file, "fasta_process.ipynb" file is used on the original data set file to get the processed dataset.
Run "data_divide.py" to produce splitted dataset.

2. Run the prediction model: Put the "data_loader.py", "modules.py", "run.py", "utils.py" and directory "data" in the same directory, and run "python run.py --device "gpu" --batch_size 64 --epoch 10 --K 1 --n_experts 3  --n_experts_share 3  --expert_dim 1024" in the command line.
