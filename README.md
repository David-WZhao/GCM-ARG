# GCM-ARG

This is the code of the gating-controlled mechanism network for ARG(GCM-ARG) properties prediction in our manuscript (Subtask-aware Representation Learning for Predicting Antibiotic Resistance Gene Properties via Gating-controlled Mechanism)

### Requirement
- python 3.8
- torch == 1.12.1
- scikit-learn == 1.2.2
- numpy == 1.24.2
- pandas == 1.3.4

### Data
The process of collecting data is described in our manuscript and metadata can be accessed upon request.

### How to run the code?
"arg_v5.fasta" file is the original data set file, "fasta_process.ipynb" file is used on the original data set file to get the processed dataset.
Run "data_divide.py" to produce splitted dataset.
Put the ARG data (i.e., ) under "dataset", and run "python run.py --device "gpu" --batch_size 64 --epoch 10 --K 1 --n_experts 3  --n_experts_share 3  --expert_dim 1024" in the command line.
