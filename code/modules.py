import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
import sys

class MLTARG(nn.Module):
    def __init__(self, anti_label, mech_label, type_label, n_experts, n_experts_share, expert_dim):
        super(MLTARG, self).__init__()
        self.feature = nn.Sequential(
             # (batch * 1 * 1576 * 23) -> (batch * 32 * 1537 * 20)
             nn.Conv2d(1, 32, kernel_size=(40, 4)),
             nn.LeakyReLU(),
             # (batch * 32 * 1537 * 20) -> (batch * 32 * 1533 * 19)
             nn.MaxPool2d(kernel_size=(5, 2), stride=1),
             # (batch * 32 * 1533 * 19) -> (batch * 64 * 1504 * 16)
             nn.Conv2d(32, 64, kernel_size=(30, 4)),
             nn.LeakyReLU(),
             # (batch * 64 * 1504 * 16) -> (batch * 128 * 1475 * 13)
             nn.Conv2d(64, 128, kernel_size=(30, 4)),
             nn.LeakyReLU(),
             # (batch * 128 * 1475 * 13) -> (batch * 128 * 1471 * 12)
             nn.MaxPool2d(kernel_size=(5, 2), stride=1),
             # (batch * 128 * 1471, 12) -> (batch * 256 * 1452 * 10)
             nn.Conv2d(128, 256, kernel_size=(20, 3)),
             nn.LeakyReLU(),
             # (batch * 256 * 1452 * 10) -> (batch * 256 * 1433 * 8)
             nn.Conv2d(256, 256, kernel_size=(20, 3)),
             nn.LeakyReLU(),
             # (batch * 256 * 1433 * 8) -> (batch * 256 * 1430 * 8)
             nn.MaxPool2d(kernel_size=(4, 1), stride=1),
             # (batch * 256 * 1430 * 8) -> (batch * 1 * 1411 * 6)
             nn.Conv2d(256, 1, kernel_size=(20, 3)),
             nn.LeakyReLU(),
             # (batch * 1 * 1411 * 6) -> (batch * 1 * 1410 * 6)
             nn.MaxPool2d(kernel_size=(2, 1), stride=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(8460, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU()
        )
        self.cgc = CGCModule(1024, expert_dim, 3, n_experts, n_experts_share)
        self.arg = ARGModule(type_label, expert_dim)
        self.antibiotic = AntibioticModule(anti_label, expert_dim)
        self.mechanism = MechanismModule(mech_label, expert_dim)

    def forward(self, seq_map):
        x = self.feature(seq_map)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        cgc_outs= self.cgc(x)
        arg_output = self.arg(cgc_outs[0])
        antibiotic_output = self.antibiotic(cgc_outs[1])
        mechanism_output = self.mechanism(cgc_outs[2])
        return arg_output, antibiotic_output, mechanism_output

class CGCModule(nn.Module):
    def __init__(self, input_dim, expert_dim, n_task, n_experts, n_experts_share):
        super(CGCModule, self).__init__()
        self.n_task = n_task
        self.expert_layers = nn.ModuleList([])
        for i in range(n_task):
            sub_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.LeakyReLU()
                ) for j in range(n_experts)
            ])
            self.expert_layers.append(sub_experts)
        self.share_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.LeakyReLU()
                ) for j in range(n_experts_share)
            ])
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, n_experts + n_experts_share),
                nn.Softmax(dim=1)
            ) for j in range(n_task)
        ])

    def forward(self, x):
        expert_features = [[expert(x) for expert in sub_experts] for sub_experts in self.expert_layers]
        share_features = [expert(x) for expert in self.share_layers]
        outs = []
        for i in range(self.n_task):
            g = self.gate_layers[i](x).unsqueeze(2)
            e = share_features + expert_features[i]
            e = torch.cat([expert[:, np.newaxis, :] for expert in e], dim=1)
            out = torch.matmul(e.transpose(1, 2), g).squeeze(2)  
            outs.append(out)
        return outs



class ARGModule(nn.Module):
    def __init__(self, type_label, expert_dim):
        super(ARGModule, self).__init__()
        self.fc = nn.Linear(expert_dim, type_label)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc(x))

class AntibioticModule(nn.Module):
    def __init__(self, anti_label, expert_dim):
        super(AntibioticModule, self).__init__()
        self.fc = nn.Linear(expert_dim, anti_label)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc(x))


class MechanismModule(nn.Module):
    def __init__(self, mech_label, expert_dim):
        super(MechanismModule, self).__init__()
        self.fc = nn.Linear(expert_dim, mech_label)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.fc(x))



