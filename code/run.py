import torch
import numpy as np
import pandas as pd
import random
import torch.optim as optim
import torch.nn as nn
from argparse import ArgumentParser
from data_loader import ARGDataLoader
from modules import MLTARG
from utils import evaluate
import warnings
import random
from random import randint
warnings.filterwarnings("ignore")

parser = ArgumentParser("MLTARG")
# runtime args
parser.add_argument("--device", type=str, help='cpu or gpu', default="cpu")
parser.add_argument("--train_rate", type=float, help='train rate', default=0.8)
parser.add_argument("--batch_size", type=int, help='batch size', default=16)
parser.add_argument("--lr", type=float, help='learning rate', default=1e-4)
parser.add_argument("--epoch", type=int, help='epoch', default=5)
parser.add_argument("--K", type=int, help='K fold', default=5)
parser.add_argument("--n_experts", type=int, help='n_experts', default=1)
parser.add_argument("--n_experts_share", type=int, help='n_experts_share', default=1)
parser.add_argument("--expert_dim", type=int, help='expert_dim', default=1024)
args = parser.parse_args()

device = args.device
if args.device != 'cpu':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

train_rate = args.train_rate
batch_size = args.batch_size
lr = args.lr
epoch = args.epoch
K = args.K
n_experts, n_experts_share, expert_dim = args.n_experts, args.n_experts_share, args.expert_dim
print(str(args))

dataloader = ARGDataLoader()
# train_val_dataloader = dataloader.get_train_val_dataloader()
# assert K == len(train_val_dataloader)

anti_count, mech_count, type_count = dataloader.get_data_shape()

alpha, beta ,yita = 1.0, 0.2, 0.2

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

seed_num = randint(1,1000)
setup_seed(seed_num)
print(seed_num)
# setup_seed(42)

# Train
t_arg_acc, t_arg_precision, t_arg_recall, t_arg_f1 = 0, 0, 0, 0
t_anti_acc, t_anti_precision, t_anti_recall, t_anti_f1 = 0, 0, 0, 0
t_mech_acc, t_mech_precision, t_mech_recall, t_mech_f1 = 0, 0, 0, 0

test_dataloader = dataloader.load_test_dataSet(batch_size)

for k in range(K):
    print('Cross ', k + 1, ' of ', K)
    model = MLTARG(anti_count, mech_count, type_count, n_experts, n_experts_share, expert_dim)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    arg_loss_function = nn.NLLLoss()
    anti_loss_function = nn.NLLLoss()
    mech_loss_function = nn.NLLLoss()

    # train_dataloader = train_val_dataloader[k]['train']
    # val_dataloader = train_val_dataloader[k]['val']
    train_dataloader, val_dataloader = dataloader.load_n_cross_data(k + 1, batch_size)

    running_loss = 0.0
    for e in range(epoch):
        df = pd.DataFrame()
        model.train()
        print('train batch: ', len(train_dataloader))
        for index, (seq_map, anti_label, mech_label, arg_label) in enumerate(train_dataloader):
            seq_map, anti_label, mech_label, arg_label = seq_map.view(-1, 1, 1576, 23).to(device), anti_label.to(device), mech_label.to(device), arg_label.to(device)
            optimizer.zero_grad()
            arg_output, antibiotic_output, mechanism_output = model.forward(seq_map)
            loss_arg = arg_loss_function(torch.log(arg_output + 0.000001 ) , arg_label)
            loss_anti = anti_loss_function(torch.log(antibiotic_output + 0.000001)  , anti_label)
            loss_mech = mech_loss_function(torch.log(mechanism_output + 0.000001) , mech_label)
            # loss = loss_function(torch.log(output), r.long())
            loss = alpha * loss_anti + beta * loss_arg + yita * loss_mech
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            df = df.append({'loss_arg': loss_arg.item(), 'loss_anti': loss_anti.item(), 'loss_mech': loss_mech.item(), 'loss': loss.item(), 'running_loss': running_loss}, ignore_index=True)
            if index % 50 == 49:
                print('[%d, %2d, %5d] loss: %.3f' % (k + 1, e + 1, index + 1, running_loss / 50))
                running_loss = 0.0

        df.to_csv('./res/loss_cross' + str(k + 1) + '_epoch' + str(e) + '.csv')
        model.eval()
        val_arg_pred, val_arg_label = np.empty(shape=[0, type_count]), np.array([])
        val_anti_pred, val_anti_label = np.empty(shape=[0, anti_count]), np.array([])
        val_mech_pred, val_mech_label = np.empty(shape=[0, mech_count]), np.array([])

        for index, (seq_map, anti_label, mech_label, arg_label) in enumerate(val_dataloader):
            seq_map, anti_label, mech_label, arg_label = seq_map.view(-1, 1, 1576, 23).to(device), anti_label.to(device), mech_label.to(device), arg_label.to(device)
            arg_output, antibiotic_output, mechanism_output = model.forward(seq_map)

            arg_output, arg_label = arg_output.cpu().detach().numpy(), arg_label.cpu().numpy()
            val_arg_pred = np.append(val_arg_pred, arg_output, axis=0)
            val_arg_label = np.concatenate((val_arg_label, arg_label))

            antibiotic_output, anti_label = antibiotic_output.cpu().detach().numpy(), anti_label.cpu().numpy()
            val_anti_pred = np.append(val_anti_pred, antibiotic_output, axis=0)
            val_anti_label = np.concatenate((val_anti_label, anti_label))

            mechanism_output, mech_label = mechanism_output.cpu().detach().numpy(), mech_label.cpu().numpy()
            val_mech_pred = np.append(val_mech_pred, mechanism_output, axis=0)
            val_mech_label = np.concatenate((val_mech_label, mech_label))

        print('-------------Val: epoch ' + str(e + 1) + '-----------------')
        acc, macro_p, macro_r, macro_f1 = evaluate(val_arg_pred, val_arg_label, type_count)
        print('arg -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
        acc, macro_p, macro_r, macro_f1 = evaluate(val_anti_pred, val_anti_label, anti_count)
        print('antibiotic -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
        acc, macro_p, macro_r, macro_f1 = evaluate(val_mech_pred, val_mech_label, mech_count)
        print('mechanism -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))

    model.eval()
    test_arg_pred, test_arg_label = np.empty(shape=[0, type_count]), np.array([])
    test_anti_pred, test_anti_label = np.empty(shape=[0, anti_count]), np.array([])
    test_mech_pred, test_mech_label = np.empty(shape=[0, mech_count]), np.array([])
    for index, (seq_map, anti_label, mech_label, arg_label) in enumerate(test_dataloader):
        seq_map, anti_label, mech_label, arg_label = seq_map.view(-1, 1, 1576, 23).to(device), anti_label.to(device), mech_label.to(device), arg_label.to(device)
        arg_output, antibiotic_output, mechanism_output = model.forward(seq_map)

        arg_output, arg_label = arg_output.cpu().detach().numpy(), arg_label.cpu().numpy()
        test_arg_pred = np.append(test_arg_pred, arg_output, axis=0)
        test_arg_label = np.concatenate((test_arg_label, arg_label))

        antibiotic_output, anti_label = antibiotic_output.cpu().detach().numpy(), anti_label.cpu().numpy()
        test_anti_pred = np.append(test_anti_pred, antibiotic_output, axis=0)
        test_anti_label = np.concatenate((test_anti_label, anti_label))

        mechanism_output, mech_label = mechanism_output.cpu().detach().numpy(), mech_label.cpu().numpy()
        test_mech_pred = np.append(test_mech_pred, mechanism_output, axis=0)
        test_mech_label = np.concatenate((test_mech_label, mech_label))

    print('========Test: Cross ' + str(k + 1) + '===============')
    acc, macro_p, macro_r, macro_f1 = evaluate(test_arg_pred, test_arg_label, type_count)
    print('arg -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
    t_arg_acc += acc
    t_arg_precision += macro_p
    t_arg_recall += macro_r
    t_arg_f1 += macro_f1

    acc, macro_p, macro_r, macro_f1 = evaluate(test_anti_pred, test_anti_label, anti_count)
    print('antibiotic -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
    t_anti_acc += acc
    t_anti_precision += macro_p
    t_anti_recall += macro_r
    t_anti_f1 += macro_f1

    acc, macro_p, macro_r, macro_f1 = evaluate(test_mech_pred, test_mech_label, mech_count)
    print('mechanism -> acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, macro_p, macro_r, macro_f1))
    t_mech_acc += acc
    t_mech_precision += macro_p
    t_mech_recall += macro_r
    t_mech_f1 += macro_f1

    torch.save(model.state_dict(), './res/model{}.pth'.format(k))
    # torch.save(model,'./res/modeltotal.pth')
print('arg => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_arg_acc / K, t_arg_precision / K,
                                                                                                    t_arg_recall / K,
                                                                                                    t_arg_f1 / K))
print('antibiotic => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_anti_acc / K, t_anti_precision / K,
                                                                                                    t_anti_recall / K,
                                                                                                    t_anti_f1 / K))
print('mechanism => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_mech_acc / K, t_mech_precision / K,
                                                                                                    t_mech_recall / K,
                                                                                                    t_mech_f1 / K))
with open('./res/result.txt', 'a', encoding='utf8') as f:
    f.write(str(args))
    f.write('\n seed =>{}\n'.format(seed_num))
    f.write('\n arg => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_arg_acc / K, t_arg_precision / K,
                                                                                                    t_arg_recall / K,
                                                                                                    t_arg_f1 / K))
    f.write('\n antibiotic => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_anti_acc / K,
                                                                                                        t_anti_precision / K,
                                                                                                        t_anti_recall / K,
                                                                                                        t_anti_f1 / K))
    f.write('\n mechanism => final acc: {}, final precision: {}, final recall: {}, final f1: {}\n'.format(t_mech_acc / K,
                                                                                                        t_mech_precision / K,
                                                                                                        t_mech_recall / K,
                                                                                                        t_mech_f1 / K))
    f.write('----------------------------------------------------------------------------------------\n')

# def save_snapshot(model, filename):
#     torch.save(model.state_dict(), filename)
#     f.close()

