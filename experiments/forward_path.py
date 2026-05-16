import sys

sys.path.append('../')
import os 
import torch
import random
import numpy as np
from oracle import count_parameters
from utils import dataset_prepare
from scipy.io import loadmat
from model import Hammerstein

# Determine experiment name and create its directory
# exp_name = "mnm"
# exp_name = "mnm_damped"
# exp_name = "conj_grad"
# exp_name = "cg_iter_5"
# exp_name = "dcd"
# exp_name = "sgd_auto"
exp_name = "mnm_forward_del_3"

add_folder = os.path.join("")
curr_path = os.getcwd()
load_path = os.path.join(curr_path, add_folder, exp_name)
# os.mkdir(save_path)

device = "cuda:5"
# device = "cpu"
seed = 964
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# torch.use_deterministic_algorithms(True)
if device != "cpu":
    torch.backends.cudnn.deterministic = True

# Load PA input and output data
mat = loadmat("../data/LTE60M_334RB_fs122k.mat")

# Define data type
# dtype = torch.complex64
dtype = torch.complex128

# Memory FIR filter number of taps
tap_num = 51
# Padding type:
# padding == 'same' means that all convolutions in model keep dimensionaity of tensors
# padding == 'valid' means that all convolutions in model reduce dimensionaity of tensors
padding = 'valid'
# Non-linerities order
nonlin_order = 8
# Delays applied to the signal before non-linearity application.
# If non-linearity is 1-dimensional, each nested list must consist of 2 delays:
# 1-st delay corresponds to the delay of pure signal x_n, 
# 2-nd delay corresponds to the delay of signal magnitude |x_n|
# If non-linearity is 2-dimensional, each nested list must consist of 3 delays etc.
delays = [[0, 0]]
slot_num = 1
# Indices of slots which are chosen to be included in train/test set (must be of a range type).
# Elements of train_slots_ind, test_slots_ind must be higher than 0 and lower, than slot_num
# In full-batch mode train, validation and test dataset are the same.
# In mini-batch mode validation and test dataset are the same.
train_slots_ind, validat_slots_ind, test_slots_ind = range(1), range(1), range(1)
# Delay of target signal w.r.t. the input signal
delay_d = 1
# batch_size == None is equal to batch_size = 1.
# block_size == None is equal to block_size = signal length.
# Block size is the same as chunk size 
batch_size = 1
chunk_num = 1
chunk_size = int(79667/chunk_num)
# L2 regularization parameter
alpha = 0.0
# Configuration file
config_train = None
# Input signal is padded with pad_zeros zeros at the beginning and ending of input signal.
# pad_zeros = 0
if padding == 'valid':
    pad_zeros = int(tap_num // 2)
elif padding == 'same':
    pad_zeros = 0
else:
    raise ValueError(f"Padding type must equal either \'same\' or \'valid\', but {padding} is given.")
dataset = dataset_prepare(mat, dtype, device, slot_num=slot_num, delay_d=delay_d,
                          train_slots_ind=train_slots_ind, test_slots_ind=test_slots_ind, validat_slots_ind=validat_slots_ind,
                          pad_zeros=pad_zeros, batch_size=batch_size, block_size=chunk_size)

train_dataset, validate_dataset, test_dataset = dataset

# Show sizes of batches in train dataset, size of validation and test dataset
# for i in range(len(dataset)):
#     for j, batch in enumerate(dataset[i]):
#         if j == 0:
#             print(batch[0].size())
#             print(batch[1].size())
# print(f"Number of batches: {j + 1}")
# sys.exit()

def batch_to_tensors(a):
    x = a[0]
    d = a[1][:, :1, :]
    nf = a[1][:, 1:, :]
    return x, d, nf

def complex_mse_loss(d, y, model):
    error = (d - y)[..., None if pad_zeros == 0 else pad_zeros: None if pad_zeros == 0 else -pad_zeros]
    # error = (d - y)
    return error.abs().square().sum() + alpha * sum(torch.norm(p)**2 for p in model.parameters())

def loss(model, signal_batch):
    x, y, _ = batch_to_tensors(signal_batch)
    return complex_mse_loss(model(x), y, model)
# This function is used only for telecom task.
# Calculates NMSE on base of accumulated on every batch loss function
@torch.no_grad()
# To avoid conflicts for classification task you can write:
# def quality_criterion(loss_val):
#     return loss_val
# def quality_criterion(model, dataset):
#     loss_val = 0
#     for batch in dataset:
#         loss_val += loss(model, batch)
#     return 10.0 * torch.log10(loss_val).item()
def quality_criterion(model, dataset):
    input_pow, nf_pow, loss_val = 0, 0, 0
    for batch in dataset:
        x, _, nf = batch_to_tensors(batch)
        nf_pow += nf[..., None if pad_zeros == 0 else pad_zeros: None if pad_zeros == 0 else -pad_zeros].abs().square().sum()
        input_pow += x[..., None if pad_zeros == 0 else pad_zeros: None if pad_zeros == 0 else -pad_zeros].abs().square().sum()
        # targ_pow += d[..., None if pad_zeros == 0 else pad_zeros: None if pad_zeros == 0 else -pad_zeros].abs().square().sum()
        loss_val += loss(model, batch)
    return 10.0 * torch.log10((loss_val - nf_pow) / (input_pow - nf_pow)).item()

def load_weights(path_name, device=device):
    return torch.load(path_name, map_location=torch.device(device))

def set_weights(model, weights):
    model.load_state_dict(weights)

def get_nested_attr(module, names):
    for i in range(len(names)):
        module = getattr(module, names[i], None)
        if module is None:
            return
    return module

# Define DPD model
model = Hammerstein(delays=delays, tap_num=tap_num, nonlin_order=nonlin_order, 
                    padding=padding, device=device, dtype=dtype)

model.to(device)

weight_names = list(name for name, _ in model.state_dict().items())

print(f"Current model parameters number is {count_parameters(model)}")

# Set model parameters
set_weights(model, load_weights(load_path + r'/weights_best_test_' + exp_name))
# weights = load_weights("../classic/omp_delays_2_branches_10_order_nl_mixed_ls_cheby_basis/weights_best_test_omp_delays_2_branches_10_order_nl_mixed_ls_cheby_basis")
# weights_keys_load_from = [f'nonlin.nonlin.{i}' for i in range(len(list(weights.keys())))]
# for i in range(len(list(weights.keys()))):
#     model.nonlin[i].nonlin[0].data = weights[weights_keys_load_from[i]].data

model.eval()
with torch.no_grad():
    # train_dataset, validate_dataset, test_dataset
    dataset = test_dataset
    NMSE = quality_criterion(model, dataset)
    print(NMSE)
    for j, batch in enumerate(dataset):
        data = batch_to_tensors(batch)
    y = model(data[0])

y = y[0, 0, :].detach().cpu().numpy()
d = data[1][0, 0, :].detach().cpu().numpy()
x = data[0][0, 0, None if pad_zeros == 0 else pad_zeros: None if pad_zeros == 0 else -pad_zeros].detach().cpu().numpy()
np.save(load_path + r'/d.npy', d)
np.save(load_path + r'/y.npy', y)