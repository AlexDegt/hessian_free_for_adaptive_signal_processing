import sys

sys.path.append('../')
import os
import yaml, shutil

import torch
import random
import numpy as np
from oracle import count_parameters
from trainer import train
from utils import dataset_prepare
from scipy.io import loadmat
from model import Hammerstein

# Import config file
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Determine experiment name and create its directory
exp_name = config["exp_name"]

add_folder = os.path.join(config["add_folder"])
curr_path = os.getcwd()
save_path = os.path.join(curr_path, add_folder, exp_name)
os.makedirs(save_path, exist_ok=True)

# Save config file
shutil.copyfile('config.yaml', os.path.join(save_path, 'config.yaml'))

device = config["device"]
seed = config["seed"]
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# torch.use_deterministic_algorithms(True)
if device != "cpu":
    torch.backends.cudnn.deterministic = True

# Load PA input and output data
mat = loadmat(config["data_path"])

# Define data type
dtype = getattr(torch, config["dtype"])

# Memory FIR filter number of taps
tap_num = config["tap_num"]
# Padding type:
# padding == 'same' means that all convolutions in model keep dimensionaity of tensors
# padding == 'valid' means that all convolutions in model reduce dimensionaity of tensors
padding = config["padding"]
# Non-linerities order
nonlin_order = config["nonlin_order"]
# Delays applied to the signal before non-linearity application.
# If non-linearity is 1-dimensional, each nested list must consist of 2 delays:
# 1-st delay corresponds to the delay of pure signal x_n, 
# 2-nd delay corresponds to the delay of signal magnitude |x_n|
# If non-linearity is 2-dimensional, each nested list must consist of 3 delays etc.
delays = config["delays"]
slot_num = 1
# Indices of slots which are chosen to be included in train/test set (must be of a range type).
# Elements of train_slots_ind, test_slots_ind must be higher than 0 and lower, than slot_num
# In full-batch mode train, validation and test dataset are the same.
# In mini-batch mode validation and test dataset are the same.
train_slots_ind, validat_slots_ind, test_slots_ind = range(1), range(1), range(1)
# Delay of target signal w.r.t. the input signal
delay_d = config["delay_d"]
# batch_size == None is equal to batch_size = 1.
# block_size == None is equal to block_size = signal length.
# Block size is the same as chunk size 
batch_size = 1
chunk_num = config["chunk_num"]
chunk_size = int(78960/chunk_num)
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
#     print(f"Number of batches: {j + 1}")
# sys.exit()

def batch_to_tensors(a):
    x = a[0]
    d = a[1][:, :1, :]
    nf = a[1][:, 1:, :]
    return x, d, nf

def complex_mse_loss(d, y):
    error = (d - y)[..., None if pad_zeros == 0 else pad_zeros: None if pad_zeros == 0 else -pad_zeros]
    return error.abs().square().sum()

def loss(model, signal_batch):
    x, y, _ = batch_to_tensors(signal_batch)
    return complex_mse_loss(model(x), y)
# This function is used only for telecom task.
# Calculates NMSE on base of accumulated on every batch loss function
@torch.no_grad()
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

# Define DPD model. Model parameters names: ['nonlin.nonlin.0', 'fir.conv_complex.weight']
model = Hammerstein(delays=delays, tap_num=tap_num, nonlin_order=nonlin_order, 
                    padding=padding, device=device, dtype=dtype)

model.to(device)

# Set parameters, which implied to be trainable
weight_names = config["weight_names"]

print(f"Current model parameters number is {count_parameters(model, count_non_differentiable=False)}")
param_names = [name for name, p in model.named_parameters()]
params = [(name, p.size(), p.dtype) for name, p in model.named_parameters()]
# params = [(name, p) for name, p in model.named_parameters()]
# print(params)
# ss.exit()

# Train type shows which algorithm is used for optimization.
train_type = config["train_type"]
# train_type='sgd_auto' # gradient-based optimizer.
# train_type='mnm_damped' # Damped Mixed Newton. Work only with models with complex parameters!
# train_type='mnm_block' # Block Mixed Newton. Work only with models with complex parameters!
# train_type='mnm_lev_marq' # Levenberg-Marquardt on base of Mixed Newton. Work only with models with complex parameters!
# train_type='mnm_ls' # LS method with mixed hessian.
# train_type='conj_grad' # Conjugate gradient method, using properties of mixed hessian.
# train_type='conj_grad_block' # Block conjugate gradient method, using properties of mixed hessian.
# train_type='dcd' # Dichotomous Coordinate Descent method. Hessian and gradient are calculated for holomorphic errors.
# train_type='newton_damped' # Damped Newton. Can be used for models with real and complex parameters.
# train_type='newton_lev_marq' # Levenberg-Marquardt on base of Newton. Can be used for models with real and complex parameters.
# train_type='cubic_newton' # Cubic Newton. Currently work only with models with complex parameters!
# train_type='cubic_newton_simple' # Simplified cubic Newton. Currently work only with models with complex parameters!
learning_curve, best_criterion = train(model, train_dataset, loss, quality_criterion, config_train, batch_to_tensors, validate_dataset, test_dataset, 
                                       train_type=train_type, chunk_num=chunk_num, exp_name=exp_name, save_every=1, save_path=save_path, 
                                       weight_names=weight_names, device=device, config=config)
# print([(name, p) for name, p in model.named_parameters()])
print(f"Best NMSE: {best_criterion} dB")