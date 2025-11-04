import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable
import numpy as np

import sys
sys.path.append('../../')

from utils import Timer

OptionalInt = Union[int, None]
OptionalStr = Union[str, None]
DictOptional = Union[dict, None]
OptimizerType = torch.optim.Optimizer
DataLoaderType = torch.utils.data.dataloader.DataLoader
LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]

def train_sgd_auto(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType,
              test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, 
              batch_to_tensors: BatchTensorType, config_train: dict, save_path: OptionalStr = None, exp_name: OptionalStr = None, 
              save_every: OptionalInt = None, config: DictOptional = None):
    """
    Function optimizes model parameters using common stochastic gradient descent, loss.backward() method.

    Args:
        model (nn.Module): The model with differentiable parameters.
        train_dataset (torch DataLoader type): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
        validate_dataset (torch DataLoader type, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate intermediate quality criterion values. 
            Attention! Validate dataset must have only 1 batch containing whole signal.
        test_dataset (DataLoader, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate quality criterion for test data.
            Attention! Test dataset must have only 1 batch containing whole signal, the same as for validation dataset.
        loss_fn (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
            instances. Returns differentiable Tensor scalar.
        quality_criterion (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
            instances. Returns differentiable Tensor scalar. quality_criterion is not used in the model differentiation
            process, but it`s only used to estimate model quality in more reasonable units comparing to the loss_fn.
        batch_to_tensors (Callable): Function which acquires signal batch as an input and returns tuple of tensors, where
            the first tensor corresponds to model input, the second one - to the target signal.
        config_train (dictionary): Dictionary with configurations of training procedure. Includes learning rate, training type,
            optimizers parameters etc. Implied to be loaded from .yaml config file.
        save_path (str, optional): Folder path to save function product. Defaults to "None".
        exp_name (str, optional): Name of simulation, which is reflected in function product names. Defaults to "None".
        save_every (int, optional): The number which reflects following: the results would be saved every save_every epochs.
            If save_every equals None, then results will be saved at the end of learning. Defaults to "None".
        config (dict, optional): dictionary with all configurations.

    Returns:
        Learning curve (list), containing quality criterion calculated each epoch of learning.
    """
    epochs = int(config["epochs"])

    if save_every is None:
        save_every = epochs - 1

    lr = config["lr"]
    betas = config["betas"]
    start_factor = config["start_factor"]
    end_factor = config["end_factor"]

    lrs = []
    learning_curve_test = []
    weight_decay = 0 # 1e-5
    block_num = len(train_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
    lambda_lin = lambda epoch: 1#1 - (1 - 1e-1)*epoch/epochs
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lin)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=epochs * block_num)

    print_every = 1
    timer = Timer()
    general_timer = Timer()
    general_timer.__enter__()
    def accum_loss(dataset):
        loss_val = 0
        for batch in dataset:
            loss_val += loss_fn(model, batch).item()
        return loss_val
            
    # Calculate initial values of loss and quality criterion on validation and test dataset
    with torch.no_grad():
        loss_val_test = accum_loss(test_dataset)
        criterion_val_test = quality_criterion(model, test_dataset)
        best_criterion_test = criterion_val_test
        learning_curve_test.append(criterion_val_test)
        print("Begin: loss = {:.4e}, quality_criterion_test = {:.8f} dB.".format(loss_val_test, criterion_val_test))
        loss_val_train = accum_loss(train_dataset)
        criterion_val_train = quality_criterion(model, train_dataset)
        print("Begin: loss = {:.4e}, quality_criterion_train = {:.8f} dB.".format(loss_val_train, criterion_val_train))
        loss_val_validate = accum_loss(validate_dataset)
        criterion_val_validate = quality_criterion(model, validate_dataset)
        print("Begin: loss = {:.4e}, quality_criterion_validate = {:.8f} dB.".format(loss_val_validate, criterion_val_validate))

    for epoch in range(epochs):
        timer.__enter__()
        for j, batch in enumerate(train_dataset):
            def closure():
                optimizer.zero_grad()
                loss_val = loss_fn(model, batch)
                loss_val.backward()
                return loss_val
            optimizer.step(closure)
            scheduler.step()
            # scheduler.step(criterion_val)
            with torch.no_grad():
                criterion_val_test = quality_criterion(model, test_dataset)
                if criterion_val_test < best_criterion_test:
                    best_criterion_test = criterion_val_test
                    torch.save(model.state_dict(), save_path+'weights_best_test'+exp_name)
                lrs.append(optimizer.param_groups[0]['lr'])
                learning_curve_test.append(criterion_val_test)
                assert ~np.isnan(criterion_val_test), f"Algorithm diverged at the epoch {epoch}."
            timer.__exit__()
        if epoch % save_every == 0:
            np.save(save_path + f'lc_test{exp_name}.npy', np.array(learning_curve_test))
            np.save(save_path + f'lrs{exp_name}.npy', np.array(lrs))
        if (epoch % print_every == 0) or (((j + 1) == len(train_dataset))):
            # pass
            print("Epoch is {}, ".format(epoch + 1) +\
                "Block is {},".format(j + 1) +\
                " quality_criterion = {:.8f} dB, stepsize = {:.12e},".format(criterion_val_test, lrs[-1]) +\
                " time elapsed: {:.4e} s,".format(timer.interval))
            
    general_timer.__exit__()
    print(f"Total time elapsed: {general_timer.interval} s")

    return learning_curve_test, best_criterion_test