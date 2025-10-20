import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np

import sys
sys.path.append('../../')

from utils import Timer
from oracle import Oracle

OptionalInt = Union[int, None]
OptionalStr = Union[str, None]
StrOrList = Union[str, List[str], Tuple[str], None]
DataLoaderType = torch.utils.data.dataloader.DataLoader
LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]

def train_conjugate_gradient(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType, 
                                   test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, 
                                   batch_to_tensors: BatchTensorType, chunk_num: OptionalInt = None, 
                                   save_path: OptionalStr = None, exp_name: OptionalStr = None, save_every: OptionalInt = None, 
                                   save_signals: bool = False, weight_names: StrOrList = None):
    """
    Function implements conjugate gradient method using mixed hessian for holomorphic functions. 
    Current function uses oracle.Oracle.direction_through_jacobian function which firstly accumulates model output 
    jacobian J w.r.t. the model parameters on the whole batch, then calculates Hessian as matrix 
    multiplication (J^H @ J). Gradient is calculated as vector-jacobian multiplication
    (J^H @ e), where e - model error on current batch. 
    
    Attention!
    Mixed Newton performs better on the long batches - with big sample size. Batch size could be chosen as 1.
    For the big sample size (>~ 50000) jacobian matrix requires huge memory resources, that`s why function
    oracle.Oracle.direction_through_jacobian contains chunk-mechanism under the hood. Chunk-mechanism divides
    whole batch into the chunks with chunk_size to save GPU memory.

    Levenberg–Marquardt algorithm is an effective mechanism which, in fact, is an adaptive regularization. In current implementation 
    it checks if quality criterion in current iteration degraded, then reqgularization parameter alpha increased: (H + alpha * I)^(-1) * grad,
    otherwise it decreased. Thus in case of high numerical mistakes in Hessian calculation due to the bag-conditioning, then algorithm
    tries make more gradient-like step by increasing regularization parameter.

    The stop criteria is determined by gradient norm. If it is lower than min_grad_norm than algorithm stops.

    Args:
        model (nn.Module): The model with differentiable parameters.
        train_dataset (torch DataLoader type): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
        validate_dataset (torch DataLoader type, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate intermediate quality criterion values. 
            Attention! Validate dataset must have only 1 batch containing whole signal.
            Newton-based training methods usually work on the whole signal dataset. 
            Therefore train and validation datasets are implied to be the same.
        test_dataset (DataLoader, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate quality criterion for test data.
            Attention! Test dataset must have only 1 batch containing whole signal, the same as for validation dataset.
        loss_fn (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
                instances. Returns differentiable Tensor scalar.
        quality_criterion (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
                instances. Returns differentiable Tensor scalar. quality_criterion is not used in the model differentiation
                process, but it`s only used to estimate model quality in more reasonable units comparing to the loss_fn.
        batch_to_tensors (Callable): Function which acquires signal batch as an input and returns tuple of tensors, where
            the first tensor corresponds to model input, the second one - to the target signal. This function is used to
            obtain differentiable model output tensor to calculate jacobian.
        chunk_num (int, optional): The number of chunks in dataset. Defaults to "None".
        save_path (str, optional): Folder path to save function product. Defaults to "None".
        exp_name (str, optional): Name of simulation, which is reflected in function product names. Defaults to "None".
        save_every (int, optional): The number which reflects following: the results would be saved every save_every epochs.
            If save_every equals None, then results will be saved at the end of learning. Defaults to "None".
        save_signals (bool): The flag that shows, whether to save training signals or not. Defaults to False.
        weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
            for several named parameters. Defaults to "None".

    Returns:
        Learning curve (list), containing quality criterion calculated each epoch of learning.
    """
    # Algorithm stop criteria parameters
    epochs = int(1000)

    # Conjugate gradient algorithm number of steps per 1 epoch
    step_num_conj_grad = 20

    if save_every is None:
        save_every = epochs - 1

    epoch, print_every = 0, 1

    SICOracle = Oracle(model, loss_fn)

    mu = 1.e-2
    alpha = 1.
    eps = 1e-4
    reg_param_curve = []
    learning_curve_train = []
    learning_curve_test = []
    learning_curve_validate = []
    learning_curve_train_qcrit = []
    learning_curve_test_qcrit = []
    learning_curve_validate_qcrit = []
    grad_norm_curve = []
    weights_norm_curve = []
    grad_norm = None
    timer = Timer()
    general_timer = Timer()
    general_timer.__enter__()

    def accum_loss(dataset):
        loss_val = 0
        for batch in dataset:
            loss_val += SICOracle.loss_function_val(batch).item()
        return loss_val
            
    # Calculate initial values of loss and quality criterion on validation and test dataset
    with torch.no_grad():
        loss_val_test = accum_loss(test_dataset)
        criterion_val_test = quality_criterion(model, test_dataset)
        best_criterion_test = criterion_val_test
        learning_curve_test.append(loss_val_test)
        learning_curve_test_qcrit.append(criterion_val_test)
        print("Begin: loss = {:.4e}, quality_criterion_test = {:.8f} dB.".format(loss_val_test, criterion_val_test))
        loss_val_train = accum_loss(train_dataset)
        criterion_val_train = quality_criterion(model, train_dataset)
        learning_curve_train.append(loss_val_train)   
        learning_curve_train_qcrit.append(criterion_val_train)
        print("Begin: loss = {:.4e}, quality_criterion_train = {:.8f} dB.".format(loss_val_train, criterion_val_train))
        loss_val_validate = accum_loss(validate_dataset)
        criterion_val_validate = quality_criterion(model, validate_dataset)
        learning_curve_validate.append(loss_val_validate)
        learning_curve_validate_qcrit.append(criterion_val_validate)
        print("Begin: loss = {:.4e}, quality_criterion_validate = {:.8f} dB.".format(loss_val_validate, criterion_val_validate))

    epoch = 0
    min_grad_norm = 1e-8
    for epoch in range(epochs):
    # while grad_norm is None or grad_norm >= min_grad_norm:
        timer.__enter__()
        # Accumulate hessian and gradient on the whole training dataset.
        # Combination of all batches on train dataset should be equal validation dataset
        for j, batch in enumerate(train_dataset):

            delta_hess, delta_grad = SICOracle.direction_through_jacobian(batch, batch_to_tensors, weight_names=weight_names)

            with torch.no_grad():
                if j % chunk_num == 0:
                    hess = torch.zeros_like(delta_hess)
                    grad = torch.zeros_like(delta_grad)
                hess += delta_hess
                grad += delta_grad
                del delta_hess, delta_grad
                torch.cuda.empty_cache()

        # Implement conjugate gradient iterations using mixed hessian properties
        p, q = grad.clone(), grad.clone()
        direction = 0 #SICOracle.get_flat_params(name_list=weight_names)
        x = SICOracle.get_flat_params(name_list=weight_names)
        for _ in range(step_num_conj_grad):

            # Direct way of xi calculation
            xi = hess @ p
            # Alternative way of xi calculation
            # def func1(d, y):
            #     return (d - y).abs().square().sum()
            
            # def func2(d, y, xi_conj):
            #     return torch.conj(xi_conj).view(-1) @ (d - y).view(-1)

            # def func(model, signal_batch):
            #     radius = 1.e-2
            #     inp, target, _ = batch_to_tensors(signal_batch)
            #     model_out = model(inp)

            #     # backward не работает из-за get_flat_params/set_flat_params !!!!!!!!!!!!!!!

            #     # weights = model.state_dict()
            #     # for name, param in model.state_dict().items():
            #     #     model.state_dict()[name] += shift
            #     # weights_shifted = 

            #     params_curr = SICOracle.get_flat_params(name_list=weight_names)
            #     SICOracle.add_flat_params(radius * p, name_list=weight_names)
            #     # SICOracle.set_flat_params(params_curr + radius * p, name_list=weight_names)
            #     model_out_vicinity = model(inp)
            #     SICOracle.set_flat_params(params_curr, name_list=weight_names)
            #     direct_deriv = -1 * (model_out_vicinity - model_out) / radius
            #     # return func1(target, model_out)
            #     return func2(target, model_out, direct_deriv)

            # # def func(model, signal_batch):
            # #     x, d, _ = batch_to_tensors(signal_batch)
            # #     return (d - model(x)).abs().square().sum()
            
            # func_aux = func(model, batch)

            # # def func(d, y):
            # #     return (d - y).abs().square().sum()
            # # func_aux = func(target, model_out)
            # # func_aux = (target - model_out).abs().square().sum()
            # # func_aux = direct_deriv @ torch.conj(target - model_out)
            # # print([p.requires_grad for p in model.parameters()])
            # # print(func_aux.size())
            # # print(func_aux)
            # # sys.exit()
            # # print([(name, p.grad) for name, p in model.named_parameters()])
            # # torch.autograd.set_detect_anomaly(True)
            # func_aux.backward()
            # # print([(name, p.grad) for name, p in model.named_parameters()])

            # xi = torch.cat([p.grad.view(-1) for p in model.parameters()])
            # # print(xi)
            # # sys.exit()

            # alpha = -1 * (torch.norm(q) ** 2) / (torch.conj(p) @ xi)
            # direction += alpha * p
            # q_prev = q.clone()
            # q += alpha * xi
            # beta = (torch.norm(q) / torch.norm(q_prev)) ** 2
            # p = q + beta * p

            alpha = (-1) * (p.conj() @ q) / (p.conj() @ xi)
            direction += alpha * p
            q_prev = q.clone()
            q += alpha * xi
            beta = (-1) * (q.conj() @ xi).conj() / (p.conj() @ xi)
            p = (1) * q + beta * p

        # direction = params - x

        curr_params = x + mu * direction
        SICOracle.set_flat_params(curr_params, name_list=weight_names)

        # Update model parameters using damped line search
        with torch.no_grad():
            tmp_loss_val = accum_loss(train_dataset)
            tmp_criterion_val = quality_criterion(model, train_dataset)
        if tmp_loss_val <= loss_val_train + eps:
            mu *= 1.6
            if mu > 1.:
                mu = 1.
        else:
            while tmp_loss_val > loss_val_train + eps:
                mu /= 1.2
                curr_params = x + mu * direction
                SICOracle.set_flat_params(curr_params, name_list=weight_names)
                with torch.no_grad():
                    tmp_loss_val = accum_loss(train_dataset)
                    tmp_criterion_val = quality_criterion(model, train_dataset)
                if epoch % print_every == 0:
                    print(f"Deverges, epoch is {epoch + 1}, quality_criterion_train = {tmp_criterion_val:.8f} dB," + \
                            f"loss_val = {tmp_loss_val:.4f}, stepsize = {mu:.6e}")
        loss_val_train = tmp_loss_val
        criterion_val_train = tmp_criterion_val

        # Track algorithm parameters
        reg_param_curve.append(alpha)
        grad_norm = torch.norm(grad).item()
        grad_norm_curve.append(grad_norm)
        weights_norm_curve.append(torch.norm(curr_params).item())

        hess.detach()
        grad.detach()
        del grad, hess
        torch.cuda.empty_cache()

        # Track NMSE values on validation and test dataset and save gradient, model parameters norm and 
        # algorithm regularization history
        with torch.no_grad():
            loss_val_test = accum_loss(test_dataset)
            criterion_val_test = quality_criterion(model, test_dataset)
            loss_val_validate = accum_loss(validate_dataset)
            criterion_val_validate = quality_criterion(model, validate_dataset)

            learning_curve_test.append(loss_val_test)
            learning_curve_train.append(loss_val_train)
            learning_curve_validate.append(loss_val_validate)
            learning_curve_test_qcrit.append(criterion_val_test)
            learning_curve_train_qcrit.append(criterion_val_train)
            learning_curve_validate_qcrit.append(criterion_val_validate)

            if criterion_val_test < best_criterion_test:
                best_criterion_test = criterion_val_test
                torch.save(model.state_dict(), save_path+'weights_best_test'+exp_name)
            if epoch % save_every == 0:
                np.save(save_path + f'lc_train{exp_name}.npy', np.array(learning_curve_train))
                np.save(save_path + f'lc_test{exp_name}.npy', np.array(learning_curve_test))
                np.save(save_path + f'lc_validate{exp_name}.npy', np.array(learning_curve_validate))
                np.save(save_path + f'lc_qcrit_train{exp_name}.npy', np.array(learning_curve_train_qcrit))
                np.save(save_path + f'lc_qcrit_test{exp_name}.npy', np.array(learning_curve_test_qcrit))
                np.save(save_path + f'lc_qcrit_validate{exp_name}.npy', np.array(learning_curve_validate_qcrit))
                np.save(save_path + f'grad_norm{exp_name}.npy', np.array(grad_norm_curve))
                np.save(save_path + f'param_norm{exp_name}.npy', np.array(weights_norm_curve))
        timer.__exit__()
        if epoch % print_every == 0:
            print(f"Epoch is {epoch + 1}, " + \
                f"loss_train = {loss_val_train:.8f}, " + \
                f"quality_criterion_train = {criterion_val_train:.8f} dB, stepsize = {mu:.6e}, " + \
                f"|grad| = {grad_norm:.4e}, time elapsed: {timer.interval:.2e}")
        epoch += 1

        general_timer.__exit__()
        print(f"Total time elapsed: {general_timer.interval} s")

    return learning_curve_test, best_criterion_test