import numpy as np
import torch
import torch.nn as nn
from torch.nn import ReLU
import time
import sys

class ChebyPolynom(nn.Module):
    def __init__(self, order, channel_num, dtype=torch.complex128, device='cuda'):
        super(ChebyPolynom, self).__init__()
        self.order = order
        self.dtype = dtype
        self.device = device
        self.channel_num = channel_num
        self.nonlin = nn.ParameterList()
        for _ in range(channel_num):
            self.nonlin.append(torch.nn.Parameter(1.e-3*torch.rand(order, dtype=dtype, device=device), requires_grad=True))
    def forward(self, x):
        """
            Signal input must be 4-dimensional:
            1-st dim. - batch size,
            2-nd dim. - number of branches,
            3-d dim. - dimensionality of non-linearity, for 1D Chebyshev polynomial must equal 1,
            4-th dim. - signal block length.
        """
        x_carrier = x[..., 0, :]
        nl_input = torch.abs(x[..., 1, :]).to(self.dtype)
        output = []
        for c in range(self.channel_num):
            tmp = torch.kron(torch.arccos(nl_input[:, c, :]).unsqueeze(-1), torch.arange(self.order, device=self.device))
            tmp = torch.cos(tmp) @ self.nonlin[c]
            output.append((x_carrier[:, c, :] * tmp))
        output = sum(output).unsqueeze(1)
        return output

class Cheby3D(nn.Module):
    def __init__(self, order=8, complex_coef=True, uniCalc=False, dtype=torch.complex128, device='cuda'):
        super().__init__()
        self.order = order
        self.dtype = dtype
        self.device = device
        self.vand = None
        self.ChebyLUT3D = torch.nn.Parameter(torch.zeros(order**3, dtype=dtype, device=device), requires_grad=True)
        self.ChebyLUT3D.data = 1.e-2*(torch.rand(order**3, dtype=dtype, device=device) + 1j*torch.rand(order**3, dtype=dtype, device=device) - 1/2 - 1j/2)

    def forward(self, input):  # input is always real in range [0; 1]
        input = torch.abs(input)

        # if self.vand == None:
        ind = torch.arange(self.order, device=self.device)

        T_0 = torch.cos(ind[:, None] * torch.arccos(input[0, :1, :]))
        T_1 = torch.cos(ind[:, None] * torch.arccos(input[0, 1:2, :]))
        T_2 = torch.cos(ind[:, None] * torch.arccos(input[0, 2:, :]))

        T01 = (T_0[:,None,:] * T_1[None, :,:]).reshape(-1, T_0.shape[-1])
        self.vand = (T01[:,None,:] * T_2[None, :,:]).reshape(-1, T_0.shape[-1]).T.to(self.dtype)

        approx = (self.vand @ self.ChebyLUT3D)[None, None, :]

        return approx
    
class Cheby2D(nn.Module):
    def __init__(self, order=8, complex_coef=True, uniCalc=False, dtype=torch.complex128, device='cuda'):
        super().__init__()
        self.order = order
        self.dtype = dtype
        self.device = device
        self.vand = None
        self.Cheby2D = torch.nn.Parameter(torch.zeros(order**2, dtype=dtype, device=device), requires_grad=True)
        self.Cheby2D.data = 1.e-2*(torch.rand(order**2, dtype=dtype, device=device) + 1j*torch.rand(order**2, dtype=dtype, device=device) - 1/2 - 1j/2)

    def forward(self, input):  # input is always real in range [0; 1]
        input = torch.abs(input)

        # if self.vand == None:
        ind = torch.arange(self.order, device=self.device)

        T_0 = torch.cos(ind[:, None] * torch.arccos(input[0, :1, :]))
        T_1 = torch.cos(ind[:, None] * torch.arccos(input[0, 1:2, :]))

        self.vand = (T_0[:,None,:] * T_1[None, :,:]).reshape(-1, T_0.shape[-1]).T.to(self.dtype)

        approx = (self.vand @ self.ChebyLUT3D)[None, None, :]

        return approx