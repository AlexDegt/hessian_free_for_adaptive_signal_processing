import torch
from .layers import CentralFIR, ChebyPolynom, Delay
from itertools import chain
import sys

class Hammerstein(torch.nn.Module):
    '''
        Class implements Hammerstein model. Input signal is implied to single channel. Takes pure signal as an input.
    '''
    def __init__(self, delays=[[0]], tap_num=51, nonlin_order=8, padding='same', device=None, dtype=torch.complex128):
        super().__init__()

        self._dtype = dtype
        self._device = device

        self._delay_num = len(list(chain(*delays)))
        self._branch_num = len(delays)
        self._tap_num = tap_num
        self._nonlin_order = nonlin_order
        
        # Define branches
        self.delay = torch.nn.ModuleList()
        self.nonlin = torch.nn.ModuleList()
        self.fir = torch.nn.ModuleList()
        # Delays of the signal applied after FIR
        self.delay = Delay(delays, device=device, dtype=dtype)
        # PA non-linearity model. Choose non-linearity basis
        self.nonlin = ChebyPolynom(nonlin_order, self._branch_num, dtype=dtype, device=device)
        # Filter, which takes into account PA memory effects
        self.fir = CentralFIR(N=tap_num, mode=padding, device=device, dtype=dtype)
        
    def forward(self, x_in):
        x_curr = self.delay(x_in)
        x_curr = self.nonlin(x_curr)
        output = self.fir(x_curr)
        return output