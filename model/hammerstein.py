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
        
        # Delays of the signal applied after FIR
        self.delay = Delay(delays, device=device, dtype=dtype)
        # PA non-linearity model. Choose non-linearity basis
        self.nonlin = ChebyPolynom(nonlin_order, self._branch_num, dtype=dtype, device=device)
        # Filter, which takes into account PA memory effects
        self.fir = CentralFIR(N=tap_num, mode=padding, device=device, dtype=dtype)
        
    def forward(self, x_in):
        x_curr = self.delay(x_in)

        # # AFIR normalization for block 2-nd order methods convergence
        # with torch.no_grad():
        #     afir_param = self.fir.conv_complex.weight.data
        #     nonlin_param = self.nonlin.nonlin[0].data
        #     alpha = afir_param.norm().item()
        #     afir_param /= alpha
        #     nonlin_param *= alpha

        x_curr = self.nonlin(x_curr)
        output = self.fir(x_curr)
        return output