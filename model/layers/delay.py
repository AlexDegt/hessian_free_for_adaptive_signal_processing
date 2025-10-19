import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import numpy as np
import sys

class Delay(nn.Module):
    def __init__(self, delays, device=None, dtype=torch.complex128):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.delays_num = len(delays[0])
        self.branch_num = len(delays)
        # Define delay filters
        self.delay_filt_tap = 2*max(np.abs(list(chain(*delays)))) + 1
        self.delay_filter = torch.zeros(self.branch_num, self.delays_num, self.delay_filt_tap, dtype=self.dtype, device=self.device)
        
        self.branch_ind = torch.arange(self.branch_num)
        self.delays_ind = torch.arange(self.delays_num)

        indices = int(np.floor(self.delay_filt_tap/2)) + torch.tensor(delays, dtype=int)
        row_indices = torch.arange(indices.shape[0])[:, None]
        col_indices = torch.arange(indices.shape[1])
        self.delay_filter[row_indices, col_indices, indices] = 1

        self.delays = delays

    def forward(self, x):
        """
            Input signal x muse have dimensionality (batch_size, channel_num, sequence length)
        """
        batch_size = x.size(dim=0)
        tmp = torch.ones((batch_size, self.branch_num, self.delays_num, 1), dtype=self.dtype, device=x.device)
        tmp = torch.mul(tmp, x)
        tmp_view =  tmp.view(batch_size, self.branch_num * self.delays_num, -1)
        weight =  self.delay_filter.view(self.branch_num * self.delays_num, 1, self.delay_filt_tap)
        output = F.conv1d(tmp_view, weight, padding='same', groups=self.branch_num * self.delays_num).view(batch_size, self.branch_num, self.delays_num, -1)
        return output