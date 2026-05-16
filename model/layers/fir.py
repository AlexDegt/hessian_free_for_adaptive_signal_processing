import torch
import torch.nn.functional as F


class CausalFIR(torch.nn.Module):
    def __init__(self, N, in_=1, out_=1, dilation=1, device=None, dtype=torch.complex128):
        super().__init__()
        self.conv_complex = torch.nn.Conv1d(in_channels=in_, out_channels=out_, kernel_size=N, padding=0, dilation=dilation,
                                            groups=in_, bias=False, device=device, dtype=dtype)
        self.conv_complex.weight.data.real.zero_()
        self.conv_complex.weight.data.real[:, :, -1] = 1
        self.conv_complex.weight.data.imag.zero_()
        self.padding_zeros_num = (N - 1) * dilation

    def forward(self, x):
        return self.conv_complex(F.pad(x, (self.padding_zeros_num, 0)))


class CentralFIR(torch.nn.Module):
    def __init__(self, N, in_=1, out_=1, dilation=1, mode='same', device=None, dtype=torch.complex128):
        super().__init__()
        assert N % 2 == 1
        self.mode = mode
        assert mode == 'same' or mode == 'full' or mode == 'valid', \
            f"Parameter mode must equal \'same\', \'full\' or \'valid\', but input is {mode}"
        self.conv_complex = torch.nn.Conv1d(in_channels=in_, out_channels=out_, kernel_size=N, padding=0, dilation=dilation,
                                            groups=in_, bias=False, device=device, dtype=dtype)
        self.conv_complex.weight.data.real.zero_()
        self.conv_complex.weight.data.real[:, :, (N - 1) // 2] = 1.0
        self.conv_complex.weight.data.imag.zero_()

        self.same_padding_zeros_num = (N - 1) * dilation // 2
        self.full_padding_zeros_num = (N - 1) * dilation
        self.valid_padding_zeros_num = 0

        if self.mode == 'same':
            self.zero_padding = self.same_padding_zeros_num
        elif self.mode == 'full':
            self.zero_padding = self.full_padding_zeros_num
        elif self.mode == 'valid':
            self.zero_padding = self.valid_padding_zeros_num

    def forward(self, x):
        return self.conv_complex(F.pad(x, (self.zero_padding, self.zero_padding)))
        
class RealComplexFIR(torch.nn.Module):
    """
        Class of complex FIR (1D Convolution), implemented by means of real float-point parameters.
    """
    def __init__(self, N, in_=1, out_=1, mode='same', device=None, dtype=torch.float64):
        """
        Constructor of the RealComplexFIR class. 
        Complex-valued 1D convolution consists of 2 real-velued 1D convolutions.
        RealComplexFIR is implemented in real parameters, but works in the same way as complex 1D convolution.

        Args:
            N (int): Number of complex FIR taps.
            in_ (int): Number of input channels. FIR requires 1 input channel. Defaults to 1.
            out_ (int): Number of output channels. FIR requires 1 output channel. Defaults to 1.
            mode (str): Mode of convolution.
                'same' -- provides the same number of output samples.
                'valid' -- The output consists only of those elements that do not rely on the zero-padding.
                'full' -- The output is the full discrete linear convolution of the input.
                Default to 'same'.
            device (str, optional): Parameter shows which device to use for calculation on.
                'cpu', None -- CPU usage.
                'cuda' -- GPU usage.
            dtype (torch.float32 or torch.float64): Parameter type. Default to torch.float64.
        """
        super().__init__()
        assert N % 2 == 1
        self.mode = mode
        assert mode == 'same' or mode == 'full' or mode == 'valid', \
            f"Parameter mode must equal \'same\', \'full\' or \'valid\', but input is {mode}"
        
        self.conv_re = torch.nn.Conv1d(in_channels=in_, out_channels=out_, kernel_size=N, padding=0, dilation=1,
                                        groups=in_, bias=False, device=device, dtype=dtype)
        self.conv_im = torch.nn.Conv1d(in_channels=in_, out_channels=out_, kernel_size=N, padding=0, dilation=1,
                                        groups=in_, bias=False, device=device, dtype=dtype)
        self.conv_re.weight.data.zero_()
        self.conv_re.weight.data[:, :, (N - 1) // 2] = 1.0
        self.conv_im.weight.data.zero_()

        self.same_padding_zeros_num = (N - 1) // 2
        self.full_padding_zeros_num = (N - 1)
        self.valid_padding_zeros_num = 0

        if self.mode == 'same':
            self.zero_padding = self.same_padding_zeros_num
        elif self.mode == 'full':
            self.zero_padding = self.full_padding_zeros_num
        elif self.mode == 'valid':
            self.zero_padding = self.valid_padding_zeros_num

    def forward(self, x):
        y_re = self.conv_re(F.pad(x.real, (self.zero_padding, self.zero_padding))) - \
            self.conv_im(F.pad(x.imag, (self.zero_padding, self.zero_padding)))
        y_im = self.conv_re(F.pad(x.imag, (self.zero_padding, self.zero_padding))) + \
            self.conv_im(F.pad(x.real, (self.zero_padding, self.zero_padding)))
        return y_re + 1j * y_im