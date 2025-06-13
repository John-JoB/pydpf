import torch
from pydpf import pydpf
from math import ceil
from torch import Tensor
from torch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out, uniform_
import math
'''
The authors thank Xiongjie Chen for kindly providing the code for his paper 'Normalizing Flow-based Differentiable Particle Filters' which
we have heavily based our neural networks on.
'''

class SeedableConv2D(torch.nn.Conv2d):
    '''
    Pytorch doesn't allow Conv2D to be initialised with a random generator.
    We implement this behaviour.
    '''

    def __init__(self, *args, generator, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_reset_parameters(generator)

    def seeded_reset_parameters(self, generator) -> None:
        kaiming_uniform_(self.weight, a=math.sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                uniform_(self.bias, -bound, bound, generator=generator)


class SeedableLinear(torch.nn.Linear):
    '''
        Pytorch doesn't allow Linear to be initialised with a random generator.
        We implement this behaviour.
        '''

    def __init__(self, *args, generator, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_reset_parameters(generator)

    def seeded_reset_parameters(self, generator) -> None:
        kaiming_uniform_(self.weight, a=math.sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            uniform_(self.bias, -bound, bound, generator=generator)

class SeedableConvTranspose2D(torch.nn.ConvTranspose2d):
    '''
    Pytorch doesn't allow ConvTranspose2D to be initialised with a random generator.
    We implement this behaviour.
    '''

    def __init__(self, *args, generator, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_reset_parameters(generator)

    def seeded_reset_parameters(self, generator) -> None:
        kaiming_uniform_(self.weight, a=math.sqrt(5), generator=generator)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                uniform_(self.bias, -bound, bound, generator=generator)

class SeedableDropoutNd(pydpf.Module):
        __constants__ = ["p", "inplace"]
        p: float
        inplace: bool


        def __init__(self, p: float = 0.5, inplace: bool = False, generator:torch.Generator = torch.default_generator) -> None:
            super().__init__()
            if p < 0 or p > 1:
                raise ValueError(
                    f"dropout probability has to be between 0 and 1, but got {p}"
                )
            self.p = p
            if self.p < 0.0 or self.p > 1.0:
                raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
            self.inplace = inplace
            self.generator = generator

        def extra_repr(self) -> str:
            return f"p={self.p}, inplace={self.inplace}"


class SeedableDropout(SeedableDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input
        mask = (torch.rand(input.size(), device=input.device, dtype=torch.float32, generator = self.generator) > self.p).to(input.dtype)
        return (
            input.multiply_(mask) if self.inplace else input * mask
        )

class SeedableDropout2d(SeedableDropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        if not self.training:
            return input
        if input.dim() != 4:
            raise ValueError('SeedableDropout2d only supports Batch,Channel,Height,Width tensors')
        mask = pydpf.multiple_unsqueeze((torch.rand((input.size(0), input.size(1)), device=input.device, dtype=torch.float32, generator = self.generator) > self.p).to(input.dtype), 2, -1)
        return (
            input.multiply_(mask) if self.inplace else input * mask
        )


class FCNN(pydpf.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, generator):
        super().__init__()
        self.network = torch.nn.Sequential(
            SeedableLinear(in_dim, hidden_dim, generator = generator, device=generator.device),
            torch.nn.Tanh(),
            SeedableLinear(hidden_dim, hidden_dim, generator = generator, device=generator.device),
            torch.nn.Tanh(),
            SeedableLinear(hidden_dim, out_dim, generator = generator, device=generator.device),
        )

    def forward(self, x):
        return self.network(x)


class ObservationEncoder(pydpf.Module):
    def __new__(cls, hidden_size, dropout_keep_ratio = 0.7, generator = torch.default_generator):
        encode = torch.nn.Sequential(  # input: 3*24*24
            SeedableConv2D(3, 16, kernel_size=4, stride=2, padding=1, bias=False, generator = generator, device=generator.device),  # 16*12*12
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(16),
            SeedableConv2D(16, 32, kernel_size=4, stride=2, padding=1, bias=False, generator = generator, device=generator.device),  # 32*6*6
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            SeedableConv2D(32, 64, kernel_size=4, stride=2, padding=1, bias=False, generator = generator, device=generator.device),  # 64*3*3
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(64),
            torch.nn.Flatten(),
            SeedableDropout(p=1 - dropout_keep_ratio, generator = generator),
            SeedableLinear(64 * 3 * 3, hidden_size, generator = generator, device=generator.device),
        )
        return encode


class ObservationDecoder(pydpf.Module):
    def __new__(cls, hidden_size, dropout_keep_ratio = 0.7, generator = torch.default_generator):
        decode = torch.nn.Sequential(
            SeedableLinear(hidden_size, 3 * 3 * 64, generator = generator, device=generator.device),
            torch.nn.ReLU(True),
            torch.nn.Unflatten(-1, (64, 3, 3)),  # -1 means the last dim, (64, 3, 3)
            SeedableConvTranspose2D(64, 32, kernel_size=4, padding=1, stride=2, bias=False, generator = generator, device=generator.device),  # (32, 6,6)
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            SeedableConvTranspose2D(32, 16, kernel_size=4, padding=1, stride=2, bias=False, generator = generator, device=generator.device),  # (16, 12,12)
            SeedableDropout2d(p=1 - dropout_keep_ratio, generator=generator),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm2d(16),
            SeedableConvTranspose2D(16, 3, kernel_size=4, padding=1, stride=2, bias=False, generator = generator, device=generator.device),  # (3, 24, 24)
            SeedableDropout2d(p=1 - dropout_keep_ratio, generator=generator),
            torch.nn.BatchNorm2d(3),
            torch.nn.Sigmoid()
        )
        return decode

class StateEncoder(pydpf.Module):
    def __new__(cls, hidden_size, dropout_keep_ratio = 0.7, generator = torch.default_generator):
        particle_encode = torch.nn.Sequential(
            SeedableLinear(4, 16, generator = generator, device=generator.device),
            torch.nn.ReLU(True),
            SeedableLinear(16, 32, generator = generator, device=generator.device),
            torch.nn.ReLU(True),
            SeedableLinear(32, 64, generator = generator, device=generator.device),
            torch.nn.ReLU(True),
            SeedableDropout(p=1 - dropout_keep_ratio, generator=generator),
            SeedableLinear(64, hidden_size, generator = generator, device=generator.device),
        )
        return particle_encode

class RealNVP_cond(pydpf.Module):

    def __init__(self, dim, hidden_dim=8, base_network=FCNN, condition_on_dim=None, generator = torch.default_generator, zero_i = False):
        super().__init__()
        self.dim = dim
        self.condition_on_dim = condition_on_dim
        self.t1 = base_network(dim // 2 + self.condition_on_dim, ceil(dim / 2), hidden_dim, generator)
        self.t2 = base_network(ceil(dim / 2) + self.condition_on_dim, dim // 2, hidden_dim, generator)
        self.generator = generator
        if zero_i:
            self.zero_initialization()

    def zero_initialization(self, std=0.01):
        for layer in self.t1.network:
            if layer.__class__.__name__ == 'SeedableLinear':
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                layer.bias.data.fill_(0)
        for layer in self.t2.network:
            if layer.__class__.__name__ == 'SeedableLinear':
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                layer.bias.data.fill_(0)

    def forward(self, x, condition_on):
        lower, upper = x[..., :self.dim // 2], x[..., self.dim // 2:]
        lower_extended = torch.cat([lower, condition_on], dim=-1)
        t1_transformed = self.t1(lower_extended)
        upper = t1_transformed + upper
        upper_extended = torch.cat([upper, condition_on], dim=-1)
        t2_transformed = self.t2(upper_extended)
        lower = t2_transformed + lower
        z = torch.cat([lower, upper], dim=-1)
        return z, 0

    def inverse(self, z, condition_on):
        lower, upper = z[..., :self.dim // 2], z[..., self.dim // 2:]

        upper_extended = torch.cat([upper, condition_on], dim=-1)
        t2_transformed = self.t2(upper_extended)
        lower = lower - t2_transformed
        lower_extended = torch.cat([lower, condition_on], dim=-1)
        t1_transformed = self.t1(lower_extended)
        upper = upper - t1_transformed
        x = torch.cat([lower, upper], dim=-1)
        log_det = 0
        return x, log_det


class RealNVP_cond_(pydpf.Module):

    def __init__(self, dim, hidden_dim=8, base_network=FCNN, condition_on_dim=None, generator = torch.default_generator, zero_i = False):
        super().__init__()
        self.dim = dim
        self.condition_on_dim = condition_on_dim
        self.t1 = base_network(dim // 2 + self.condition_on_dim, ceil(dim / 2), hidden_dim, generator)
        self.s1 = base_network(dim // 2 + self.condition_on_dim, ceil(dim / 2), hidden_dim, generator)
        self.t2 = base_network(ceil(dim / 2) + self.condition_on_dim, dim // 2, hidden_dim, generator)
        self.s2 = base_network(ceil(dim / 2) + self.condition_on_dim, dim // 2, hidden_dim, generator)
        self.generator = generator
        self.zero_i = zero_i
        if zero_i:
            self.zero_initialization()

    def zero_initialization(self, std=0.01):
        for layer in self.t1.network:
            if layer.__class__.__name__ == 'SeedableLinear':
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                #layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s1.network:
            if layer.__class__.__name__ == 'SeedableLinear':
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                #layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.t2.network:
            if layer.__class__.__name__ == 'SeedableLinear':
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s2.network:
            if layer.__class__.__name__ == 'SeedableLinear':
                torch.nn.init.normal_(layer.weight, std=std, generator=self.generator)
                #layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x, condition_on):
        lower, upper = x[..., :self.dim // 2], x[..., self.dim // 2:]
        lower_extended = torch.cat([lower, condition_on], dim=-1)
        t1_transformed = self.t1(lower_extended)
        if self.zero_i:
            s1_transformed = torch.tensor(0, device=x.device)
        else:
            s1_transformed = self.s1(lower_extended)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        upper_extended = torch.cat([upper, condition_on], dim=-1)
        t2_transformed = self.t2(upper_extended)
        if self.zero_i:
            s2_transformed = torch.tensor(0, device=x.device)
        else:
            s2_transformed = self.s2(upper_extended)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=-1)
        log_det = torch.sum(s1_transformed, dim=-1) + torch.sum(s2_transformed, dim=-1)
        return z, log_det

    def inverse(self, z, condition_on):
        lower, upper = z[..., :self.dim // 2], z[..., self.dim // 2:]

        upper_extended = torch.cat([upper, condition_on], dim=-1)
        t2_transformed = self.t2(upper_extended)
        if self.zero_i:
            s2_transformed = torch.tensor(0, device=z.device)
        else:
            s2_transformed = self.s2(upper_extended)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        lower_extended = torch.cat([lower, condition_on], dim=-1)
        t1_transformed = self.t1(lower_extended)
        if self.zero_i:
            s1_transformed = torch.tensor(0, device=z.device)
        else:
            s1_transformed = self.s1(lower_extended)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=-1)
        log_det = torch.sum(-s1_transformed, dim=-1) + torch.sum(-s2_transformed, dim=-1)
        return x, log_det

class NormalizingFlowModel_cond(pydpf.Module):

    def __init__(self, prior, flows, device='cuda:0'):
        super().__init__()
        self.prior = prior
        self.device = device
        self.flows = torch.nn.ModuleList(flows).to(self.device)

    def forward(self, x, condition_on):
        b, m, d = x.shape
        log_det = torch.zeros((b, m)).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x, condition_on)
            log_det += ld
        return x, log_det

    def inverse(self, z, condition_on):
        b, m, d = z.shape
        log_det = torch.zeros((b, m)).to(self.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, condition_on)
            log_det += ld
        x = z
        return x, log_det

    def log_density(self, x, condition_on):
        z, log_det = self.forward(x, condition_on)
        prior_prob = self.prior.log_density(z)

        return prior_prob + log_det

    def sample(self, sample_size, condition_on):
        z = self.prior.sample(sample_size, device = self.device)
        return self.inverse(z, condition_on)[0]
