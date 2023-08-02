import torch
from torch.autograd import Function


class OpenGLRound(Function):
    @staticmethod
    def forward(ctx, raw_result):
        # round at the end of the effect pipeline to 8 bit precision
        result = torch.clamp(raw_result, 0.0, 1.0)
        result *= 255.0
        result = torch.round(result).float()
        return result / 255.0

    @staticmethod
    # straight through
    def backward(ctx, grad_output):
        return grad_output


class StraightTroughFloor(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    # straight through
    def backward(ctx, grad_output):
        return grad_output


class LeakyClamp(Function):
    LEAKY_FACTOR = 0.1
    MIN = -0.5
    MAX = 0.5

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=LeakyClamp.MIN, max=LeakyClamp.MAX)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_output = torch.where(x < LeakyClamp.MIN, grad_output * LeakyClamp.LEAKY_FACTOR, grad_output)
        grad_output = torch.where(x > LeakyClamp.MAX, grad_output * LeakyClamp.LEAKY_FACTOR, grad_output)
        return grad_output, None, None


class StraightThroughClamp(Function):
    MIN = -0.5
    MAX = 0.5

    @staticmethod
    def forward(ctx, x):
        return torch.clamp(x, min=LeakyClamp.MIN, max=LeakyClamp.MAX)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def activation_f(x, name):
    if name == "linear":
        return x
    elif name == "sigmoid":
        return torch.sigmoid(x)
    else:
        raise ValueError("activation")


def custom_floor(x, gamma=0.95):
    nominator = torch.arctan((-gamma * torch.sin(2 * pi * x)) / (1 - gamma * torch.cos(2 * pi * x)))
    fraction = nominator / pi
    return x - 0.5 - fraction


opengl_round = OpenGLRound.apply
straight_through_floor = StraightTroughFloor.apply
leaky_clamp = LeakyClamp.apply
straight_through_clamp = StraightThroughClamp.apply
