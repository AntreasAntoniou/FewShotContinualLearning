import torch
import torch.nn as nn
import torch.nn.functional as F


def BatchConv2DLayer(x, weight, stride, padding, dilation, bias=None):
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == x.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    b_i, b_j, c, h, w = x.shape
    b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape

    out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
    weight = weight.view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

    out = F.conv2d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                   padding=padding)


    out = out.view(b_j, b_i, out_channels, out.shape[-2], out.shape[-1])

    out = out.permute([1, 0, 2, 3, 4])

    if bias is not None:
        out = out + bias.unsqueeze(1).unsqueeze(3).unsqueeze(3)

    return out

def BatchConv1DLayer(x, weight, stride, padding, dilation, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == x.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        b_i, b_j, c, h = x.shape
        b_i, out_channels, in_channels, kernel_width_size = weight.shape

        out = x.permute([1, 0, 2, 3]).contiguous().view(b_j, b_i * c, h)
        weight = weight.view(b_i * out_channels, in_channels, kernel_width_size)

        out = F.conv1d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=b_i,
                       padding=padding)

        out = out.view(b_j, b_i, out_channels, out.shape[-1])

        out = out.permute([1, 0, 2, 3])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3)

        return out

def BatchLinearLayer(x, weight, bias=None):
    if bias is None:
        assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
    else:
        print(x.shape, weight.shape, bias.shape)
        assert x.shape[0] == weight.shape[0] and bias.shape[0] == x.shape[
            0], "dim=0 of bias must be equal in size to dim=0 of weight"

    out = torch.bmm(x, weight)

    if bias is not None:
        out = out + bias.unsqueeze(1)

    return out
