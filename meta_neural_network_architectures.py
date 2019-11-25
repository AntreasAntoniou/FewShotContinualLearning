import logging
import math
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

import batch_layers as B


def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    return output_dict


def extract_params_and_check_for_missing_keys(current_dict, layer_dict):
    params_dict = extract_top_level_dict(current_dict=current_dict)
    for key in layer_dict.keys():
        if key not in params_dict:
            params_dict[key] = None

    return params_dict


class MetaConv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1):
        """
        A MetaConv1D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv1dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

        self.groups = groups

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv1d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        return out


class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1):
        """
        A MetaConv1D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters), requires_grad=True)

        self.groups = groups

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            # print([key for key in params.keys()])
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        return out


class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()
        self.input_shape = input_shape
        b, c = input_shape[:2]

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.empty(num_filters, c))
        nn.init.xavier_uniform_(self.weights)

        logging.debug("debug message", self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        # print(x.shape)
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
            # print(x.shape, params['weights'].shape)
        else:
            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        # print(x.shape)

        out = F.linear(input=x, weight=weight, bias=bias)
        # print(out.shape, weight.shape, self.input_shape)
        return out

    def reset_parameters(self):
        self.weights.data = self.weights.data * 0.
        fan_in, fan_out = _calculate_fan_in_and_fan_out(self.weights)
        std = 1. * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        a_array = torch.ones(self.weights.shape) * a
        a_array.to(self.weights.device)
        self.weights.data = self.weights.data + torch.distributions.Uniform(low=-a_array, high=a_array).rsample().to(
            self.weights.device)
        if self.use_bias:
            self.bias.data = self.bias.data * 0.


class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, num_support_set_steps, num_target_set_steps,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 use_per_step_bn_statistics=False, learnable_bn_gamma=True, learnable_bn_beta=True):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting. Also has the additional functionality of being able to store per step running stats and per step beta and gamma.
        """
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.num_features = num_features
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.learnable_gamma = learnable_bn_gamma
        self.learnable_beta = learnable_bn_beta

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(
                torch.zeros(num_support_set_steps + num_target_set_steps + 1, num_features),
                requires_grad=False)
            self.running_var = nn.Parameter(
                torch.ones(num_support_set_steps + num_target_set_steps + 1, num_features),
                requires_grad=False)
            self.bias = nn.Parameter(
                torch.zeros(num_support_set_steps + num_target_set_steps + 1, num_features),
                requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(
                torch.ones(num_support_set_steps + num_target_set_steps + 1, num_features),
                requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(self, input, num_step, training=False, backup_running_statistics=False):
        """
        Forward propagates by applying a bach norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            weight, bias = self.weight[num_step], self.bias[num_step]
            # print(num_step)
        else:
            running_mean = self.running_mean
            running_var = self.running_var
            weight, bias = self.weight, self.bias

        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum
        # print(running_mean.shape, running_var.shape)
        output = F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=True, momentum=momentum, eps=self.eps)

        return output

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean, requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var, requires_grad=False)

        self.to(self.weight.device)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class MetaConvNormLayerLeakyReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, per_step_bn_statistics,
                 num_support_set_steps, num_target_set_steps,
                 use_normalization=True, groups=1):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_shape: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaConvNormLayerLeakyReLU, self).__init__()
        self.input_shape = input_shape
        self.use_normalization = use_normalization
        self.use_per_step_bn_statistics = per_step_bn_statistics
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_support_set_steps = num_support_set_steps
        self.num_target_set_steps = num_target_set_steps
        self.stride = stride
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x

        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding, use_bias=self.use_bias,
                                    groups=self.groups)

        out = self.conv(out)
        if type(out) == tuple:
            out, _ = out

        if self.use_normalization:
            self.norm_layer = MetaBatchNormLayer(num_features=out.shape[1], track_running_stats=True,
                                                 use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                 num_support_set_steps=self.num_support_set_steps,
                                                 num_target_set_steps=self.num_target_set_steps)
            # print(out.shape)
            out = self.norm_layer.forward(out, num_step=0)

        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        conv_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            conv_params = params['conv']

        # if params is not None:
        #     print([key for key in params.keys()])
        # else:
        #     print(None)

        out = x

        out = self.conv(out, params=conv_params)

        if type(out) == tuple:
            out, _ = out

        if self.use_normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = F.leaky_relu(out)
        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()

class VGGActivationNormNetwork(nn.Module):
    def __init__(self, input_shape, num_output_classes, use_channel_wise_attention,
                 num_stages, num_filters, num_support_set_steps, num_target_set_steps):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(VGGActivationNormNetwork, self).__init__()

        self.total_layers = 0
        self.upscale_shapes = []
        self.num_filters = num_filters
        self.num_stages = num_stages
        self.input_shape = input_shape
        self.use_channel_wise_attention = use_channel_wise_attention
        self.num_output_classes = num_output_classes
        self.num_support_set_steps = num_support_set_steps
        self.num_target_set_steps = num_target_set_steps
        self.build_network()

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_stages):

            self.layer_dict['conv_{}'.format(i)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                              num_filters=self.num_filters,
                                                                              kernel_size=3, stride=1,
                                                                              padding=1,
                                                                              use_bias=True,
                                                                              groups=1, per_step_bn_statistics=True,
                                                                              num_support_set_steps=self.num_support_set_steps,
                                                                              num_target_set_steps=self.num_target_set_steps)

            out = self.layer_dict['conv_{}'.format(i)](out, training=True, num_step=0)

            out = F.max_pool2d(input=out, kernel_size=2, stride=2, padding=0)

        out = out.view((out.shape[0], -1))

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=out.shape,
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, dropout_training=None, params=None, training=False,
                backup_running_statistics=False, return_features=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            params = {key: value for key, value in params.items()}
            # print([key for key, value in param_dict.items()])
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        # print([key for key, value in param_dict.items() if value is not None])

        for i in range(self.num_stages):


            out = self.layer_dict['conv_{}'.format(i)](out, params=param_dict['conv_{}'.format(i)], training=training,
                                                       backup_running_statistics=backup_running_statistics,
                                                       num_step=num_step)

            out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)


        features = out

        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])

        if return_features:
            return out, features
        else:
            return out

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

class VGGActivationNormNetworkWithAttention(nn.Module):
    def __init__(self, input_shape, num_output_classes, use_channel_wise_attention,
                 num_stages, num_filters, num_support_set_steps, num_target_set_steps):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(VGGActivationNormNetworkWithAttention, self).__init__()

        self.total_layers = 0
        self.upscale_shapes = []
        self.num_filters = num_filters
        self.num_stages = num_stages
        self.input_shape = input_shape
        self.use_channel_wise_attention = use_channel_wise_attention
        self.num_output_classes = num_output_classes
        self.num_support_set_steps = num_support_set_steps
        self.num_target_set_steps = num_target_set_steps
        self.build_network()

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_stages):

            if self.use_channel_wise_attention:
                out_channels = F.avg_pool2d(out, out.shape[-1]).squeeze()
                out_channels = out_channels.view(out_channels.shape[0], -1)

                self.layer_dict['channel_wise_attention_linear_pred_network_{}'.format(i)] = MetaLinearLayer(
                    input_shape=out_channels.shape, num_filters=out_channels.shape[-1], use_bias=True)
                channel_wise_attention_regions = self.layer_dict[
                    'channel_wise_attention_linear_pred_network_{}'.format(i)].forward(
                    out_channels, params=None)

                channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)
                # print(out.shape, channel_wise_attention_regions.shape)
                out = out * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)
            # print(i, out.shape)
            self.layer_dict['conv_{}'.format(i)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                              num_filters=self.num_filters,
                                                                              kernel_size=3, stride=1,
                                                                              padding=1,
                                                                              use_bias=True,
                                                                              groups=1, per_step_bn_statistics=True,
                                                                              num_support_set_steps=self.num_support_set_steps,
                                                                              num_target_set_steps=self.num_target_set_steps)

            out = self.layer_dict['conv_{}'.format(i)](out, training=True, num_step=0)

            out = F.avg_pool2d(input=out, kernel_size=2)

        if self.use_channel_wise_attention:
            out_channels = F.avg_pool2d(out, out.shape[-1]).squeeze()

            out_channels = out_channels.view(out_channels.shape[0], -1)

            self.layer_dict['channel_wise_attention_linear_pred_network_out'] = MetaLinearLayer(
                input_shape=out_channels.shape, num_filters=out_channels.shape[-1], use_bias=True)

            channel_wise_attention_regions = self.layer_dict[
                'channel_wise_attention_linear_pred_network_out'].forward(
                out_channels, params=None)

            channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)
            out = out * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view((out.shape[0], -1))

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, dropout_training=None, params=None, training=False,
                backup_running_statistics=False, return_features=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            params = {key: value for key, value in params.items()}
            # print([key for key, value in param_dict.items()])
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        # print([key for key, value in param_dict.items() if value is not None])

        for i in range(self.num_stages):

            if self.use_channel_wise_attention:
                out_channels = F.avg_pool2d(out, out.shape[-1]).squeeze()
                b, c = out_channels.shape[:2]
                out_channels = out_channels.view(out_channels.shape[0], c)
                # if param_dict['channel_wise_attention_linear_pred_network_{}'.format(i)] is not None:
                # print(out_channels.shape, [(key, value.shape) for key, value in param_dict['channel_wise_attention_linear_pred_network_{}'.format(i)].items()])
                channel_wise_attention_regions = self.layer_dict[
                    'channel_wise_attention_linear_pred_network_{}'.format(i)].forward(
                    out_channels, params=param_dict['channel_wise_attention_linear_pred_network_{}'.format(i)])
                channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)
                # print(out.shape, channel_wise_attention_regions.shape)
                out = out * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

            out = self.layer_dict['conv_{}'.format(i)](out, params=param_dict['conv_{}'.format(i)], training=training,
                                                       backup_running_statistics=backup_running_statistics,
                                                       num_step=num_step)

            out = F.avg_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if self.use_channel_wise_attention:
            out_channels = F.avg_pool2d(out, out.shape[-1]).squeeze()
            out_channels = out_channels.view(out_channels.shape[0], -1)

            channel_wise_attention_regions = self.layer_dict[
                'channel_wise_attention_linear_pred_network_out'].forward(
                out_channels, params=param_dict['channel_wise_attention_linear_pred_network_out'])

            channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)

            out = out * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

        features = out
        out = F.avg_pool2d(out, out.shape[-1])

        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])

        if return_features:
            return out, features
        else:
            return out

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


class FCCActivationNormNetwork(nn.Module):
    def __init__(self, im_shape, num_output_classes, args, device, use_bn, num_stages=None, use_bias=True,
                 meta_classifier=True):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(FCCActivationNormNetwork, self).__init__()
        self.device = device
        self.args = args
        self.input_shape = list(im_shape)
        self.num_output_classes = num_output_classes
        self.meta_classifier = meta_classifier
        self.num_stages = num_stages
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.build_network()

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        out = out.view(out.size(0), -1)
        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_stages):
            self.layer_dict['fcc_{}'.format(i)] = MetaLinearLayer(input_shape=out.shape, num_filters=40, use_bias=False)
            out = self.layer_dict['fcc_{}'.format(i)].forward(out)
            if self.use_bn:
                self.layer_dict['fcc_bn_{}'.format(i)] = MetaBatchNormLayer(num_features=out.shape[1], args=self.args,
                                                                            use_per_step_bn_statistics=True)
                out = self.layer_dict['fcc_bn_{}'.format(i)].forward(out, num_step=0)
            out = F.leaky_relu(out)

        out = out.view(out.shape[0], -1)

        self.layer_dict['preds_linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                          num_filters=self.num_output_classes, use_bias=self.use_bias)

        out = self.layer_dict['preds_linear'](out)
        print("FCCActivationNormNetwork build", out.shape)

    def forward(self, x, num_step, params=None, training=False,
                backup_running_statistics=False, return_features=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x
        out = out.view(out.size(0), -1)
        for i in range(self.num_stages):
            out = self.layer_dict['fcc_{}'.format(i)](out, params=param_dict['fcc_{}'.format(i)])
            if self.use_bn:
                out = self.layer_dict['fcc_bn_{}'.format(i)].forward(out, num_step=num_step,
                                                                     params=None, training=training,
                                                                     backup_running_statistics=backup_running_statistics)
            out = F.leaky_relu(out)
            features = out

        out = out.view(out.size(0), -1)
        out = self.layer_dict['preds_linear'](out, param_dict['preds_linear'])

        if return_features:
            return out, features
        else:
            return out

    def reset_parameters(self):
        for name, module in self.named_modules():
            if type(module) == MetaLinearLayer:
                # print("reset", name)
                module.reset_parameters()

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


class DenseNetActivationNormNetwork(nn.Module):
    def __init__(self, im_shape, num_output_filters, num_stages, num_blocks_per_stage, num_filters, num_output_classes,
                 args):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(DenseNetActivationNormNetwork, self).__init__()
        self.args = args
        self.num_filters = num_filters
        self.num_output_filters = num_output_filters
        self.input_shape = list(im_shape)
        self.num_stages = num_stages
        self.num_output_classes = num_output_classes
        self.num_blocks_per_stage = num_blocks_per_stage
        self.layer_dict = nn.ModuleDict()
        self.build_network()

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['stem_conv'] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                  num_filters=self.num_filters,
                                                                  kernel_size=3, stride=1,
                                                                  padding=1,
                                                                  use_bias=True, args=self.args,
                                                                  normalization=True,
                                                                  meta_layer=True,
                                                                  no_bn_learnable_params=False)
        out = self.layer_dict['stem_conv'](out, training=True, num_step=0)
        out = F.avg_pool2d(out, 2)
        self.layer_dict['transition_init_conv'] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                             num_filters=out.shape[1],
                                                                             kernel_size=1, stride=1,
                                                                             padding=1,
                                                                             use_bias=False, args=self.args,
                                                                             normalization=True,
                                                                             meta_layer=True,
                                                                             no_bn_learnable_params=False)
        out = self.layer_dict['transition_init_conv'](out, training=True, num_step=0)

        for i in range(self.num_stages):
            for j in range(self.num_blocks_per_stage):
                self.layer_dict['conv_{}_{}'.format(i, j)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                                        num_filters=self.num_filters,
                                                                                        kernel_size=3, stride=1,
                                                                                        padding=1,
                                                                                        use_bias=False, args=self.args,
                                                                                        normalization=True,
                                                                                        meta_layer=True,
                                                                                        no_bn_learnable_params=False)
                cur = self.layer_dict['conv_{}_{}'.format(i, j)](out, training=True, num_step=0)
                out = torch.cat([out, cur], dim=1)

            out = F.avg_pool2d(out, 2)
            self.layer_dict['transition_conv_{}'.format(i)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                                         num_filters=out.shape[1],
                                                                                         kernel_size=1, stride=1,
                                                                                         padding=0,
                                                                                         use_bias=False, args=self.args,
                                                                                         normalization=True,
                                                                                         meta_layer=True,
                                                                                         no_bn_learnable_params=False)
            out = self.layer_dict['transition_conv_{}'.format(i)](out, training=True, num_step=0)

        out = F.avg_pool2d(out, out.shape[2])

        out = out.view(out.shape[0], -1)

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, params=None, training=False,
                backup_running_statistics=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        out = self.layer_dict['stem_conv'](out, params=param_dict['stem_conv'], training=training,
                                           backup_running_statistics=backup_running_statistics,
                                           num_step=num_step)
        out = F.avg_pool2d(out, 2)
        out = self.layer_dict['transition_init_conv'](out, params=param_dict['transition_init_conv'], training=training,
                                                      backup_running_statistics=backup_running_statistics,
                                                      num_step=num_step)
        for i in range(self.num_stages):
            for j in range(self.num_blocks_per_stage):
                cur = self.layer_dict['conv_{}_{}'.format(i, j)](out, params=param_dict['conv_{}_{}'.format(i, j)],
                                                                 training=training,
                                                                 backup_running_statistics=backup_running_statistics,
                                                                 num_step=num_step)
                out = torch.cat([out, cur], dim=1)

            out = F.avg_pool2d(out, 2)
            out = self.layer_dict['transition_conv_{}'.format(i)](out, training=training,
                                                                  backup_running_statistics=backup_running_statistics,
                                                                  num_step=num_step,
                                                                  params=param_dict['transition_conv_{}'.format(i)])

        out = F.avg_pool2d(out, out.shape[2])

        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])

        return out

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

        # if self.args.norm_layer == "batch_norm":
        #     self.layer_dict['fc_norm_layer'].restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


class MetaParallelConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaParallelConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

        self.groups = groups

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            weight = self.weight.unsqueeze(0).repeat([x.shape[0]] + [1 for i in range(len(self.weight.shape))])

            if self.use_bias:
                bias = self.bias.unsqueeze(0).repeat([x.shape[0]] + [1 for i in range(len(self.bias.shape))])

            else:
                bias = None

        out = B.BatchConv2DLayer(x=x, weight=weight, bias=bias, stride=self.stride, padding=self.padding,
                                 dilation=self.dilation_rate)
        return out


class MetaParallelConv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaParallelConv1dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

        self.groups = groups

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            weight = self.weight.unsqueeze(0).repeat([x.shape[0]] + [1 for i in range(len(self.weight.shape))])

            if self.use_bias:
                bias = self.bias.unsqueeze(0).repeat([x.shape[0]] + [1 for i in range(len(self.bias.shape))])

            else:
                bias = None

        out = B.BatchConv1DLayer(x=x, weight=weight, bias=bias, stride=self.stride, padding=self.padding,
                                 dilation=self.dilation_rate)
        return out


class MetaParallelLinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaParallelLinearLayer, self).__init__()
        num_filters = out_channels
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(in_channels, num_filters))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            weight = self.weight.unsqueeze(0).repeat([x.shape[0]] + [1 for i in range(len(self.weight.shape))])

            if self.use_bias:
                bias = self.bias.unsqueeze(0).repeat([x.shape[0]] + [1 for i in range(len(self.bias.shape))])

            else:
                bias = None

        out = B.BatchLinearLayer(x=x, weight=weight, bias=bias)
        return out


class MetaParallelBatchNormLayer(nn.Module):
    def __init__(self, input_shape, num_support_set_steps, num_target_set_steps):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaParallelBatchNormLayer, self).__init__()

        self.meta_bn_layer = MetaBatchNormLayer(num_features=input_shape[2],
                                                num_support_set_steps=num_support_set_steps,
                                                num_target_set_steps=num_target_set_steps)

    def forward(self, x, step_idx):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        input_shape = x.shape
        target_shape = [x.shape[0] * x.shape[1]] + [item for item in x.shape[2:]]

        x = x.contiguous().view(target_shape)
        out = self.meta_bn_layer.forward(input=x, num_step=step_idx)

        out = out.view(input_shape)
        return out


class MetaParallelSqueezeExciteAttention(nn.Module):
    def __init__(self, input_shape, num_filters, num_layers):
        super(MetaParallelSqueezeExciteAttention, self).__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.ones(self.input_shape)
        out = x

        out_channels = out.mean(3).mean(3).squeeze()
        out = out_channels
        for i in range(self.num_layers - 1):
            self.layer_dict['channel_wise_attention_linear_pred_network_{}'.format(i)] = MetaParallelLinearLayer(
                in_channels=out.shape[2], out_channels=self.num_filters, use_bias=True)

            out = self.layer_dict[
                'channel_wise_attention_linear_pred_network_{}'.format(i)].forward(
                out, params=None)

            out = F.leaky_relu(out)

        self.layer_dict['channel_wise_attention_output_layer'] = MetaParallelLinearLayer(
            in_channels=out.shape[2], out_channels=x.shape[2], use_bias=True)

        out = self.layer_dict[
            'channel_wise_attention_output_layer'].forward(
            out, params=None)

        channel_wise_attention_regions = out
        channel_wise_attention_regions = channel_wise_attention_regions.sigmoid()
        # print(x.shape, channel_wise_attention_regions.unsqueeze(3).unsqueeze(3).shape)
        out = x * channel_wise_attention_regions.unsqueeze(3).unsqueeze(3)

        print('Built {} with output'.format(type(self)), out.shape, '\nModules\n', self, '\nParameters',
              "".join(["\n{}_{}".format(name, param.shape) for name, param in self.named_parameters()]))

    def forward(self, x, params=None):
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        out_channels = out.mean(3).mean(3).squeeze()
        for i in range(self.num_layers - 1):
            out = self.layer_dict[
                'channel_wise_attention_linear_pred_network_{}'.format(i)].forward(
                out_channels, params=param_dict['channel_wise_attention_linear_pred_network_{}'.format(i)])

            out = F.leaky_relu(out)

        out = self.layer_dict[
            'channel_wise_attention_output_layer'].forward(
            out, params=param_dict['channel_wise_attention_output_layer'])

        channel_wise_attention_regions = out
        channel_wise_attention_regions = channel_wise_attention_regions.sigmoid()
        out = x * channel_wise_attention_regions.unsqueeze(3).unsqueeze(3)
        return out


class ParallelVGGActivationNormNetworkWithAttention(nn.Module):
    def __init__(self, input_shape, num_output_classes, use_channel_wise_attention, num_support_set_steps,
                 num_target_set_steps,
                 num_stages, num_filters):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(ParallelVGGActivationNormNetworkWithAttention, self).__init__()

        self.num_filters = num_filters
        self.input_shape = input_shape
        self.use_channel_wise_attention = use_channel_wise_attention
        self.num_stages = num_stages
        self.num_support_set_steps = num_support_set_steps
        self.num_target_set_steps = num_target_set_steps
        self.num_output_classes = num_output_classes
        self.build_module()

        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_module(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_stages):

            if self.use_channel_wise_attention:
                self.layer_dict['squeeze_excite_attention_{}'.format(i)] = MetaParallelSqueezeExciteAttention(
                    input_shape=out.shape, num_layers=2, num_filters=64)
                out = self.layer_dict['squeeze_excite_attention_{}'.format(i)].forward(out, params=None)

            self.layer_dict['conv_{}'.format(i)] = MetaParallelConv2dLayer(in_channels=out.shape[2],
                                                                           out_channels=self.num_filters, padding=1,
                                                                           stride=1, dilation_rate=1, use_bias=True,
                                                                           kernel_size=3)

            out = self.layer_dict['conv_{}'.format(i)](out, params=None)

            self.layer_dict['bn_{}'.format(i)] = MetaParallelBatchNormLayer(input_shape=out.shape,
                                                                            num_support_set_steps=self.num_support_set_steps,
                                                                            num_target_set_steps=self.num_target_set_steps)
            out = self.layer_dict['bn_{}'.format(i)].forward(out, step_idx=0)

            out = F.leaky_relu(out)

        if self.use_channel_wise_attention:
            self.layer_dict['squeeze_excite_attention_pre_linear'] = MetaParallelSqueezeExciteAttention(
                input_shape=out.shape, num_layers=2, num_filters=64)
            out = self.layer_dict['squeeze_excite_attention_pre_linear'].forward(out, params=None)

        out_shape = out.shape
        out = out.view((-1, out_shape[2], out_shape[3], out_shape[4]))
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view((out_shape[0], out_shape[1], -1))

        self.layer_dict['linear'] = MetaParallelLinearLayer(in_channels=out.shape[2],
                                                            out_channels=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, dropout_training=None, params=None, training=False,
                backup_running_statistics=False, arg_max_softmax=False, return_features=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        for i in range(self.num_stages):

            if self.use_channel_wise_attention:
                out = self.layer_dict['squeeze_excite_attention_{}'.format(i)].forward(out, params=param_dict[
                    'squeeze_excite_attention_{}'.format(i)])

            out = self.layer_dict['conv_{}'.format(i)](out, params=param_dict['conv_{}'.format(i)])

            out = self.layer_dict['bn_{}'.format(i)].forward(out, step_idx=num_step)

            out = F.leaky_relu(out)

        if self.use_channel_wise_attention:
            out = self.layer_dict['squeeze_excite_attention_pre_linear'].forward(out, params=param_dict[
                'squeeze_excite_attention_pre_linear'])

        out_shape = out.shape
        out = out.view((-1, out_shape[2], out_shape[3], out_shape[4]))
        out = F.avg_pool2d(out, out.shape[-1])
        features = out
        out = out.view((out_shape[0], out_shape[1], -1))

        out = self.layer_dict['linear'].forward(out, params=param_dict['linear'])

        if return_features:
            return out, features
        else:
            return out

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


class SqueezeExciteLayer(nn.ModuleDict):
    def __init__(self, input_shape, num_filters, num_layers, num_support_set_steps, num_target_set_steps):
        super(SqueezeExciteLayer, self).__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_support_set_steps = num_support_set_steps
        self.num_target_set_steps = num_target_set_steps
        self.build_block()

    def build_block(self):
        self.layer_dict = nn.ModuleDict()
        x_dummy = torch.zeros(self.input_shape)
        out = x_dummy
        out = F.avg_pool2d(out, out.shape[-1]).squeeze()

        for i in range(self.num_layers - 1):
            self.layer_dict['attention_network_hidden_{}'.format(i)] = MetaLinearLayer(input_shape=out.shape,
                                                                                       use_bias=True,
                                                                                       num_filters=self.num_filters)

            out = self.layer_dict['attention_network_hidden_{}'.format(i)].forward(out, params=None)
            self.layer_dict['LeakyReLU_{}'.format(i)] = nn.LeakyReLU()
            out = self.layer_dict['LeakyReLU_{}'.format(i)].forward(out)

        self.layer_dict['attention_network_output_layer'] = MetaLinearLayer(input_shape=out.shape, use_bias=True,
                                                                            num_filters=x_dummy.shape[1])

        channel_wise_attention_regions = self.layer_dict[
            'attention_network_output_layer'].forward(
            out, params=None)

        channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)
        out = x_dummy * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

        print('Built', type(self), 'with output', out.shape, self)

    def forward(self, x, num_step=0, params=None):

        param_dict = dict()

        if params is not None:
            params = {key: value for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x
        out = F.avg_pool2d(out, out.shape[-1]).squeeze()

        for i in range(self.num_layers - 1):
            # print(out.shape)
            out = self.layer_dict[
                'attention_network_hidden_{}'.format(i)].forward(
                out, params=param_dict['attention_network_hidden_{}'.format(i)])
            out = self.layer_dict['LeakyReLU_{}'.format(i)].forward(out)
            # print(out.shape)
        channel_wise_attention_regions = self.layer_dict[
            'attention_network_output_layer'].forward(
            out, params=param_dict['attention_network_output_layer'])

        channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)
        out = x * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

        return out


# class VGGActivationNormNetworkWithAttention(nn.Module):
#     def __init__(self, input_shape, num_output_classes, use_channel_wise_attention,
#                  num_stages, num_filters, num_support_set_steps, num_target_set_steps, num_blocks_per_stage):
#         """
#         Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
#         used at inference time. Enables inner loop optimization readily.
#         :param im_shape: The input image batch shape.
#         :param num_output_classes: The number of output classes of the network.
#         :param args: A named tuple containing the system's hyperparameters.
#         :param device: The device to run this on.
#         :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
#         be enabled.
#         """
#         super(VGGActivationNormNetworkWithAttention, self).__init__()
#
#         self.total_layers = 0
#         self.upscale_shapes = []
#         self.num_filters = num_filters
#         self.num_stages = num_stages
#         self.input_shape = input_shape
#         self.use_channel_wise_attention = use_channel_wise_attention
#         self.num_output_classes = num_output_classes
#         self.num_support_set_steps = num_support_set_steps
#         self.num_target_set_steps = num_target_set_steps
#         self.build_network()
#
#     def build_network(self):
#         """
#         Builds the network before inference is required by creating some dummy inputs with the same input as the
#         self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
#         sets output shapes for each layer.
#         """
#         x = torch.zeros(self.input_shape)
#         out = x
#         self.layer_dict = nn.ModuleDict()
#
#         for i in range(self.num_stages):
#
#             if self.use_channel_wise_attention:
#                 self.layer_dict['attention_layer_{}'.format(i)] = SqueezeExciteLayer(input_shape=out.shape,
#                                                                                      num_filters=32,
#                                                                                      num_layers=2)
#                 out = self.layer_dict['attention_layer_{}'.format(i)].forward(out)
#
#             self.layer_dict['conv_{}'.format(i)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
#                                                                               num_filters=self.num_filters,
#                                                                               kernel_size=3, stride=1,
#                                                                               padding=1,
#                                                                               use_bias=True,
#                                                                               groups=1, per_step_bn_statistics=True,
#                                                                               num_support_set_steps=self.num_support_set_steps,
#                                                                               num_target_set_steps=self.num_target_set_steps)
#
#             out = self.layer_dict['conv_{}'.format(i)](out, training=True, num_step=0)
#
#             out = F.avg_pool2d(input=out, kernel_size=2)
#
#         if self.use_channel_wise_attention:
#             self.layer_dict['attention_pre_logit_layer'] = SqueezeExciteLayer(input_shape=out.shape,
#                                                                               num_filters=32,
#                                                                               num_layers=2)
#             out = self.layer_dict['attention_pre_logit_layer'].forward(out)
#
#         out = F.avg_pool2d(out, out.shape[-1])
#         out = out.view((out.shape[0], -1))
#
#         self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
#                                                     num_filters=self.num_output_classes, use_bias=True)
#
#         out = self.layer_dict['linear'](out)
#         print("VGGNetwork build", out.shape)
#
#     def forward(self, x, num_step, dropout_training=None, params=None, training=False,
#                 backup_running_statistics=False, return_features=False):
#         """
#         Forward propages through the network. If any params are passed then they are used instead of stored params.
#         :param x: Input image batch.
#         :param num_step: The current inner loop step number
#         :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
#          same as the layer names then they will be used instead.
#         :param training: Whether this is training (True) or eval time.
#         :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
#         then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
#         :return: Logits of shape b, num_output_classes.
#         """
#         param_dict = dict()
#
#         if params is not None:
#             params = {key: value[0] for key, value in params.items()}
#             # print([key for key, value in param_dict.items()])
#             param_dict = extract_top_level_dict(current_dict=params)
#
#         for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
#             path_bits = name.split(".")
#             layer_name = path_bits[0]
#             if layer_name not in param_dict:
#                 param_dict[layer_name] = None
#
#         out = x
#
#         # print([key for key, value in param_dict.items() if value is not None])
#
#         for i in range(self.num_stages):
#
#             if self.use_channel_wise_attention:
#                 out = self.layer_dict['attention_layer_{}'.format(i)].forward(out, params=param_dict[
#                     'attention_layer_{}'.format(i)])
#
#             out = self.layer_dict['conv_{}'.format(i)](out, params=param_dict['conv_{}'.format(i)], training=training,
#                                                        backup_running_statistics=backup_running_statistics,
#                                                        num_step=num_step)
#
#             out = F.avg_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
#
#         if self.use_channel_wise_attention:
#             out = self.layer_dict['attention_pre_logit_layer'].forward(out, params=param_dict[
#                 'attention_pre_logit_layer'])
#
#         features = out
#         out = F.avg_pool2d(out, out.shape[-1])
#
#         out = out.view(out.size(0), -1)
#         out = self.layer_dict['linear'](out, param_dict['linear'])
#
#         if return_features:
#             return out, features
#         else:
#             return out
#
#     def restore_backup_stats(self):
#         """
#         Reset stored batch statistics from the stored backup.
#         """
#         for name, module in self.named_modules():
#             if type(module) == MetaBatchNormLayer:
#                 module.restore_backup_stats()
#
#     def zero_grad(self, params=None):
#         if params is None:
#             for param in self.parameters():
#                 if param.requires_grad == True:
#                     if param.grad is not None:
#                         if torch.sum(param.grad) > 0:
#                             print(param.grad)
#                             param.grad.zero_()
#         else:
#             for name, param in params.items():
#                 if param.requires_grad == True:
#                     if param.grad is not None:
#                         if torch.sum(param.grad) > 0:
#                             print(param.grad)
#                             param.grad.zero_()
#                             params[name].grad = None


class DenseNetActivationNormNetworkWithAttention(nn.Module):
    def __init__(self, input_shape, num_output_classes, use_channel_wise_attention,
                 num_stages, num_filters, num_blocks_per_stage, num_support_set_steps, num_target_set_steps):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(DenseNetActivationNormNetworkWithAttention, self).__init__()

        self.total_layers = 0
        self.upscale_shapes = []
        self.num_filters = num_filters
        self.num_stages = num_stages
        self.input_shape = input_shape
        self.use_channel_wise_attention = use_channel_wise_attention
        self.num_output_classes = num_output_classes
        self.num_blocks_per_stage = num_blocks_per_stage
        self.num_support_set_steps = num_support_set_steps
        self.num_target_set_steps = num_target_set_steps
        self.build_network()

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_stages):
            for j in range(self.num_blocks_per_stage):

                block_input = out

                if self.use_channel_wise_attention:
                    self.layer_dict['attention_layer_{}_{}'.format(i, j)] = SqueezeExciteLayer(input_shape=out.shape,
                                                                                               num_filters=32,
                                                                                               num_layers=2,
                                                                                               num_support_set_steps=self.num_support_set_steps,
                                                                                               num_target_set_steps=self.num_target_set_steps)
                    out = self.layer_dict['attention_layer_{}_{}'.format(i, j)].forward(out)

                self.layer_dict['bottleneck_conv_{}_{}'.format(i, j)] = MetaConvNormLayerLeakyReLU(
                    input_shape=out.shape,
                    num_filters=self.num_filters,
                    kernel_size=1, stride=1,
                    padding=0,
                    use_bias=True,
                    groups=1, per_step_bn_statistics=True,
                    num_support_set_steps=self.num_support_set_steps,
                    num_target_set_steps=self.num_target_set_steps)

                out = self.layer_dict['bottleneck_conv_{}_{}'.format(i, j)](out, training=True, num_step=0)

                self.layer_dict['conv_{}_{}'.format(i, j)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                                        num_filters=self.num_filters,
                                                                                        kernel_size=3, stride=1,
                                                                                        padding=1,
                                                                                        use_bias=True,
                                                                                        groups=1,
                                                                                        per_step_bn_statistics=True,
                                                                                        num_support_set_steps=self.num_support_set_steps,
                                                                                        num_target_set_steps=self.num_target_set_steps)

                out = self.layer_dict['conv_{}_{}'.format(i, j)](out, training=True, num_step=0)

                out = torch.cat([out, block_input], dim=1)

            out = F.avg_pool2d(input=out, kernel_size=2)

            self.layer_dict['transition_conv_{}'.format(i)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                                         num_filters=self.num_filters,
                                                                                         kernel_size=1, stride=1,
                                                                                         padding=0,
                                                                                         use_bias=True,
                                                                                         groups=1,
                                                                                         per_step_bn_statistics=True,
                                                                                         num_support_set_steps=self.num_support_set_steps,
                                                                                         num_target_set_steps=self.num_target_set_steps)

            out = self.layer_dict['transition_conv_{}'.format(i)](out, training=True, num_step=0)

        if self.use_channel_wise_attention:
            self.layer_dict['attention_pre_logit_layer'] = SqueezeExciteLayer(input_shape=out.shape,
                                                                              num_filters=32,
                                                                              num_layers=2,
                                                                              num_support_set_steps=self.num_support_set_steps,
                                                                              num_target_set_steps=self.num_target_set_steps)
            out = self.layer_dict['attention_pre_logit_layer'].forward(out)

        features_avg = F.avg_pool2d(out, out.shape[-1]).squeeze()
        # self.layer_dict['relational_pool'] = MetaBatchRelationalModule(input_shape=out.shape, use_coordinates=True,
        #                                                                num_target_set_steps=self.num_target_set_steps,
        #                                                                num_support_set_steps=self.num_support_set_steps,
        #                                                                output_units=out.shape[1])
        # out = self.layer_dict['relational_pool'].forward(out, num_step=0)
        out = features_avg

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=out.shape,
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, dropout_training=None, params=None, training=False,
                backup_running_statistics=False, return_features=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            params = {key: value for key, value in params.items()}
            # print([key for key, value in param_dict.items()])
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        # print([key for key, value in param_dict.items() if value is not None])

        for i in range(self.num_stages):
            for j in range(self.num_blocks_per_stage):

                block_input = out

                if self.use_channel_wise_attention:
                    out = self.layer_dict['attention_layer_{}_{}'.format(i, j)].forward(out, num_step=num_step,
                                                                                        params=param_dict[
                                                                                            'attention_layer_{}_{}'.format(
                                                                                                i, j)])

                out = self.layer_dict['bottleneck_conv_{}_{}'.format(i, j)](out, training=True, num_step=num_step,
                                                                            params=param_dict[
                                                                                'bottleneck_conv_{}_{}'.format(i, j)])

                out = self.layer_dict['conv_{}_{}'.format(i, j)](out, training=True, num_step=num_step,
                                                                 params=param_dict['conv_{}_{}'.format(i, j)])

                out = torch.cat([out, block_input], dim=1)

            out = F.avg_pool2d(input=out, kernel_size=2)

            out = self.layer_dict['transition_conv_{}'.format(i)](out, training=True, num_step=num_step,
                                                                  params=param_dict['transition_conv_{}'.format(i)])

        if self.use_channel_wise_attention:
            out = self.layer_dict['attention_pre_logit_layer'].forward(out, params=param_dict[
                'attention_pre_logit_layer'])
        features = out
        features_avg = F.avg_pool2d(out, out.shape[-1]).squeeze()

        # out = F.avg_pool2d(out, out.shape[-1])

        # out = self.layer_dict['relational_pool'].forward(out, params=param_dict['relational_pool'], num_step=num_step)

        out = features_avg

        out = self.layer_dict['linear'](out, param_dict['linear'])

        if return_features:
            return out, features
        else:
            return out

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


class ResNetActivationNormNetworkWithAttention(nn.Module):
    def __init__(self, input_shape, num_output_classes, use_channel_wise_attention,
                 num_stages, num_filters, num_blocks_per_stage, num_support_set_steps, num_target_set_steps):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(ResNetActivationNormNetworkWithAttention, self).__init__()

        self.num_filters = num_filters
        self.num_stages = num_stages
        self.input_shape = input_shape
        self.use_channel_wise_attention = use_channel_wise_attention
        self.num_output_classes = num_output_classes
        self.num_support_set_steps = num_support_set_steps
        self.num_target_set_steps = num_target_set_steps
        self.num_blocks_per_stage = num_blocks_per_stage
        self.build_network()

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()

        for i in range(self.num_stages):
            for j in range(self.num_blocks_per_stage):
                if self.use_channel_wise_attention:
                    self.layer_dict['attention_layer_{}_{}'.format(i, j)] = SqueezeExciteLayer(input_shape=out.shape,
                                                                                               num_filters=32,
                                                                                               num_layers=2)
                    out = self.layer_dict['attention_layer_{}_{}'.format(i, j)].forward(out)

                block_input = out
                self.layer_dict['bottleneck_conv_{}_{}'.format(i, j)] = MetaConvNormLayerLeakyReLU(
                    input_shape=out.shape,
                    num_filters=self.num_filters,
                    kernel_size=1, stride=1,
                    padding=0,
                    use_bias=False,
                    groups=1, per_step_bn_statistics=True,
                    num_support_set_steps=self.num_support_set_steps,
                    num_target_set_steps=self.num_target_set_steps)

                out = self.layer_dict['bottleneck_conv_{}_{}'.format(i, j)](out, training=True, num_step=0)

                self.layer_dict['conv_{}_{}'.format(i, j)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                                        num_filters=self.num_filters,
                                                                                        kernel_size=3, stride=1,
                                                                                        padding=1,
                                                                                        use_bias=False,
                                                                                        groups=1,
                                                                                        per_step_bn_statistics=True,
                                                                                        num_support_set_steps=self.num_support_set_steps,
                                                                                        num_target_set_steps=self.num_target_set_steps)

                out = self.layer_dict['conv_{}_{}'.format(i, j)](out, training=True, num_step=0)

                if i != 0 and j != 0:
                    out = block_input + out

            if self.use_channel_wise_attention:
                self.layer_dict['attention_layer_{}'.format(i)] = SqueezeExciteLayer(input_shape=out.shape,
                                                                                     num_filters=32,
                                                                                     num_layers=2)
                out = self.layer_dict['attention_layer_{}'.format(i)].forward(out)

            block_input = out
            self.layer_dict['bottleneck_conv_{}'.format(i)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                                         num_filters=self.num_filters,
                                                                                         kernel_size=1, stride=1,
                                                                                         padding=0,
                                                                                         use_bias=False,
                                                                                         groups=1,
                                                                                         per_step_bn_statistics=True,
                                                                                         num_support_set_steps=self.num_support_set_steps,
                                                                                         num_target_set_steps=self.num_target_set_steps)

            out = self.layer_dict['bottleneck_conv_{}'.format(i)](out, training=True, num_step=0)

            out = F.avg_pool2d(input=out, kernel_size=2)

            self.layer_dict['conv_{}'.format(i)] = MetaConvNormLayerLeakyReLU(input_shape=out.shape,
                                                                              num_filters=self.num_filters,
                                                                              kernel_size=3, stride=1,
                                                                              padding=1,
                                                                              use_bias=False,
                                                                              groups=1,
                                                                              per_step_bn_statistics=True,
                                                                              num_support_set_steps=self.num_support_set_steps,
                                                                              num_target_set_steps=self.num_target_set_steps)

            out = self.layer_dict['conv_{}'.format(i)](out, training=True, num_step=0)

            block_input = F.avg_pool2d(block_input, kernel_size=2)

            out = block_input + out

        if self.use_channel_wise_attention:
            self.layer_dict['attention_pre_logit_layer'] = SqueezeExciteLayer(input_shape=out.shape,
                                                                              num_filters=32,
                                                                              num_layers=2)
            out = self.layer_dict['attention_pre_logit_layer'].forward(out)

        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view((out.shape[0], -1))

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                    num_filters=self.num_output_classes, use_bias=True)

        out = self.layer_dict['linear'](out)
        print('Build', self._get_name(), 'with shape', out.shape)

    def forward(self, x, num_step, dropout_training=None, params=None, training=False,
                backup_running_statistics=False, return_features=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            params = {key: value for key, value in params.items()}
            # print([key for key, value in param_dict.items()])
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        # print([key for key, value in param_dict.items() if value is not None])

        for i in range(self.num_stages):
            for j in range(self.num_blocks_per_stage):
                block_input = out
                if self.use_channel_wise_attention:
                    out = self.layer_dict['attention_layer_{}_{}'.format(i, j)].forward(out, params=param_dict[
                        'attention_layer_{}_{}'.format(i, j)])

                out = self.layer_dict['bottleneck_conv_{}_{}'.format(i, j)](out, training=True, num_step=num_step,
                                                                            params=param_dict[
                                                                                'bottleneck_conv_{}_{}'.format(i, j)])

                out = self.layer_dict['conv_{}_{}'.format(i, j)](out, training=True, num_step=num_step,
                                                                 params=param_dict['conv_{}_{}'.format(i, j)])

                if i != 0 and j != 0:
                    out = block_input + out

            if self.use_channel_wise_attention:
                out = self.layer_dict['attention_layer_{}'.format(i)].forward(out, params=param_dict[
                    'attention_layer_{}'.format(i)])

            block_input = out

            out = self.layer_dict['bottleneck_conv_{}'.format(i)](out, training=True, num_step=num_step,
                                                                  params=param_dict['bottleneck_conv_{}'.format(i)])

            out = F.avg_pool2d(input=out, kernel_size=2)

            out = self.layer_dict['conv_{}'.format(i)](out, training=True, num_step=num_step,
                                                       params=param_dict['conv_{}'.format(i)])

            block_input = F.avg_pool2d(block_input, kernel_size=2)

            out = block_input + out

        if self.use_channel_wise_attention:
            out = self.layer_dict['attention_pre_logit_layer'].forward(out, params=param_dict[
                'attention_pre_logit_layer'])

        features = out
        out = F.avg_pool2d(out, out.shape[-1])

        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])

        if return_features:
            return out, features
        else:
            return out

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for name, module in self.named_modules():
            if type(module) == MetaBatchNormLayer:
                module.restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


class MetaBatchRelationalModule(nn.Module):
    def __init__(self, input_shape, use_coordinates=True, num_support_set_steps=0, num_target_set_steps=0,
                 output_units=32):
        super(MetaBatchRelationalModule, self).__init__()

        self.input_shape = input_shape
        self.layer_dict = nn.ModuleDict()
        self.first_time = True
        self.use_coordinates = use_coordinates
        self.num_target_set_steps = num_target_set_steps
        self.num_support_set_steps = num_support_set_steps
        self.output_units = output_units
        self.build_block()

    def build_block(self):
        out_img = torch.zeros(self.input_shape)
        """g"""
        if len(out_img.shape) > 3:
            b, c, h, w = out_img.shape
            if h > 5:
                out_img = F.adaptive_avg_pool2d(out_img, output_size=5)
            print(out_img.shape)
            b, c, h, w = out_img.shape
            out_img = out_img.view(b, c, h * w)

        out_img = out_img.permute([0, 2, 1])  # h*w, c
        b, length, c = out_img.shape
        print(out_img.shape)
        # x_flat = (64 x 25 x 24)
        if self.use_coordinates:
            self.coord_tensor = []
            for i in range(length):
                self.coord_tensor.append(torch.Tensor(np.array([i])))

            self.coord_tensor = torch.stack(self.coord_tensor, dim=0).unsqueeze(0)

            if self.coord_tensor.shape[0] != out_img.shape[0]:
                self.coord_tensor = self.coord_tensor[0].unsqueeze(0).repeat([out_img.shape[0], 1, 1])

            out_img = torch.cat([out_img, self.coord_tensor], dim=2)

        x_i = torch.unsqueeze(out_img, 1)  # (1xh*wxc)
        x_i = x_i.repeat(1, length, 1, 1)  # (h*wxh*wxc)
        x_j = torch.unsqueeze(out_img, 2)  # (h*wx1xc)
        x_j = x_j.repeat(1, 1, length, 1)  # (h*wxh*wxc)

        # concatenate all together
        per_location_feature = torch.cat([x_i, x_j], 3)  # (h*wxh*wx2*c)

        out = per_location_feature.view(
            per_location_feature.shape[0] * per_location_feature.shape[1] * per_location_feature.shape[2],
            per_location_feature.shape[3])
        # print(out.shape)
        for idx_layer in range(2):
            # print('test', out.shape)
            self.layer_dict['g_fcc_{}'.format(idx_layer)] = MetaLinearLayer(input_shape=out.shape, num_filters=64,
                                                                            use_bias=True)
            out = self.layer_dict['g_fcc_{}'.format(idx_layer)].forward(out)
            self.layer_dict['LeakyReLU_{}'.format(idx_layer)] = nn.LeakyReLU()
            out = self.layer_dict['LeakyReLU_{}'.format(idx_layer)].forward(out)

        # reshape again and sum
        print(out.shape)
        out = out.view(per_location_feature.shape[0], per_location_feature.shape[1], per_location_feature.shape[2], -1)
        out = out.sum(1).sum(1)
        print('here', out.shape)
        """f"""
        self.layer_dict['post_processing_layer'] = MetaLinearLayer(input_shape=out.shape, num_filters=64, use_bias=True)
        out = self.layer_dict['post_processing_layer'].forward(out)
        self.layer_dict['LeakyReLU_post_processing'] = nn.LeakyReLU()
        out = self.layer_dict['LeakyReLU_post_processing'].forward(out)
        self.layer_dict['output_layer'] = MetaLinearLayer(input_shape=out.shape, num_filters=self.output_units,
                                                          use_bias=True)
        out = self.layer_dict['output_layer'].forward(out)
        self.layer_dict['LeakyReLU_output'] = nn.LeakyReLU()
        out = self.layer_dict['LeakyReLU_output'].forward(out)
        print('Block built with output volume shape', out.shape)

    def forward(self, x_img, num_step, params=None):

        param_dict = dict()

        if params is not None:
            params = {key: value for key, value in params.items()}
            # print([key for key, value in param_dict.items()])
            param_dict = extract_top_level_dict(current_dict=params)
            # print(list(params.keys()))

        for name, param in list(self.layer_dict.named_parameters()) + list(self.layer_dict.items()):
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out_img = x_img
        # print("input", out_img.shape)
        """g"""
        if len(out_img.shape) > 3:
            b, c, h, w = out_img.shape
            if h > 5:
                out_img = F.adaptive_avg_pool2d(out_img, output_size=5)
            b, c, h, w = out_img.shape
            out_img = out_img.view(b, c, h * w)


        out_img = out_img.permute([0, 2, 1])  # h*w, c
        b, length, c = out_img.shape

        if self.use_coordinates:
            if self.coord_tensor.shape[0] != out_img.shape[0]:
                self.coord_tensor = self.coord_tensor[0].unsqueeze(0).repeat([out_img.shape[0], 1, 1])

            out_img = torch.cat([out_img, self.coord_tensor.to(x_img.device)], dim=2)
        # x_flat = (64 x 25 x 24)
        # print('out_img', out_img.shape)
        x_i = torch.unsqueeze(out_img, 1)  # (1xh*wxc)
        x_i = x_i.repeat(1, length, 1, 1)  # (h*wxh*wxc)
        x_j = torch.unsqueeze(out_img, 2)  # (h*wx1xc)
        x_j = x_j.repeat(1, 1, length, 1)  # (h*wxh*wxc)

        # concatenate all together
        per_location_feature = torch.cat([x_i, x_j], 3)  # (h*wxh*wx2*c)
        out = per_location_feature.view(
            per_location_feature.shape[0] * per_location_feature.shape[1] * per_location_feature.shape[2],
            per_location_feature.shape[3])
        # print(out.shape)
        for idx_layer in range(2):
            # print('test', out.shape)
            # print(param_dict['g_fcc_{}'.format(idx_layer)])
            out = self.layer_dict['g_fcc_{}'.format(idx_layer)].forward(out,
                                                                        params=param_dict['g_fcc_{}'.format(idx_layer)])
            # print('test', out.shape)
            out = self.layer_dict['LeakyReLU_{}'.format(idx_layer)].forward(out)

        # reshape again and sum
        # print(out.shape)
        out = out.view(per_location_feature.shape[0], per_location_feature.shape[1], per_location_feature.shape[2], -1)
        out = out.sum(1).sum(1)

        """f"""
        out = self.layer_dict['post_processing_layer'].forward(out, params=param_dict['post_processing_layer'])
        out = self.layer_dict['LeakyReLU_post_processing'].forward(out)
        out = self.layer_dict['output_layer'].forward(out, params=param_dict['output_layer'])
        out = self.layer_dict['LeakyReLU_output'].forward(out)
        return out

# test_input = torch.zeros((32, 3, 32, 32))
# # module = BatchRelationalModule(input_shape=test_input.shape, use_coordinates=True, num_support_set_steps=1,
# #                                num_target_set_steps=1)
# module = DenseNetActivationNormNetworkWithAttention(input_shape=test_input.shape, num_output_classes=1000,
#                                                     use_channel_wise_attention=True, num_stages=3, num_filters=32,
#                                                     num_blocks_per_stage=1, num_support_set_steps=1,
#                                                     num_target_set_steps=1)
# out = module.forward(test_input, num_step=0, params={key: value.unsqueeze(0) for key, value in module.named_parameters()})
# print(out.shape)
