from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dNormLeakyReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, dilation=1, stride=1, groups=1, padding=0, use_bias=False,
                 normalization=True, weight_attention=False):
        super(Conv2dNormLeakyReLU, self).__init__()
        self.input_shape = list(input_shape)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.normalization = normalization
        self.dilation = dilation
        self.weight_attention = weight_attention
        self.groups = groups
        self.layer_dict = nn.ModuleDict()
        self.build_network()

    def build_network(self):
        x = torch.ones(self.input_shape)
        out = x

        self.layer_dict['conv'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters,
                                            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                            dilation=self.dilation, groups=self.groups, bias=self.use_bias)

        out = self.layer_dict['conv'].forward(out)

        if self.normalization:
            self.layer_dict['norm_layer'] = nn.BatchNorm2d(num_features=out.shape[1])
            out = self.layer_dict['norm_layer'](out)

        self.layer_dict['relu'] = nn.LeakyReLU()
        out = self.layer_dict['relu'](out)
        print(out.shape)

    def forward(self, x):
        out = x

        out = self.layer_dict['conv'].forward(out)

        if self.normalization:
            out = self.layer_dict['norm_layer'](out)

        out = self.layer_dict['relu'](out)
        return out

class SqueezeExciteDenseNet(nn.Module):
    def __init__(self, im_shape, num_filters, num_stages, num_blocks_per_stage, dropout_rate, average_pool_output,
                 reduction_rate, output_spatial_dim, use_channel_wise_attention):
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
        super(SqueezeExciteDenseNet, self).__init__()
        self.input_shape = list(im_shape)
        self.num_filters = num_filters
        self.num_stages = num_stages
        self.dropout_rate = dropout_rate
        self.reduction_rate = reduction_rate
        self.average_pool_output = average_pool_output
        # self.num_output_classes = num_output_classes
        self.num_blocks_per_stage = num_blocks_per_stage
        self.output_spatial_dim = output_spatial_dim
        self.conv_type = Conv2dNormLeakyReLU
        self.layer_dict = nn.ModuleDict()
        self.use_channel_wise_attention = use_channel_wise_attention
        self.build_network()

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        out = x
        print('Building', self.__class__.__name__)
        self.layer_dict['stem_conv'] = Conv2dNormLeakyReLU(input_shape=out.shape, num_filters=64,
                                                           kernel_size=3, padding=1, groups=1)

        out = self.layer_dict['stem_conv'](out)

        for i in range(self.num_stages):
            for j in range(self.num_blocks_per_stage):
                if self.use_channel_wise_attention:
                    attention_network_out = F.avg_pool2d(out, out.shape[-1]).squeeze()

                    self.layer_dict['channel_wise_attention_output_fcc_{}_{}'.format(j, i)] = nn.Linear(
                        in_features=attention_network_out.shape[1], out_features=out.shape[1], bias=True)
                    channel_wise_attention_regions = self.layer_dict[
                        'channel_wise_attention_output_fcc_{}_{}'.format(j, i)].forward(attention_network_out)

                    channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)
                    out = out * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

                self.layer_dict['conv_bottleneck_{}_{}'.format(i, j)] = self.conv_type(input_shape=out.shape,
                                                                                       num_filters=self.num_filters,
                                                                                       kernel_size=1, padding=0)

                cur = self.layer_dict['conv_bottleneck_{}_{}'.format(i, j)](out)
                self.layer_dict['conv_{}_{}'.format(i, j)] = self.conv_type(input_shape=cur.shape,
                                                                            num_filters=self.num_filters,
                                                                            kernel_size=3, padding=1, groups=1)

                cur = self.layer_dict['conv_{}_{}'.format(i, j)](cur)
                cur = F.dropout(cur, p=self.dropout_rate, training=True)
                out = torch.cat([out, cur], dim=1)

            out = F.avg_pool2d(out, 2)
            print(out.shape)
            self.layer_dict['transition_layer_{}'.format(i)] = Conv2dNormLeakyReLU(input_shape=out.shape,
                                                                                   num_filters=int(out.shape[
                                                                                                       1] * self.reduction_rate),
                                                                                   kernel_size=1, padding=0)

            out = self.layer_dict['transition_layer_{}'.format(i)](out)

        if self.average_pool_output:
            out = F.avg_pool2d(out, out.shape[2])
            out = out.view(out.shape[0], -1)
        else:
            out = F.adaptive_avg_pool2d(out, output_size=(self.output_spatial_dim, self.output_spatial_dim))

        print('Done', out.shape)

    def forward(self, x, dropout_training):
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
        out = x

        out = self.layer_dict['stem_conv'](out)
        for i in range(self.num_stages):
            for j in range(self.num_blocks_per_stage):
                # out_channels = F.avg_pool2d(out, out.shape[-1]).squeeze()
                if self.use_channel_wise_attention:
                    out_channels = F.avg_pool2d(out, out.shape[-1]).squeeze()

                    channel_wise_attention_regions = self.layer_dict[
                        'channel_wise_attention_output_fcc_{}_{}'.format(j, i)].forward(out_channels)

                    channel_wise_attention_regions = F.sigmoid(channel_wise_attention_regions)
                    out = out * channel_wise_attention_regions.unsqueeze(2).unsqueeze(2)

                cur = self.layer_dict['conv_bottleneck_{}_{}'.format(i, j)](out)
                cur = self.layer_dict['conv_{}_{}'.format(i, j)](cur)
                cur = F.dropout(cur, p=self.dropout_rate, training=dropout_training)
                out = torch.cat([out, cur], dim=1)

            out = F.avg_pool2d(out, 2)
            out = self.layer_dict['transition_layer_{}'.format(i)](out)

        if self.average_pool_output:
            out = F.avg_pool2d(out, out.shape[2])
            out = out.view(out.shape[0], -1)
        else:
            out = F.adaptive_avg_pool2d(out, output_size=(self.output_spatial_dim, self.output_spatial_dim))

        return out

