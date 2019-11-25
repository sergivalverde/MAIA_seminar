import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from termcolor import colored


class ResCoreElement(nn.Module):
    """
    Residual Core element used inside the NN. Control the number of filters
    and batch normalization.
    """
    def __init__(self,
                 input_size,
                 num_filters,
                 use_batchnorm=True,
                 use_leaky=True,
                 leaky_p=0.2):

        super(ResCoreElement, self).__init__()
        self.use_bn = use_batchnorm
        self.use_lr = use_leaky
        self.leaky_p = leaky_p

        self.conv1 = nn.Conv3d(input_size,
                               num_filters,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv3d(input_size,
                               num_filters,
                               kernel_size=1,
                               padding=0)
        self.bn_add = nn.BatchNorm3d(num_filters)

    def forward(self, x):
        """
        include residual model
        """
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_sum = self.bn_add(x_1 + x_2) if self.use_bn is True else x_1 + x_2
        return F.leaky_relu(x_sum,
                            self.leaky_p) if self.use_lr else F.relu(x_sum)


class ConvElement(nn.Module):
    """
    Residual Core element used inside the NN. Control the number of filters
    and batch normalization.
    """
    def __init__(self,
                 input_size,
                 num_filters,
                 use_leaky=True,
                 stride=1,
                 leaky_p=0.2):

        super(ConvElement, self).__init__()
        self.use_lr = use_leaky
        self.leaky_p = leaky_p
        self.conv1 = nn.Conv3d(input_size,
                               num_filters,
                               kernel_size=3,
                               padding=1,
                               stride=stride)

    def forward(self, x):
        """
        include residual model
        """
        x_1 = self.conv1(x)
        return F.leaky_relu(x_1, self.leaky_p) if self.use_lr else F.relu(x_1)


class Pooling3D(nn.Module):
    """
    3D pooling layer by striding.
    """
    def __init__(self, input_size,
                 use_batchnorm=True,
                 use_leaky=True,
                 leaky_p=0.2):
        super(Pooling3D, self).__init__()
        self.use_bn = use_batchnorm
        self.use_lr = use_leaky
        self.leaky_p = leaky_p
        self.conv1 = nn.Conv3d(input_size,
                               input_size,
                               kernel_size=2,
                               stride=2)
        self.bn = nn.BatchNorm3d(input_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x) if self.use_bn is True else x
        return F.leaky_relu(x, self.leaky_p) if self.use_lr else F.relu(x)


class UpdateStatus(object):
    """
    Update training/testing colors
    Register new elements iteratively by setting them by name

    """
    def __init__(self, pat_interval=0):
        super(UpdateStatus, self).__init__()

        # network parameters
        self.elements = {}

    def register_new_element(self, element_name, mode):
        """
        Register new elements to show. We control each element
        using a dictionary with two elements:
        - value: actual value to update
        - mode: value type, so accuracy is incremental and losses decremental
        """
        if mode == 'incremental':
            self.elements[element_name] = {'value': 0, 'mode': 'incremental'}
        else:
            self.elements[element_name] = {'value': np.inf,
                                           'mode': 'decremental'}

    def update_element(self, element_name, current_value):
        """
        update element
        """

        update = current_value
        if element_name in self.elements.keys():
            update = self.process_update(element_name, current_value)
        else:
            print('ERROR:', element_name,
                  "element is not currently registered")
        return update

    def process_update(self, element_name, current_value):
        """
        update the value for a particu
        """
        best_value = self.elements[element_name]['value']
        mode = self.elements[element_name]['mode']
        update = '{:.4f}'.format(current_value)

        if (mode == 'decremental') and (best_value > current_value):
            update = colored(update, 'green')
            self.elements[element_name]['value'] = current_value
        elif (mode == 'incremental') and (best_value < current_value):
            update = colored(update, 'green')
            self.elements[element_name]['value'] = current_value
        else:
            update = colored(update, 'red')

        return update


class EarlyStopping(object):
    """
    Control early stopping with several parameters
    check early stopping conditions and save the model. If the
    current validation loss is lower than the previous one, the
    model is saved back and the early stopping iteration
    is set to 0. If not, the number of iterations without
    decrease in the val_loss is update. When the number
    iterations is > than patience, training is stopped.

    """
    def __init__(self, epoch=1, metric='acc', patience=20):
        super(EarlyStopping, self).__init__()

        self.epoch = epoch
        self.patience = patience
        self.patience_iter = 0
        self.metric = metric
        self.best = None

        # initialize the best value
        self.__initialize_best()

    def __initialize_best(self):
        """
        Initialize the best value taking into account the kind of metric
        """
        if self.metric == 'acc':
            self.best = 0
        if self.metric == 'dsc':
            self.best = 0
        if self.metric == 'loss':
            self.best = np.inf

    def __compare_metric(self, current_value):
        """
        Check status for the current epoch
        """
        if self.metric == 'acc':
            is_best = current_value > self.best
        if self.metric == 'dsc':
            is_best = current_value > self.best
        if self.metric == 'loss':
            is_best = current_value < self.best

        return is_best

    def save_epoch(self, current_value, current_epoch):
        """
        check if the current_value for a given epoch
        """

        self.epoch = current_epoch
        is_best = self.__compare_metric(current_value)
        if is_best:
            self.best = current_value
            self.patience_iter = 0
        else:
            self.patience_iter += 1

        return is_best

    def stop_model(self):
        """
        check if the maximum number of iterations has been raised
        """
        return self.patience_iter > self.patience

    def get_best_value(self):
        """
        Return best value
        """
        return self.best


class UnetGridGatingSignal3(nn.Module):
    """
    To doc
    """

    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size=(1, 1, 1),
                 is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size,
                          out_size,
                          kernel_size,
                          (1, 1, 1),
                          (0, 0, 0)),
                nn.BatchNorm3d(out_size),
                nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size,
                          out_size,
                          kernel_size,
                          (1, 1, 1),
                          (0, 0, 0)),
                nn.ReLU(inplace=True),)
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


def weights_init_normal(m):
    """
    Inititalize the weights using Normal
    """
    classname = m.__class__.__name__

    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    """
    Initialize the weights using Xavier
    """
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    """
    Initialize the weights using Xavier
    """
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    """
    Initialize the weights using Orthogonal
    """
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    """
    Helper function to initialize the weights
    """
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] \
        is not implemented' % init_type)


class GridAttentionBlockND(nn.Module):
    """
    General GridAttentionBlock
    """
    def __init__(self, in_channels,
                 gating_channels,
                 inter_channels=None,
                 dimension=3,
                 mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation',
                        'concatenation_debug',
                        'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplementedError

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size,
                             stride=self.sub_sample_factor,
                             padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1, stride=1,
                           padding=0,
                           bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels,
                           out_channels=1,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           bias=True)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) ->
        #    f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g),
                              size=theta_x_size[2:],
                              mode=self.upsample_mode,
                              align_corners=True)

        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f,
                                   size=input_size[2:],
                                   mode=self.upsample_mode,
                                   align_corners=True)

        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw)
        # -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g),
                              size=theta_x_size[2:],
                              mode=self.upsample_mode,
                              align_corners=True)
        f = F.softplus(theta_x + phi_g)

        #  Psi^T * F -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f,
                                   size=input_size[2:],
                                   mode=self.upsample_mode,
                                   align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias)
        # -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g),
                              size=theta_x_size[2:],
                              mode=self.upsample_mode,
                              align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size,
                                              1,
                                              *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f,
                                   size=input_size[2:],
                                   mode=self.upsample_mode,
                                   align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f
