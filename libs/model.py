import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model_utils import UpdateStatus, EarlyStopping, ResCoreElement, Pooling3D
from model_utils import GridAttentionBlockND, UnetGridGatingSignal3, init_weights


class ResUnet(nn.Module):
    """
    Basic U-net model using residual layers. Control the input channels,
    output channels, and scaling of output filters.
    Attention model
    """

    def __init__(self, input_channels,
                 output_channels,
                 scale,
                 classification=False,
                 use_bn=True,
                 use_leaky=False,
                 use_attention=False,
                 attention_dsample=(2, 2, 2),
                 nonlocal_mode='concatenation',
                 leaky_p=0.2):


        super(ResUnet, self).__init__()
        self.use_bn = use_bn
        self.use_leaky = use_leaky
        self.leaky_p = leaky_p
        self.classification = classification
        self.nonlocal_mode = nonlocal_mode
        self.attention_dsample = attention_dsample
        self.attention = use_attention

        # conv  down
        self.conv1 = ResCoreElement(input_channels,
                                    int(scale * 32),
                                    use_bn,
                                    use_leaky,
                                    leaky_p)
        self.pool1 = Pooling3D(int(scale * 32), use_bn)

        # conv 2 down
        self.conv2 = ResCoreElement(int(scale * 32), int(scale * 64),
                                    use_bn, use_leaky, leaky_p)

        self.pool2 = Pooling3D(int(scale * 64), use_leaky, leaky_p)

        # conv 3 down
        self.conv3 = ResCoreElement(int(scale * 64), int(scale * 128),
                                    use_bn, use_leaky, leaky_p)
        self.pool3 = Pooling3D(int(scale * 128), use_bn, use_leaky, leaky_p)

        # conv 4 down
        self.conv4 = ResCoreElement(int(scale*128), int(scale*256),
                                    use_bn, use_leaky, leaky_p)

        # up 1
        self.up1 = nn.ConvTranspose3d(int(scale*256),
                                      int(scale*128),
                                      kernel_size=2,
                                      stride=2)
        # conv 5
        self.conv5 = ResCoreElement(int(scale*128), int(scale*128),
                                    use_bn, use_leaky, leaky_p)

        # conv 6 up
        self.bn_add35 = nn.BatchNorm3d(int(scale*128))
        self.conv6 = ResCoreElement(int(scale*128), int(scale*128),
                                    use_bn, use_leaky, leaky_p)
        self.up2 = nn.ConvTranspose3d(int(scale*128),
                                      int(scale*64),
                                      kernel_size=2,
                                      stride=2)

        # conv 7 up
        self.bn_add22 = nn.BatchNorm3d(int(scale*64))
        self.conv7 = ResCoreElement(int(scale*64), int(scale*64),
                                    use_bn, use_leaky, leaky_p)
        self.up3 = nn.ConvTranspose3d(int(scale*64),
                                      int(scale*32),
                                      kernel_size=2,
                                      stride=2)

        # conv 8 up
        self.bn_add13 = nn.BatchNorm3d(int(scale*32))
        self.conv8 = ResCoreElement(int(scale*32), int(scale*32),
                                    use_bn, use_leaky, leaky_p)

        # reconstruction
        self.conv9 = nn.Conv3d(int(scale * 32),
                               output_channels,
                               kernel_size=1)

        # Attention modules using a 3D gridBlock approach
        if self.attention:
            # attention gate
            self.gating = UnetGridGatingSignal3(int(scale*256),
                                                int(scale*128),
                                                kernel_size=(1, 1, 1),
                                                is_batchnorm=self.use_bn)

            # att block 2
            self.attconv2 = GridAttentionBlock3D(in_channels=int(scale*64),
                                                 gating_channels=int(scale*128),
                                                 inter_channels=int(scale*64),
                                                 sub_sample_factor=self.attention_dsample,
                                                 mode=self.nonlocal_mode)

            # att block 3
            self.attconv3 = GridAttentionBlock3D(in_channels=int(scale*128),
                                                 gating_channels=int(scale*128),
                                                 inter_channels=int(scale*128),
                                                 sub_sample_factor=self.attention_dsample,
                                                 mode=self.nonlocal_mode)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

        # check parameters
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print('\n--------------------------------------------------')
        print('Model information:')
        print("ResUnet3D network with {} parameters".format(nparams))
        print('--------------------------------------------------\n')

    def forward(self, x, encoder=False):

        # encoder
        x1 = self.conv1(x)
        x1d = self.pool1(x1)
        x2 = self.conv2(x1d)
        x2d = self.pool2(x2)
        x3 = self.conv3(x2d)
        x3d = self.pool3(x3)
        x4 = self.conv4(x3d)

        # attention layers
        if self.attention:
            gating = self.gating(x4)
            x3, att3 = self.attconv3(x3, gating)
            x2, att2 = self.attconv2(x2, gating)

        # decoder
        up1 = self.up1(x4)
        x5 = self.conv5(up1)

        add_35 = self.bn_add35(x5 + x3) if self.use_bn is True else x5 + x3
        x6 = self.conv6(add_35)
        up2 = self.up2(x6)

        add_22 = self.bn_add22(up2 + x2) if self.use_bn is True else x2 + up2
        x7 = self.conv7(add_22)
        up3 = self.up3(x7)

        add_13 = self.bn_add13(up3 + x1) if self.use_bn is True else x1 + up3
        x8 = self.conv8(add_13)

        return F.softmax(self.conv9(x8), dim=1) if self.classification \
            else self.conv9(x8)


class GridAttentionBlock3D(GridAttentionBlockND):
    """
    Grid Attention Block for 3D processing

    """
    def __init__(self, in_channels,
                 gating_channels,
                 inter_channels=None,
                 mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):

        super(GridAttentionBlock3D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            gating_channels=gating_channels,
            dimension=3, mode=mode,
            sub_sample_factor=sub_sample_factor)



class LesionNet(nn.Module):
    """
    Quick and dirty Voxelmorph implementation
    """

    def __init__(self,
                 input_channels=1,
                 output_channels=1,
                 patch_shape=(32, 32, 32),
                 scale=0.5,
                 training_epochs=200,
                 shuffle_data=True,
                 patience=50,
                 pat_interval=0.00,
                 batch_size=32,
                 train_split=0.3,
                 model_name=None,
                 gpu_mode=True,
                 gpu_list=[0],
                 attention=False,
                 use_bn=False,
                 use_attention=False,
                 load_weights=False,
                 loss_weights=None,
                 model_path=None,
                 resume_training=False):

        super(LesionNet, self).__init__()

        # network parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.patch_size = patch_shape
        self.num_epochs = training_epochs
        self.shuffle_data = shuffle_data
        self.patience = patience
        self.pat_interval = pat_interval
        self.batch_size = batch_size
        self.train_split = train_split
        self.model_name = model_name
        self.use_bn = use_bn
        self.resume_training = resume_training
        self.gpu_mode = gpu_mode
        # model path

        if model_path is None:
            self.model_path = os.path.join(os.getcwd(), 'models')
        else:
            self.model_path = model_path

        self.lesion_net = ResUnet(input_channels=self.input_channels,
                                  output_channels=2,
                                  scale=self.scale,
                                  use_bn=True,
                                  use_attention=use_attention,
                                  classification=True)

        # GPU devices. Configure the number of GPUS to use if more that one are
        # available. Pytorch gpu ids have to be sorted from 0,1,... so system
        # GPUS visubility is set before to match this
        self.parallel = False

        # if len(gpu_list) > 1:
        #     print('Setting parallel training')
        #     os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x)
        #                                                   for x in gpu_list)
        #     gpu_list = list(range(len(gpu_list)))
        #     self.device = torch.device('cuda:' + str(gpu_list[0]))
        #     self.parallel = True

        if self.gpu_mode:
            self.device = torch.device('cuda:' + str(gpu_list[0]))
        else:
            self.device = torch.device('cpu')

        self.gpu_list = gpu_list

    def train_model(self, t_dataloader, v_dataloader):
        """
        train the wnet model
        """
        training = True
        # train_samples = len(t_dataloader)
        # val_samples = len(v_dataloader)

        # send models to device
        self.lesion_net = self.lesion_net.to(self.device)

        # parallel models
        if self.parallel:
            self.lesion_net = nn.DataParallel(self.lesion_net,
                                              device_ids=self.gpu_list)

        # optimizers
        net_optimizer = optim.Adadelta(self.lesion_net.parameters())

        epoch = 1

        # train
        # start early stopping
        early_stopper = EarlyStopping(epoch=epoch,
                                      metric='acc',
                                      patience=self.patience)

        # color handling for training / testing
        update = UpdateStatus(pat_interval=self.pat_interval)

        # register measures
        update.register_new_element('train_loss', 'decremental')
        update.register_new_element('train_accuracy', 'incremental')
        update.register_new_element('train_dsc', 'incremental')
        update.register_new_element('val_loss', 'decremental')
        update.register_new_element('val_accuracy', 'incremental')
        update.register_new_element('val_dsc', 'incremental')

        try:
            while training:
                train_loss = 0
                train_accuracy = 0
                val_loss = 0
                val_accuracy = 0
                train_dsc = 0
                val_dsc = 0
                # train_custom = 0
                # val_custom = 0

                self.lesion_net.train()

                epoch_time = time.time()

                # train on batch
                for b, batch in enumerate(t_dataloader):

                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)

                    net_optimizer.zero_grad()

                    pred = self.lesion_net(x)

                    loss = F.cross_entropy(
                          torch.log(torch.clamp(pred, 1E-7, 1.0)),
                          y.squeeze(dim=1).long(), ignore_index=2)

                    # loss = dsc_loss(pred, y)

                    train_loss += loss.item()

                    loss.backward()
                    net_optimizer.step()

                    # relative accuracy

                    train_dsc += self.DSC_score(pred, y).item()

                    pred = pred.max(1, keepdim=True)[1]
                    train_accuracy += pred.eq(
                        y.view_as(pred).long()).sum().item() / np.prod(y.shape)

                    # clear cache
                    # cuda_memory = torch.cuda.memory_allocated(self.device)
                    # if b % 10 == 0:
                    #     print(b, cuda_memory)

                    # torch.cuda.empty_cache()

                # update losses
                train_loss /= (b+1)
                train_accuracy /= (b+1)
                train_dsc /= (b + 1)

                # --------------------------------------------------
                # compute validation
                # --------------------------------------------------

                self.lesion_net.eval()

                for b, batch in enumerate(v_dataloader):

                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)

                    # --------------------------------------------------
                    # lesion
                    # --------------------------------------------------
                    with torch.no_grad():
                        pred = self.lesion_net(x)
                        # loss = dsc_loss(pred, y)
                        loss = F.cross_entropy(
                              torch.log(torch.clamp(pred, 1E-7, 1.0)),
                              y.squeeze(dim=1).long(), ignore_index=2)
                        # loss = focal_loss(pred, y)
                        val_loss += loss.item()

                        # relative accuracy
                        pred = pred.max(1, keepdim=True)[1]
                        val_accuracy += pred.eq(
                            y.view_as(pred).long()).sum().item() / np.prod(y.shape)
                        val_dsc += self.DSC_score(pred, y).item()

                # update losses
                val_loss /= (b+1)
                val_accuracy /= (b+1)
                val_dsc /= (b+1)

                t_time = time.time() - epoch_time

                print('Epoch: {} Time: {:.2f}'.format(epoch, t_time),
                      'Train lesion loss: {}'.format(
                          update.update_element('train_loss',
                                                train_loss)),
                      'Train lesion acc: {}'.format(
                          update.update_element('train_accuracy',
                                                train_accuracy)),
                      'Train lesion DSC: {}'.format(
                          update.update_element('train_dsc',
                                                train_dsc)),
                      'Val lesion loss: {}'.format(
                          update.update_element('val_loss',
                                                val_loss)),
                      'Val lesion acc: {}'.format(
                          update.update_element('val_accuracy',
                                                val_accuracy)),
                      'Val lesion DSC: {}'.format(
                          update.update_element('val_dsc',
                                                val_dsc)))

                # update epochs
                epoch += 1

                # check epoch for early stopping:
                if early_stopper.save_epoch(val_dsc, epoch):
                    # save the model
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'data_parallel': self.parallel,
                        'state_dict_les': self.lesion_net.state_dict()})
                # check if the model has to be stoped
                if early_stopper.stop_model():
                    training = False
                    print("--------------------------------------------------")
                    print("Stopping training at epoch", epoch,
                          "best val_loss:", early_stopper.get_best_value())
                    print("--------------------------------------------------")

                if epoch > self.num_epochs:
                    training = False

        except KeyboardInterrupt:
            pass

    def save_checkpoint(self, state):
        """
        save the best net state
        """

        filename = os.path.join(self.model_path, self.model_name)
        torch.save(state, filename)

    def load_weights(self, model_path=None, model_name=None):
        """
        load network weights

        """
        if model_path is not None:
            self.model_path = model_path
        if model_name is not None:
            self.model_name = model_name

        # send models to device

        self.lesion_net = self.lesion_net.to(self.device)

        # remap weights if CPU: necessary for Docker
        map_location = 'cuda' if self.gpu_mode else 'cpu'

        # load weights
        # filename = self.model_path + '/models/' + self.model_name
        filename = os.path.join(self.model_path, self.model_name)

        # filename = './models/' + self.model_name
        # print("--------------------------------------------------")
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=map_location)
            # check if the network was trained using data parallelism
            if checkpoint['data_parallel']:
                self.lesion_net = nn.DataParallel(self.lesion_net)
            # load weights
            self.lesion_net.load_state_dict(checkpoint['state_dict_les'])

            # print("=> loaded weights '{}'".format(self.model_name))
        else:
            print("=> no checkpoint found at '{}'".format(self.model_name))
        # print("--------------------------------------------------")

    def test_net(self, test_input):
        """
        Testing the network
        To doc
        """
        # output reconstruction
        bs, cs, xs, ys, zs = test_input.shape
        lesion_out = np.zeros((bs, 1, xs, ys, zs)).astype('float32')

        self.lesion_net.eval()
        with torch.no_grad():
            for b in range(0, len(lesion_out), self.batch_size):
                x = torch.tensor(
                    test_input[b:b+self.batch_size]).to(self.device)
                pred = self.lesion_net(x)
                output = pred[:, 1].unsqueeze(dim=1)
                # save the result back
                lesion_out[b:b+self.batch_size] = output.cpu().numpy()

        return lesion_out

    def test_net_3D(self, test_input):
        """
        Testing the network
        To doc
        """
        # output reconstruction
        # compute validation
        # --------------------------------------------------
        self.lesion_net.eval()

        for x in test_input:
            with torch.no_grad():
                x = x[0].to(self.device)
                with torch.no_grad():
                    output = self.lesion_net(x)[:, 1]

        return output.cpu().numpy()

    def DSC_score(self, pred, label, smooth=1.):
        """
        DSC loss
        """
        pred = pred.float()  # only lesion probabilities
        label = label.float()
        label[label == 2] = 0
        dice_numerator = 2.0 * torch.sum(pred * label, dim=0)
        dice_denominator = torch.sum(pred, dim=0) + torch.sum(label, dim=0)
        dice_score = (dice_numerator + smooth) / (dice_denominator + smooth)
        return torch.mean(dice_score) * smooth

    def accuracy(self, pred, label, smooth=1.):
        """
        Accuracy
        """
        pred = pred.float()  # only lesion probabilities
        label = label.float()
        pred_max = pred.max(1, keepdim=True)[1].float()
        numerator = torch.abs(pred_max - label).sum()
        denominator = label.sum()
        return numerator / denominator
