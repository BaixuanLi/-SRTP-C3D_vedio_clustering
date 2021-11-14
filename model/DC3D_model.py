import torch
import torch.nn as nn
import numpy as np
from Path_file import Path


class DC3D(nn.Module):
    def __init__(self,pretrained=False):
        super(DC3D, self).__init__()

        self.features = nn.Sequential(
            nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 2
            nn.ReLU(),
            nn.ConvTranspose3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 4

            nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 7
            nn.ReLU(),
            nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 9

            nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 12
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 14

            nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 17

            nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 20
        )

        self.conv3deconv_indices = {
            0: 20, 3: 17, 6: 14, 8: 12,
            11: 9, 13: 7, 16: 4, 18: 2
        }

        self.unpool3pool_indices = {
            18: 2, 15: 5, 10: 10, 5: 15, 0: 20
        }

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    '''def forward(self, x):

        x = self.unpool5(x)
        x = self.relu(x)
        x = self.dconv5b(x)
        x = self.relu(x)
        x = self.dconv5a(x)

        x = self.unpool4(x)
        x = self.relu(x)
        x = self.dconv4b(x)
        x = self.relu(x)
        x = self.dconv4a(x)

        x = self.unpool3(x)
        x = self.relu(x)
        x = self.dconv3b(x)
        x = self.relu(x)
        x = self.dconv3a(x)

        x = self.unpool2(x)
        x = self.relu(x)
        x = self.dconv2(x)

        x = self.unpool1(x)
        x = self.relu(x)
        x = self.dconv1(x)
        '''
    def forward(self, x, layer, activation_idx, pool_locs):
        if layer in self.conv3deconv_indices:
            start_idx = self.conv3deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool3d):
                x = self.features[idx]\
                (x, pool_locs[self.unpool3pool_indices[idx]])
            else:
                x = self.features[idx](x)
        return x

    def __load_pretrained_weights(self):
        """Initialize network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "features.20.weight",
                        #"features.0.bias": "features.12.bias",
                        # Conv2
                        "features.3.weight": "features.17.weight",
                        #"features.3.bias": "features.10.bias",
                        # Conv3a
                        "features.6.weight": "features.14.weight",
                        #"features.6.bias": "features.8.bias",
                        # Conv3b
                        "features.8.weight": "features.12.weight",
                        #"features.8.bias": "features.7.bias",
                        # Conv4a
                        "features.11.weight": "features.9.weight",
                        #"features.11.bias": "features.5.bias",
                        # Conv4b
                        "features.13.weight": "features.7.weight",
                        #"features.13.bias": "features.4.bias",
                        # Conv5a
                        "features.16.weight": "features.4.weight",
                        #"features.16.bias": "features.2.bias",
                        # Conv5b
                        "features.18.weight": "features.2.weight",
                        #"features.18.bias": "features.1.bias",
                        }

        p_dict = torch.load(Path.model_dir(), map_location=lambda storage, loc: storage)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            print('预存参数')
            print(p_dict[name].size())
            print('现网络结构')
            print(s_dict[corresp_name[name]].size())
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)
        print('model load successfully!')

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == '__main__':

    net = DC3D(pretrained=False)
    print(net.state_dict())