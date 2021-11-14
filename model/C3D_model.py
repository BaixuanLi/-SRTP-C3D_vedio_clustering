import torch
import torch.nn as nn
from Path_file import Path

from collections import OrderedDict


class C3D(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True),

            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), return_indices=True),

            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1), return_indices=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.conv_layer_indices = [0, 3, 6, 8, 11, 13, 16, 18]
        # feature maps
        self.feature_maps = OrderedDict()
        # switch
        self.pool_locs = OrderedDict()
        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    '''def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits'''

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool3d):
                x, location = layer(x)
                # self.pool_locs[idx] = location
            else:
                x = layer(x)

        # reshape to (1, 512 * 7 * 7)
        x = x.view(x.size()[0], -1)
        return x

    def __load_pretrained_weights(self):
        """Initialize network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "features.0.weight",
                        "features.0.bias": "features.0.bias",
                        # Conv2
                        "features.3.weight": "features.3.weight",
                        "features.3.bias": "features.3.bias",
                        # Conv3a
                        "features.6.weight": "features.6.weight",
                        "features.6.bias": "features.6.bias",
                        # Conv3b
                        "features.8.weight": "features.8.weight",
                        "features.8.bias": "features.8.bias",
                        # Conv4a
                        "features.11.weight": "features.11.weight",
                        "features.11.bias": "features.11.bias",
                        # Conv4b
                        "features.13.weight": "features.13.weight",
                        "features.13.bias": "features.13.bias",
                        # Conv5a
                        "features.16.weight": "features.16.weight",
                        "features.16.bias": "features.16.bias",
                        # Conv5b
                        "features.18.weight": "features.18.weight",
                        "features.18.bias": "features.18.bias",
                        # fc6
                        "classifier.0.weight": "classifier.0.weight",
                        "classifier.0.bias": "classifier.0.bias",
                        # fc7
                        "classifier.2.weight": "classifier.2.weight",
                        "classifier.2.bias": "classifier.2.bias",
                        }

        p_dict = torch.load(Path.model_dir(), map_location=lambda storage, loc: storage)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
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


if __name__ == "__main__":

    net = C3D(num_classes=101, pretrained=False)
    print(net.state_dict())