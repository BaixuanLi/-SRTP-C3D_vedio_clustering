import torch
import torch.nn as nn
from torch import device
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
import cv2
from functools import partial
import matplotlib

from dataset import VideoDataset

matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

from model import C3D_model
from model import DC3D_model


def store(model):
    """
    make hook for feature map
    """

    def hook(module, input, output, key):
        if isinstance(module, nn.MaxPool3d):
            model.feature_maps[key] = output[0]
            model.pool_locs[key] = output[1]
        else:
            model.feature_maps[key] = output

    for idx, layer in enumerate(model._modules.get('features')):
        # _modules returns an OrderedDict
        layer.register_forward_hook(partial(hook, key=idx))


def vis_layer(layer, C3D_model, DC3D_model):
    """
    visualing the layer deconv result
    """
    # print(C3D_model.feature_maps.keys())
    num_feat = C3D_model.feature_maps[layer].shape[1]
    print(num_feat)

    # set other feature map activations to zero
    new_feat_map = C3D_model.feature_maps[layer].clone()
    # choose the max activations map
    act_lst = []
    for i in range(0, num_feat):
        choose_map = new_feat_map[0, i, :, :, :]
        activation = torch.max(choose_map)
        act_lst.append(activation.item())

    act_lst = np.array(act_lst)
    mark = np.argmax(act_lst)

    choose_map = new_feat_map[0, mark, :, :, :]
    max_activation = torch.max(choose_map)

    # make zeros for other feature maps
    if mark == 0:
        new_feat_map[:, 1:, :, :, :] = 0
    else:
        new_feat_map[:, :mark, :, :, :] = 0
        # print(new_feat_map)
        '''print('mark:')
        print(mark)
        print('shape:')
        print(C3D_model.feature_maps[layer].shape[1])'''
        '''if mark != C3D_model.feature_maps[layer].shape[1] - 1:
            new_feat_map[:, mark + 1:, :, :, :] = 0'''
            # print(new_feat_map)
    '''choose_map = torch.where(choose_map == max_activation,
                             choose_map,
                             torch.zeros(choose_map.shape)
                             )'''
    # make zeros for ther activation
    new_feat_map[0, mark, :, :, :] = choose_map

    # print(torch.max(new_feat_map[0, mark, :, :]))

    deconv_output = DC3D_model(new_feat_map, layer, mark, C3D_model.pool_locs)

    new_img_set = []
    for i in range(0, 16):
        new_img = deconv_output.data.numpy()[0][:, i, :, :]
        new_img = new_img.transpose(1, 2, 0)
        # normalize
        new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
        new_img = new_img.astype(np.uint8)
        new_img_set.append(new_img)
    return new_img_set, int(max_activation)


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # forward processing
    features = np.zeros((1, 8192))
    model = C3D_model.C3D(num_classes=101, pretrained=True)
    model.to(device)
    model.eval()
    store(model)


    dataset = VideoDataset(dataset='ucf101', split='test', clip_len=16, preprocess=False)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    for i, (frame, _) in enumerate(dataloader):
        inputs = np.array(frame).astype(np.float32)
        inputs = torch.from_numpy(inputs)
        inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
        with torch.no_grad():
            # print(inputs)
            outputs = model(inputs)
            outputs = outputs.numpy()
        # print(outputs)
        features = np.append(features, outputs, axis=0)

    pool_locs = model.pool_locs

    # init model
    # checkpoint = torch.load('./pretrained_model/c3d-pretrained.pth', map_location=lambda storage, loc: storage)
    # print(list(checkpoint.keys()))
    # model.load_state_dict(checkpoint)'''


    # backward processing
    deconv = DC3D_model.DC3D(pretrained=True)
    deconv.to(device)
    deconv.eval()


    for idx, layer in enumerate([0, 3, 6, 8, 11, 13, 16, 18]):
        img_set, activation = vis_layer(layer, model, deconv)
        plt.title(f'{layer} layer, the max activations is {activation}')
        for i in range(0, 16):
            img = img_set[i]
            img = img*255
            new_img = Image.fromarray(np.uint8(img))
            new_img.save('result_'+'_layer_' +str(idx) +'_'+str(i)+'.jpg')
            # img_size = 112*112*3
