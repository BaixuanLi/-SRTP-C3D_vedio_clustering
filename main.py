import torch
import numpy as np
from model import C3D_model
import matplotlib.pyplot as plt
from evaluate import Evaluate
import sys
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from dataset import VideoDataset
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main(arg1, arg2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    '''with open('utils/UCF_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()'''

    features = np.zeros((1, 101))
    # print(features.shape)

    # init model
    model = C3D_model.C3D(num_classes=101, pretrained=True)
    # checkpoint = torch.load('./pretrained_model/c3d-pretrained.pth', map_location=lambda storage, loc: storage)
    # print(list(checkpoint.keys()))
    # model.load_state_dict(checkpoint)'''
    model.to(device)
    model.eval()
    '''video = 'D:/UCF-101/UCF_test/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
    # read video
    dataset = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=False)  #todo
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
    for i, (video, _) in enumerate(dataloader):
    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model(inputs)
                outputs = outputs.numpy()
            # print(outputs)
            features = np.append(features, outputs, axis=0)
            # print(features.shape)'''
    dataset = VideoDataset(dataset='ucf101', split='test', clip_len=16, preprocess=False)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    for i, (frame, _) in enumerate(dataloader):
        inputs = np.array(frame).astype(np.float32)
        # inputs = np.expand_dims(inputs, axis=0)
        # inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
        inputs = torch.from_numpy(inputs)
        inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.numpy()
        # print(outputs)
        features = np.append(features, outputs, axis=0)
        # print(features.shape)'''

    # cap.release()
    # cv2.destroyAllWindows()

    features = features[1:, :]
    # print(features.shape[0])
    features = np.array(features)

    tsne = TSNE(n_components=arg2, init='pca', n_iter=5000, verbose=True)  # T-SNE降维可视化
    tsne.fit_transform(features)
    features = tsne.embedding_

    re = KMeans(n_clusters=arg1, verbose=True).fit(features)  # KMeans聚类
    label_pred = re.labels_

    Evaluate(dataset=dataset, cluster_labels=label_pred)

    if arg2 == 3:
        ax = plt.subplot(projection='3d')  # 创建3D绘图工程
        ax.set_title('3d_visualization')
        ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=label_pred, cmap=plt.cm.get_cmap('Spectral'))
        ax.set_xlabel('X')
        ax.set_xlabel('Y')
        ax.set_xlabel('Z')
        plt.savefig('./visualization/3d_clustering.jpg', dpi=300)
        plt.show()
    elif arg2 == 2:
        # 2d绘图工程
        plt.scatter(features[:, 0], features[:, 1], c=label_pred, cmap=plt.cm.get_cmap('Spectral'))
        plt.savefig('./visualization/2d_clustering.jpg', dpi=300)
        plt.show()
    else:
        print('Only 2D and 3D drawings are supported !')


if __name__ == '__main__':
    n_clusters = sys.argv[1]  # the number of clusters
    n_d = sys.argv[2]  # the dim that you want after dim reduction
    n_clusters = int(n_clusters)
    n_d = int(n_d)
    main(n_clusters, n_d)
