import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from evaluate import Evaluate
from collections import Counter
from sklearn.manifold import TSNE

'''X = np.array(([1, 2, 3, 4, 5], [3, 2, 1, 4, 3], [2, 3, 4, 5, 1]))
Y = np.array(([[1, 2, 2, 2, 2]]))

re = KMeans(n_clusters=2).fit(X)

y_pred = re.labels_

print(y_pred)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

a = []
a.append([1, 2, 2])
a.append([2, 3, 4])
a = np.array(a)

print(Y.shape)
print(Y[0, :].shape)
X = np.append(X, Y, axis=0)
print(X)
re = KMeans(n_clusters=2).fit(X)
y_pred = re.labels_ 

print(y_pred)

dir = 'D:/UCF-101/'
dir1 = os.path.join(dir, '/', 'video')
dir2 = os.path.join(dir, 'video')
print(dir1)
print(dir2)'''
'''a = [1, 1, 2, 4, 2, 1]
b = Counter(a)
c = []
print(b)
for i in range(len(a)):
    print(a[i])

d = sum(c)
print(d)
print('The num of {}'.format(a[0]))
a = ['test', 'test', 'test']
print(list(set(a)))
'''
'''
X = np.array([[0, 0, 0, 1, 2], [0, 1, 1, 3, 2], [1, 0, 1, 1, 2], [1, 1, 1, 3, 2]])

tsne = TSNE(n_components=3, init='pca', verbose=True)  # T-SNE降维可视化
tsne.fit_transform(X)
dim_redu_feature = tsne.embedding_

ax = plt.subplot(projection='3d')  # 创建3D绘图工程
ax.set_title('3d_visualization')
label_pred = (0, 1, 2, 3)
ax.scatter(dim_redu_feature[:, 0], dim_redu_feature[:, 1], dim_redu_feature[:, 2], c=label_pred, cmap=plt.cm.get_cmap('Spectral'))
ax.set_xlabel('X')
ax.set_xlabel('Y')
ax.set_xlabel('Z')
plt.savefig('./test/3d_clustering.jpg', dpi=300)
plt.show()
'''
'''with open('./utils/UCF_labels.txt', 'r') as f:
    class_names = f.readlines()
    f.close()

for i in range(10):
    print(class_names[i])'''
