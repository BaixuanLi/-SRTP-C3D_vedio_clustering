# for evaluation

from collections import Counter
import numpy as np


class Evaluate:
    def __init__(self, dataset, cluster_labels):
        cluster_acc = []
        temp_true = []
        temp_cluster = []
        for i, sample in enumerate(dataset):
            if i == 0:
                true_label = sample[1]
                cluster_label = cluster_labels[i]
                temp_true.append(true_label)
                temp_cluster.append(cluster_label)
            else:
                true_label = sample[1]
                cluster_label = cluster_labels[i]
                if true_label == temp_true[-1]:
                    temp_true.append(true_label)
                    temp_cluster.append(cluster_label)
                else:
                    temp_acc = self.calculate_acc(cluster_labels=temp_cluster)
                    cluster_acc.append(temp_acc)
                    temp_true = []
                    temp_cluster = []
                    temp_true.append(true_label)
                    temp_cluster.append(cluster_label)

        temp_acc = self.calculate_acc(cluster_labels=temp_cluster)
        cluster_acc.append(temp_acc)

        max_acc = max(cluster_acc)
        min_acc = min(cluster_acc)
        mean_acc = np.mean(cluster_acc)

        for i in range(len(cluster_acc)):
            print('The accuracy of cluster ({}) is {}%'.format(i, cluster_acc[i]*100))
        print('---------------------------------------------------------------------------------------')
        print('The max accuracy of the model is {}%.'.format(max_acc*100))
        print('The min accuracy of the model is {}%.'.format(min_acc*100))
        print('The mean accuracy of the model is {}%.'.format(mean_acc*100))
        print('---------------------------------------------------------------------------------------')

    def calculate_acc(self, cluster_labels):
        temp = []
        counter = Counter(cluster_labels)
        for key in counter:
            temp.append(counter[key])

        sum_temp = sum(temp)
        for i in range(len(temp)):
            temp[i] /= sum_temp

        for i in range(len(temp)):
            temp[i] *= temp[i]

        acc = sum(temp)

        return acc
