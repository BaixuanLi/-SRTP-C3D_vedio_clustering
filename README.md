# 实验环境：

- python 3.8

- pytorch 1.7.1


- matplotlib 3.4.2  （可能存在与捆绑安装的pyqt包不兼容导致绘图失败问题，建议卸载pyqt）


- sklearn

- OpenCV

- FFmpeg, FFprobe（可选项，与OpenCV在该项目中发挥同样功能）


# 实验数据：

**UCF-101**

本实验使用UCF-101公开视频数据集

可自行从[官网](https://www.crcv.ucf.edu/data/UCF101.php)下载后调用 `dataset.py` 进行数据处理（注意事先完成实验步骤1中路径的配置)

注意处理数据需配置 VedioDataset 中的参数 `preprocess=True`

若为已处理后的数据，则设置`preprocess=False`

终端命令输入：

```
activate envs （envs为实验所需虚拟环境名）
python dataset.py
```

**注意**  确保所下载的UCF-101内结构如下：

```
UCF-101
├── ApplyEyeMakeup
│   ├── v_ApplyEyeMakeup_g01_c01.avi
│   └── ...
├── ApplyLipstick
│   ├── v_ApplyLipstick_g01_c01.avi
│   └── ...
└── Archery
│   ├── v_Archery_g01_c01.avi
│   └── ...
```



**其他/自定义数据集**

本项目除修改dataset.py中部分参数生成数据集外，还为大家提供了另一种数据处理的可行方法，大家还可使用本实验提供的FFmpeg, FFprobe处理视频的方法：

可调用utils中的`video_to_jpg.py`将avi格式数据提取为jpg格式的连续帧集

```
activate envs （envs为实验所需虚拟环境名）
python utils/video_to_jpg.py avi_video_directory jpg_video_directory
```

生成n_frame文件使用utils中的`generate_n_frame.py`

```
python utils/generate_n_frame.py jpg_video_directory
```

**注意**  使用该方法主要考虑的是使用自定义其他数据集进行网络测试的情况，故并不包含train/val/test集合的划分，如特殊需要则需另自定义模块划分，或重写`dataset.py`进行实现

# 实验步骤：

1.配置`Path_file.py`中静态地址：（因OpenCV无法识别中文路径，故请采用英文命名路径）

`root_dir`：原始avi数据地址

`output_dir`：处理后帧（jpg）数据地址，包含train/val/test目录 （按以下形式预创对应目录）

```
preprocess_ucf（可自定义该输出文件夹名称）
├── train
├── val
├── test
```

`model_dir`：预训练网络所在路径 （使用本项目中自带预训练模型则不需要修改路径）

2.终端命令输入：

```
activate envs （envs为实验所需虚拟环境名）
python main.py cluster_num dim_num
(其中cluster_num为需要聚类数目，本实验中因所使用的测试集类别数定为101)
(dim_num为降维后所得到维度，建议参数选取为2或3，因目前绘图方面只支持2D和3D绘图)
示例命令：python main.py 101 2
```

**注意**  事先在项目目录下创建 visualization 目录以存放绘图结果

# 实验结果：

首先我们考察对于UCF-101测试集聚类为101个cluster时的性能：

<img src="D:\C3D_Vedio_Clustering\visualization\2d_clustering.jpg" alt="2d_clustering" style="zoom: 33%;" />

![101](C:\Users\LEGION\Pictures\Camera Roll\101.png)

经过评估计算，类内最高准确率可达100%，而最低只有9.3%，101类平均准确率为45.7%

以上为对于UCF-101整体的聚类结果，现展示抽取其中几类组成小样本后的性能：

**对于5类小样本：**

<img src="D:\C3D_Vedio_Clustering\visualization\2d_clustering_5.jpg" alt="2d_clustering_5" style="zoom:33%;" />

![5](C:\Users\LEGION\Pictures\Camera Roll\5.png)

抽取UCF-101其中5个类别视频进行评估时，可以看到聚类效果有了极大的进步，最高类内准确率达到100%，最低类内准确率达到61.8%，平均准确率为81.9%

**对于10类小样本：**

<img src="D:\C3D_Vedio_Clustering\visualization\2d_clustering_10.jpg" alt="2d_clustering_10" style="zoom:33%;" />

![10](C:\Users\LEGION\Pictures\Camera Roll\10.png)

类内最高准确率可达100%，最低有59.68%，10类平均准确率为81.3%

相比于5类小样本时准确率略微降低
