# 使用WSL配置mmdetection环境  

Editor : LZS  

## 1 Ubuntu环境配置(WSL)  

参考连接  
[配置wsl2-Ubuntu视频](https://www.bilibili.com/video/BV1o8411C7wm/)  
[wsl前期准备](https://blog.csdn.net/B11050729/article/details/132580410)  
[官方教程](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)  
[公孙启博客](https://www.gongsunqi.xyz/posts/3c995b2a/)  
[wsl迁移](https://zhuanlan.zhihu.com/p/406917270)  

### 1.1 安装wsl  

**step1**  
开启wsl相关功能 然后重启电脑 [wsl前期准备](https://blog.csdn.net/B11050729/article/details/132580410)  

**step2**  
下载linux内核更新包 `wsl_update_x64.msi` [官方教程](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)  

**step3**  
微软商店下载`ubuntu20.04`  

**step4**  
wsl更新到wsl2[官方教程](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)  
设置wsl2为默认版本(以管理员打开powershell)  
`wsl --set-default-version 2`  

**step5 (optional)**  
wsl迁移  
wsl默认路径是C盘 可以迁移到其他地方  
即先备份到一个位置 再加载到另一个位置  
以管理员运行powershell  
关闭wsl  
`wsl --shutdown`  
导出  
`wsl --export Ubuntu-20.04 D:/ubuntu/ubuntu2004.tar`  
卸载原有的Linux  
`wsl --unregister Ubuntu-20.04`  
导入wsl  
`wsl --import Ubuntu-20.04 D:/ubuntu/ubuntu2004 D:/ubuntu/ubuntu2004.tarr --version 2`  
修改默认加载的用户名  
`ubuntu2004 config --default-user lzs`  

### 1.2 安装cuda环境  

**step1**  
安装cudatoolkit  
[cudatoolkit官网下载](https://developer.nvidia.com/cuda-toolkit-archive)  
选择`linux` `x86_64` `wsl-Ubuntu` `version-2.0` `deb(local)`  
复制命令到wsl  

**step2**  
cuda环境变量配置  
`sudo nano ~/.bashrc`  
将下面两行添加进文件最后(**注意改路径中的cuda版本号**)  
`export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}`  
``export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}``  
保存退出后(Ctrl+x)  
更新一下环境变量  
`source ~/.bashrc`  
执行 `nvcc -V` 显示cuda版本  

**step3**  
下载cudnn  
[cudnn官网下载](https://developer.nvidia.com/rdp/cudnn-archive)  
下载对应版本的cudnn  
将下载的 `.tar.xz` 文件复制到ubuntu的home目录下  
解压  
`sudo tar -xvf cudnn**`  
[若报错 删去identifier文件再执行解压命令]  

**step4**  
配置cudnn  
把解压得到的文件分别拷贝到对应的文件夹(**注意改路径中的cuda版本号**)  
进入解压文件夹  
`cd cudnn-linux-x86_64-8.9.4.25_cuda11-archive/`  
复制到对应文件夹  
`sudo cp -r /lib/* /usr/local/cuda-11.8/lib64/`  
`sudo cp -r /include/* /usr/local/cuda-11.8/include/`  
更改读取权限  
`sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*`  
`sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*`  
检查cudnn是否安装成功  
`cat /usr/local/cuda-11.8/include/cudnn_version.h | grep CUDNN_MAJOR -A 2`  
`nvidia-smi`  

### 1.3 配置conda环境  

**step1**  
[anaconda官网](https://www.anaconda.com/download)  
官网复制Linux版本下载链接  
ubuntu终端输入 `wget <下载链接>` 例如  
`wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh`  

执行  
`sh Anaconda3-2023.03-Linux-x86_64.sh`  
(输入 `sh A` 按 `tab` 自动补全)  

**step2**  
配置pytorch  
`conda create --name mmdet python=3.8`  
`conda activate mmdet`  
[pytorch官网](https://pytorch.org/get-started/previous-versions/)  
选择对应pytorch版本下载  

## 2 mmdetection配置  

参考连接  
[配置mmdetection视频](https://www.bilibili.com/video/BV1jV411U7zb/)  
[mmdetection的github仓库](https://github.com/open-mmlab/mmdetection)  
[mmdetection官方安装教程](https://mmdetection.readthedocs.io/zh-cn/latest/get_started.html)  
[使用mmdetection训练自己的数据集(VOC格式)](https://juejin.cn/post/7011481869472514056)  

### 2.1 安装mmdetection  

**step1**  
使用 MIM 安装 MMEngine 和 MMCV  
进入conda的mmdet环境  
`pip install -U openmim`  
`mim install mmengine`  
`mim install "mmcv>=2.0.0"`  

**step2**  
创建文件夹  
`mkdir mmdet`  
`cd mmdet/`  
克隆仓库  
`git clone https://github.com/open-mmlab/mmdetection.git`  
`cd mmdetection`  
`pip install -v -e .`  

**step3**  
验证安装  
下载配置文件和模型权重文件  
`mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .`  
运行demo程序  
`python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device gpu`  
成功后 `outputs/vis` 文件夹中会有新的图像 `demo.jpg`  

**step4**  
使用vscode打开  
到 `mmdet` 路径下  
`code mmdetection`  
或者在vscode中打开 左下角打开远程窗口 连接到wsl  

### 2.2 训练VOC数据集  

[使用mmdetection训练自己的数据集(VOC格式)](https://juejin.cn/post/7011481869472514056)  

**step1**  
在克隆的 `mmdetection` 文件夹下新建 `data` 文件夹  
将VOC数据集放入 `data` 文件夹  

**step2**  
修改类别名称列表(2处)  
`mmdetection/mmdet/datasets/voc.py` 中的 `VOCDataset(XMLDataset)`  
`mmdetection/mmdet/evaluation/functional/class_names.py` 中的 `voc_classes()`  

**step3**  
修改类别数(2处)  
`mmdetection/configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py`  
`mmdetection/configs/base/models/faster_rcnn_r50_fpn.py`  
两个文件中的 `num_classes`  

**step4**  
改数据集加载  
`mmdetection/configs/base/datasets/voc0712.py`  
中修改  
`data_root = 'data/SAR-AIRcraft-1.0/'`  
`ann_file='ImageSets/Main/train.txt',`  
`data_prefix=dict(sub_data_root=''),`  

**step5 (optional)**  
修改epoch数以及优化器  
`mmdetection/configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py`  

**step6 (optional)**  
修改模型(nms, anchor)  
`mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py`  

**step7**  
开始训练  
先进入克隆的 `mmdetection` 文件夹  
重新初始化  
`python setup.py install`  
训练  
`python tools/train.py configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py`  

### 2.3 可视化  

**绘制曲线** `loss_rpn_cls`  
`python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster-rcnn_r50_fpn_1x_voc0712/20231222_133834/vis_data/20231222_133834.json --key loss_rpn_cls`  

**显示测试集上的标注图片**  
`python tools/test.py work_dirs/faster-rcnn_r50_fpn_1x_voc0712/20231222_133834/vis_data/config.py work_dirs/faster-rcnn_r50_fpn_1x_voc0712/epoch_6.pth --show`  

**仅显示测试集上的mAP**  
`python tools/test.py work_dirs/faster-rcnn_r50_fpn_1x_voc0712/config.py work_dirs/faster-rcnn_r50_fpn_1x_voc0712/epoch_4.pth`  
