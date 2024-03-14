# 使用WSL配置 mmdetection 环境  

Editor : LZS  
[github博客地址](https://github.com/lzsion/blog/blob/main/mmdetection_environment_configuration_by_wsl/mmdetection_environment_configuration_by_wsl.md)  

**补充:**  
添加了 mmrotate 环境配置步骤  

## 1. Ubuntu 环境配置(WSL)  

参考链接  
[配置wsl2-Ubuntu视频](https://www.bilibili.com/video/BV1o8411C7wm/)  
[wsl前期准备](https://blog.csdn.net/B11050729/article/details/132580410)  
[官方教程](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)  
[公孙启博客](https://www.gongsunqi.xyz/posts/3c995b2a/)  
[wsl迁移](https://zhuanlan.zhihu.com/p/406917270)  

### 1.1 安装wsl  

**step1**  
[官方教程](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual)  
开启wsl相关功能  
[wsl前期准备](https://blog.csdn.net/B11050729/article/details/132580410)  

打开控制面板 选择 `程序` -> `启动或关闭Windows功能`  
勾选`适用与Linux的Windows子系统` 和 `虚拟机平台`  
![开启wsl相关功能](https://github.com/lzsion/image-hosting/blob/master/blog/Snipaste_2024-03-14_15-15-53.png?raw=true)
然后重启电脑  

或者以管理员身份打开 `PowerShell` 输入  

```shell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

然后重启电脑  

**step2**  
下载 Linux 内核更新包 `wsl_update_x64.msi`  
[linux内核更新包](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)  
运行更新包进行安装  

**step3**  
微软商店下载`ubuntu20.04`  

**step4**  
wsl更新到wsl2  
设置wsl2为默认版本(以管理员打开powershell)  

```shell
wsl --set-default-version 2
```

启动wsl *第一次会安装 Ubuntu  需要输入用户名以及密码*  

```shell
wsl
```

**step5 (optional)**  
wsl迁移  
wsl默认路径是C盘 可以迁移到其他地方  
即先备份到一个位置 再加载到另一个位置  

以管理员运行 powershell  
关闭wsl  

```shell
wsl --shutdown
```

导出wsl  

```shell
wsl --export Ubuntu-20.04 D:/ubuntu/ubuntu2004.tar
```

卸载原有的Ubuntu (卸载后空间会自动释放)  

```shell
wsl --unregister Ubuntu-20.04
```

导入wsl  

```shell
wsl --import Ubuntu-20.04 D:/ubuntu/ubuntu2004 D:/ubuntu/ubuntu2004.tar --version 2
```

修改默认加载的用户名  

```shell
ubuntu2004 config --default-user lzs
```

### 1.2 安装 cuda 环境  

**step1**  
安装 cudatoolkit  
进入官网选择对应版本的 cudatoolkit  
[cudatoolkit官网](https://developer.nvidia.com/cuda-toolkit-archive)  
选择`linux`, `x86_64`, `wsl-Ubuntu`, `version-2.0`, `deb(local)`  
复制命令输入到wsl  

**step2**  
cuda 环境变量配置  

```shell
sudo nano ~/.bashrc
```

将下面两行添加进文件最后(**注意改路径中的cuda版本号**)  

```txt
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

保存退出后(Ctrl+x)  
更新环境变量  

```shell
source ~/.bashrc
```

显示 cuda 版本  

```shell
nvcc -V
```

**step3**  
下载 cudnn  
[cudnn官网下载](https://developer.nvidia.com/rdp/cudnn-archive)  
下载对应版本的 cudnn  
将下载的 `.tar.xz` 文件复制到 Ubuntu 的 `home` 目录下  
解压  

```shell
sudo tar -xvf cudnn**
```

若报错 删去 `identifier` 文件再执行解压命令  

**step4**  
配置 cudnn  
把解压得到的文件分别拷贝到对应的文件夹(**注意改路径中的cuda版本号**)  
进入解压文件夹  

```shell
cd cudnn-linux-x86_64-8.9.4.25_cuda11-archive/
```

复制到对应文件夹  

```shell
sudo cp -r /lib/* /usr/local/cuda-11.8/lib64/
sudo cp -r /include/* /usr/local/cuda-11.8/include/
```

更改读取权限  

```shell
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*
sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*
```
  
检查 cudnn 是否安装成功  

```shell
cat /usr/local/cuda-11.8/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
nvidia-smi
```

没有报错即为安装成功  

### 1.3 配置 conda 环境  

**step1**  
[anaconda官网](https://www.anaconda.com/download)  
官网复制 Linux 版本下载链接  
ubuntu终端输入 `wget <下载链接>` 例如  

```shell
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
```

执行  

```shell
sh Anaconda3-2023.03-Linux-x86_64.sh
```

(输入 `sh A` 按 `tab` 自动补全)  

## 2. mmdetection 配置  

参考链接  
[配置mmdetection视频](https://www.bilibili.com/video/BV1jV411U7zb/)  
[mmdetection的github仓库](https://github.com/open-mmlab/mmdetection)  
[mmdetection官方安装教程](https://mmdetection.readthedocs.io/zh-cn/latest/get_started.html)  
[使用mmdetection训练自己的数据集(VOC格式)](https://juejin.cn/post/7011481869472514056)  

### 2.1 安装 mmdetection  

**step1**  
配置 pytorch  

```shell
conda create --name mmdet python=3.8
conda activate mmdet
```

[pytorch官网](https://pytorch.org/get-started/previous-versions/)  
选择对应 pytorch 版本下载  

**step2**  
进入 conda 的 `mmdet` 环境  
使用 `mim` 安装 mmengine 和 mmcv  

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**step3**  
克隆 mmdetection 仓库 (也可以下载到本地解压)  

```shell
git clone https://github.com/open-mmlab/mmdetection.git
```

将mmdetection库的解析位置设为本地的克隆项目文件  

```shell
cd mmdetection
pip install -v -e .
```

**step4**  
验证安装  
下载配置文件和模型权重文件 运行demo程序  

```shell
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device gpu
```

成功后 `outputs/vis` 文件夹中会有新的图像 `demo.jpg`  

**step5**  
使用vscode打开  
到 `mmdet` 路径下  

```shell
code mmdetection
```

或者在 vscode 中打开  
点击左下角打开远程窗口 连接到wsl  

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
`ann_file='ImageSets/Main/train.txt'`  
`data_prefix=dict(sub_data_root='')`  

**step5 (optional)**  
修改epoch数以及优化器  
`mmdetection/configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py`  

**step6 (optional)**  
修改模型 (nms, anchor)  
`mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py`  

**step7**  
开始训练  
先进入克隆的 `mmdetection` 文件夹  
重新初始化  

```shell
python setup.py install
```

训练  

```shell
python tools/train.py configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py
```

### 2.3 可视化  

绘制曲线 `loss_rpn_cls`  

```shell
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster-rcnn_r50_fpn_1x_voc0712/20231222_133834/vis_data/20231222_133834.json --key loss_rpn_cls
```

显示测试集上的标注图片  

```shell
python tools/test.py work_dirs/faster-rcnn_r50_fpn_1x_voc0712/20231222_133834/vis_data/config.py work_dirs/faster-rcnn_r50_fpn_1x_voc0712/epoch_6.pth --show
```

仅显示测试集上的mAP  

```shell
python tools/test.py work_dirs/faster-rcnn_r50_fpn_1x_voc0712/config.py work_dirs/faster-rcnn_r50_fpn_1x_voc0712/epoch_4.pth
```

## 3. mmrotate 配置  

参考链接  
[mmrotate的github仓库](https://github.com/open-mmlab/mmrotate)  
[mmrotate官方教程](https://mmrotate.readthedocs.io/zh-cn/latest/install.html#id2)  
[mmrotate dev-1.x官方教程](https://github.com/open-mmlab/mmrotate/blob/dev-1.x/docs/zh_cn/get_started.md)  
[mmrotate dev-1.x常见问题解答](https://github.com/open-mmlab/mmrotate/blob/dev-1.x/docs/zh_cn/notes/faq.md)  
[mmrotate dev-1.x安装教程 博客1](https://www.cnblogs.com/lzqdeboke/p/17335742.html)  
[mmrotate dev-1.x安装教程 博客2](https://blog.csdn.net/qq_41627642/article/details/128713683)  

### 3.1 配置 mmrotate  

**step1**  
配置 pytorch  

```shell
conda create --name mmrotate python=3.8
conda activate mmrotate
```

[pytorch官网](https://pytorch.org/get-started/previous-versions/)  
选择对应 pytorch 版本下载  

**step2**  
进入 conda 的 `mmrotate` 环境  
使用 `mim` 安装 mmengine 和 mmcv  

```shell
pip install -U openmim
mim install mmcv-full
mim install mmdet\<3.0.0
```

克隆mmrotate仓库(也可以下载到本地解压)  
*默认的main分支mmrotate版本为0.3.4*  

```shell
git clone https://github.com/open-mmlab/mmrotate.git
```

将mmrotate库的解析位置设为本地的克隆项目文件  

```shell
cd mmrotate
pip install -v -e .
```

检查安装的版本 [版本兼容表格](https://github.com/open-mmlab/mmrotate/blob/main/docs/zh_cn/faq.md)  

```shell
mim list
```

**step3**  
验证安装  
下载配置文件和模型权重文件 运行demo程序  

```shell
mim download mmrotate --config oriented_rcnn_r50_fpn_1x_dota_le90 --dest .
python demo/image_demo.py demo/demo.jpg oriented_rcnn_r50_fpn_1x_dota_le90.py oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file result.jpg
```

成功后当前文件夹中会有新的图像 `result.jpg`  

### 3.2 配置 mmrotate dev-1.x  

**step1**  
配置 pytorch  

```shell
conda create --name mmrotate—dev1.x python=3.8
conda activate mmrotate—dev1.x
```

[pytorch官网](https://pytorch.org/get-started/previous-versions/)  
选择对应 pytorch 版本下载  
也可以克隆之前配置好的 `mmrotate` 的环境  

```shell
conda create --name mmrotate—dev1.x --clone mmrotate
conda activate mmrotate—dev1.x
```

**step2**  
进入 conda 的 `mmrotate—dev1.x` 环境  
使用 `mim` 安装 mmengine 和 mmcv  

```shell
pip install -U openmim
mim install "mmengine==0.10.3"
mim install "mmcv==2.0.0rc4"
```

安装 mmdet 可以直接用 min 安装  

```shell
mim install "mmdet==3.0.0rc6"
```

也可以克隆或下载对应版本的mmdetection到本地后  
将mmdet库的解析位置设为本地克隆的项目文件 *这样便于修改*  

```shell
git clone https://github.com/open-mmlab/mmdetection.git -b dev-3.x
cd mmdetection
pip install -v -e .
```

克隆mmrotate dev-1.x仓库(也可以下载到本地解压)  

```shell
git clone -b dev-1.x https://github.com/open-mmlab/mmrotate.git
```

将mmrotate库的解析位置设为本地的克隆项目文件  

```shell
cd mmrotate
pip install -v -e .
```

配置mmrotate dev-1.x 的时候 一些mmlab的模块可能不兼容  
检查安装的版本 [mmrotate dev-1.x版本兼容表格](https://github.com/open-mmlab/mmrotate/blob/dev-1.x/docs/zh_cn/notes/faq.md)  

```shell
mim list
```

**注:**  
如果有低版本的 mmcv-full 需要删除  

```shell
mim uninstall mmcv-full
```

**step3**  
验证安装  
下载配置文件和模型权重文件 运行demo程序  

```shell
mim download mmrotate --config rotated_rtmdet_s-3x-dota --dest .
python demo/image_demo.py demo/demo.jpg rotated_rtmdet_s-3x-dota.py rotated_rtmdet_s-3x-dota-11f6ccf5.pth --out-file result.jpg
```

成功后当前文件夹中会有新的图像 `result.jpg`  

### 3.3 mmrotate dev-1.x 遇到的一些问题  

写在最后  
我在 `mmrotate dev-1.x` 版本上遇到了一些问题  
问题与 github 上的一篇 issues 描述一致  
[[Bug] With MMrotate dev1.x branch, train rotated faster RCNN on DOTA loss=nan](https://github.com/open-mmlab/mmrotate/issues/988)  
这篇 issue 发布于2023年6月 距今已超过半年 问题还没有得到解决
*若没遇到类似的问题，可以忽略*  

在 `mmrotate dev-1.x` 版本上，我使用自己标注的DOTA格式数据集在 `Rotated Faster R-CNN` 模型以及 `oriented R-CNN` 模型上训练的时候，在训练几百个batch后，出现了 `grad_norm:nan` 和 `loss:nan` 的问题。 (我只修改了模型配置文件中加载数据集的部分)  

- **然而**：  
    1. 同样的数据集，在 `mmrotate dev-1.x` 版本上，使用 `Rotated RTMDet` 模型可以正常训练。  
    2. 同样的数据集，在默认版本 `mmrotate 0.3.4` 上，使用 `Rotated Faster R-CNN` 模型以及 `oriented R-CNN` 模型均能正常训练。  
    3. 使用 `DOTA-v1` 数据集，在 `mmrotate dev-1.x` 版本上，使用 `Rotated Faster R-CNN` 模型进行训练的时候，也没有问题。  

该 issues 的提问者在 issue 的最后也写到  
> Additional information  
> I tried mmrotate0.3.4 on the same dataset and it worked well. Then I tried rotated retinanet on mmrotatedev1.x, it still works well.  
> I also tried to decrease LR, but the same problem happened.  
> I suspect there may be some problem with my environment but cannot figure it out, which is CUDA 11.6 Pytorch1.12.1 MMEngine0.10.3 mmcv2.0.1 mmdet3.0.0rc6 mmrotate1.0.0rc1  

**关于这个问题我的一些排查与尝试**  
(1) 官方方法  
首先我尝试了官方的方法  
在 `mmrotate dev-1.x` 的官方的 doc 中有写到 loss 为 nan 的一些解决办法  
[mmrotate dev-1.x 常见问题解答](https://github.com/open-mmlab/mmrotate/blob/dev-1.x/docs/zh_cn/notes/faq.md)  

> "Loss goes Nan"  
>  
> 1. 检查数据的标注是否正常，长或宽为 0 的框可能会导致回归 loss 变为 nan，一些小尺寸（宽度或高度小于 1）的框在数据增强（例如，instaboost）后也会导致此问题。因此，可以检查标注并过滤掉那些特别小甚至面积为 0 的框，并关闭一些可能会导致 0 面积框出现数据增强。  
> 2. 降低学习率：由于某些原因，例如 batch size 大小的变化，导致当前学习率可能太大。您可以降低为可以稳定训练模型的值。  
> 3. 延长 warm up 的时间：一些模型在训练初始时对学习率很敏感，您可以把 `warmup_iters` 从 500 更改为 1000 或 2000。  
> 4. 添加 gradient clipping: 一些模型需要梯度裁剪来稳定训练过程。默认的 `grad_clip` 是 `None`，你可以在 config 设置 `optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))`。 如果你的 config 没有继承任何包含 `optimizer_config=dict(grad_clip=None)`，你可以直接设置 `optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))`。  

在各个 mmrotate 版本中，该段文档基本一致。  

- **我的改变**  
    1. 我尝试过降低学习率，没有效果。
    2. 也尝试过把 `warmup_iters` 改大，都无法解决。  
    3. 模型配置文件中 `grad_clip` 默认就是 `optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))` 我尝试过把 `max_norm` 改小，但都无济于事。  

(2) mmengine  
在排查官方的解决办法无果后，我怀疑是计算loss的时候出了问题  
我对比 `mmrotate dev-1.x` 和 `mmrotate 0.3.4` 能运行的环境中的各种 `mmlab` 模块的版本，发现 `mmengine` 的版本均为 `0.10.3`  
版本不一样的模块是 `mmcv` 和 `mmdet`  

(3) mmdet  
`mmrotate dev-1.x` 用的是 `mmdet 3.0.0rc6`  
`mmrotate 0.3.4` 用的是 `mmdet 2.28.2`  

我发现 `Rotated Faster R-CNN` 模型的配置文件中，计算损失函数用到了 `mmdet` 模块中的 `mmdet.CrossEntropyLoss` 和 `mmdet.SmoothL1Loss` 来计算损失，而 `Rotated RTMDet` 模型的配置文件中没有用到这两个损失。  
我怀疑是这两个损失函数的计算有问题，我找到了 `mmdet 2.28.2` 中对应的损失函数代码，做了迁移替换，但似乎没有什么效果。  
我也查看了 `max_iou_assigners.py` 代码，但似乎没发现问题。  

(4) mmrotate  
我对比了 `mmrotate dev-1.x` 和 `mmrotate 0.3.4` 两个版本的 `rotated_iou_loss.py` 代码，也没发现问题。  

(5) mmcv  
`mmrotate dev-1.x` 用的是 `mmcv 2.0.0rc4`  
`mmrotate 0.3.4` 用的是 `mmcv-full 1.7.2`  

我尝试过更换 mmcv 版本为 2.0.1 仍然没有解决  

寻找这个问题的原因耗费了我大量的时间和经历，我太累了而放弃了比较 mmcv 两个版本的差异，以及其余的排查问题工作，也许是 mmrotate 最新开发版的不稳定导致的，我希望 mmlab 官方能尽快更新。  
