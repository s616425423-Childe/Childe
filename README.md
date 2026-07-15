斯坦福大学李菲菲教授的cs231n课程assignment1

一、所需的下载
首先需要的安装的软件：PyCharm和Anaconda
其次需要从https://cs231n.github.io/assignments2026/assignment1/下载assignment1需要的压缩包
另外由于下载的压缩包没有所需的数据集，因此还需要从python官网下载作业所需的数据集，可以在PyCharm中任意打开一个新的项目，输入：
import torchvision
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform,download=True)
来下载需要的数据集CIFAR10，路径就在该py项目下。

二、环境配置
由于是本地完成作业，不用按照官网一样在Google Colab完成，因此需要在本地配置环境。
需要在anaconda prompt中创建新环境，类似于：conda create -n cs231n-a1 python=3.11  #我所创建的环境名是cs231n-a1，python版本是3.11
然后安装jupyter。由于作业大都是notebook文档，所以正确的jupyter安装很重要。
然后就可以开始完成作业了。

由于是入门学习，所以很多都弄不懂，花了很多时间，多用AI解决问题吧。
