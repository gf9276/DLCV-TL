# 交通信号灯分类

## 数据集

将 TL_Dataset 文件放置在与该项目同级别的目录下，如下图所示:
![文件目录](/ref_img/1.png)
[pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) 是我参考的源文件（在这里可以忽略）

## train

1. 修改配置文件：configs下的yaml文件，配置合适参数，configs/gf_v1.yaml 是一个例子（建议直接用这个，试了几个这个效果最好）
2. cd 到该项目文件夹下，执行命令

```
python train.py --file=configs/gf_v1.yaml
```

开始执行训练，最终得到的best_model输出保存在output里

## test

cd 到该项目文件夹下，执行命令

```
python test.py --file=path/to/your/ckpt --name=your/like/txt/name
```

--file 表示模型参数保存文件的路径，例如

--name 表示输出最终txt文件的名字，默认是result.txt

例子如下：

```
 python test.py --file=output/DenseNet121_11_07_2022__18_50_58/best_model.ckpt --name=哈哈.txt
```
