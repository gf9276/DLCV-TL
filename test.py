# -*- coding = utf-8 -*-
# @time:2022/11/7 16:31
# @File:test.py
# @Software:PyCharm

"""
训练TL数据集
"""
import os
import sys

from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # cuda选择，服务器上有用
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse

from utils import *
from dataset.test_set import *
from train import evaluation, setup_datasets_and_dataloaders


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='checkpoint')
    parser.add_argument('--file', dest='file', help='checkpoint文件相对路径',
                        default='output/SimpleDLA_11_07_2022__17_42_00/best_model.ckpt', type=str)
    parser.add_argument('--name', dest='name', help='.txt文件保存的名字', default='result.txt', type=str)
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), 'You need to provide a .ckpt of .yaml file'
    print('Called with args:')
    print(args)
    return args


def main(model_file):
    # 获取预训练模型，里面有两部分 'state_dict' 和 'config'
    pretrained_model_path = Path(curPath) / model_file
    checkpoint = torch.load(str(pretrained_model_path))
    model_args = checkpoint['config'].model.params
    config = checkpoint['config']
    config.datasets.test.save_class_file_name = args.name  # 就是.txt文件保存的名字，默认我写成result的

    # 获取模型并载入参数
    model = get_model(config.model.params.type).cuda()  # 模型
    pre = checkpoint
    model_dict = model.state_dict()  # 模型参数
    pretrained_dict = {k: v for k, v in pre['state_dict'].items() if k in model_dict}  # 选取名字一样的
    model_dict.update(pretrained_dict)  # 更新一下。。
    model.load_state_dict(model_dict)  # 直接载入
    model = model.cuda()  # 扔到 GPU 里去
    model.eval()  # ！！！！！！别忘了

    print('Loaded model from {}'.format(pretrained_model_path))
    print('model params {}'.format(model_args))

    # # 测试一下结果对不对，事实证明是对的
    # train_dataset, train_loader = setup_datasets_and_dataloaders(config.datasets, 50000)
    # criterion = nn.CrossEntropyLoss()  # 损失函数，交叉熵 自带 softmax 的那种
    # config.model.save_checkpoint = False  # 小测一波
    # evaluation(config, train_dataset, model, criterion, config.arch.start_epochs)

    # 载入数据，一个个来也没什么问题。。。
    batch_size = 200  # 20000的整数，不然除不尽可能会有问题的。。。
    test_dataset = TestSetLoader(config.datasets.test.path)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             pin_memory=False,
                             shuffle=False,
                             num_workers=0,
                             worker_init_fn=None,
                             sampler=None)
    print('Loaded {} image pairs '.format(len(test_loader)))

    with torch.no_grad():
        n_test_batches = len(test_loader)
        txt_save_path = Path(config.model.checkpoint_path) / config.datasets.test.save_class_file_name
        pbar = tqdm(enumerate(test_loader, 0),
                    unit=' images',
                    unit_scale=batch_size,
                    total=n_test_batches,
                    smoothing=0,
                    disable=False,
                    ncols=135)  # 调整长度

        with open(str(txt_save_path), "w") as f:
            for (batch_idx, data) in pbar:
                # calculate loss
                data_cuda = sample_to_cuda(data)
                outputs = model(data_cuda['img'])  # 输入图片
                _, predicted = outputs.max(1)
                for i in range(len(data['name'])):
                    f.write(data['name'][i] + ' ' + str(predicted[i].item()) + '\n')


if __name__ == "__main__":
    args = parse_args()
    main(args.file)
