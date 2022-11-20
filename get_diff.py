import os
import sys
from datetime import datetime

from torch import optim
from tqdm import tqdm
from collections import Counter
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # cuda选择，服务器上有用
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from pathlib import Path  # 我觉得这个更好用，呵呵呵
import argparse
import os
import random

import numpy as np  # 其实我没用到，不过也无所谓了，随便固定一下种子
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.train_set import TrainSetLoader
from dataset.val_set import ValSetLoader
from configs.get_config import parse_train_file
from utils import *

if __name__ == "__main__":
    src_file = Path('/home/guof/fuckpython/TL/result10.txt')
    src_f = open(src_file)
    contents = src_f.readlines()
    for i in range(len(contents)):
        if contents[i][-2] == '9':
            tmp = list(contents[i])
            tmp[-2] = '8'
            contents[i] = ''.join(tmp)
    src_f.close()

    with open('/home/guof/fuckpython/TL/result9.txt', "w") as f:
        for i in range(len(contents)):
            f.write(contents[i])

    # file = [Path('output/DenseNet121_11_08_2022__16_21_14/result.txt'), Path('/home/guof/fuckpython/TL/result.txt'),
    #         Path('/home/guof/fuckpython/TL/9class_tmp2/output/DLA_9_11_19_2022__18_07_45/result.txt'),
    #         Path('/home/guof/fuckpython/TL/9class_tmp1/output/DenseNet121_9_11_19_2022__17_02_36/result.txt')]
    # f = []
    # contents = []
    #
    # for i in file:
    #     tmp = open(str(i))
    #     f.append(tmp)
    #     contents.append(tmp.readlines())
    #
    # error_images = []
    # for i in range(20000):
    #     tmp = []
    #     for j in contents:
    #         class_name = j[i].split()[-1]
    #         if class_name == '9':
    #             class_name = '8'
    #         tmp.append(class_name)
    #     cnt = Counter(tmp)
    #     first_time = cnt.most_common()[0][1]
    #     if (int(contents[0][i].split(' ')[-1]) != 1 and int(contents[0][i].split(' ')[-1]) != 6) and first_time <= 3:
    #         error_images.append(contents[0][i].split(' ')[0])
    #
    # print(len(error_images))
    #
    # tmp1 = Path('../ErrorDataset/').exists()
    # if tmp1:
    #     shutil.rmtree('../ErrorDataset/')
    # Path.mkdir(Path('../ErrorDataset/'))
    # for i in error_images:
    #     shutil.copy('../TL_Dataset/Testset/' + i, '../ErrorDataset/' + i)
    #
    # for i in f:
    #     i.close()
