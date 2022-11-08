"""
加载，训练数据集
因为没有测试集，所以我把整个数据集分成了train和val两部分
"""

import random
from pathlib import Path  # 我觉得这个更好用，呵呵呵

import torch
from torch.utils.data import Dataset

from dataset.data_utils import *


class TestSetLoader(Dataset):
    """
    Coco dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    """

    def __init__(self, root_dir: str, data_size=50000, shape=(32, 32), is_pad=False):
        super().__init__()

        self.dir = Path(root_dir) / 'Testset'  # 测试集
        self.limited = data_size  # 限制张数上限，留着可能有用
        self.shape = shape
        self.is_pad = is_pad

        # 获取所有的文件名字，并升序排列
        tmp_files = []
        for i in list(self.dir.glob('*.png')):
            tmp_files.append(int(i.stem))
        tmp_files.sort()

        # 处理后获取绝对路径
        self.files = []
        for i in tmp_files:
            self.files.append(self.dir / (str(i) + '.png'))

    def __len__(self):
        """
        返回训练集长度
        """
        return min(len(self.files), self.limited)

    def __getitem__(self, idx):
        """
        获取一张图片，
        额，包括他的 one hot 表示
        CHW
        ONE HOT
        """
        # 读取图片，并限制大小
        img = read_rgb_file(self.files[idx])  # 读取图片
        img = size_transforms(img, self.shape, self.is_pad)  # 直接resize一下大小
        # 转成tensor
        sample = to_tensor_sample({'img': img, 'name': self.files[idx].name}, 'torch.FloatTensor')

        return sample


if __name__ == '__main__':
    pass
    haha = TestSetLoader('/home/guof/fuckpython/TL/TL_Dataset/')
    print('hehe')
