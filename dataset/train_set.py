"""
加载，训练数据集
因为没有测试集，所以我把整个数据集分成了train和val两部分
"""

import random
from pathlib import Path  # 我觉得这个更好用，呵呵呵

import torch
from torch.utils.data import Dataset

from dataset.data_utils import *


def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     比例，训练集集比例
    :param shuffle:   是否打乱
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


class TrainSetLoader(Dataset):
    """
    Coco dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    """

    def __init__(self, root_dir: str, data_size=50000, shape=(32, 32), is_pad_probability=0):
        super().__init__()

        self.dir = Path(root_dir)  # TL_Dataset 的文件夹
        self.limited = data_size  # 限制张数上限，留着可能有用
        self.shape = shape
        self.is_pad_probability = is_pad_probability

        # 指定类型，并获取数据集文件夹路径
        self.data_class = {'Green Circle': 1, 'Green Left': 3, 'Green Negative': 9, 'Green Right': 7, 'Green Up': 5,
                           'Red Circle': 0, 'Red Left': 2, 'Red Negative': 8, 'Red Right': 6, 'Red Up': 4}
        tmp_dir_list = [path for path in self.dir.iterdir()]  # 文件夹下面一级的所有文件
        self.dir_list = []
        for tmp_dir in tmp_dir_list:
            if tmp_dir.stem in self.data_class.keys():
                self.dir_list.append(tmp_dir)  # 获取包含10个数据集的文件夹，其实不用屯着，不过万一有啥用呢
        # 结束

        self.train_files = []  # 训练集，全是 .png 文件哦，读取png注意第四维度（透明维度）的问题
        self.val_files = []  # 验证集，直接从整个数据集里和训练集按照一定比例拉出来的
        for tmp_dir in self.dir_list:
            tmp_train_files, tmp_val_files = data_split(list(tmp_dir.glob('*.png')), 0.9)  # 就不打乱了，无所谓的
            # 下面两个 合计 26426 张
            self.train_files = self.train_files + tmp_train_files  # 合并列表
            self.val_files = self.val_files + tmp_val_files  # 这一部分在 val_set.py 里用到

    def __len__(self):
        """
        返回训练集长度
        """
        return min(len(self.train_files), self.limited)

    def __getitem__(self, idx):
        """
        获取一张图片，
        额，包括他的 one hot 表示
        CHW
        ONE HOT
        """
        img = read_rgb_file(self.train_files[idx])  # 读取图片
        is_pad = np.random.rand(1).item() < self.is_pad_probability  # 按照概率决定是否添加pad
        img = size_transforms(img, self.shape, is_pad)  # gogogo
        class_name = self.train_files[idx].parent.stem  # 获取类别名称

        # # one hot 版本
        # one_hot = torch.zeros([len(self.dir_list)]).long()  # 来个向量
        # one_hot[self.data_class[class_name]] = 1  # 指定位置 置1
        # sample = to_tensor_sample({'img': img, 'class': one_hot}, 'torch.FloatTensor')

        # 普通版本
        sample = to_tensor_sample({'img': img, 'class': self.data_class[class_name]}, 'torch.FloatTensor')

        return sample


if __name__ == '__main__':
    pass
    haha = TrainSetLoader('/home/guof/PyScript/TL/TL_Dataset/')
    print('hehe')
