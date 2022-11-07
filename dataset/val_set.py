"""
没有测试集，我只能从训练集里搞点出来当作验证集
所以这个是基于训练集的
"""

from pathlib import Path  # 我觉得这个更好用，呵呵呵
from PIL import Image
from torch.utils.data import Dataset
import random
import torch
from dataset.data_utils import read_rgb_file, to_tensor_sample
from dataset.train_set import TrainSetLoader


class ValSetLoader(Dataset):
    """
    Coco dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    """

    def __init__(self, train_dataset: TrainSetLoader):
        super().__init__()
        self.train_dataset = train_dataset

    def __len__(self):
        """
        返回训练集长度
        """
        return len(self.train_dataset.val_files)

    def __getitem__(self, idx):
        """
        获取一张图片，
        额，包括他的 one hot 表示
        CHW
        ONE HOT
        """
        img = read_rgb_file(self.train_dataset.val_files[idx])
        class_name = self.train_dataset.val_files[idx].parent.stem  # 获取类别名称
        sample = to_tensor_sample({'img': img, 'class': self.train_dataset.data_class[class_name]},
                                  'torch.FloatTensor')

        return sample


if __name__ == '__main__':
    pass
    haha = TrainSetLoader('/home/guof/PyScript/TL/TL_Dataset/')
    hehe = ValSetLoader(haha)
    print(len(hehe))
