"""
随便处理一下图像
"""

from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt


def cv22pil(img):
    # 带变色的，别乱玩
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return image


def pil2cv2(image):
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return img


def size_transforms(img: Image.Image, shape, is_pad=True):
    """
    处理一下大小
    :param img: PIL
    :param shape:
    :param is_pad:
    :return:
    """
    width, height = img.size  # 获取图片大小，他返回的是宽和高，但是 shape 表示高和宽
    if (width < shape[1] or height < shape[0]) and is_pad:  # 如果比shape小并且允许pad
        if width < shape[1] // 2 or height < shape[0] // 2:
            img = img.resize((shape[1] // 2, shape[0] // 2), Image.ANTIALIAS)  # 大小不能差太多，不然全是黑的
        width, height = img.size
        left = (shape[1] - width) // 2
        right = shape[1] - width - left
        up = (shape[0] - height) // 2
        down = shape[0] - height - up
        transform = transforms.Pad((left, up, right, down), fill=0, padding_mode="constant")  # 左上右下
        img = transform(img)  # 填充完成
    else:
        img = img.resize(shape, Image.ANTIALIAS)
    return img


def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.
    数据从 PIL 转成 tensor 格式的

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    # transform = transforms.ToTensor()  # 这个帮我归一化了
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 这玩意不好使
    ])
    sample['img'] = transform(sample['img']).type(tensor_type)
    for i in range(3):  # 标准化。。。 加上效果好很多
        mean = sample['img'][i].mean()
        std = sample['img'][i].std()
        sample['img'][i] = (sample['img'][i] - mean) / std
    return sample


def read_rgb_file(filename):
    """
    按照 RGB 格式读取图片
    .png 默认读取是 RGBA 的，最后一维是透明度，这里直接干掉。
    大小不一定
    这种10分类任务本来就很简单的
    """
    return Image.open(filename).convert('RGB')  # 不做大小的修改


if __name__ == "__main__":
    pass
    img = read_rgb_file('../../TL_Dataset/Testset/5.png')
    img = pil2cv2(img)
    plt.imshow(img)
    plt.show()
    img = cv22pil(img)
    img = size_transforms(img, (64, 64), True)
    img = pil2cv2(img)
    plt.imshow(img)
    plt.show()
    print('hehe')
