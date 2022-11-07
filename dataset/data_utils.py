"""
随便处理一下图像
"""

from PIL import Image
import torchvision.transforms as transforms


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
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    sample['img'] = transform(sample['img']).type(tensor_type)
    return sample


def read_rgb_file(filename):
    """
    按照 RGB 格式读取图片
    .png 默认读取是 RGBA 的，最后一维是透明度，这里直接干掉。
    为了和 model 统一，大小缩放为 32*32 ，无所谓的啦，反正图片本来就不大，不会丢失多少信息的
    这种10分类任务本来就很简单的
    """
    return Image.open(filename).convert('RGB').resize((32, 32), Image.ANTIALIAS)
