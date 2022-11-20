from termcolor import colored  # 好看
from models import *


def printcolor(message, color="white"):
    """Print a message in a certain color (only rank 0)"""
    print(colored(message, color))


def sample_to_cuda(data):
    """
    将数据 放到cuda 里
    """
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        data_cuda = {}
        for key in data.keys():
            data_cuda[key] = sample_to_cuda(data[key])
        return data_cuda
    elif isinstance(data, list):
        data_cuda = []
        for key in data:
            data_cuda.append(sample_to_cuda(key))
        return data_cuda
    else:
        return data.to('cuda')


def get_model(model_type: str):
    """
    返回指定类型的网络，if else 呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵，笨办法也挺好的，呵呵呵呵呵呵呵
    我觉得比字典好。。。丢人。。。
    """
    model = None

    if model_type == 'VGG19':
        model = VGG('VGG19')
    elif model_type == 'ResNet18':
        model = ResNet18()
    elif model_type == 'PreActResNet18':
        model = PreActResNet18()
    elif model_type == 'GoogLeNet':
        model = GoogLeNet()
    elif model_type == 'DenseNet121':
        model = DenseNet121()
    elif model_type == 'ResNeXt29_2x64d':
        model = ResNeXt29_2x64d()
    elif model_type == 'MobileNet':
        model = MobileNet()
    elif model_type == 'MobileNetV2':
        model = MobileNetV2()
    elif model_type == 'DPN92':
        model = DPN92()
    elif model_type == 'ShuffleNetG2':
        model = ShuffleNetG2()
    elif model_type == 'SENet18':
        model = SENet18()
    elif model_type == 'ShuffleNetV2':
        model = ShuffleNetV2(1)
    elif model_type == 'EfficientNetB0':
        model = EfficientNetB0()
    elif model_type == 'RegNetX_200MF':
        model = RegNetX_200MF()
    elif model_type == 'SimpleDLA':
        model = SimpleDLA()
    elif model_type == 'DLA':
        model = DLA()
    elif model_type == 'DenseNet121_9':
        model = DenseNet121(9)
    elif model_type == 'DLA_9':
        model = DLA(num_classes=9)
    elif model_type == 'DPN92_9':
        model = DPN92(num_classes=9)
    elif model_type == 'SimpleDLA_9':
        model = SimpleDLA(num_classes=9)

    return model
