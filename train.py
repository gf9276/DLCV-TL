"""
训练TL数据集
"""
import os
import sys
from datetime import datetime

from torch import optim
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # cuda选择，服务器上有用
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

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


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='TL training script')
    parser.add_argument('--file', dest='file', help='配置文件相对路径', default='configs/gf_v1.yaml', type=str)
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), 'You need to provide a .ckpt of .yaml file'
    print('Called with args:')
    print(args)
    return args


def set_seeds(seed=42):
    """Set Python random seeding and PyTorch seeds.
    固定随机数种子

    Parameters
    ----------
    seed: int, default: 42
        Random number generator seeds for PyTorch and python
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_datasets_and_dataloaders(config, data_size=None):
    """
    Prepare datasets for training, validation and test.
    """

    def _worker_init_fn(worker_id):
        """
        Worker init fn to fix the seed of the workers
        用来固定数据加载过程中的线程随机数种子的
        """
        # seed = 43 + worker_id
        seed = torch.initial_seed() % 2 ** 32 + worker_id  # worker_id 可以不加，每个epoch都不一样，**优先级很高的
        np.random.seed(seed)
        random.seed(seed)
        # print(str(torch.initial_seed()) + '\n')
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

    train_dataset = TrainSetLoader(config.train.path, data_size=data_size,
                                   shape=config.augmentation.image_shape,
                                   is_pad_probability=config.augmentation.is_pad_probability)
    sampler = None  # 这个是在多GPU上用的，我没搞多GPU

    train_loader = DataLoader(train_dataset,
                              batch_size=config.train.batch_size,
                              pin_memory=True,
                              shuffle=True,
                              num_workers=config.train.num_workers,
                              worker_init_fn=_worker_init_fn,
                              sampler=sampler,
                              drop_last=True)

    return train_dataset, train_loader  # 数据集获取成功，外面一般用的都是 train_loader


def main(args):
    # Parse config
    file = args.file
    file = curPath + '/' + file
    config = parse_train_file(file)
    print(config)

    if config.arch.seed is not None:
        set_seeds(config.arch.seed)

    printcolor('-' * 25 + ' MODEL PARAMS ' + '-' * 25)
    printcolor(config.model.params, 'red')

    # 获取数据集
    train_dataset, train_loader = setup_datasets_and_dataloaders(config.datasets, 50000)
    printcolor('({}) length: {}'.format("Train", len(train_dataset)))

    # 获取网络
    model = get_model(config.model.params.type).cuda()  # 模型
    optimizer = optim.Adam(model.parameters(), lr=config.model.optimizer.learning_rate,
                           weight_decay=config.model.optimizer.weight_decay)  # 优化器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.model.scheduler.decay,
                                                       config.arch.start_epochs)
    criterion = nn.CrossEntropyLoss()  # 损失函数，交叉熵 自带 softmax 的那种

    # checkpoint model
    date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  # 加上时间
    date_time = config.model.params.type + '_' + date_time  # 模型名字加时间吧
    config.model.checkpoint_path = os.path.join(config.model.checkpoint_path, date_time)  # emm记录数据的文件所在地
    # log_path = os.path.join(config.model.checkpoint_path, 'logs')
    # os.makedirs(log_path, exist_ok=True)  # 这个文件 暂时没用，我没写tensorboard
    print('Saving models at {}'.format(config.model.checkpoint_path))
    os.makedirs(config.model.checkpoint_path, exist_ok=True)
    evaluation(config, train_dataset, model, criterion, config.arch.start_epochs)
    # Train，start_epochs是上次训练到一半的断点，正常训练不用管的
    for epoch in range(config.arch.start_epochs + 1, config.arch.epochs):
        # train for one epoch (only log if eval to have aligned steps...)
        train(config, train_loader, model, optimizer, criterion, epoch)
        evaluation(config, train_dataset, model, criterion, epoch)
        for param_group in optimizer.param_groups:
            if param_group['lr'] > config.model.optimizer.min_lr:
                scheduler.step()  # 学习率衰减，加着玩玩呗
            else:
                param_group['lr'] = config.model.optimizer.min_lr

    printcolor('Training complete, models saved in {}'.format(config.model.checkpoint_path), "green")


def train(config, train_loader, model, optimizer, criterion, epoch):
    printcolor('\n' + '-' * 50)
    printcolor('epoch ' + str(epoch), 'red')
    for param_group in optimizer.param_groups:
        printcolor('Changing learning rate to {:8.6f}'.format(param_group['lr']), 'red')
    printcolor('\n' + '-' * 50)

    # Set to train mode
    model.train()

    n_train_batches = len(train_loader)
    pbar = tqdm(enumerate(train_loader, 0),
                unit=' images',
                unit_scale=config.datasets.train.batch_size,
                total=n_train_batches,
                smoothing=0,
                disable=False,
                ncols=135)  # 调整长度
    correct = 0
    total = 0
    train_loss = 0

    for (batch_idx, data) in pbar:
        # calculate loss
        optimizer.zero_grad()
        data_cuda = sample_to_cuda(data)
        outputs = model(data_cuda['img'])  # 输入图片
        targets = data_cuda['class']

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description('{:<8d}: Train [ Epoch {}, Loss {:.4f}, Acc {:.3f}: {:d}/{:d}'.format(
            batch_idx, epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def evaluation(config, train_dataset, model, criterion, epoch):
    """
    我只是借用一下数据集里划分出来的
    """
    # Set to eval mode
    model.eval()
    model.training = False

    val_loss = 0
    correct = 0
    total = 0

    batch_size = 128
    val_dataset = ValSetLoader(train_dataset, is_pad=config.datasets.test.is_pad)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            pin_memory=False,
                            shuffle=False,
                            num_workers=0,
                            worker_init_fn=None,
                            sampler=None)
    print('Loaded {} image pairs '.format(len(val_loader)))

    with torch.no_grad():
        n_val_batches = len(val_loader)
        pbar = tqdm(enumerate(val_loader, 0),
                    unit=' images',
                    unit_scale=batch_size,
                    total=n_val_batches,
                    smoothing=0,
                    disable=False,
                    ncols=135)  # 调整长度

        for (batch_idx, data) in pbar:
            # calculate loss
            data_cuda = sample_to_cuda(data)
            outputs = model(data_cuda['img'])  # 输入图片
            targets = data_cuda['class']
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description('{:<8d}: Train [ Epoch {}, Loss {:.4f}, Acc {:.3f}: {:d}/{:d}'.format(
                batch_idx, epoch, val_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    if config.model.save_checkpoint:
        acc = 100. * correct / total
        if acc > config.model.best_acc:
            config.model.best_acc = acc
            current_model_path = os.path.join(config.model.checkpoint_path, 'best_model.ckpt')
            printcolor('\nSaving model (epoch:{}) at {}'.format(epoch, current_model_path), 'green')
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'config': config
                }, current_model_path)
        config.arch.start_epochs = epoch


if __name__ == "__main__":
    args = parse_args()
    main(args)
