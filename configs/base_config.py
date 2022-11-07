from yacs.config import CfgNode as CN

########################################################################################################################
cfg = CN()
cfg.name = ''  # Run name
cfg.debug = True  # Debugging flag
########################################################################################################################
### ARCH  我不知道为啥要交arch hhh
########################################################################################################################
cfg.arch = CN()
cfg.arch.seed = 1244  # Random seed for Pytorch/Numpy initialization 4,294,967,295 上限
cfg.arch.epochs = 50  # Maximum number of epochs  轮次
cfg.arch.start_epochs = -1  # Maximum number of epochs  轮次
########################################################################################################################
### MODEL
########################################################################################################################
cfg.model = CN()
cfg.model.checkpoint_path = 'output/'  # 最终权重保存的根目录
cfg.model.save_checkpoint = True  # 开启
cfg.model.best_acc = 0  # 记录最佳精确度 base on vals
########################################################################################################################
### MODEL.SCHEDULER  学习率衰减
########################################################################################################################
cfg.model.scheduler = CN()
cfg.model.scheduler.decay = 0.5  # Scheduler decay rate
cfg.model.scheduler.lr_epoch_divide_frequency = 5  # 学习率更改的频率
########################################################################################################################
### MODEL.OPTIMIZER
########################################################################################################################
cfg.model.optimizer = CN()
cfg.model.optimizer.learning_rate = 0.001
cfg.model.optimizer.weight_decay = 0.0
cfg.model.optimizer.min_lr = 0.0
########################################################################################################################
### MODEL.PARAMS  这里没什么参数
########################################################################################################################
cfg.model.params = CN()
cfg.model.params.type = 'SimpleDLA'
########################################################################################################################
### DATASETS
########################################################################################################################
cfg.datasets = CN()
########################################################################################################################
### DATASETS.AUGMENTATION  这数据集不需要增强，这部分我没写
########################################################################################################################
cfg.datasets.augmentation = CN()
cfg.datasets.augmentation.image_shape = (32, 32)  # Image shape 其实我没用这个
########################################################################################################################
### DATASETS.TRAIN 训练集
########################################################################################################################
cfg.datasets.train = CN()
cfg.datasets.train.batch_size = 128  # Training batch size
cfg.datasets.train.num_workers = 8  # Training number of workers
cfg.datasets.train.path = '../TL_Dataset/'  # 训练集路径
########################################################################################################################
### DATASETS.TEST 测试集
########################################################################################################################
cfg.datasets.test = CN()
cfg.datasets.test.path = '../TL_Dataset/'
cfg.datasets.test.save_class_file_name = 'haha.txt'
########################################################################################################################
### THESE SHOULD NOT BE CHANGED
########################################################################################################################
cfg.configs = ''  # Run configuration file
cfg.default = ''  # Run default configuration file


########################################################################################################################


def get_cfg_defaults():
    return cfg.clone()
