from yacs.config import CfgNode
from configs.base_config import get_cfg_defaults


def get_default_config(cfg_default):
    """
    Get default configuration from file
    """
    config = get_cfg_defaults()
    config.merge_from_list(['default', cfg_default])
    return config


def merge_cfg_file(config, cfg_file=None):
    """Merge configuration file"""
    if cfg_file is not None:
        config.merge_from_file(cfg_file)
        config.merge_from_list(['configs', cfg_file])
    return config


def parse_train_config(cfg_default, cfg_file):
    """
    Parse model configuration for training

    Parameters
    ----------
    cfg_default : str
        Default **.py** configuration file
    cfg_file : str
        Configuration **.yaml** file to override the default parameters

    Returns
    -------
    configs : CfgNode
        Parsed model configuration
    """
    # Loads default configuration
    config = get_default_config(cfg_default)
    # Merge configuration file
    config = merge_cfg_file(config, cfg_file)
    # Return prepared configuration
    return config


def parse_train_file(file):
    """
    Parse file for training

    Parameters
    ----------
    file : str
        File, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file

    Returns
    -------
    configs : CfgNode
        Parsed model configuration
    ckpt : str
        Parsed checkpoint file
    """
    # If it's a .yaml configuration file
    if file.endswith('yaml'):
        cfg_default = 'configs/default_config'
        return parse_train_config(cfg_default, file)
    # We have a problem
    else:
        raise ValueError('You need to provide a .yaml or .ckpt to train')
