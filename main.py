import yaml
from types import SimpleNamespace
import argparse
import torch
from torch.utils.data import DataLoader
import os
from models.IIMGF import IIMGF
from src.dataloader import load_dataset,dataset
from train import train_model

def parse_args():
    parser = argparse.ArgumentParser(description="Training script with YAML config")
    parser.add_argument('--config', type=str, default='/home/wyl/work/Rebuttal/IIMGF_yaml/config/train.yaml',
                        required=True, help='Path to the YAML config file')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    # 将 dict 转成可以用 . 访问的形式，例如 config.train.epochs
    return SimpleNamespace(**{k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in config_dict.items()})

def main():
    args = parse_args()
    config = load_config(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.model.gpu_idx
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = IIMGF(config=config).to(device)

    # load dataset
    derm_data_group = load_dataset(dir_release=config.data.derm7pt_path)
    train_iterator = DataLoader(dataset(derm=derm_data_group, shape=(224, 224), mode='train'),
                                batch_size=config.data.batch_size, shuffle=True, num_workers=config.data.num_workers)
    valid_iterator = DataLoader(dataset(derm=derm_data_group, shape=(224, 224), mode='valid'),
                                batch_size=1, shuffle=False, num_workers=2)
    test_iterator = DataLoader(dataset(derm=derm_data_group, shape=(224, 224), mode='test'),
                               batch_size=1, shuffle=False, num_workers=2)

    train_model(config, model, train_iterator, valid_iterator, test_iterator)


if __name__ == '__main__':
    main()
