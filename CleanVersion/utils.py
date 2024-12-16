import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import TrafficDataset
import os
import logging

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers):
    data_folder = os.path.join(dataset_dir, dataset_name)
    graph_csv = dataset_name + '.csv'
    data_file = dataset_name + '.npz'
    if dataset_name == 'PEMS03':
        id_file = dataset_name + '.txt'
    else:
        id_file = None

    train_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'train', id_file=id_file)
    val_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'val', id_file=id_file)
    test_set = TrafficDataset(data_folder, graph_csv, data_file, T, t_in, stride, 'test', id_file=id_file)

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader

def create_logger(log_dir, log_filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# def 
class WeightedMSELoss(nn.Module):
    def __init__(self, t, T, weights=None) -> None:
        super().__init__()
        self.t = t
        self.T = T
        if weights is None:
            self.weights = t / (T - t)
        else:
            self.weights = weights

    def forward(self, inputs, target):
        rec_loss = nn.MSELoss()(inputs[:,:self.t], target[:,:self.t])# ((inputs[:,:self.t] - target[:,:self.t]) ** 2).mean()
        pred_loss = nn.MSELoss()(inputs[:,self.t:], target[:,self.t:])# ((inputs[:,self.t:] - target[:,self.t:]) ** 2).mean()
        return rec_loss * self.weights + pred_loss