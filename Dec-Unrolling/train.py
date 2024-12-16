import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from lib.unrolling_model import UnrollingModel
from tqdm import tqdm
import os
import math
from utils import *

seed_everything(3407)
# Hyper-parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 8
learning_rate = 2e-3
num_epochs = 30
num_workers = 4

# load datasets
loss_name = 'MSE'

if loss_name == 'MSE':
    loss_fn = nn.MSELoss()
elif loss_name == 'Huber':
    loss_fn = nn.HuberLoss(delta=1)
elif loss_name == 'Mix':
    loss_fn = WeightedMSELoss(6, 12)

dataset_dir = './datasets/PEMS0X_data/'
experiment_name = 'k_hop'
dataset_name = 'PEMS04'
T = 12
t_in = 6
stride = 3

train_set, val_set, test_set, train_loader, val_loader, test_loader = create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers)
# print(len(train_loader), len(val_loader), len(test_loader))

# normalization:
train_mean, train_std = train_set.data.mean(), train_set.data.std()

num_admm_blocks = 5
num_heads = 4
feature_channels = 6
k_hop = 5
ADMM_info = {
                 'ADMM_iters':24,
                 'CG_iters': 3,
                 'PGD_iters': 3,
                 'mu_u_init':3,
                 'mu_d1_init':3,
                 'mu_d2_init':3,
                 }

model_pretrained_path = None

model = UnrollingModel(num_admm_blocks, device, T, t_in, num_heads, train_set.signal_channel, feature_channels, graph_info=train_set.graph_info, ADMM_info=ADMM_info, k_hop=k_hop).to(device)
# 'UnrollingForecasting/MainExperiments/models/v2/PEMS04/direct_4b_4h_6f/val_15.pth'

if model_pretrained_path is not None:
    model = model.load_state_dict(torch.load(model_pretrained_path))

ADMM_iters = ADMM_info['ADMM_iters']
# optimizer
import torch.optim as optim
from torch.optim import lr_scheduler

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 创建文件处理器
log_dir = f'./logs/{experiment_name}'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{dataset_name}_{loss_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.log'
logger = create_logger(log_dir, log_filename)

logger.info('#################################################')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'pretrained path: {model_pretrained_path}')
logger.info(f'learning k hop: {k_hop}')
logger.info(f'batch size: {batch_size}')
logger.info(f'learning rate: {learning_rate}')
logger.info(f'Loss function: {loss_name}')
logger.info(f"Total parameters: {total_params}")
logger.info(f'device: {device}')
logger.info('PARAMTER SETTINGS:')
logger.info(f'ADMM info: {ADMM_info}')
logger.info(f'graph info: nodes {train_set.n_nodes}, edges {train_set.n_edges}')
logger.info('--------BEGIN TRAINING PROCESS------------')

model_dir = os.path.join(f'./models/{experiment_name}', f'{dataset_name}/{loss_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f')
os.makedirs(model_dir, exist_ok=True)
masked_flag = False
# train models
# test = True
for epoch in range(num_epochs):
    # TODO: remember to / 50 # don't need now
    model.train()
    running_loss = 0
    rec_mse = 0
    pred_mse = 0
    pred_mae = 0
    pred_mape = 0
    nearest_loss = 0

    for y, x in tqdm(train_loader):
        # print(y.shape, x.shape)
        optimizer.zero_grad()
        y, x = y.to(device), x.to(device) # y in (B, t, nodes, 1)
        # normalization
        y = (y - train_mean) / train_std

        output = model(y) # in (B, T, nodes, 1)

        output = output * train_std + train_mean

        rec_mse += ((x[:,:t_in] - output[:,:t_in]) ** 2).mean().item()
        # only unknowns
        if masked_flag:
            x, output = x[:,t_in:], output[:,t_in:]
        loss = loss_fn(output, x)
        loss.backward()       
        optimizer.step()
        # clamp param
        model.clamp_param(0.20, 0.20)
        # loggers
        running_loss += loss.item()
        # with torch.no_grad():
        if masked_flag:
            pred_mse += ((x - output) ** 2).mean().item()
            pred_mae += (torch.abs(output - x)).mean().item()
            pred_mape += (torch.abs(output - x) / (x + 1e-6)).mean().item() * 100
            nearest_loss += ((x[:, 0] - output[:, 0]) ** 2).mean().item()
        else:
            pred_mse += ((x[:,t_in:] - output[:,t_in:]) ** 2).mean().item()
            pred_mae += (torch.abs(output[:, t_in:] - x[:,t_in:])).mean().item()
            pred_mape += (torch.abs(output[:, t_in:] - x[:,t_in:]) / (x[:,t_in:] + 1e-6)).mean().item() * 100
            nearest_loss += ((x[:, t_in] - output[:, t_in]) ** 2 / y.size(0)).mean().item()

        glm = model.model_blocks[0]['graph_learning_module']
        admm_block = model.model_blocks[0]['ADMM_block']
        # break
    
    logger.info(f'output: ({output.max().item()}, {output.min().item()})')

    total_loss = running_loss / len(train_loader)
    nearest_rmse = math.sqrt(nearest_loss / len(train_loader))
    rec_rmse = math.sqrt(rec_mse / len(train_loader))
    pred_rmse = math.sqrt(pred_mse / len(train_loader))
    pred_mae = pred_mae / len(train_loader)
    pred_mape = pred_mape / len(train_loader)
    metrics = {
        'rec_RMSE': rec_rmse,
        'pred_RMSE': pred_rmse,
        'pred_MAE': pred_mae,
        'pred_MAPE(%)': pred_mape 
    }

    logger.info(f'Training: Epoch [{epoch + 1}/{num_epochs}], Loss:{total_loss:.4f}, rec_RMSE: {rec_rmse:.4f}, RMSE_next:{nearest_rmse:.4f}, RMSE:{pred_rmse:.4f}, MAE:{pred_mae:.4f}, MAPE(%):{pred_mape:.4f}')
    logger.info(f'multiQ1, multiQ2, multiM: {glm.multiQ1.max().item()}, {glm.multiQ2.max().item()}, {glm.multiM.max().item()}')
    logger.info(f'rho, rho_u, rho_d: {admm_block.rho.max().item()}, {admm_block.rho_u.max().item()}, {admm_block.rho_d.max().item()}')
    logger.info(f'max alphas, {admm_block.alpha_x.max().item():.4f}, {admm_block.alpha_zu.max().item():.4f}, {admm_block.alpha_zd.max().item():.4f}')
    logger.info(f'max betas, {admm_block.beta_x.max().item():.4f}, {admm_block.beta_zu.max().item():.4f}, {admm_block.beta_zd.max().item():.4f}')

    # validation
    if (epoch + 1) % 6 == 0:
        model.eval()
        with torch.no_grad():
            running_loss = 0
            nearest_loss = 0
            rec_mse = 0
            pred_mse = 0
            pred_mape = 0
            pred_mae = 0
            for y, x in tqdm(val_loader):
                y, x = y.to(device), x.to(device)

                y = (y - train_mean) / train_std

                output = model(y)

                output = output * train_std + train_mean

                rec_mse += ((x[:,:t_in] - output[:,:t_in]) ** 2).mean().item()
                if masked_flag:
                    x, output = x[:,t_in:], output[:,t_in:]
                loss = loss_fn(output, x)

                running_loss += loss.item()
                if masked_flag:
                    pred_mse += ((x - output) ** 2).mean().item()
                    pred_mae += (torch.abs(output - x)).mean().item()
                    pred_mape += (torch.abs(output - x) / x).mean().item() * 100
                    nearest_loss += ((x[:, 0] - output[:, 0]) ** 2).mean().item()
                else:
                    pred_mse += ((x[:,t_in:] - output[:,t_in:]) ** 2).mean().item()
                    pred_mae += (torch.abs(output[:, t_in:] - x[:,t_in:])).mean().item()
                    pred_mape += (torch.abs(output[:, t_in:] - x[:,t_in:]) / (x[:,t_in:] + 1e-6)).mean().item() * 100
                    nearest_loss += ((x[:, t_in] - output[:, t_in]) ** 2).mean().item()

        total_loss = running_loss / len(val_loader)
        nearest_rmse = math.sqrt(nearest_loss / len(val_loader))
        rec_rmse = math.sqrt(rec_mse / len(val_loader))
        pred_rmse = math.sqrt(pred_mse / len(val_loader))
        pred_mae = pred_mae / len(val_loader)
        pred_mape = pred_mape / len(val_loader)
        metrics = {
            'rec_RMSE': rec_rmse,
            'pred_RMSE': pred_rmse,
            'pred_MAE': pred_mae,
            'pred_MAPE(%)': pred_mape 
        }
        
        logger.info(f'Validation: Epoch [{epoch + 1}/{num_epochs}], Loss:{total_loss:.4f}, rec_RMSE:{rec_rmse:.4f} RMSE_next:{nearest_rmse:.4f}, RMSE:{pred_rmse:.4f}, MAE:{pred_mae:.4f}, MAPE(%):{pred_mape:.4f}')
        # save models
        torch.save(model, os.path.join(model_dir, f'val_{epoch+1}.pth'))

    # rmse_total = math.sqrt(avg_mse_loss)