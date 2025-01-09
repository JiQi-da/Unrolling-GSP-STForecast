import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from lib.unrolling_model import UnrollingModel
# from lib.graph_learning_module import Swish
from lib.backup_modules import visualise_graph
from tqdm import tqdm
import os
import math
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', help='CUDA device', type=int)
parser.add_argument('--dataset', help='dataset name', type=str)
parser.add_argument('--batchsize', help='batch size', type=int)
parser.add_argument('--tin', help='time input', default=6, type=int)
parser.add_argument('--tout', help='time output', default=6, type=int)
parser.add_argument('--hop', help='k for kNN', default=6, type=int)
parser.add_argument('--numblock', help='number of admm blocks', default=5, type=int)
parser.add_argument('--numlayer', help='number of admm layers', default=25, type=int)
parser.add_argument('--cgiter', help='CGD iterations', default=3, type=int)
parser.add_argument('--seed', help='random seed', default=3407, type=int)
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--debug', help='if debug, save model every iteration', default=False, type=bool)
parser.add_argument('--optim', help='optimizer', default='adam', type=str)
parser.add_argument('--mode', help='normalization mode', type=str)

args = parser.parse_args()

seed_everything(args.seed)
# Hyper-parameter[s
device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
batch_size = args.batchsize
learning_rate = args.lr# 1e-3
num_epochs = 30
num_workers = 4

# load datasets
loss_name = 'MSE'

if loss_name == 'MSE':
    loss_fn = nn.MSELoss()
elif loss_name == 'Huber':
    loss_fn = nn.HuberLoss(delta=1)
elif loss_name == 'Mix':
    loss_fn = WeightedMSELoss(args.tin, args.tin + args.tout)

def get_degrees(u_edges:torch.Tensor):
    '''
    u_edges: ndarray, in (n_edges, 2), already bidirectional
    '''
    n_edges = u_edges.size(0)
    degrees = np.zeros((n_edges,), dtype=int)
    for i in range(n_edges):
        degrees[u_edges[i,0]] += 1
        # degrees[u_edges[i,1]] += 1
    return degrees

k_hop = args.hop
dataset_dir = '/mnt/qij/datasets/PEMS0X_data/'
experiment_name = f'{k_hop}_hop_concatFE_{args.tin}_{args.tout}'
dataset_name = args.dataset
T = args.tin + args.tout
t_in = args.tin
stride = 3

return_time = True

train_set, val_set, test_set, train_loader, val_loader, test_loader = create_dataloader(dataset_dir, dataset_name, T, t_in, stride, batch_size, num_workers, return_time)
# print(len(train_loader), len(val_loader), len(test_loader))
signal_channels = train_set.signal_channel

print('signal channels', signal_channels)

# visualise_graph(train_set.graph_info['u_edges'], train_set.graph_info['u_dist'], dataset_name, dataset_name + '.png')
# normalization:
# train_mean, train_std = train_set.data.mean(), train_set.data.std()
# train_min, train_max = train_set.data.min(), train_set.data.max()
data_normalization = Normalization(train_set, args.mode)

num_admm_blocks = args.numblock
num_heads = 4
feature_channels = 6
ADMM_info = {
                 'ADMM_iters':args.numlayer,
                 'CG_iters': args.cgiter,
                 'PGD_iters': 3,
                 'mu_u_init':3,
                 'mu_d1_init':3,
                 'mu_d2_init':3,
                 }
# graph_sigma = 6

model_pretrained_path = None



model = UnrollingModel(num_admm_blocks, device, T, t_in, num_heads, train_set.signal_channel, feature_channels, GNN_layers=2, graph_info=train_set.graph_info, ADMM_info=ADMM_info, k_hop=k_hop).to(device)
# 'UnrollingForecasting/MainExperiments/models/v2/PEMS04/direct_4b_4h_6f/val_15.pth'

if model_pretrained_path is not None:
    model = model.load_state_dict(torch.load(model_pretrained_path))

ADMM_iters = ADMM_info['ADMM_iters']
# optimizer
import torch.optim as optim
from torch.optim import lr_scheduler

if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif args.optim == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.2) # TODO: step size

# 创建文件处理器
log_dir = f'/mnt/qij/Dec-Results/logs/{experiment_name}'
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{dataset_name}_{loss_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.log'
# logger = create_logger(log_dir, log_filename)

grad_logger_dir = f'/mnt/qij/Dec-Results/grad_logs/{experiment_name}'
os.makedirs(grad_logger_dir, exist_ok=True)
# grad_logger = create_logger(grad_logger_dir, log_filename, add_console_handler=False)

logger = setup_logger('logger1', os.path.join(log_dir, log_filename), logging.DEBUG, to_console=True)
grad_logger = setup_logger('logger2', os.path.join(grad_logger_dir, log_filename), logging.INFO, to_console=False)

debug_model_path = os.path.join(f'/mnt/qij/Dec-Results/debug_models/{experiment_name}', f'{dataset_name}/{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.pth')

print('log dir', log_dir)
logger.info('#################################################')
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'pretrained path: {model_pretrained_path}')
logger.info(f'learning k hop: {k_hop}')
logger.info(f'feature channels: {feature_channels}')
# logger.info(f'graph sigma: {graph_sigma}')
logger.info(f'batch size: {batch_size}')
logger.info(f'learning rate: {learning_rate}')
logger.info(f'Loss function: {loss_name}')
logger.info(f"Total parameters: {total_params}")
logger.info(f'device: {device}')
logger.info('PARAMTER SETTINGS:')
logger.info(f'ADMM blocks: {num_admm_blocks}')
logger.info(f'ADMM info: {ADMM_info}')
logger.info(f'graph info: nodes {train_set.n_nodes}, edges {train_set.n_edges}')
logger.info('--------BEGIN TRAINING PROCESS------------')

grad_logger.info('------BEGIN TRAINING PROCESS-------')

model_dir = os.path.join(f'/mnt/qij/Dec-Results/models/{experiment_name}', f'{dataset_name}/{loss_name}_{num_admm_blocks}b{ADMM_iters}_{num_heads}h_{feature_channels}f.pth')
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

    iteration_count = 0

    for y, x, t_list in tqdm(train_loader):
        # print(y.shape, x.shape)
        optimizer.zero_grad()
        y, x, t_list = y.to(device), x.to(device), t_list.to(device) # y in (B, t, nodes, 1)
        # normalization
        # y = (y - train_mean) / train_std
        y = data_normalization.normalize_data(y)
        # y = (y - train_min) / (train_max - train_min)
        try:
            output = model(y, t_list) # in (B, T, nodes, 1)
        except ValueError as ve:
            print(f'Error in [Epoch {epoch}, Iter {iteration_count}] - {ve}')
        # except AssertionError as ae:
           #  print(f'Error in [Epoch {epoch}, Iter {iteration_count}] - {ae}')
        output = data_normalization.recover_data(output)
        if args.mode == 'normalize':
        # output = output * (train_max - train_min) + train_min # output * train_std + train_mean
            output = nn.ReLU()(output)# [:,:,:,:signal_channels]

        rec_mse += ((x[:,:t_in] - output[:,:t_in]) ** 2).mean().item()
        # only unknowns
        if masked_flag:
            x, output = x[:,t_in:], output[:,t_in:]
        loss = loss_fn(output, x)
        loss.backward()       
        optimizer.step()
        iteration_count += 1

        grad_logger.info(f'[Epoch {epoch}, Iter {iteration_count}]')
        for name, param in model.named_parameters():
            # if not model.use_old_extrapolation:
            if 'agg_fc.weight' in name:
                grad_logger.info(f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')
                if iteration_count % 30 == 9:
                    print(f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')
            if model.use_old_extrapolation:
                if 'linear_extrapolation' in name:
                    grad_logger.info(f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')
                    if iteration_count % 30 == 9:
                        print(f'{name}: ({param.min():.4f}, {param.max():.4f})\t grad (L2 norm): {param.grad.data.norm(2).item():.4f}')
        # save model for debug
        if args.debug:
            torch.save(model.state_dict(), debug_model_path)

        # max_grad = 0.0 
        # max_grad_param_name = None
        # for name, param in model.named_parameters(): 
        #     if param.grad is not None: 
        #         param_grad_norm = param.grad.data.norm(2).item() 
        #         if param_grad_norm > max_grad: 
        #             max_grad = param_grad_norm 
        #             max_grad_param_name = name
        # print(f'Max gradient parameter name: {max_grad_param_name}, Max gradient value: {max_grad}')

        # total_norm = 0.0
        # for p in model.parameters(): 
        #     param_norm = p.grad.data.norm(2) 
        #     total_norm += param_norm.item() ** 2 
        # total_norm = total_norm ** (1. / 2) 

        # print(f'Gradient norm: {total_norm}')
        # clamp param
        model.clamp_param(0.18, 0.18)
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
    logger.info(f'min alphas, {admm_block.alpha_x.min().item():.4f}, {admm_block.alpha_zu.min().item():.4f}, {admm_block.alpha_zd.min().item():.4f}')
    logger.info(f'max betas, {admm_block.beta_x.max().item():.4f}, {admm_block.beta_zu.max().item():.4f}, {admm_block.beta_zd.max().item():.4f}')
    logger.info(f'min betas, {admm_block.beta_x.min().item():.4f}, {admm_block.beta_zu.min().item():.4f}, {admm_block.beta_zd.min().item():.4f}')

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
            for y, x, t_list in tqdm(val_loader):
                y, x, t_list = y.to(device), x.to(device), t_list.to(device)

                # y = (y - train_mean) / train_std
                # y = (y - train_min) / (train_max - train_min)
                y = data_normalization.normalize_data(y)
                output = model(y, t_list)
                output = data_normalization.recover_data(output)
                if args.mode == 'normalize':
                    output = nn.ReLU()(output)
                # output = output * (train_max - train_min) + train_min
                # output = output * train_std + train_mean

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