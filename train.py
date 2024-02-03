# -*- coding: utf-8 -*-
import os
import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric import datasets
import torch_geometric.utils as utils

import sys
sys.path.append("./sat")
from sat.models_fingerprint import GraphTransformer
from sat.data import GraphDataset
from sat.utils import count_parameters
from sat.position_encoding import POSENCODINGS
from sat.gnn_layers import GNN_TYPES
from timeit import default_timer as timer

def load_args():
    parser = argparse.ArgumentParser(
        description='Structure-Aware Transformer on Metal chelating agent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=158,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="Metal",
                        help='name of dataset')
    parser.add_argument('--num-heads', type=int, default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=6, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=64, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--batch-size', type=int, default= 128,
                        help='batch size')
    parser.add_argument('--abs-pe', type=str, default= None, choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=20, help='dimension for absolute PE')
    parser.add_argument('--outdir', type=str, default='./SATCMF_train_result',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=50, help="number of iterations for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--use-edge-attr', action='store_false', default=True, help='use edge features')
    parser.add_argument('--use-fp-density-morgan', action='store_true', default=False, help='use fp density morgan')

    parser.add_argument('--edge-dim', type=int, default=32, help='edge features hidden dim')
    parser.add_argument('--gnn-type', type=str, default='gcn',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--k-hop', type=int, default=3 , 
        help="Number of hops to use when extracting subgraphs around each node")
    parser.add_argument('--global-pool', type=str, default='cls', choices=['mean', 'cls', 'add'],
                        help='global pooling method')
    parser.add_argument('--se', type=str, default="khopgnn", 
            help='Extractor type: khopgnn, or gnn')
    parser.add_argument('--early_stop', type=int, default=50, help='The patience of early stopping')
    parser.add_argument('--cuda-device', type=int, default= 0 ,
                    help='CUDA device index')
    parser.add_argument('--data-path', type=str, default='./dataset/10_fold_cv/fold10', help='Path to the data directory')
    parser.add_argument('--n-tags', type=int, default= 145, help='Number of tags')
    parser.add_argument('--num-edge-features', type=int, default=15, help='Number of edge features')
    parser.add_argument('--cross_val', type=int, default=None, help='Number of folds for cross-validation. If None, no cross-validation is performed.')


    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm


    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass

        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.use_edge_attr:
            outdir = outdir + '/edge_attr'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass

        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        bn = 'BN' if args.batch_norm else 'LN'
        if args.se == "khopgnn":
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.se, args.gnn_type, args.k_hop, args.dropout, args.warmup, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, args.global_pool, bn, date_time,
            )

        else:
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn, date_time,
    )
            
        if args.use_fp_density_morgan:
            outdir = outdir + '_use_fp_density_morgan'


        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args



def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0
    running_rmse = 0.0
    running_mae = 0.0

    tic = timer()
    for i, data in enumerate(loader):
        size = len(data.y)
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        if args.abs_pe == 'lap':
            sign_flip = torch.rand(data.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.abs_pe = data.abs_pe * sign_flip.unsqueeze(0)

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        rmse = torch.sqrt(torch.mean((output - data.y) ** 2))
        mae = torch.mean(torch.abs(output - data.y))

        running_loss += loss.item() * size
        running_rmse += rmse.item() * size
        running_mae += mae.item() * size

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_rmse = running_rmse / n_sample
    epoch_mae = running_mae / n_sample
    print('Train loss: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}, time: {:.2f}s'.format(
          epoch_loss, epoch_rmse, epoch_mae, toc - tic))
    return epoch_loss, epoch_rmse, epoch_mae


import math
def eval_epoch(model, loader, criterion, use_cuda=False, split='Val'):
    model.eval()

    running_loss = 0.0
    mae_loss = 0.0
    mse_loss = 0.0

    tic = timer()
    with torch.no_grad():
        for data in loader:
            size = len(data.y)
            if use_cuda:
                data = data.cuda()

            output = model(data)
            loss = criterion(output, data.y)
            mse_loss += F.mse_loss(output, data.y).item() * size
            mae_loss += F.l1_loss(output, data.y).item() * size

            running_loss += loss.item() * size
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_mae = mae_loss / n_sample
    epoch_rmse = math.sqrt(mse_loss / n_sample)
    print('{} loss: {:.4f} RMSE: {:.4f} MAE: {:.4f} time: {:.2f}s'.format(
          split, epoch_loss, epoch_rmse, epoch_mae, toc - tic))  
    return epoch_loss, epoch_mae, epoch_rmse

import pandas as pd

def eval_epoch(model, loader, criterion, use_cuda=False, split='Val', save_csv=False, csv_path='output.csv'):
    model.eval()

    running_loss = 0.0
    mae_loss = 0.0
    mse_loss = 0.0

    tic = timer()

    output_list = []
    target_list = []
    morgan_list = []
    with torch.no_grad():
        for data in loader:
            size = len(data.y)
            if use_cuda:
                data = data.cuda()

            output = model(data)
            loss = criterion(output, data.y)
            mse_loss += F.mse_loss(output, data.y).item() * size
            mae_loss += F.l1_loss(output, data.y).item() * size

            running_loss += loss.item() * size

            if save_csv:
                output_list += output.cpu().numpy().tolist()  
                target_list += data.y.cpu().numpy().tolist()  

    if save_csv:
        df = pd.DataFrame({
            'target': target_list,
            'output': output_list,
            
        })
        df.to_csv(csv_path, index=False)

    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_mae = mae_loss / n_sample
    epoch_rmse = math.sqrt(mse_loss / n_sample)  
    print('{} loss: {:.4f} RMSE: {:.4f} MAE: {:.4f} time: {:.2f}s'.format(
          split, epoch_loss, epoch_rmse, epoch_mae, toc - tic))  
    return epoch_loss, epoch_mae, epoch_rmse



def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = args.data_path

    n_tags = args.n_tags
    num_edge_features = args.num_edge_features
    input_size = n_tags

    train_data = torch.load(data_path + "/train.pt")
    val_data = torch.load( data_path + "/valid.pt")
    test_data = torch.load(data_path + "/test.pt") 


    train_dset = GraphDataset(train_data, degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr) 
    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
            shuffle=True)
        
    val_dset = GraphDataset(val_data, degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)

    test_dset = GraphDataset(test_data, degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(train_dset)
            abs_pe_encoder.apply_to(val_dset)

    deg = torch.cat([
        utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for
        data in train_dset])

    model = GraphTransformer(in_size=input_size,
                             num_class=1,
                             d_model=args.dim_hidden,
                             dim_feedforward=2*args.dim_hidden,
                             dropout=args.dropout,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             batch_norm=args.batch_norm,
                             abs_pe=args.abs_pe,
                             abs_pe_dim=args.abs_pe_dim,
                             gnn_type=args.gnn_type,
                             use_edge_attr=args.use_edge_attr,
                             num_edge_features=num_edge_features,
                             edge_dim=args.edge_dim,
                             k_hop=args.k_hop,
                             se=args.se,
                             deg=deg,
                             use_fp_density_morgan= args.use_fp_density_morgan,
                             global_pool=args.global_pool) 
    print(model)

    output_filename = 'model_args.txt'

    with open(os.path.join(args.outdir, output_filename), 'w') as output_file:
        for arg in vars(args):
            output_file.write("{}: {}\n".format(arg, getattr(args, arg)))
        print(model, file=output_file)


    if args.use_cuda:
        device = torch.device("cuda:{}".format(args.cuda_device))
        torch.cuda.set_device(device)
        model.to(device)

    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.5,
                                                            patience=15,
                                                            min_lr=1e-05,
                                                            verbose=False)
    else:
        lr_steps = (args.lr - 1e-6) / args.warmup
        decay_factor = args.lr * args.warmup ** .5
        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-6 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr


    
    #FIXME
    if abs_pe_encoder is not None:
        abs_pe_encoder.apply_to(test_dset)

    print("Training...")
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    early_stop_counter = 0
    start_time = timer()


    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss, train_rmse, train_mae = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)
        val_loss, val_mae, val_rmse = eval_epoch(model, val_loader, criterion, args.use_cuda, split='Val')
        test_loss, test_mae, test_rmse = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

        if args.warmup is None:
            lr_scheduler.step(val_loss)

        logs = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'test_loss': test_loss,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
        }

        if args.save_logs:
            logs_df = pd.DataFrame(logs, index=[0])
            if epoch == 0:
                logs_df.to_csv(args.outdir + '/logs.csv', header=True, index=False)
            else:
                logs_df.to_csv(args.outdir + '/logs.csv', mode='a', header=False, index=False)

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.early_stop:
            print('Early stopping after {} epochs without improvement'.format(args.early_stop))
            break




    total_time = timer() - start_time
    print("best epoch: {} best val loss: {:.4f}".format(best_epoch, best_val_loss))
    model.load_state_dict(best_weights)
    print()
    print("Training...")
    train_loss, train_mae, train_rmse = eval_epoch(model, train_loader, criterion, args.use_cuda, split='Train', save_csv=True, csv_path= args.outdir + '/train_output.csv')

    print("Train Loss: {:.4f}, Train MAE: {:.4f}, Train RMSE: {:.4f}".format(train_loss, train_mae, train_rmse))

    print()
    print("Validating...")
    val_loss, val_mae, val_rmse = eval_epoch(model, val_loader, criterion, args.use_cuda, split='Val', save_csv=True, csv_path= args.outdir + '/val_output.csv')

    print("Validation Loss: {:.4f}, Validation MAE: {:.4f}, Validation RMSE: {:.4f}".format(val_loss, val_mae, val_rmse))

    print()
    print("Testing...")
    test_loss, test_mae, test_rmse = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test', save_csv=True, csv_path= args.outdir + '/test_output.csv')

    print("Test Loss: {:.4f}, Test MAE: {:.4f}, Test RMSE: {:.4f}".format(test_loss, test_mae, test_rmse))

    print(args)

    if args.save_logs:
        results = {
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }

        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(args.outdir + '/results.csv', header=['value'], index_label='name')

        torch.save(
            {'args': args,
            'state_dict': best_weights},
            args.outdir + '/model.pth')



if __name__ == "__main__":
    global args
    
    args = load_args()

    main()