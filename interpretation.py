import os
import argparse
import torch
from torch.utils.data import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from torch_geometric.loader import DataLoader
from torch_geometric import utils
import sys
sys.path.append("./sat")
from sat.models_fingerprint import GraphTransformer
from sat.data import GraphDataset
from sat.position_encoding import POSENCODINGS


def load_args():
    parser = argparse.ArgumentParser(
        description='Model visualization: SATCMF',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--visu', action='store_true', help='perform visualization')
    parser.add_argument('--graph-idx', type=int, default=0, help='graph to interpret')
    parser.add_argument('--outpath', type=str, default='./interpretation_analysis',
                        help='visualization output path')
    parser.add_argument('--dataset-path', type=str, default='./interpretation_analysis/data_interpretation_DOTA_random_edge.pt',
                        help='path to the dataset file')
    parser.add_argument('--model-path', type=str, default='./interpretation_analysis/SATCMF_gcn_K=3_model.pth',
                        help='path to the model file')
    
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    return args

def load_model(datapath, dataset):
    args = load_args()
    model = torch.load(datapath, map_location=torch.device('cuda'))
    args, state_dict = model['args'], model['state_dict']
    n_tags = args.n_tags
    num_edge_features = args.num_edge_features
    input_size = n_tags

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
                             deg=None,
                             use_fp_density_morgan= args.use_fp_density_morgan,
                             global_pool=args.global_pool) 
    model.load_state_dict(state_dict)
    return model, args

def compute_attn(datapath, dataset):
    model, args = load_model(datapath, dataset)
    model.eval() 
    graph_dset = GraphDataset(dataset, degree=True, k_hop=args.k_hop, se=args.se, use_subgraph_edge_attr=args.use_edge_attr)
    graph_loader = DataLoader(graph_dset, batch_size=1, shuffle=False)

    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(graph_dset)  

    attns = [] 

    def get_attns(module, input, output):
        attns.append(output[1])  

    for i in range(args.num_layers):
        model.encoder.layers[i].self_attn.register_forward_hook(get_attns)

    for g in graph_loader:
        with torch.no_grad():  
            y_pred = model(g, return_attn=True)  
            y_pred = y_pred.argmax(dim=-1)  
            y_pred = y_pred.item()
        if y_pred != 0:
            return None  

    attn = attns[-1].mean(dim=-1)[-1]
    return attn

def draw_graph_with_attn(graph, outdir, filename, nodecolor=["tag", "attn"], dpi=1000, edge_vmax=None, args=None, eps=1e-6):
    if len(graph.edges) == 0:
        return  

    plt.switch_backend("agg")
    fig = plt.figure(figsize=(4*len(nodecolor), 4), dpi=dpi)

    node_colors = defaultdict(list)  

    titles = {
        'tag': 'molecule',
        'attn1': 'SATCMF',
    }

    for i in graph.nodes():
        for key in nodecolor:
            node_colors[key].append(graph.nodes[i][key])

    vmax = {}  
    cmap = {}  
    for key in nodecolor:
        vmax[key] = 19  
        cmap[key] = 'tab20'  
        if 'attn' in key:
            vmax[key] = max(node_colors[key]) if node_colors[key] else vmax[key]  
            cmap[key] = 'Reds'  

    pos_layout = nx.kamada_kawai_layout(graph, weight=None)
    
    for i, key in enumerate(nodecolor):
        ax = fig.add_subplot(1, len(nodecolor), i+1)  
        ax.set_title(titles[key], fontweight='bold')  
        nx.draw(
            graph,
            pos=pos_layout,
            with_labels=False,
            font_size=4,
            node_color=node_colors[key],
            vmin=0,
            vmax=vmax[key],
            cmap=cmap[key],
            width=1.3,
            node_size=100,
            alpha=1.0,
        )
        if 'attn' in key:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap[key], norm=plt.Normalize(vmin=0, vmax=vmax[key]))
            sm._A = []
            plt.colorbar(sm, cax=cax)

    fig.axes[0].xaxis.set_visible(False)  
    fig.canvas.draw()  

    save_path = os.path.join(outdir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    plt.savefig(save_path)  

def main():
    all_args = load_args()

    dataset = torch.load(all_args.dataset_path)

    graph_idx = all_args.graph_idx
    graph = dataset[graph_idx]

    attn = compute_attn(all_args.model_path, dataset)
    graph.tag = graph.x.argmax(dim=-1)
    graph.attn1 = attn

    last_node_index = graph.num_nodes - 1
    mask = (graph.edge_index[0] != last_node_index) & (graph.edge_index[1] != last_node_index)
    graph.edge_index = graph.edge_index[:, mask]
    if graph.edge_attr is not None:
        graph.edge_attr = graph.edge_attr[mask]

    graph = utils.to_networkx(graph, node_attrs=['tag', 'attn1'], to_undirected=True)

    input_filename = os.path.basename(all_args.dataset_path)
    output_filename = os.path.splitext(input_filename)[0] + '.png'

    draw_graph_with_attn(
        graph, all_args.outpath,
        output_filename,
        nodecolor=['tag', 'attn1']
        )
    
    print(f"Attention distribution has been computed. The graph has been saved at {output_filename}")

if __name__ == "__main__":
    main()