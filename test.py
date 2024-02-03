import torch
from torch_geometric.data import DataLoader
import sys
sys.path.append("./sat")
from sat.data import GraphTestDataset
from sat.models_fingerprint import GraphTransformer
import argparse
import pandas as pd


def main(args):
    # Load test data
    test_data = torch.load(args.test_data_path)
    # test_dset = GraphTestDataset(test_data, degree=True, k_hop=args.k_hop, se=args.se,
    #                         use_subgraph_edge_attr=args.use_edge_attr)
    # test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model_checkpoint = torch.load(args.model_path)
    model_args = model_checkpoint['args']
    model_state_dict = model_checkpoint['state_dict']

    model = GraphTransformer(
        in_size=model_args.n_tags,
        num_class=1,
        d_model=model_args.dim_hidden,
        dim_feedforward=2 * model_args.dim_hidden,
        dropout=model_args.dropout,
        num_heads=model_args.num_heads,
        num_layers=model_args.num_layers,
        batch_norm=model_args.batch_norm,
        abs_pe=model_args.abs_pe,
        abs_pe_dim=model_args.abs_pe_dim,
        gnn_type=model_args.gnn_type,
        use_edge_attr=model_args.use_edge_attr,
        num_edge_features=model_args.num_edge_features,
        edge_dim=model_args.edge_dim,
        k_hop=model_args.k_hop,
        se=model_args.se,
        deg=None,
        use_fp_density_morgan=model_args.use_fp_density_morgan,
        global_pool=model_args.global_pool
    )
    print(model_args)

    test_dset = GraphTestDataset(test_data, degree=True, k_hop=model_args.k_hop, se=model_args.se,
                            use_subgraph_edge_attr=model_args.use_edge_attr)
    test_loader = DataLoader(test_dset, batch_size=model_args.batch_size, shuffle=False)

    model.load_state_dict(model_state_dict)
    if args.use_cuda:
        model.cuda()

    # Set model to evaluation mode
    model.eval()

    # Predict and save results
    predictions = []
    smiles_list = []
    metal_list = []
    with torch.no_grad():
        for data in test_loader:
            if args.use_cuda:
                data = data.cuda()
            output = model(data)
            predictions.append(output.cpu().numpy())
            smiles_list += data.smiles
            metal_list += data.metal

    # Save predictions, Smiles and metal to CSV file
    output_list = [item for sublist in predictions for item in sublist]
    df = pd.DataFrame({'Smiles': smiles_list, 'metal': metal_list, 'prediction': output_list})
    df.to_csv(args.output_path, index=False)
    print(f"Prediction successful, results saved at: {args.output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, default="./test/data_test.pt", help="Path to the test data file.")
    parser.add_argument("--model_path", type=str, default="./test/SATCMF_gcn_K=3_model.pth", help="Path to the model file.")
    parser.add_argument("--output_path", type=str, default="./test/predictions.csv", help="Path where the prediction output will be saved.")
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    main(args)