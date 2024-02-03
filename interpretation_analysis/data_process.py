import re
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from rdkit import RDLogger, Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from mendeleev import element
import os

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import argparse

warnings.filterwarnings("ignore", category=UserWarning)


def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    if index != -1:
        encoding[index] = 1
    return encoding


def GetNum(x, allowed_set):
    one_hot_encoded = onek_encoding_unk(x, allowed_set)
    return one_hot_encoded


def get_atom_graphformer_feature(m):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        feat = atom_feature_attentive_FP(m.GetAtomWithIdx(i), explicit_H=False, use_chirality=True)
        H.append(feat)
    try:
        H = np.stack(H, axis=0)
    except ValueError as e:
        print("Error stacking the atom features:", e)
        print("Atom features:")
        for i, feat in enumerate(H):
            print(f"Atom {i} feature:", feat)
    return H


def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        fbond = [1] + [0] * (15 - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        fbond.append(0)
    return fbond


def metal_bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        fbond = [1] + [0] * (15 - 1)
    else:
        fbond = [0] * (15 - 1) + [1]
    return fbond


def atom_feature_attentive_FP(atom, explicit_H=False, use_chirality=True):
    features = GetNum(atom.GetAtomicNum(), list(range(1, 119))) + GetNum(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    features += [abs(int(atom.GetFormalCharge())), abs(int(atom.GetNumRadicalElectrons()))]
    features += GetNum(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])
    features.append(atom.GetIsAromatic())
    if not explicit_H:
        features += GetNum(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            features += GetNum(atom.GetProp('_CIPCode'), ['R', 'S'])
            features.append(atom.HasProp('_ChiralityPossible'))
        except:
            features += [False, False]
            features.append(atom.HasProp('_ChiralityPossible'))
    expected_length = 145
    if len(features) < expected_length:
        features += [0] * (expected_length - len(features))
    return np.array(features)


def create_graph_data(smiles: str, metal: str, metal_valence: int, logP_SA_cycle_normalized: float,
                      explicit_H=False) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if not explicit_H:
        mol = Chem.RemoveHs(mol)
    metal_edge = []
    atom_feature = get_atom_graphformer_feature(mol)
    metal_mol = Chem.MolFromSmiles(metal)
    metal_feature = assign_metal_node_features(metal, int(metal_valence))
    atom_feature = np.append(atom_feature, metal_feature, axis=0)
    atom_feature = torch.tensor(atom_feature, dtype=torch.float32)
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    num_atoms = mol.GetNumAtoms()
    num_features = 15
    adj_matrix = np.zeros((num_atoms, num_atoms))
    bond_feature_matrix = np.zeros((num_atoms, num_atoms, num_features))
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        neighbors = [a.GetSymbol() for a in atom.GetNeighbors()]
        bond_order = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
        if symbol == 'O' and 1.0 in bond_order and 'C' in neighbors:
            metal_edge.append(atom.GetIdx())
        if symbol == 'O' and 1.0 in bond_order and 'P' in neighbors:
            metal_edge.append(atom.GetIdx())
        if symbol == 'O' and atom.GetTotalNumHs() == 0:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'N' and neighbor.IsInRing():
                    ring_info = mol.GetRingInfo()
                    if ring_info.IsAtomInRingOfSize(neighbor.GetIdx(), 6):
                        metal_edge.append(atom.GetIdx())
                        break
                elif neighbor.GetSymbol() == 'C' and neighbor.IsInRing():
                    for n_neighbor in neighbor.GetNeighbors():
                        if n_neighbor.GetSymbol() == 'N' and n_neighbor.IsInRing():
                            ring_info = mol.GetRingInfo()
                            if ring_info.IsAtomInRingOfSize(n_neighbor.GetIdx(), 5):
                                metal_edge.append(atom.GetIdx())
                                break
                elif neighbor.GetSymbol() == 'C' and neighbor.IsInRing():
                    ring_info = mol.GetRingInfo()
                    if ring_info.IsAtomInRingOfSize(neighbor.GetIdx(), 5):
                        for n_neighbor in neighbor.GetNeighbors():
                            if n_neighbor.GetSymbol() == 'O' and n_neighbor.GetTotalNumHs() == 0:
                                bond = mol.GetBondBetweenAtoms(neighbor.GetIdx(), n_neighbor.GetIdx())
                                if bond.GetBondType() == BondType.DOUBLE:
                                    metal_edge.append(atom.GetIdx())
                                    break
        if symbol == 'N' and 'C' in neighbors:
            metal_edge.append(atom.GetIdx())
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_matrix[i, j] = adj_matrix[j, i] = 1
        bond_feature_matrix[i, j] = bond_feature_matrix[j, i] = bond_features(bond)
    adj_matrix = np.pad(adj_matrix, ((0, 1), (0, 1)), 'constant')
    bond_feature_matrix = np.pad(bond_feature_matrix, ((0, 1), (0, 1), (0, 0)), 'constant')
    for idx in metal_edge:
        adj_matrix[idx, -1] = adj_matrix[-1, idx] = 1
        bond_feature_matrix[idx, -1] = bond_feature_matrix[-1, idx] = metal_bond_features(bond)
    adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    edge_attr = torch.tensor(bond_feature_matrix, dtype=torch.float)
    target = torch.tensor([logP_SA_cycle_normalized])
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    fp_tensor = torch.tensor(fp_array, dtype=torch.float32).view(1, -1)
    graph_data = {
        'num_atom': len(atom_feature),
        'atom_feature': atom_feature,
        'bond_type': adj_matrix_tensor,
        'edge_attr': edge_attr,
        'logP_SA_cycle_normalized': target,
        'fp_density_morgan': fp_tensor,
        'smiles': smiles,
        'metal': metal
    }
    return graph_data


def assign_metal_node_features(metal, valence):
    atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(metal)
    degree = valence
    features = GetNum(atomic_number, list(range(1, 119))) + GetNum(degree, [0, 1, 2, 3, 4, 5])
    expected_length = 145
    if len(features) < expected_length:
        features += [0] * (expected_length - len(features))
    return np.array(features)[np.newaxis, :]


def extract_metal_symbol_and_valence(metal_string):
    match = re.match(r"([A-Za-z]+)(\+*)", metal_string)
    if match:
        symbol, valence_str = match.groups()
        valence = len(valence_str)
        return symbol, valence
    else:
        raise ValueError(f"Invalid metal string: {metal_string}")


def scaffold_split(smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    return train_idx, valid_idx, test_idx


def split_data(smiles_list, use_scaffold=False, test_size=0.2, random_state=42):
    if use_scaffold:
        train_idx, valid_idx, test_idx = scaffold_split(smiles_list)
    else:
        train_idx, remaining_idx = train_test_split(range(len(smiles_list)), test_size=test_size, random_state=random_state)
        valid_idx, test_idx = train_test_split(remaining_idx, test_size=0.5, random_state=random_state)
    return train_idx, valid_idx, test_idx


def get_graph_objects(smiles_list, metal_list, labels_list, split_idx=None):
    split_graph_objects = []
    if split_idx is not None:
        idx_list = split_idx
    else:
        idx_list = range(len(smiles_list))
    for idx in tqdm(idx_list, desc="Processing graphs"):
        metal_symbol, metal_valence = metal_list[idx]
        try:
            graph_object = create_graph_data(smiles_list[idx], metal_symbol, metal_valence, labels_list[idx])
        except:
            print(smiles_list[idx])
        split_graph_objects.append(graph_object)
    return split_graph_objects


def process_dataset(indices, mols, savedir):
    pbar = tqdm(total=len(indices))
    pbar.set_description(f'Processing dataset')
    data_list = []
    max_feature_value = float('-inf')
    for idx in indices:
        mol = mols[idx]
        x = mol['atom_feature'].to(torch.long)
        if (x < 0).any():
            print(f"Negative values found in x for molecule {idx}")
        current_max_value = x.max().item()
        max_feature_value = max(max_feature_value, current_max_value)
        y = mol['logP_SA_cycle_normalized'].to(torch.float)
        adj = mol['bond_type']
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        row_indices, col_indices = edge_index[0], edge_index[1]
        edge_attr = mol['edge_attr'][row_indices, col_indices]
        fp_density_morgan = mol['fp_density_morgan'].to(torch.float)
        smiles = mol['smiles']
        metal = mol['metal']
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, fp_density_morgan=fp_density_morgan,
                    smiles=smiles, metal=metal)
        if edge_index.size(1) != edge_attr.size(0):
            print(f"edge_index and edge_attr dimensions do not match for molecule {idx}")
            print(data)
        data_list.append(data)
        pbar.update(1)
    pbar.close()
    return data_list, max_feature_value


def main(args):
    filtered_df = pd.read_csv(args.data_path)
    smiles_list = list(filtered_df['SMILES'])
    metal_list = [extract_metal_symbol_and_valence(Metal) for Metal in filtered_df["Metal"]]
    labels_list = np.zeros(len(smiles_list))
    savedir = args.save_dir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    graph_objects = get_graph_objects(smiles_list, metal_list, labels_list)
    max_values = []
    indices = range(len(graph_objects))
    data_list, current_max_value = process_dataset(indices, graph_objects, savedir)
    max_values.append(current_max_value)
    global_max_value = max(max_values)
    embedding_input_dim = global_max_value + 1
    print('Embedding input dimension:', embedding_input_dim)
    torch.save(data_list, os.path.join(savedir, "data_interpretation_DOTA_based_chemical_edge.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./DOTA_smiles.csv", help="Path to the input data file.")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory where the processed data will be saved.")
    args = parser.parse_args()
    main(args)