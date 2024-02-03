import os
import pickle
import random
import re
import warnings
import numpy as np
import pandas as pd
import torch
from mendeleev import element
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data
from tqdm import tqdm
import argparse
import random

# Suppress specific RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Ignore UserWarnings
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

    
def get_atom_feature(m):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        feat = atom_feature_SATCMF(m.GetAtomWithIdx(i),
                                             explicit_H=False,
                                             use_chirality=True)
        # print(feat.shape)
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
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
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
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (15 - 1)
    else:
        fbond = [0] * (15 - 1) + [1]

    return fbond


def atom_feature_SATCMF(atom, explicit_H=False, use_chirality=True):

    # Initial feature vector
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

        # Ensure the feature vector has the correct length
    expected_length = 145
    if len(features) < expected_length:
        features += [0] * (expected_length - len(features))

    return np.array(features)



def create_graph_data(smiles: str, metal: str, metal_valence: int, logP_SA_cycle_normalized: float, explicit_H=False) -> dict:
    """
    Create graph data from a SMILES string representing a molecule and metal information.

    Parameters:
    - smiles (str): SMILES string of the molecule
    - metal (str): SMILES string of the metal atom
    - metal_valence (int): Valence of the metal atom
    - logP_SA_cycle_normalized (float): Normalized logP SA cycle value
    - explicit_H (bool): Flag to indicate whether to include explicit hydrogen atoms in the molecule

    Returns:
    - dict: A dictionary containing graph data for the molecule
    """

    mol = Chem.MolFromSmiles(smiles)
    if not explicit_H:
        mol = Chem.RemoveHs(mol)
    atom_feature = get_atom_feature(mol)
    metal_mol = Chem.MolFromSmiles(metal)
    metal_feature = assign_metal_node_features(metal, int(metal_valence))
    atom_feature = np.append(atom_feature, metal_feature, axis=0)
    atom_feature = torch.tensor(atom_feature, dtype=torch.float32)
    num_atoms = mol.GetNumAtoms()
    num_features = 15
    adj_matrix = np.zeros((num_atoms, num_atoms))
    bond_feature_matrix = np.zeros((num_atoms, num_atoms, num_features))

    all_atoms = list(range(num_atoms))
    num_edges = random.randint(1, num_atoms)
    metal_edge = random.sample(all_atoms, num_edges)

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
        'fp_density_morgan': fp_tensor
    }
    
    return graph_data


def assign_metal_node_features(metal, valence):
    atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(metal)
    degree = valence 
    features = GetNum(atomic_number, list(range(1, 119))) + GetNum(degree, [0, 1, 2, 3, 4, 5])
    # Set all other features to 0
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
    

def generate_scaffold(smiles, include_chirality=False):
    
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def scaffold_split(smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1):

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
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

def get_graph_objects(split_idx, smiles_list, metal_list, labels_list):
    split_graph_objects = []

    for idx in tqdm(split_idx, desc="Processing graphs"):
        metal_symbol, metal_valence = metal_list[idx]
        try:
            graph_object = create_graph_data(smiles_list[idx], metal_symbol, metal_valence, labels_list[idx])
        except:
            print(smiles_list[idx])
            
        split_graph_objects.append(graph_object)
    
    return split_graph_objects

def process_dataset(indices, mols, savedir, dataset_name):
    """
    Process a dataset of molecules into a format suitable for graph-based machine learning models.

    Parameters:
    - indices (list): List of indices to process from the mols dataset.
    - mols (list): List of dictionaries containing molecular information.
    - savedir (str): Directory path where the processed dataset will be saved.
    - dataset_name (str): Name of the dataset being processed.
    """
    pbar = tqdm(total=len(indices))
    pbar.set_description(f'Processing {dataset_name} dataset')

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
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, fp_density_morgan=fp_density_morgan)

        if edge_index.size(1) != edge_attr.size(0):
            print(f"edge_index and edge_attr dimensions do not match for molecule {idx}")
            print(data)
        data_list.append(data)
        pbar.update(1)

    pbar.close()
    torch.save(data_list, savedir + f"/{dataset_name}.pt")
    return data_list, max_feature_value


def main():
    # Load the dataset
    filtered_df = pd.read_csv(args.data_path)
    smiles_list = list(filtered_df['SMILES'])
    metal_list = [extract_metal_symbol_and_valence(Metal) for Metal in filtered_df["Metal"]]
    filtered_df['LogK1'] = filtered_df['LogK1'].round(2)
    labels_list = np.asarray(filtered_df['LogK1'])

    # Create graph objects for all data
    all_graph_objects = get_graph_objects(range(len(smiles_list)), smiles_list, metal_list, labels_list)

    # 10-fold split for the entire dataset
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    args.save_fold_dir
    fold_dir = args.save_fold_dir
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    for i, (train_valid_idx, test_idx) in enumerate(kf.split(all_graph_objects)):
        # Split the train_valid_idx further into train and valid sets (90% train, 10% valid)
        train_idx, valid_idx = train_test_split(train_valid_idx, test_size=1/9, random_state=42) # 1/9 since one fold out of the remaining nine should be for validation

        # Get graph objects for train, valid, and test
        train_graph_objects = [all_graph_objects[idx] for idx in train_idx]
        valid_graph_objects = [all_graph_objects[idx] for idx in valid_idx]
        test_graph_objects = [all_graph_objects[idx] for idx in test_idx]

        fold_dir_name = f'fold{i+1}'
        fold_savedir = os.path.join(fold_dir, fold_dir_name)
        if not os.path.exists(fold_savedir):
            os.makedirs(fold_savedir)

        # Save as pickle
        with open(os.path.join(fold_savedir, 'train.pickle'), 'wb') as f:
            pickle.dump(train_graph_objects, f)
        with open(os.path.join(fold_savedir, 'valid.pickle'), 'wb') as f:
            pickle.dump(valid_graph_objects, f)
        with open(os.path.join(fold_savedir, 'test.pickle'), 'wb') as f:
            pickle.dump(test_graph_objects, f)

        # Save as pt
        torch.save(train_graph_objects, os.path.join(fold_savedir, 'train.pt'))
        torch.save(valid_graph_objects, os.path.join(fold_savedir, 'valid.pt'))
        torch.save(test_graph_objects, os.path.join(fold_savedir, 'test.pt'))

        # Process and save the fold train, valid and test data
        for dataset_name, dataset_objects in zip(['train', 'valid', 'test'], [train_graph_objects, valid_graph_objects, test_graph_objects]):
            indices = range(len(dataset_objects))
            data_list, _ = process_dataset(indices, dataset_objects, fold_savedir, dataset_name)

        print(f"Fold {i+1} saved and processed at {fold_savedir}")

    print("All folds have been saved and processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default = "./dataset.csv", help="Path to the input data file.")
    parser.add_argument("--save_fold_dir", type=str, default = "10_fold_cv_random_edge", help="Directory where the processed data will be saved.")
    args = parser.parse_args()
    main()





