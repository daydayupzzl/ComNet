import os
import torch
import pickle
import argparse
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import *
import networkx as nx
from rdkit import Chem
import concurrent.futures
from operator import index
from rdkit.Chem import AllChem
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


def atom_features(atom, atom_rxyz):
    atom_idx = atom.GetIdx()
    xyz = atom_rxyz[atom_idx, :] if atom_rxyz is not None else np.zeros(3, dtype=np.float32)

    one_hot_symbol = one_hot_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                           'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
                                           'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                                           'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'X'])
    one_hot_props = [
        one_hot_encoding(atom.GetDegree(), range(11)),
        one_hot_encoding(atom.GetTotalNumHs(), range(11)),
        one_hot_encoding(atom.GetImplicitValence(), range(11)),
        [atom.GetIsAromatic()]
    ]
    features = np.concatenate([np.array(one_hot_symbol, dtype=float)] +
                              [np.array(prop, dtype=float) for prop in one_hot_props] +
                              [xyz])
    return features


class GraphConvConstants:
    @classmethod
    def load_constants(cls, config_path='constants.json'):
        with open(config_path, 'r') as file:
            constants = json.load(file)
            for key, value in constants.items():
                setattr(cls, key, value)


GraphConvConstants.load_constants()


def bond_features_deep_chemm(bond, use_chirality=False, use_extended_chirality=False):
    bond_type = bond.GetBondType()
    bond_stereo = bond.GetStereo()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats += one_hot_encoding_unk(str(bond_stereo), GraphConvConstants.possible_bond_stereo)
    if use_extended_chirality:
        bond_feats += one_hot_encoding_unk(int(bond_stereo), list(range(6)))
    return np.array(bond_feats, dtype=float)


def euclidean_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)


def get_spatial_matrix(coords, adjacency_matrix):
    distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=-1)
    spatial_mat = np.where(adjacency_matrix == 1, distances, 0)
    return spatial_mat


def get_adjacency_matrix(mol):
    num_atoms = mol.GetNumAtoms()
    adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    return adjacency_matrix


def normalize_edge_weights(matrix):
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    normalized_matrix = matrix / row_sums[:, np.newaxis]
    return normalized_matrix


def embed_molecule(mol):
    if AllChem.EmbedMolecule(mol) == -1:
        raise ValueError("Embedding failed")
    return mol


def get_mol_3D_matrix(mol):
    mol = Chem.AddHs(mol)
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(embed_molecule, mol)
            mol = future.result(timeout=10)
    except (concurrent.futures.TimeoutError, ValueError) as e:
        print(f"Embedding failed: {e}")
        return None, None
    mol = Chem.RemoveHs(mol)
    atom_num = mol.GetNumAtoms()
    atom_rxyz = np.zeros((atom_num, 3), dtype=np.float32)
    try:
        conformer = mol.GetConformer()
    except ValueError:
        print('No 3D conformation available')
        return None, None
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_rxyz[atom_idx, :] = conformer.GetAtomPosition(atom_idx)
    adjacency_matrix = get_adjacency_matrix(mol)
    spatial_matrix = get_spatial_matrix(atom_rxyz, adjacency_matrix).astype(np.float32)
    return normalize_edge_weights(spatial_matrix), atom_rxyz


def generate_drug_data(id, mol, smiles):
    num_atoms = mol.GetNumAtoms()
    edge_weight_matrix, atom_rxyz = get_mol_3D_matrix(mol)
    features = [atom_features(atom, atom_rxyz) / sum(atom_features(atom, atom_rxyz)) for atom in mol.GetAtoms()]
    edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    g = nx.Graph(edges).to_directed()
    mol_adj = np.zeros((num_atoms, num_atoms), dtype=int)
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = mol_adj[e2, e1] = 1
    np.fill_diagonal(mol_adj, 1)
    index_row, index_col = np.where(mol_adj >= 0.5)
    edge_index = np.array(list(zip(index_row, index_col)), dtype=np.int64).T
    edge_weight = []
    edges_attr = []
    for i, j in zip(index_row, index_col):
        bond = mol.GetBondBetweenAtoms(i, j)
        if i == j:
            edge_weight.append(1)
            edges_attr.append([0] * 6)
        else:
            edge_weight.append(edge_weight_matrix[i, j] if edge_weight_matrix is not None else 1)
            edges_attr.append(bond_features_deep_chemm(bond, use_chirality=False, use_extended_chirality=False))
    data = CustomData(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edges_attr, dtype=torch.float32),
        edge_weight=torch.tensor(edge_weight, dtype=torch.float32)
    )
    return data


def load_drug_mol_data(args):
    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    drug_smile_dict = {
        id1: smiles1
        for id1, id2, smiles1, smiles2, relation in
        zip(data[args.c_id1], data[args.c_id2], data[args.c_s1], data[args.c_s2], data[args.c_y])
    }
    drug_id_mol_tup = [
        (id, Chem.MolFromSmiles(smiles.strip()), smiles)
        for id, smiles in drug_smile_dict.items()
        if Chem.MolFromSmiles(smiles.strip()) is not None
    ]
    for id, smiles in drug_smile_dict.items():
        if Chem.MolFromSmiles(smiles.strip()) is None:
            print(f"Warning: Unable to parse molecule for SMILES '{smiles}'.")
    drug_data = {id: generate_drug_data(id, mol, smiles) for id, mol, smiles in
                 tqdm(drug_id_mol_tup, desc='Processing drugs')}
    save_data(drug_data, 'drug_data.pkl', args)
    return drug_data


def generate_pair_triplets(args):
    with open(f'{args.dirname}/{args.dataset.lower()}/drug_data.pkl', 'rb') as f:
        drug_ids = list(pickle.load(f).keys())
    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    pos_triplets = [
        [id1, id2, relation - 1 if args.adjust_relation else relation]
        for id1, id2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_y])
        if id1 in drug_ids and id2 in drug_ids
    ]
    if not pos_triplets:
        raise ValueError('All tuples are invalid.')
    pos_triplets = np.array(pos_triplets)
    data_statistics = load_data_statistics(pos_triplets)
    drug_ids = np.array(drug_ids)
    neg_samples = []
    for pos_item in tqdm(pos_triplets, desc='Generating Negative sample'):
        h, t, r = pos_item[:3]

        neg_heads, neg_tails = _generate_negative_samples(h, t, r, data_statistics, drug_ids, args)

        temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + [str(neg_t) + '$t' for neg_t in neg_tails]
        neg_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))
    df = pd.DataFrame({'Drug1_ID': pos_triplets[:, 0],
                       'Drug2_ID': pos_triplets[:, 1],
                       'Y': pos_triplets[:, 2],
                       'Neg samples': neg_samples})
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df.to_csv(filename, index=False)
    save_data(data_statistics, 'data_statistics.pkl', args)


def _generate_negative_samples(h, t, r, data_statistics, drug_ids, args):
    if hasattr(args, 'negative_sampling_strategy') and args.negative_sampling_strategy == 'normal':
        return _normal_batch(h, t, r, args.neg_ent, data_statistics, drug_ids, args)
    else:
        existing_drug_ids = np.asarray(list(set(
            np.concatenate(
                [data_statistics["ALL_TRUE_T_WITH_HR"][(h, r)], data_statistics["ALL_TRUE_H_WITH_TR"][(h, r)]], axis=0)
        )))
        return _corrupt_ent(existing_drug_ids, args.neg_ent, drug_ids, args)


def load_data_statistics(all_tuples):
    statistics = {
        "ALL_TRUE_H_WITH_TR": defaultdict(list),
        "ALL_TRUE_T_WITH_HR": defaultdict(list),
        "FREQ_REL": defaultdict(int),
        "ALL_H_WITH_R": defaultdict(dict),
        "ALL_T_WITH_R": defaultdict(dict),
        "ALL_TAIL_PER_HEAD": {},
        "ALL_HEAD_PER_TAIL": {}
    }
    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        statistics["FREQ_REL"][r] += 1
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1
    for key in ["ALL_TRUE_H_WITH_TR", "ALL_TRUE_T_WITH_HR"]:
        for pair in statistics[key]:
            statistics[key][pair] = np.unique(statistics[key][pair])
    for r in statistics["FREQ_REL"]:
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])
    return statistics


def _corrupt_ent(existing_ents, max_num, drug_ids, args):
    corrupted_ents = []
    while len(corrupted_ents) < max_num:
        candidates = args.random_num_gen.choice(drug_ids, (max_num - len(corrupted_ents)) * 2, replace=False)
        invalid_drug_ids = np.concatenate([existing_ents, corrupted_ents])
        mask = ~np.isin(candidates, invalid_drug_ids)
        corrupted_ents.extend(candidates[mask])
    return np.array(corrupted_ents)[:max_num]


def _normal_batch(h, t, r, neg_size, data_statistics, drug_ids, args):
    prob = data_statistics["ALL_TAIL_PER_HEAD"][r] / (
            data_statistics["ALL_TAIL_PER_HEAD"][r] + data_statistics["ALL_HEAD_PER_TAIL"][r]
    )
    neg_size_h, neg_size_t = 0, 0
    for _ in range(neg_size):
        if args.random_num_gen.random() < prob:
            neg_size_h += 1
        else:
            neg_size_t += 1
    neg_heads = _corrupt_ent(data_statistics["ALL_TRUE_H_WITH_TR"][t, r], neg_size_h, drug_ids, args)
    neg_tails = _corrupt_ent(data_statistics["ALL_TRUE_T_WITH_HR"][h, r], neg_size_t, drug_ids, args)
    return neg_heads, neg_tails


def save_data(data, filename, args):
    directory = os.path.join(args.dirname, args.dataset)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {file_path}!')
