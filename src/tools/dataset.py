import os
import torch
import pickle
import yaml
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from data_preprocessing import CustomData


# %%
def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


with open("dic/smiles_dict.yaml", "r") as f:
    smiles_dict = yaml.safe_load(f)


def encode_smiles(smiles, smiles_dict, max_len):
    if not isinstance(smiles, str):
        raise ValueError("smiles must be a string")
    if not isinstance(smiles_dict, dict):
        raise ValueError("smiles_dict must be a dictionary")
    if not isinstance(max_len, int) or max_len <= 0:
        raise ValueError("max_len must be a positive integer")

    indexed_smiles = [smiles_dict.get(char, 0) for char in smiles]
    if len(indexed_smiles) < max_len:
        return indexed_smiles + [0] * (max_len - len(indexed_smiles))
    return indexed_smiles[:max_len]


def get_smiles_transformer_input(smiles, smiles_dict, max_len=100):
    encoded_smiles = encode_smiles(smiles, smiles_dict, max_len)
    input_ids = torch.tensor([encoded_smiles], dtype=torch.long)
    attention_mask = torch.tensor([[1 if x != 0 else 0 for x in encoded_smiles]], dtype=torch.long)
    return input_ids, attention_mask


def get_fingerprint(d1_smiles, radius=2, bits=1024):
    if not isinstance(d1_smiles, str):
        raise ValueError("d1_smiles must be a string")
    if not isinstance(radius, int) or radius <= 0:
        raise ValueError("radius must be a positive integer")
    if not isinstance(bits, int) or bits <= 0:
        raise ValueError("bits must be a positive integer")

    molecule = Chem.MolFromSmiles(d1_smiles)
    if molecule is None:
        return None

    bit_vector = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits)
    return torch.tensor(list(bit_vector), dtype=torch.float32).unsqueeze(0)


class DrugDataset(Dataset):
    def __init__(self, data_df, drug_graph, dataset):
        if not isinstance(data_df, pd.DataFrame):
            raise ValueError("data_df must be a pandas DataFrame")
        if not isinstance(drug_graph, dict):
            raise ValueError("drug_graph must be a dictionary")

        self.data_df = data_df
        self.drug_graph = drug_graph
        self.drug_smile_dict = {}

        base_path = os.getenv('DATA_PATH', 'data')

        dataset_configs = {
            'drugbank': {'file': os.path.join(base_path, 'drugbank.tab'), 'delimiter': '\t',
                         'columns': ['ID1', 'ID2', 'X1', 'X2', 'Y'], 'id_head': 2},
            'twosides': {'file': os.path.join(base_path, 'twosides_shrink.csv'), 'delimiter': ',',
                         'columns': ['Drug1_ID', 'Drug2_ID', 'Drug1', 'Drug2', 'New Y'], 'id_head': 3},
            'dataset1': {'file': os.path.join(base_path, 'dataset1.csv'), 'delimiter': ',',
                         'columns': ['d1', 'd2', 'smiles1', 'smiles2', 'type'], 'id_head': 2},
        }

        config = dataset_configs[dataset]
        data = pd.read_csv(config['file'], delimiter=config['delimiter'])
        c_id1, c_id2, c_s1, c_s2, c_y = config['columns']
        self.id_head = config['id_head']

        for id1, id2, smiles1, smiles2, relation in zip(data[c_id1], data[c_id2], data[c_s1], data[c_s2], data[c_y]):
            self.drug_smile_dict[id1] = smiles1
            self.drug_smile_dict[id2] = smiles2

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        if not isinstance(index, int) or index < 0 or index >= len(self.data_df):
            raise IndexError("Index out of bounds")
        return self.data_df.iloc[index]

    def collate_fn(self, batch):
        if not isinstance(batch, list):
            raise ValueError("batch must be a list of data samples")

        head_list, head_input_ids, head_mask, head_fingerprint_list, head_smiles = [], [], [], [], []
        tail_list, tail_input_ids, tail_mask, tail_fingerprint_list, tail_smiles = [], [], [], [], []
        label_list, rel_list = [], []

        for row in batch:
            Drug1_ID, Drug2_ID, Y, Neg_samples = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']
            Neg_ID, Ntype = Neg_samples.split('$')

            d1_smiles = self.drug_smile_dict.get(Drug1_ID)
            d2_smiles = self.drug_smile_dict.get(Drug2_ID)
            Neg_smiles = self.drug_smile_dict.get(Neg_ID)

            if None in [d1_smiles, d2_smiles, Neg_smiles]:
                raise ValueError("SMILES not found for one or more drugs")

            h_graph = self.drug_graph.get(Drug1_ID)
            t_graph = self.drug_graph.get(Drug2_ID)
            n_graph = self.drug_graph.get(Neg_ID)

            if None in [h_graph, t_graph, n_graph]:
                raise ValueError("Graph data not found for one or more drugs")

            pos_pair_h, pos_pair_t = h_graph, t_graph

            head_list.append(pos_pair_h)
            head_fingerprint_list.append(get_fingerprint(d1_smiles))
            head_input_ids.append(get_smiles_transformer_input(d1_smiles)[0])
            head_mask.append(get_smiles_transformer_input(d1_smiles)[1])
            head_smiles.append(torch.tensor([int(Drug1_ID[self.id_head])]).unsqueeze(0))

            tail_list.append(pos_pair_t)
            tail_fingerprint_list.append(get_fingerprint(d2_smiles))
            tail_input_ids.append(get_smiles_transformer_input(d2_smiles)[0])
            tail_mask.append(get_smiles_transformer_input(d2_smiles)[1])
            tail_smiles.append(torch.tensor([int(Drug2_ID[self.id_head])]).unsqueeze(0))

            if Ntype == 'h':
                neg_pair_h, neg_pair_t = n_graph, t_graph
            else:
                neg_pair_h, neg_pair_t = h_graph, n_graph

            head_list.append(neg_pair_h)
            head_fingerprint_list.append(get_fingerprint(Neg_smiles))
            head_input_ids.append(get_smiles_transformer_input(Neg_smiles)[0])
            head_mask.append(get_smiles_transformer_input(Neg_smiles)[1])
            head_smiles.append(torch.tensor([int(Neg_ID[self.id_head])]).unsqueeze(0))

            tail_list.append(neg_pair_t)
            tail_fingerprint_list.append(get_fingerprint(d2_smiles if Ntype == 'h' else Neg_smiles))
            tail_input_ids.append(get_smiles_transformer_input(d2_smiles if Ntype == 'h' else Neg_smiles)[0])
            tail_mask.append(get_smiles_transformer_input(d2_smiles if Ntype == 'h' else Neg_smiles)[1])
            tail_smiles.append(
                torch.tensor([int(Drug2_ID[self.id_head]) if Ntype == 'h' else int(Neg_ID[self.id_head])]).unsqueeze(0))

            rel_list.extend([torch.LongTensor([Y]), torch.LongTensor([Y])])
            label_list.extend([torch.FloatTensor([1]), torch.FloatTensor([0])])

        head_pairs = Batch.from_data_list(head_list, follow_batch=['x'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['x'])

        return {
            'head_pairs': head_pairs,
            'tail_pairs': tail_pairs,
            'head_fins': torch.cat(head_fingerprint_list, dim=0).long(),
            'head_input_ids': torch.cat(head_input_ids, dim=0).long(),
            'head_mask': torch.cat(head_mask, dim=0),
            'head_smiles': torch.cat(head_smiles, dim=0),
            'tail_fins': torch.cat(tail_fingerprint_list, dim=0).long(),
            'tail_input_ids': torch.cat(tail_input_ids, dim=0).long(),
            'tail_mask': torch.cat(tail_mask, dim=0),
            'tail_smiles': torch.cat(tail_smiles, dim=0),
            'rel': torch.cat(rel_list, dim=0),
            'label': torch.cat(label_list, dim=0)
        }


class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        if not isinstance(data, DrugDataset):
            raise ValueError("data must be an instance of DrugDataset")
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def split_train_valid(data_df, fold, val_ratio=0.2):
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("data_df must be a pandas DataFrame")
    if 'Y' not in data_df.columns:
        raise KeyError("data_df must contain a 'Y' column")

    cv_split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=data_df, y=data_df['Y'])))

    train_df = data_df.iloc[train_index].reset_index(drop=True)
    val_df = data_df.iloc[val_index].reset_index(drop=True)

    return train_df, val_df


def load_ddi_dataset(root, batch_size, dataset='drugbank'):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"The specified root directory does not exist: {root}")

    drug_data_file = os.path.join(root, 'drug_data.pkl')
    train_file = os.path.join(root, 'pair_pos_neg_triplets_train.csv')
    val_file = os.path.join(root, 'pair_pos_neg_triplets_validation.csv')
    test_file = os.path.join(root, 'pair_pos_neg_triplets_test.csv')

    for file_path in [drug_data_file, train_file, val_file, test_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required file: {file_path}")

    drug_graph = read_pickle(drug_data_file)
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    if train_df.empty:
        raise ValueError("The training dataset is empty")
    if val_df.empty:
        raise ValueError("The validation dataset is empty")
    if test_df.empty:
        raise ValueError("The test dataset is empty")

    train_set = DrugDataset(train_df, drug_graph, dataset)
    val_set = DrugDataset(val_df, drug_graph, dataset)
    test_set = DrugDataset(test_df, drug_graph, dataset)

    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=10)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=10)

    return train_loader, val_loader, test_loader
