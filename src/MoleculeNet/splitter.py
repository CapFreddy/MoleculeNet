import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split, KFold

from dgl.data.utils import Subset
from dgllife.data import MoleculeCSVDataset

import os
import os.path as osp
from collections import defaultdict

from typing import Tuple, Iterable, Optional

from .utils import dataset_utils


"""Interface API"""
def train_val_test_split(dataset: str,
                         split_idx: int,
                         n_splits: int,
                         random_state: int, 
                         data_list: Optional[Iterable] = None) -> Tuple[Iterable, Iterable, Iterable]:
    if dataset in dataset_utils.random_split:
        return one_in_k_fold_split(dataset, split_idx, n_splits, random_state, data_list)

    assert dataset in dataset_utils.scaffold_split
    return randomized_scaffold_split(dataset, random_state, data_list)


#################### Random split: K-fold cross validation ####################
"""Obtain indices or dataset slices of one split in randomized K-fold splitting"""
def one_in_k_fold_split(dataset:str,
                        split_idx: int,
                        n_splits: int,
                        random_state: int, 
                        data_list: Optional[Iterable] = None) -> Tuple[Iterable, Iterable, Iterable]:
    assert dataset in dataset_utils.random_split, f'{dataset} is not suitable for random split.'

    df = filtered_dataset(dataset)
    if data_list is not None:
        assert len(data_list) == len(df), 'Length of `data_list` does not match filtered dataset.'

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = list(skf.split(df))[split_idx]
    train_idx, val_idx = train_test_split(train_idx, test_size=1 / (n_splits - 1), random_state=random_state)

    if isinstance(data_list, pd.DataFrame):
        return data_list.iloc[train_idx], data_list.iloc[val_idx], data_list.iloc[test_idx]
    elif isinstance(data_list, MoleculeCSVDataset):
        return Subset(data_list, train_idx), Subset(data_list, val_idx), Subset(data_list, test_idx)
    elif data_list is not None:
        return data_list[train_idx], data_list[val_idx], data_list[test_idx]

    return train_idx, val_idx, test_idx


#################### Scaffold split: Deterministic & randomized ####################
"""Deterministic scaffold split"""
def scaffold_split(dataset: str, data_list: Optional[Iterable] = None) \
                   -> Tuple[Iterable, Iterable, Iterable]:
    assert dataset in dataset_utils.scaffold_split, f'{dataset} is not suitable for scaffold split.'

    df = filtered_dataset(dataset)
    if data_list is not None:
        assert len(data_list) == len(df), 'Length of `data_list` does not match filtered dataset.'

    # Create dict of the form { scaffold_i: [idx_1, idx...] }
    all_scaffold = defaultdict(list)
    smiles_sr = df.loc[:, dataset_utils.smiles_col[dataset]]
    for idx, smiles in enumerate(smiles_sr):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles, includeChirality=True)
        all_scaffold[scaffold].append(idx)

    # Sort from largest to smallest sets
    all_scaffold = { key: sorted(value) for key, value in all_scaffold.items() }
    all_scaffold_set = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffold.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    # Get train, val, test indices
    train_idx, val_idx, test_idx = [], [], []
    train_cutoff = len(df) * 0.8
    val_cutoff = len(df) * (0.8 + 0.1)

    for scaffold_set in all_scaffold_set:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(val_idx) + len(scaffold_set) > val_cutoff:
                test_idx.extend(scaffold_set)
            else:
                val_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    if isinstance(data_list, pd.DataFrame):
        return data_list.iloc[train_idx], data_list.iloc[val_idx], data_list.iloc[test_idx]
    elif isinstance(data_list, MoleculeCSVDataset):
        return Subset(data_list, train_idx), Subset(data_list, val_idx), Subset(data_list, test_idx)
    elif data_list is not None:
        return data_list[train_idx], data_list[val_idx], data_list[test_idx]

    return train_idx, val_idx, test_idx


"""Randomized scaffold split"""
def randomized_scaffold_split(dataset: str, random_state: int, data_list: Optional[Iterable] = None) \
                              -> Tuple[Iterable, Iterable, Iterable]:
    assert dataset in dataset_utils.scaffold_split, f'{dataset} is not suitable for scaffold split.'

    df = filtered_dataset(dataset)
    if data_list is not None:
        assert len(data_list) == len(df), 'Length of `data_list` does not match filtered dataset.'

    # Create dict of the form { scaffold_i: [idx_1, idx...] }
    all_scaffold = defaultdict(list)
    smiles_sr = df.loc[:, dataset_utils.smiles_col[dataset]]
    for idx, smiles in enumerate(smiles_sr):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles, includeChirality=True)
        all_scaffold[scaffold].append(idx)

    # Randomly permutate scaffold sets
    all_scaffold_set = list(all_scaffold.values())
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(all_scaffold_set))
    all_scaffold_set = [all_scaffold_set[perm_idx] for perm_idx in perm]

    # Get train, val, test indices
    train_idx, val_idx, test_idx = [], [], []
    train_cutoff = len(df) * 0.8
    val_cutoff = len(df) * (0.8 + 0.1)

    for scaffold_set in all_scaffold_set:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(val_idx) + len(scaffold_set) > val_cutoff:
                test_idx.extend(scaffold_set)
            else:
                val_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    if isinstance(data_list, pd.DataFrame):
        return data_list.iloc[train_idx], data_list.iloc[val_idx], data_list.iloc[test_idx]
    elif isinstance(data_list, MoleculeCSVDataset):
        return Subset(data_list, train_idx), Subset(data_list, val_idx), Subset(data_list, test_idx)
    elif data_list is not None:
        return data_list[train_idx], data_list[val_idx], data_list[test_idx]

    return train_idx, val_idx, test_idx


#################### Filter unparsable SMILES ####################
ABS_DIR = osp.dirname(osp.abspath(__file__))


"""Obtain filtered dataset and cache"""
def filtered_dataset(dataset: str) -> pd.DataFrame:
    filtered_dir = osp.join(ABS_DIR, 'filtered')
    if not osp.exists(filtered_dir):
        os.mkdir(filtered_dir)

    filtered_path = osp.join(filtered_dir, f'{dataset}.csv')
    if not osp.exists(filtered_path):
        raw_path = osp.join(ABS_DIR, 'raw', dataset_utils.raw_file[dataset])
        smiles_col = dataset_utils.smiles_col[dataset]
        filtered_df = filter_unparsable(pd.read_csv(raw_path), smiles_col)
        filtered_df.to_csv(filtered_path, index=False)

    return pd.read_csv(filtered_path)


"""Filter unparsable SMILES"""
def filter_unparsable(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    smiles_sr = df.loc[:, smiles_col]
    mol_sr = smiles_sr.apply(Chem.MolFromSmiles)
    filtered_df = df.loc[~mol_sr.isna()].reset_index(drop=True)
    return filtered_df
