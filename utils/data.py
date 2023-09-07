import pickle
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def load_dataset(mode="DAVIS"):
    if mode in ["DAVIS", "BindingDB", "BIOSNAP"]:
        train_df = pd.read_csv(f"./data/{mode}_train.csv")
        valid_df = pd.read_csv(f"./data/{mode}_valid.csv")
        test_df = pd.read_csv(f"./data/{mode}_test.csv")
    elif mode == "merged":
        print("Load merged datasets")
        train_df = pd.read_csv("./data/train_dataset.csv")
        valid_df = pd.read_csv("./data/valid_dataset.csv")
        test_df = pd.read_csv("./data/test_dataset.csv")

    return train_df, valid_df, test_df


def load_cached_prot_features(max_length=1024):
    with open(f"prot_feat/{max_length}_cls.pkl", "rb") as f:
        prot_feat_teacher = pickle.load(f)

    return prot_feat_teacher


class DTIDataset(Dataset):
    def __init__(
        self,
        data,
        prot_feat_teacher,
        mol_tokenizer,
        prot_tokenizer,
        max_length,
        d_mode="merged",
    ):
        self.data = data
        self.prot_feat_teacher = prot_feat_teacher
        self.max_length = max_length
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.d_mode = d_mode

    def get_mol_feat(self, smiles):
        return self.mol_tokenizer(smiles, max_length=512, truncation=True)

    def get_prot_feat_student(self, fasta):
        return self.prot_tokenizer(
            " ".join(fasta), max_length=self.max_length + 2, truncation=True
        )

    def get_prot_feat_teacher(self, fasta):
        return self.prot_feat_teacher[fasta[:20]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = self.data.loc[index, "SMILES"]
        mol_feat = self.get_mol_feat(smiles)

        fasta = self.data.loc[index, "Target Sequence"]
        prot_feat_student = self.get_prot_feat_student(fasta)
        prot_feat_teacher = self.get_prot_feat_teacher(fasta)

        y = self.data.loc[index, "Label"]

        if self.d_mode == "merged":
            source = self.data.loc[index, "Source"]
            if source == "DAVIS":
                source = 1
            elif source == "BindingDB":
                source = 2
            elif source == "BIOSNAP":
                source = 3
        elif self.d_mode == "DAVIS":
            source = 1
        elif self.d_mode == "BindingDB":
            source = 2
        elif self.d_mode == "BIOSNAP":
            source = 3

        return mol_feat, prot_feat_student, prot_feat_teacher, y, source


class CollateBatch(object):
    def __init__(self, mol_tokenizer, prot_tokenizer):
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer

    def __call__(self, batch):
        mol_features, prot_feat_student, prot_feat_teacher, y, source = (
            [],
            [],
            [],
            [],
            [],
        )

        for mol_seq, prot_seq_student, prot_seq_teacher, y_, source_ in batch:
            mol_features.append(mol_seq)
            prot_feat_student.append(prot_seq_student)
            prot_feat_teacher.append(prot_seq_teacher.detach().cpu().numpy().tolist())
            y.append(y_)
            source.append(source_)

        mol_features = self.mol_tokenizer.pad(mol_features, return_tensors="pt")
        prot_feat_student = self.prot_tokenizer.pad(
            prot_feat_student, return_tensors="pt"
        )
        prot_feat_teacher = torch.tensor(prot_feat_teacher).float()
        y = torch.tensor(y).float()
        source = torch.tensor(source)

        return mol_features, prot_feat_student, prot_feat_teacher, y, source


def define_balanced_sampler(train_df, target_col_name="Label"):
    counts = np.bincount(train_df[target_col_name])
    labels_weights = 1.0 / counts
    weights = labels_weights[train_df[target_col_name]]
    sampler = WeightedRandomSampler(weights, len(weights))

    return sampler


def get_dataloaders(
    train_df,
    valid_df,
    test_df,
    prot_feat_teacher,
    mol_tokenizer,
    prot_tokenizer,
    max_lenght,
    d_mode="merged",
    target_col_name="Label",
    batch_size=128,
    num_workers=-1,
):
    train_dataset = DTIDataset(
        train_df,
        prot_feat_teacher,
        mol_tokenizer,
        prot_tokenizer,
        max_lenght,
        d_mode=d_mode,
    )
    valid_dataset = DTIDataset(
        valid_df,
        prot_feat_teacher,
        mol_tokenizer,
        prot_tokenizer,
        max_lenght,
        d_mode=d_mode,
    )
    test_dataset = DTIDataset(
        test_df,
        prot_feat_teacher,
        mol_tokenizer,
        prot_tokenizer,
        max_lenght,
        d_mode=d_mode,
    )

    # sampler = define_balanced_sampler(train_df, target_col_name)
    collator = CollateBatch(mol_tokenizer, prot_tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=collator,
    )
    #   sampler=sampler, collate_fn=collator)

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=collator,
    )

    return train_dataloader, valid_dataloader, test_dataloader
