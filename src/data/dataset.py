import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MMKGDataset:
    """
    Loads a multimodal KG alignment dataset (FB15K-DB15K / FB15K-YG15K format).
    Expected directory layout:
        data/<dataset>/
            ent_ids_1, ent_ids_2       -- entity id maps
            rel_ids_1, rel_ids_2       -- relation id maps
            triples_1, triples_2       -- (h, r, t) triples
            train_links                -- seed alignment pairs
            test_links                 -- test alignment pairs
            images/                    -- entity images (ent_id.jpg)
            entity_descriptions.txt   -- entity text descriptions
    """

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = data_dir
        self.split = split
        self._load()

    def _load(self):
        self.ent2id_1 = self._load_id_map("ent_ids_1")
        self.ent2id_2 = self._load_id_map("ent_ids_2")
        self.triples_1 = self._load_triples("triples_1")
        self.triples_2 = self._load_triples("triples_2")
        self.train_links = self._load_links("train_links")
        self.test_links = self._load_links("test_links")

    def _load_id_map(self, fname):
        path = os.path.join(self.data_dir, fname)
        id_map = {}
        with open(path) as f:
            for line in f:
                eid, name = line.strip().split("\t")
                id_map[int(eid)] = name
        return id_map

    def _load_triples(self, fname):
        path = os.path.join(self.data_dir, fname)
        triples = []
        with open(path) as f:
            for line in f:
                h, r, t = map(int, line.strip().split("\t"))
                triples.append((h, r, t))
        return triples

    def _load_links(self, fname):
        path = os.path.join(self.data_dir, fname)
        links = []
        with open(path) as f:
            for line in f:
                e1, e2 = map(int, line.strip().split("\t"))
                links.append((e1, e2))
        return links


class AlignmentDataset(Dataset):
    """Wraps seed pairs for training."""

    def __init__(self, links, neg_num: int = 25, num_entities_1: int = 0, num_entities_2: int = 0):
        self.links = links
        self.neg_num = neg_num
        self.n1 = num_entities_1
        self.n2 = num_entities_2

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        e1, e2 = self.links[idx]
        neg_e2 = torch.randint(0, self.n2, (self.neg_num,))
        neg_e1 = torch.randint(0, self.n1, (self.neg_num,))
        return {
            "pos_e1": e1,
            "pos_e2": e2,
            "neg_e2": neg_e2,
            "neg_e1": neg_e1,
        }
