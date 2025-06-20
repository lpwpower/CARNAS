# import os
import os.path as osp

import torch
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold
from torch import cat
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree

from assigner import DegreeDistribution
from datasets.spmotif_dataset import SPMotif


class HandleNodeAttention(object):
    def __call__(self, data):
        data.attn = torch.softmax(data.x[:, 0], dim=0)
        data.x = data.x[:, 1:]
        return data


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.uint8, device=index.device)
    mask[index] = 1
    return mask


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def derive_colors(dataset, ratio=0.1):
    labels = []
    for data in dataset:
        labels.append(data.y.item())

    from collections import Counter
    count_labels = Counter(labels)
    print(count_labels)
    data_mask = torch.zeros(len(labels), dtype=torch.uint8, device=data.y.device)

    labels = torch.tensor(labels)
    for i in range(len(count_labels)):
        idx = torch.where(labels == i)[0]
        sampled_idx = int(count_labels[i] * ratio)
        print(i, sampled_idx, len(idx))
        data_mask[idx[:sampled_idx]] = 1
    print(data_mask.sum())
    return dataset[data_mask]


def load_data(dataset_name='DD', cleaned=False, split_seed=12345, batch_size=32, bias=None, remove_large_graph=True):
    transform = DegreeDistribution()
    if 'ogbg' in dataset_name:
        dataset = PygGraphPropPredDataset(name=dataset_name, root='./ogb/dataset/', pre_transform=transform)
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        return [dataset, dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']], train_loader, val_loader, test_loader], -1

    if 'SPMotif' in dataset_name:
        datadir = '/DATA/DATANAS1/yhq/PAS-main-data/rationale2/data/'
        train_dataset = SPMotif(osp.join(datadir, f'{dataset_name}/'), mode='train')
        val_dataset = SPMotif(osp.join(datadir, f'{dataset_name}/'), mode='val')
        test_dataset = SPMotif(osp.join(datadir, f'{dataset_name}/'),  mode='test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return [train_dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader], -1

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    
    dataset.data.edge_attr = None
    # load and process
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    train_id = []
    test_id = []
    # for diffpool method: remove large graphs
    num_nodes = max_num_nodes = 0
    for i, data in enumerate(dataset):
        num_nodes += data.num_nodes
        max_num_nodes = max(data.num_nodes, max_num_nodes)

        if dataset_name == "DD":
            if data.num_nodes < 201:
                train_id.append(i)
            else:
                test_id.append(i)

    # # Filter out a few really large graphs in order to apply DiffPool.
    num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

    
    
    # split 811
    skf = StratifiedKFold(10, shuffle=True, random_state=split_seed)
    idx = [torch.from_numpy(i) for _, i in skf.split(torch.zeros(len(dataset)), dataset.data.y[:len(dataset)])]
    split = [cat(idx[:8], 0), cat(idx[8:9], 0), cat(idx[9:], 0)]

    train_dataset = dataset[split[0]]
    val_dataset = dataset[split[1]]
    test_dataset = dataset[split[2]]
    print('train:{}, val:{}, test:{}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    return [dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader], num_nodes


def load_k_fold(dataset, folds, batch_size):
    print('10fold split')
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices = []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[:len(dataset)]):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    data_10fold = []
    for i in range(folds):
        data_ith = [0, 0, 0, 0]  # align with 811 split process.
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_mask = train_mask.nonzero().view(-1)

        data_ith.append(DataLoader(dataset[train_mask], batch_size, shuffle=True))
        data_ith.append(DataLoader(dataset[val_indices[i]], batch_size, shuffle=True))
        data_ith.append(DataLoader(dataset[test_indices[i]], batch_size, shuffle=True))
        data_10fold.append(data_ith)

    return data_10fold
