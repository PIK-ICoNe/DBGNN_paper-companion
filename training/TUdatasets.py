from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
from torch_geometric.utils import one_hot


class transform_add_node_edge_feat(object):
    def __init__(self, add_node_features=True, add_edge_features=True, hot_encoding_targets = True):
        self.add_node_features = add_node_features
        self.add_edge_features = add_edge_features
        # self.hot_encoding_targets = hot_encoding_targets
        
    def __call__(self, data):
        if self.add_node_features:
            data.x = torch.ones(data.num_nodes,1)
        if self.add_edge_features:
            data.edge_attr = torch.ones(data.edge_index.shape[1],1)
        # if self.hot_encoding_targets:
        #     data.y = one_hot(data.y, 3).int()
        return data


def init_TU_dataset_collab(config):
    if config["dataset_name"] == "TU-collab":   
        name = "COLLAB"
    dataset = TUDataset(root=config[dataset_path = Path(PATHMISSING)
    train_set, valid_set, test_set = random_split(dataset, [0.7726, 0.0772, 1-0.7726-0.0772])
    return train_set, valid_set, test_set