from torch_geometric.datasets import  LINKXDataset
import torch
from torch_geometric.transforms import RandomNodeSplit



class pre_transform_func(object):
    def __init__(self,dtype="float32"):
        if dtype == "float32":
            self.dtype = torch.float32
            self.transform_rand_node_split = RandomNodeSplit(split="train_rest", num_val = .25, num_test = .25)
        
    def __call__(self, data):
        data = transform_features(data, self.dtype)
        data.y = data.y.float()
        data = self.transform_rand_node_split(data)
        return data

class pre_transform_func_MaskUnlabeled(object):
    def __init__(self,dtype="float32"):
        if dtype == "float32":
            self.dtype = torch.float32
            self.transform_rand_node_split = RandomNodeSplit(split="train_rest", num_val = .25, num_test = .25)
        
    def __call__(self, data):
        data = transform_features(data, self.dtype)
        mask = data.y == -1
        data.y = data.y.float()
        data.y[mask] = torch.nan
        data = self.transform_rand_node_split(data)
        return data


def transform_features(data, dtype):
    try:
        data.x = data.x.to(dtype)
    except:
        data.x = torch.ones(data.num_nodes,1).to(dtype)
    data.edge_attr = torch.unsqueeze(torch.ones(data.edge_index.shape[1]),1)
    data.edge_attr = data.edge_attr.to(dtype)
    return data

def init_linkx_dataset(config):
    dataset_name = config["dataset::name"][6:]
    if dataset_name == "penn94":
        linkxdataset =  LINKXDataset(name = dataset_name, root = config[dataset_path = Path(PATHMISSING)
    else:
        linkxdataset =  LINKXDataset(name = dataset_name, root = config[dataset_path = Path(PATHMISSING)
    return linkxdataset