from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader

import torch



class transform_feat_to_float(object):
    def __init__(self,dtype="float32"):
        if dtype == "float32":
            self.dtype = torch.float32
        
    def __call__(self, data):
        try:
            data.x = data.x.to(self.dtype)
        except:
            data.x = torch.ones(data.num_nodes,1).to(self.dtype)
        data.edge_attr = data.edge_attr.to(self.dtype)
        data.y = data.y.squeeze()
        # data.edge_index = data.edge_index.to(torch.int32)
        return data



def init_ogb_dataset(config):
    dataset = PygGraphPropPredDataset(name =config["dataset::name"], root = config[dataset_path = Path(PATHMISSING)
    split_idx = dataset.get_idx_split() 
    return dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]