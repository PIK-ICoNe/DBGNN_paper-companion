from torch_geometric.datasets import LRGBDataset
import torch

class transform_feat_to_float(object):
    def __init__(self, dataset_name, dtype="float32"):
        if dtype == "float32":
            self.dtype = torch.float32
        self.dataset_name = dataset_name
        
    def __call__(self, data):
        data.x = data.x.to(self.dtype)
        data.edge_attr = data.edge_attr.to(self.dtype)
        if self.dataset_name == "Peptides-func":
            # data.y = data.y.int()
            data.y = data.y.float()
        # data.y = data.y.squeeze()
        return data


def init_lrgb_dataset(config):
    dataset_name = config["dataset::name"][6:]
    dataset_path = Path(PATHMISSING)
    train_set = LRGBDataset(name =dataset_name , split = "train", root = dataset_path, pre_transform = transform_feat_to_float(dataset_name))
    valid_set = LRGBDataset(name =dataset_name, split = "val", root = dataset_path, pre_transform = transform_feat_to_float(dataset_name))
    test_set = LRGBDataset(name =dataset_name, split = "test", root = dataset_path, pre_transform = transform_feat_to_float(dataset_name))
    return train_set, valid_set, test_set