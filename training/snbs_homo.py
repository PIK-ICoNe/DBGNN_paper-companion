import os
import numpy as np
import h5py
import pandas as pd

import torch

from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset
from torch.utils.data import Dataset, DataLoader





class gnn_snbs_surv(gDataset):
    def __init__(self, grid_path, task, task_type, use_dtype_double=False, slice_index=slice(0, 0), normalize_targets=False):
        super(gnn_snbs_surv, self).__init__()
        self.normalize_targets = normalize_targets
        self.start_index = slice_index.start + 1
        self.data_len = slice_index.stop - slice_index.start + 1

        self.path = grid_path
        # self.num_classes = 1
        self.task = task
        self.task_type = task_type
        if use_dtype_double == True:
            self.dtype = 'float64'
        else:
            self.dtype = 'float32'
        self.data = {}
        self.read_in_all_data()
        # self.positive_weight = self.compute_pos_weight()

    def read_targets(self):
        all_targets = {}
        file_targets = self.path / f'{self.task}.h5'
        hf = h5py.File(file_targets, 'r')
        for index_grid in range(self.start_index, self.start_index + self.data_len):
            all_targets[index_grid] = np.array(
                hf.get(str(index_grid)), dtype=self.dtype)
        return all_targets

    def read_in_all_data(self):
        targets = self.read_targets()
        file_to_read = str(self.path) + "/input_data.h5"
        f = h5py.File(file_to_read, 'r')
        dset_grids = f["grids"]
        for index_grid in range(self.start_index, self.start_index + self.data_len):
            node_features = np.array(dset_grids[str(index_grid)].get(
                "node_features"), dtype=self.dtype).transpose()
            edge_index = (np.array(dset_grids[str(index_grid)].get(
                "edge_index"), dtype='int64') - 1).transpose()
            edge_attr = np.array(
                dset_grids[str(index_grid)].get("edge_attr"), dtype=self.dtype)
            y = torch.tensor(targets[index_grid])
            self.data[index_grid - self.start_index] = gData(x=(torch.tensor(node_features).unsqueeze(-1)), edge_index=torch.tensor(
                edge_index), edge_attr=torch.tensor(edge_attr).unsqueeze(-1), y=y)

        if self.task_type in ['classification', 'regressionThresholding']:
            targets_all_in_one_array = targets[self.start_index]
            for index_grid in range(self.start_index+1, self.start_index + self.data_len):
                targets_all_in_one_array = np.concatenate(
                    (targets_all_in_one_array, targets[index_grid]))
            if self.task_type == "regressionThresholding":
                targets_classified = np.where(
                    targets_all_in_one_array < 15., 0., 1.)
            else:
                targets_classified = targets_all_in_one_array
            self.positive_weight = torch.tensor((np.size(
                targets_classified) - np.count_nonzero(targets_classified))/np.count_nonzero(targets_classified))

    def len(self):
        return len(self.data)

    def get(self, index):
        return self.data[index]


class NetworkMeasures_MLP_Dataset(Dataset):
    def __init__(self, path_dataset, path_scaling, task, task_type, selected_measures, data_slice=slice(0, 0), scaling_slice=slice(0, 0), use_dtype_double=True):

        self.path_dataset = path_dataset
        self.path_scaling = path_scaling
        self.selected_measures = selected_measures
        self.task = task
        self.task_type = task_type
        self.data_slice = data_slice
        self.scaling_slice = scaling_slice
        if use_dtype_double == True:
            self.dtype = 'float64'
        else:
            self.dtype = 'float32'

        self.read_in_data()

    def read_in_data(self):
        data_slice = self.data_slice
        scaling_slice = self.scaling_slice
        task_type = self.task_type
        task = self.task
        path_dataset = self.path_dataset
        path_scaling = self.path_scaling

        if task_type == 'node_classification' or task_type == 'node_regression':
            N, start_index, targets = self.read_in_targets(
                path_dataset, task, task_type)
        elif task_type == 'node_regressionThresholding':
            N, start_index, targets, targets_threshold = self.read_in_targets(
                path_dataset, task, task_type)

        X = pd.read_csv(self.path_dataset / 'input_features.csv')
        try:
            X = X.drop(columns=["node_cat"])
        except:
            pass

        if data_slice.stop != 0:
            data_index_start = (data_slice.start-start_index+1)*N
            data_index_end = (data_slice.stop-start_index+2)*N
            X = X[data_index_start:data_index_end]
            targets = targets[data_index_start:data_index_end]
            if self.task_type == 'node_regressionThresholding':
                targets_threshold = targets_threshold[data_index_start:data_index_end]

        if task_type == 'node_classification' or task_type == 'node_regression':
            N_scaling, start_index_scaling, targets_scaling = self.read_in_targets(
                path_scaling, task, task_type)
        elif task_type == 'node_regressionThresholding':
            N_scaling, start_index_scaling, targets_scaling, targets_scaling_threshold = self.read_in_targets(
                path_scaling, task, task_type)

        X_scaling = pd.read_csv(path_scaling / 'input_features.csv')
        try:
            X_scaling = X_scaling.drop(columns=["node_cat"])
        except:
            pass

        if scaling_slice.stop != 0:
            scaling_index_start = (scaling_slice.start-start_index_scaling+1)*N
            scaling_index_end = (scaling_slice.stop-start_index_scaling+1)*N
            X_scaling = X_scaling[scaling_index_start:scaling_index_end]
            targets_scaling = targets_scaling[scaling_index_start:scaling_index_end]
            if self.task_type == 'node_regressionThresholding':
                targets_scaling_threshold = targets_scaling_threshold[
                    scaling_index_start:scaling_index_end]
        scaler = StandardScaler()
        scaler.fit(X_scaling)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        X_scaled = np.array(X_scaled[self.selected_measures], dtype=self.dtype)
        self.X = torch.tensor(X_scaled)
        self.targets = torch.tensor(targets)
        if self.task_type == 'node_classification':
            self.positive_weight = torch.tensor((np.size(
                targets_scaling) - np.count_nonzero(targets_scaling))/np.size(targets_scaling))
        if self.task_type == 'node_regressionThresholding':
            self.positive_weight = torch.tensor((np.size(
                targets_scaling) - np.count_nonzero(targets_scaling))/np.size(targets_scaling))
            self.targets_threshold = torch.FloatTensor(targets_threshold)

    def read_in_targets(self, dataset_path, task, task_type):
        # read in targets
        file_targets = dataset_path / f'{task}.h5'
        hf = h5py.File(file_targets, 'r')
        targets_dict = {}
        targets = np.empty(0, dtype=self.dtype)
        for key in hf.keys():
            targets_dict[key] = np.array(hf.get(key), dtype=self.dtype)
        start_index = min(list(map(int, targets_dict.keys())))
        N = len(targets_dict[list(targets_dict.keys())[0]])

        for i in range(0, len(targets_dict)):
            targets = np.append(targets, targets_dict[str(int(start_index)+i)])

        if task_type == 'node_regressionThresholding':
            file_targets_threshold = dataset_path / 'tm.h5'
            hf_threshold = h5py.File(file_targets_threshold, 'r')
            targets_threshold_dict = {}
            targets_threshold = np.empty(0)
            for key in hf.keys():
                targets_threshold_dict[key] = np.array(
                    hf_threshold.get(key), dtype=self.dtype)
            for i in range(0, len(targets_dict)):
                targets_threshold = np.append(
                    targets_threshold, targets_threshold_dict[str(int(start_index)+i)])
        if task_type == 'node_regressionThresholding':
            return N, start_index, targets, targets_threshold
        else:
            return N, start_index, targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.task_type == "node_regressionThresholding":
            return self.X[idx], self.targets[idx], self.targets_threshold[idx]
        else:
            return self.X[idx], self.targets[idx]

def init_snbs_hom_dataset(config):
    task = config["task"]
    task_type = config["task_type"]

    if "MLP::use_MLP" in config.keys():
        use_MLP = config["MLP::use_MLP"]
    else:
        use_MLP = False
    if "troublemakers::normalize_targets" in config:
        normalize_targets = config["troublemakers::normalize_targets"]
    else:
        normalize_targets = False

    if "snbs_homo::train_set::start_index" in config.keys() and "snbs_homo::train_set::end_index" in config.keys():
        train_slice = slice(
            config["snbs_homo::train_set::start_index"], config["snbs_homo::train_set::end_index"])
        print("train_slice: ", train_slice)
    else:
        train_slice = slice(0, 6999)

    if "snbs_homo::valid_set::start_index" in config.keys() and "snbs_homo::valid_set::end_index" in config.keys():
        valid_slice = slice(
            config["snbs_homo::valid_set::start_index"], config["snbs_homo::valid_set::end_index"])
        print("snbs_homo::valid_slice: ", valid_slice)
    else:
        valid_slice = slice(7000, 8499)
    if use_MLP:
        scaling_slice = config["MLP::scaling_slice"]
        train_set = NetworkMeasures_MLP_Dataset(config[dataset_path = Path(PATHMISSING)
                                                        task_type, selected_measures=config["MLP::selected_measures"], data_slice=train_slice, scaling_slice=scaling_slice)
        valid_set = NetworkMeasures_MLP_Dataset(config[dataset_path = Path(PATHMISSING)
                                                        task_type, selected_measures=config["MLP::selected_measures"], data_slice=valid_slice, scaling_slice=scaling_slice)
        test_set = NetworkMeasures_MLP_Dataset(config[dataset_path = Path(PATHMISSING)
                                                    task_type, selected_measures=config["MLP::selected_measures"], data_slice=slice(8500, 9999), scaling_slice=scaling_slice)
    else:
        train_set = gnn_snbs_surv(
            config[dataset_path = Path(PATHMISSING)
        valid_set = gnn_snbs_surv(
            config[dataset_path = Path(PATHMISSING)
        test_set = gnn_snbs_surv(
            config[dataset_path = Path(PATHMISSING)

    return train_set, valid_set, test_set