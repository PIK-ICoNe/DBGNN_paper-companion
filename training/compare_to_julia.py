from gnn_models import * 
# from snbs_homo import *
# from surv_nf import *
# from TUdatasets import *
# from ogb_datasets import *



from torch_geometric.data import DataLoader
# from torch_geometric.loader import DataLoader


from torch_geometric.data import Data as gData


from pathlib import Path


# dataset_path = Path(PATHMISSING)

# dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = dataset_path)
# split_idx = dataset.get_idx_split() 

# dataset[split_idx["test"]]

cfg = {}
# cfg[dataset_path = Path(PATHMISSING)
cfg["num_classes"] = 1
cfg["num_layers"] = 1#15
cfg["edge_dim"] = 2
cfg["final_linear_layer"] = False#True
cfg["final_sigmoid_layer"] = False
cfg["activation"] = "ReLU"#["relu", "None"]

cfg["DBGNN::dense_after_linDB"] = False
cfg["DBGNN::in_channels_n"] = 2
cfg["DBGNN::out_channels_n"] = cfg["num_classes"]
cfg["DBGNN::hidden_channels_n"] = 2
cfg["DBGNN::in_channels_e"] = cfg["edge_dim"]
cfg["DBGNN::hidden_channels_e"] = 2
cfg["DBGNN::num_steps"] = 2#30
cfg["DBGNN::Δ"] = .01


cfg["skip_connection_n"] = False
cfg["skip_connection_e"] = False


cfg["dropout_n"] = 0
cfg["dropout_e"] = 0
cfg["pool"] = False



# train_set, valid_set, test_set = init_ogb_dataset_molhiv(config)
config = cfg
model = DBGNNModel(num_layers=config["num_layers"], in_channels_n=1, out_channels_n=1, hidden_channels_n=1, in_channels_e=1, hidden_channels_e=1, num_steps = config["DBGNN::num_steps"], activation=config["activation"], dropout_n=config["dropout_n"], dropout_e=config["dropout_e"], skip_connection_n=config["skip_connection_n"], skip_connection_e=config["skip_connection_e"], Δ = config["DBGNN::Δ"], dense_after_linDB=config["DBGNN::dense_after_linDB"], pool = config["pool"], final_linear_layer=config["final_linear_layer"], final_sigmoid_layer=config["final_sigmoid_layer"])

## 1D- example by hand

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
edge_attr = torch.unsqueeze(torch.tensor([1.,2.,3.,4.]),dim=1)
# edge_attr = torch.tensor([1.,2.,3.,4.])
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = gData(x=x, edge_index=edge_index, edge_attr = edge_attr)

out_model = model(x, edge_index, edge_attr, 1)

# modify weights of model

model.dbgnn_conv.dense_e_in.weight = torch.nn.Parameter(torch.tensor([[1.]]))
model.dbgnn_conv.dense_n_in.weight = torch.nn.Parameter(torch.tensor([[1.]]))
model.dbgnn_conv.dense_e_in.bias = torch.nn.Parameter(torch.tensor([0.]))
model.dbgnn_conv.dense_n_in.bias = torch.nn.Parameter(torch.tensor([0.]))

# model.dbgnn_conv.linDB[0].W_ne.weight=torch.nn.Parameter(torch.tensor([[1.]]))
# model.dbgnn_conv.linDB[0].W_en.weight=torch.nn.Parameter(torch.tensor([[1.]]))
model.dbgnn_conv.linDB[0].W_ne=torch.nn.Parameter(torch.tensor([[1.]]))
model.dbgnn_conv.linDB[0].W_en=torch.nn.Parameter(torch.tensor([[1.]]))

model.dbgnn_conv.linDB[0].beta_n = torch.nn.Parameter(torch.tensor([[1.]]))
model.dbgnn_conv.linDB[0].beta_e = torch.nn.Parameter(torch.tensor([[1.]]))

model.dbgnn_conv.final_dense.weight = torch.nn.Parameter(torch.tensor([[1.]]))
model.dbgnn_conv.final_dense.bias = torch.nn.Parameter(torch.tensor([0.]))

out_after_mod = model(x, edge_index, edge_attr, 1)


## 2D- example by hand

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
edge_attr = torch.tensor([[1.,5],[2.,6],[3.,7],[4.,8]])
# edge_attr = torch.tensor([1.,2.,3.,4.])
x = torch.tensor([[-1, 1], [0, 4], [1, 2]], dtype=torch.float)

data = gData(x=x, edge_index=edge_index, edge_attr = edge_attr)


model = DBGNNModel(num_layers=config["num_layers"], in_channels_n=2, out_channels_n=1, hidden_channels_n=2, in_channels_e=2, hidden_channels_e=2, num_steps = config["DBGNN::num_steps"], activation=config["activation"], dropout_n=config["dropout_n"], dropout_e=config["dropout_e"], skip_connection_n=config["skip_connection_n"], skip_connection_e=config["skip_connection_e"], Δ = config["DBGNN::Δ"], dense_after_linDB=config["DBGNN::dense_after_linDB"], pool = config["pool"], final_linear_layer=config["final_linear_layer"], final_sigmoid_layer=config["final_sigmoid_layer"])
out_model = model(x, edge_index, edge_attr, 1)

# modify weights of model
weight_tensor = torch.tensor([[2.,1.], [5.,1.]])
# weight_tensor = torch.tensor([[2.,5.], [1.,1.]])
bias_tensor = torch.tensor([0., 0.])

model.dbgnn_conv.dense_e_in.weight = torch.nn.Parameter(weight_tensor)
model.dbgnn_conv.dense_n_in.weight = torch.nn.Parameter(weight_tensor)
model.dbgnn_conv.dense_e_in.bias = torch.nn.Parameter(bias_tensor)
model.dbgnn_conv.dense_n_in.bias = torch.nn.Parameter(bias_tensor)

# model.dbgnn_conv.linDB[0].W_ne.weight=torch.nn.Parameter(weight_tensor)
# model.dbgnn_conv.linDB[0].W_en.weight=torch.nn.Parameter(weight_tensor)

model.dbgnn_conv.linDB[0].W_ne=torch.nn.Parameter(weight_tensor)
model.dbgnn_conv.linDB[0].W_en=torch.nn.Parameter(weight_tensor)

model.dbgnn_conv.linDB[0].beta_n = torch.nn.Parameter(weight_tensor)
model.dbgnn_conv.linDB[0].beta_e = torch.nn.Parameter(weight_tensor)

model.dbgnn_conv.final_dense.weight = torch.nn.Parameter(weight_tensor[0,:].unsqueeze(0))
model.dbgnn_conv.final_dense.bias = torch.nn.Parameter(bias_tensor[0])

out_after_mod = model(x, edge_index, edge_attr, 1)


out_after_mod_2d = torch.tensor([[ 930.], [3585.], [2133.]])

out_after_mod == out_after_mod_2d
print("finished")