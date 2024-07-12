from gnn_models import * 
from snbs_homo import *
from surv_nf import *
from TUdatasets import *
from ogb_datasets import *
from lrgb_datasets import *

from dirichlet_energy import *

from torch_geometric.data import DataLoader
# from torch_geometric.loader import DataLoader


from torch_geometric.data import Data as gData


from pathlib import Path

# dataset_path = Path(PATHMISSING)
# dataset_path = Path(PATHMISSING)
# dataset_path = Path(PATHMISSING)
# ieee_path = Path("/Users/christiannauck/work/pik/data/dataset_v2/ieee")

dataset_path = Path(PATHMISSING)
dataset_path = Path(PATHMISSING)

cfg = {}
# dataset
cfg[dataset_path = Path(PATHMISSING)
# cfg["dataset::ieee_path"] = ieee_path
cfg["dataset::name"] = "lrgb::Peptides-struct"#"ogbg-ppa"
cfg["input_features_node_dim"] = 9
cfg["input_features_edge_dim"] = 3
cfg["output_features_node_dim"] = 11
cfg["num_classes"] = 11#37#7#3 # 1
cfg["pool"] = "mean"
cfg["task"] = "None"#"surv"#"snbs"
cfg["task_type"] = "regression" #"classification"#
cfg["criterion"] = "MSELoss"#"CELoss"#"BCEWithLogitsLoss"#"CELoss"#"MSELoss"
# cfg["list_metrics"] = ["accuracy", "f1", "fbeta", "recall", "precision"]
# cfg["list_metrics"] = ["accuracy", "f1", "fbeta", "recall", "precision", "auroc"]
# cfg["list_metrics"] = ["auroc"]
cfg["list_metrics"] = ["r2"]
cfg["list_metrics"] = ["MAE"]


metric = "valid_auroc"
metric_mode = "max"
metric_checkpoint = metric
# metric = "valid_loss"
# metric_mode = "min"
# metric_checkpoint = "valid_R2"





## datset snbs_homog
cfg["snbs_homo::train_set::start_index"] = 0
cfg["snbs_homo::train_set::end_index"] = 799

## dataset nf properties
cfg["scaling"] = "normalize"
cfg["ieee_scaling"] = "normalize"
cfg["dtype"] = "float32"
cfg["grid_type"] = "homo"#"hetero"
cfg["ieee_eval"] = False



cfg["train_slice"] = slice(1, 700)
cfg["valid_slice"] = slice(701, 850)
cfg["test_slice"] = slice(851, 1000)

# dataset batch sizes
cfg["train_set::batchsize"] = 1000
# cfg["train_set::batchsize"] = tune.choice([10, 20])
# cfg["train_set::batchsize"] = tune.randint(10,800)
cfg["test_set::batchsize"] = 150
cfg["valid_set::batchsize"] = 150
cfg["train_set::shuffle"] = True
cfg["test_set::shuffle"] = False
cfg["valid_set::shuffle"] = False



cfg["Fbeta::beta"] = 2.0


# model settings
cfg["model_name"] = "DBGNN"# "GAT"#"TAG"#"DBGNN"
cfg["num_layers"] = 1
cfg["final_linear_layer"] = False#True
cfg["final_sigmoid_layer"] = False




cfg["dropout_n"] = 0
cfg["dropout_e"] = 0



### other models
cfg["hidden_channels"] = 10
cfg["linear_layer_after_conv"] = False#True# False #True
cfg["ll_after_conv_dim"] = 500
cfg["TAG::K"] = 3
cfg["activation"] = "ReLU"#["relu", "None"]
### GAT
cfg["heads"] = 2
cfg["add_self_loops"] = False
cfg["GAT::v2"] = True


# DBGNN
cfg["DBGNN::dense_after_linDB"] = False
cfg["DBGNN::in_channels_n"] = cfg["input_features_node_dim"]
cfg["DBGNN::out_channels_n"] = cfg["output_features_node_dim"]
cfg["DBGNN::hidden_channels_n"] = 10
cfg["DBGNN::in_channels_e"] = cfg["input_features_edge_dim"]
cfg["DBGNN::hidden_channels_e"] = 10
cfg["DBGNN::num_steps"] = 12
cfg["DBGNN::Δ"] = .01


cfg["skip_connection_n"] = False
cfg["skip_connection_e"] = False



## MLP
cfg["MLP::use_MLP"] = False
cfg["MLP::num_classes"] = cfg["num_classes"]
cfg["MLP::num_hidden_layers"] = 1
cfg["MLP::num_hidden_unit_per_layer"] = 30
# cfg["MLP::selected_measures"] = [
#     "P",
#     'degree',
#     'average_neighbor_degree',
#     'clustering',
#     'current_flow_betweenness_centrality',
#     'closeness_centrality']
cfg["MLP::selected_measures"] = ['P', 'AP', 'AAP', 'AAAP', 'row_sum_AA', 'row_sum_AAA',
       'attribute_assortativity_coefficient_P', 'degree',
       'neighbor_degree_max', 'neighbor_degree_min', 'clusterƒing',
       'betweenness_centrality', 'closeness_centrality', 'load_centrality',
       'eigenvector_centrality', 'katz_centrality', 'second_order_centrality',
       'current_flow_closeness_centrality',
       'current_flow_betweenness_centrality', 'average_neighbor_degree',
       'harmonic_centrality', 'square_clustering', 'eccentricity',
       'degree_assortativity_coefficient', 'transitivity', 'diameter', 'bulk',
       'root', 'dense sprout', 'sparse sprout', 'inner tree node']

cfg["MLP::num_input_features"] = len(cfg["MLP::selected_measures"])
cfg["MLP::final_sigmoid_layer"] = False


# training settings
cfg["cuda"] = False
#cfg["num_workers"] = 1
#cfg["num_threads"] = 2
cfg["manual_seed"] = 1
# cfg["manual_seed"] = tune.choice([1,2,3,4,5])s
cfg["epochs"] = 200
cfg["optim::optimizer"] = "SGD"
# cfg["optim::LR"] = .3
cfg["optim::LR"] = 1e-4#tune.loguniform(1e-4, 2e1)
# cfg["optim::LR"] = tune.choice([1e-4, 1e-2])
cfg["optim::momentum"] = .9
cfg["optim::weight_decay"] = 1e-9
cfg["optim::scheduler"] = None
cfg["optim::ReducePlat_patience"] = 20
cfg["optim::LR_reduce_factor"] = .7
cfg["optim::stepLR_step_size"] = 30
cfg["optim::ExponentialLR::gamma"] = .99


## gradient clipping
cfg["gradient_clipping"] = True
cfg["gradient_clipping_::max_norm"] = 10


# evaluation
cfg["eval::threshold"] = .1



# ray settings
cfg["save_after_epochs"] = 10000
cfg["checkpoint_freq"] = 10000
cfg["num_samples"] = 8
cfg["ray_name"] = "TAG"

if cfg["grid_type"] == "hetero":
    hetero = True
else:
    hetero = False


num_workers = 0
pin_memory = False

config = cfg
task = cfg["task"]
task_type = cfg["task_type"]
if cfg["dataset::name"] == "snbs_homo":
    train_set, valid_set, test_set = init_snbs_hom_dataset(cfg)
elif cfg["dataset::name"] == "TU-collab":
    train_set, valid_set, test_set = init_TU_dataset_collab(cfg)
elif config["dataset::name"] == "ogbg-molhiv":
    train_set, valid_set, test_set = init_ogb_dataset(config)
elif config["dataset::name"] == "ogbg-ppa":
    train_set, valid_set, test_set = init_ogb_dataset(config)
elif config["dataset::name"] == "surv_nf":
    train_set, valid_set, test_set = init_surv_nf_dataset(config)
elif config["dataset::name"][0:4] == "lrgb":
    train_set, valid_set, test_set = init_lrgb_dataset(config)


train_loader = DataLoader(
    train_set, batch_size=config["train_set::batchsize"], shuffle=config["train_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
valid_loader = DataLoader(
    valid_set, batch_size=config["valid_set::batchsize"], shuffle=config["valid_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=config["test_set::batchsize"], shuffle=config["test_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)

cfg_ray = {"len_trainloader": len(train_loader)}
gnnModule = GNNmodule(cfg, config_ray = cfg_ray)
# if hetero == False:
#     dir_en = compute_dirichlet_energy(gnnModule.model,train_set[0])



# gnnModule.model.dbgnn_conv.final_dense.bias = torch.nn.Parameter(torch.tensor([.7]))

# gnnModule.model(train_set[0])
# gnnModule.model(train_set[2440])
# gnnModule.model(train_set[3732])

# for i in range(len(train_set)):
#     print("i: "+ str(i))
#     gnnModule.model(train_set[i])


train_loss_all_epochs = []
train_accu_all_epochs = []
train_R2_all_epochs = []
train_MAE_all_epochs = []
train_fbeta_all_epochs = []
train_auroc_all_epochs = []
valid_loss_all_epochs = []
valid_accu_all_epochs = []
valid_R2_all_epochs = []
valid_fbeta_all_epochs = []
valid_auroc_all_epochs = []
test_loss_all_epochs = []
test_accu_all_epochs = []
test_R2_all_epochs = []
test_loss_all_epochs = []
test_fbeta_all_epochs = []
test_auroc_all_epochs = []

train_fbeta_all_epochs = []
test_fbeta_all_epochs = []

for epoch in range(1,cfg["epochs"]):
    if cfg["grid_type"] == "homo":
        if cfg["task_type"] == "regression":
            train_loss, train_accu, train_R2, train_MAE = gnnModule.train_epoch_regression(train_loader, cfg["eval::threshold"])
            # valid_loss, valid_accu, valid_R2 = gnnModule.eval_model_regression(valid_loader, cfg["eval::threshold"])
            # test_loss, test_accu, test_R2 = gnnModule.eval_model_regression(test_loader, cfg["eval::threshold"])
            train_loss_all_epochs.append(train_loss)
            train_accu_all_epochs.append(train_accu)
            train_R2_all_epochs.append(train_R2)
            train_MAE_all_epochs.append(train_MAE)

            # valid_loss_all_epochs.append(valid_loss)
            # valid_accu_all_epochs.append(valid_accu)
            # valid_R2_all_epochs.append(valid_R2)

            # test_loss_all_epochs.append(test_loss)
            # test_accu_all_epochs.append(test_accu)
            # test_R2_all_epochs.append(test_R2)
            print("epoch: ", epoch, "      train_R2: ", train_R2)
            print("epoch: ", epoch, "      train_MAE: ", train_MAE)

        elif cfg["task_type"] == "classification":
            train_loss, train_acc, train_f1, train_fbeta, train_recall, train_precision, train_auroc = gnnModule.train_epoch_classification(train_loader)
            valid_loss, valid_acc, valid_f1, valid_fbeta, valid_recall, valid_precision, valid_auroc = gnnModule.eval_model_classification(valid_loader)
            test_loss, test_acc, test_f1, test_fbeta, test_recall, test_precision, test_auroc = gnnModule.eval_model_classification(test_loader)

            train_loss_all_epochs.append(train_loss)
            train_accu_all_epochs.append(train_acc)
            train_fbeta_all_epochs.append(train_fbeta)
            train_auroc_all_epochs.append(train_auroc)

            valid_loss_all_epochs.append(valid_loss)
            valid_accu_all_epochs.append(valid_acc)
            valid_fbeta_all_epochs.append(valid_fbeta)
            valid_auroc_all_epochs.append(valid_auroc)

            test_loss_all_epochs.append(test_loss)
            test_accu_all_epochs.append(test_acc)
            test_fbeta_all_epochs.append(test_fbeta)
            test_auroc_all_epochs.append(test_auroc)
            # print("epoch: ", epoch, "      train_fbeta: ", train_fbeta)
            print("epoch: ", epoch, "      train_accu: ", train_acc)
            # print("epoch: ", epoch, "      train_auroc: ", rond(train_auroc,5), "      valid_auroc: ", round(valid_auroc,5), "      test_auroc: ", round(test_auroc,5))

        elif cfg["task_type"] == "regressionThresholding":
            train_loss, train_R2, train_fbeta, train_recall = gnnModule.train_epoch_regressionThresholding(train_loader)
            test_loss, test_R2, test_fbeta, test_recall = gnnModule.eval_model_regressionThresholding(test_loader)


            print("epoch: ", epoch, "      train_fbeta: ", train_fbeta)
    else:
        train_loss, train_accu, train_R2 = gnnModule.train_epoch_regression_hetero(train_loader, cfg["eval::threshold"])
        valid_loss, valid_accu, valid_R2 = gnnModule.eval_model_regression_hetero(valid_loader, cfg["eval::threshold"])
        test_loss, test_accu, test_R2 = gnnModule.eval_model_regression_hetero(test_loader, cfg["eval::threshold"])
        train_loss_all_epochs.append(train_loss)
        train_accu_all_epochs.append(train_accu)
        train_R2_all_epochs.append(train_R2)

        valid_loss_all_epochs.append(valid_loss)
        valid_accu_all_epochs.append(valid_accu)
        valid_R2_all_epochs.append(valid_R2)

        test_loss_all_epochs.append(test_loss)
        test_accu_all_epochs.append(test_accu)
        test_R2_all_epochs.append(test_R2)
        print("epoch: ", epoch, "      train_R2: ", train_R2)


if cfg["task_type"] == "regression":
    print("train max_R2: ", max(train_R2_all_epochs))
    print("valid max_R2: ", max(valid_R2_all_epochs))
    print("test max_R2: ", max(test_R2_all_epochs))
elif cfg["task_type"] == "regressionThresholding":
    print("train max_R2: ", max(train_R2_all_epochs))
    print("valid max_R2: ", max(valid_R2_all_epochs))
    print("test max_R2: ", max(test_R2_all_epochs))
elif cfg["task_type"] == "classification":
    print("train max_accu: ", max(train_accu_all_epochs))
    print("valid max_accu: ", max(valid_accu_all_epochs))
    print("test max_accu: ", max(test_accu_all_epochs))



import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
# ax.plot(test_R2_all_epochs, label="test")
ax.plot(train_R2_all_epochs, label="train")
ax.plot(valid_R2_all_epochs, label="valid")
ax.plot(test_R2_all_epochs, label="test")
ax.set_ylabel('$R^2$',fontsize=18)
ax.set_xlabel('epochs',fontsize=18)
ax.set_xlim([1,cfg["epochs"]])
# ax.set_xlim=[5,8]
# ax.set_ylim=[0,1]
ax.set_ylim([0,1])
plt.xticks(fontsize= 14)
plt.yticks(fontsize= 14)
plt.legend()

plt.show()

print("finished")
