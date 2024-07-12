from pathlib import Path
from sys import argv
import math

import ray as ray
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper
from ray import tune, train

from ray_trainable import NN_tune_trainable, nan_stopper



dataset_path = Path(PATHMISSING)
# dataset_path = Path(PATHMISSING)
ieee_path = Path("/Users/christiannauck/work/pik/data/dataset_v2/ieee")
dataset_path = Path(PATHMISSING)

temp_dir = Path("/Users/christiannauck/work/pik/ray_temp").as_posix()


# dataset_path = Path(PATHMISSING)
# dataset_path = Path(PATHMISSING)


# dataset_path = Path(PATHMISSING)
# dataset_path = Path(PATHMISSING)



# N_cpus = int(argv[1])
# port_dashboard = int(argv[2])
# ray.init(num_cpus=N_cpus, num_gpus = 1, include_dashboard=True,dashboard_port=port_dashboard)

ray.init(_temp_dir=temp_dir,num_cpus=2, num_gpus = 0)

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
cfg["task_type"] = "regression"#"classification"#
cfg["criterion"] = "MSELoss"#"CELoss"#"BCEWithLogitsLoss"#"CELoss"#"MSELoss"
metric = "valid_MAE"
metric_mode = "min"
metric_checkpoint = metric
# cfg["list_metrics"] = ["accuracy", "f1", "fbeta", "recall", "precision"]
# cfg["list_metrics"] = ["accuracy", "f1", "fbeta", "recall", "precision", "auroc"]
# cfg["list_metrics"] = ["r2"]
cfg["list_metrics"] = ["MAE"]

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
cfg["train_set::batchsize"] = 10
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
cfg["optim::LR"] = tune.loguniform(1e-4, 2e1)
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


asha_scheduler = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric=metric,
    mode=metric_mode,
    max_t=cfg["epochs"],
    grace_period=150,
    # reduction_factor=3,
    # brackets=5,
)

optuna_search = OptunaSearch(
    metric=metric,
    mode=metric_mode,
    # points_to_evaluate=[{"manual_seed": 1}, {"manual_seed": 2}, {"manual_seed": 3}, {"manual_seed": 4}, {"manual_seed": 5}]
    )

# tune_stop = CombinedStopper(MaximumIterationStopper(max_iter=cfg["epochs"]))

stop_nan = tune.Stopper()
tune_stop = CombinedStopper(MaximumIterationStopper(max_iter=cfg["epochs"]), TrialPlateauStopper(metric=metric, num_results=100, std=0.005, grace_period=150), nan_stopper())



checkpoint_freq = cfg["checkpoint_freq"]
num_samples = cfg["num_samples"]
name = cfg["ray_name"]


analysis = tune.run(
    NN_tune_trainable,
    name=name,
    stop=tune_stop,
    config=cfg,
    num_samples=num_samples,
    local_dir=result_path,
    search_alg=optuna_search,
    # checkpoint_freq=checkpoint_freq,
    keep_checkpoints_num=1,
    checkpoint_score_attr=metric,
    checkpoint_freq=1,
    checkpoint_at_end=True,
    resources_per_trial={'cpu': 1. ,'gpu': 0},
    max_failures=1,
    scheduler=asha_scheduler,
)

print('best config: ', analysis.get_best_config(metric=metric, mode="max"))


# ray.shutdown()
print("finished")
