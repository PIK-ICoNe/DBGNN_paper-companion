from pathlib import Path
from sys import argv
import math

import ray as ray
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper
from ray import tune, train
from ray.air import CheckpointConfig 
from ray_trainable import NN_tune_trainable, nan_stopper

dataset_path = Path(PATHMISSING)
dataset_path = Path(PATHMISSING)
tmp_dir = "TMP_PATH"
N_cpus = int(argv[1])
port_dashboard = int(argv[2])
tmp_dir = tmp_dir + argv[3]
ray.init(_temp_dir=tmp_dir,num_cpus=N_cpus, num_gpus = 2, include_dashboard=True,dashboard_port=port_dashboard)

cfg = {}

# dataset
cfg[dataset_path = Path(PATHMISSING)
# cfg["dataset::ieee_path"] = ieee_path
cfg["dataset::name"] = "lrgb::Peptides-struct"
cfg["input_features_node_dim"] = 9
cfg["input_features_edge_dim"] = 3
cfg["output_features_node_dim"] = 11
cfg["num_classes"] = 11
cfg["pool"] = "mean"

cfg["task_type"] = "regression"
cfg["criterion"] = "MSELoss"
metric = "valid_MAE"
metric_mode = "min"
metric_checkpoint = metric
# cfg["list_metrics"] = ["accuracy", "f1", "fbeta", "recall", "precision"]
# cfg["list_metrics"] = ["accuracy". "f1", "fbeta", "recall", "precision", "auroc"]
# cfg["list_metrics"] = ["r2"]
cfg["list_metrics"] = ["MAE"]

## datset snbs_homog
#cfg["snbs_homo::train_set::start_index"] = 0
#cfg["snbs_homo::train_set::end_index"] = 799

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
cfg["train_set::batchsize"] = 60 #800
cfg["test_set::batchsize"] = 500
cfg["valid_set::batchsize"] = 500
cfg["train_set::shuffle"] = True
cfg["test_set::shuffle"] = False
cfg["valid_set::shuffle"] = False






# model settings
cfg["model_name"] = "DBGNN"
cfg["num_layers"] = 2
cfg["final_linear_layer"] = False#True
cfg["final_sigmoid_layer"] = False
cfg["bias_zero"] = True
cfg["activation_name_n"] = "LeakyReLU"#"ReLU"
cfg["activation_name_e"] = "LeakyReLU"#"ReLU"

cfg["dropout_n"] = 0.05543539652343399#tune.loguniform(1E-2,.2)
cfg["dropout_e"] = 0.010589825864825495#tune.loguniform(1E-2, .2) 

cfg["activation"] = "ReLU"#["relu", "None"]

cfg["DBGNN::dense_after_linDB"] = False
cfg["DBGNN::in_channels_n"] = cfg["input_features_node_dim"] 
cfg["DBGNN::out_channels_n"] = cfg["output_features_node_dim"]
cfg["DBGNN::hidden_channels_n"] = 113 
cfg["DBGNN::in_channels_e"] = cfg["input_features_edge_dim"]
cfg["DBGNN::hidden_channels_e"] = 109 
cfg["DBGNN::num_steps"] = 68 
cfg["DBGNN::Δ"] = 0.0001#tune.loguniform(1E-5, 1E-1)
cfg["DBGNN::scale_features"] = False#tune.choice([True,False])

cfg["skip_connection_n"] = True 
cfg["skip_connection_e"] = True 

# training settings
cfg["cuda"] = True
#cfg["num_workers"] = 1
#cfg["num_threads"] = 2
# cfg["manual_seed"] = 1
cfg["manual_seed"] = tune.choice([1,2,3,4])
cfg["epochs"] = 2000
cfg["optim::optimizer"] = "adamW"#"SGD"
# cfg["optim::LR"] = 0.193048 # 1.1
# cfg["optim::LR"] = 0.3
cfg["optim::LR"] = 1E10# tune.loguniform(1e-9, 1e-4)
# cfg["optim::LR"] = tune.choice([1.1])
cfg["optim::momentum"] = .9
cfg["optim::weight_decay"] = 0.0563959925155956#tune.loguniform(1E-7, 1E-1)#1E-7#1.4552148412694113e-07#tune.loguniform(1E-9, 1E-1) 
cfg["optim::scheduler"] = "OneCycleLR" 
cfg["optim::ReducePlat_patience"] = 20
cfg["optim::LR_reduce_factor"] = .7
cfg["optim::stepLR_step_size"] = 30
cfg["optim::anneal_strategy"] = "cos"
cfg["optim::div_factor"] = 151#tune.randint(5,200)
cfg["optim::final_div_factor"] = 4216749345.13404#tune.loguniform(1E3, 1E12) 
cfg["optim::max_LR"] = 0.0054818584469448145#tune.loguniform(1E-4,1E1)
# cfg["optim::scheduler"] = "stepLR"
cfg["search_alg"] = "Optuna"


## gradient clipping
cfg["gradient_clipping::grad_norm"] = True
cfg["gradient_clipping::grad_norm::max_norm"] = 1
cfg["gradient_clipping::grad_value"] = True
cfg["gradient_clipping::grad_value::clip_value"] = 1

# evaluation
cfg["eval::threshold"] = .1


# ray settings
cfg["save_after_epochs"] = 10000
cfg["checkpoint_freq"] = 10000
cfg["num_samples"] = 4 
cfg["ray_name"] = "DBGNN"

asha_scheduler = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric=metric,
    mode=metric_mode,
    max_t=cfg["epochs"],
    grace_period=500,
    #reduction_factor=3,
    #brackets=5,
)

optuna_search = OptunaSearch(
    metric=metric,
    mode=metric_mode,
    points_to_evaluate=[{"manual_seed": 1}, {"manual_seed": 2}, {"manual_seed": 3}, {"manual_seed": 4}]
   # points_to_evaluate=[{"optim::div_factor": 25, "optim::max_LR": 1, "optim::final_div_factor": 1E6, "DBGNN:scale_fatures": True}]
   # points_to_evaluate=[{"dropout_n": 1E-2, "dropout_e": 1E-2, "optim::weight_decay": 1E-7}]

)
tune_stop = CombinedStopper(MaximumIterationStopper(max_iter=cfg["epochs"]), TrialPlateauStopper(metric=metric, num_results=100, std=0.0005, grace_period=150), nan_stopper())
checkpoint_freq = cfg["checkpoint_freq"]
num_samples = cfg["num_samples"]
name = cfg["ray_name"]

checkpoint_config = CheckpointConfig(
    num_to_keep=1, checkpoint_frequency = 1, checkpoint_score_attribute=metric, checkpoint_score_order=metric_mode, checkpoint_at_end= True,
)
analysis = tune.run(
    NN_tune_trainable,
    name=name,
    stop=tune_stop,
    config=cfg,
    num_samples=num_samples,
    storage_path=result_path,
    search_alg=optuna_search,
    #scheduler=asha_scheduler,
    checkpoint_config=checkpoint_config,
    resources_per_trial={'cpu': 1. ,'gpu': .5},
    max_failures=1,
    resume=False,
)



print('best config: ', analysis.get_best_config(metric=metric, mode=metric_mode))


ray.shutdown()
print("finished")
