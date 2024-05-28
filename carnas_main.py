import argparse
import csv
import os
import random
import sys
import time
import numpy as np

import torch
from autogllight.utils import set_seed
from autogllight.utils.evaluation import Auc, Acc

from torch_geometric.data import DataLoader

from nas.space_carnas import CarnasSpace
from nas.algo_carnas import Carnas
from nas.estimator import OneShotOGBEstimator, OneShotEstimator
from carnas_dataset import load_data

sys.path.append("..")
sys.path.append(".")
os.environ["AUTOGL_BACKEND"] = "pyg"


def parser_args():
    graph_classification_dataset = [
        "spmotif",
        "ogbg-molsider",
        "ogbg-molhiv",
        "ogbg-molbace",
    ]
    parser = argparse.ArgumentParser("pas-train-search")
    parser.add_argument(
        "--data", type=str, default="ogbg-molbace", help="location of the data corpus"
    )
    parser.add_argument(
        "--eval", type=str, default="auc", help="evaluation methods: SPMotif(acc), mol(auc)"
    )
    parser.add_argument(
        "--record_time",
        action="store_true",
        default=False,
        help="used for run_with_record_time func",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="init learning rate"
    )
    parser.add_argument(
        "--learning_rate_min", type=float, default=0.001, help="min learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--gpu", type=int, default=1, help="gpu device id")
    parser.add_argument(
        "--epochs", type=int, default=100, help="num of training epochs"
    )
    parser.add_argument(
        "--model_path", type=str, default="saved_models", help="path to save the model"
    )
    parser.add_argument("--save", type=str, default="EXP", help="experiment name")
    parser.add_argument(
        "--save_file", action="store_true", default=False, help="save the script"
    )
    parser.add_argument("--seed", type=int, default=2, help="random seed")
    parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="the explore rate in the gradient descent process",
    )
    parser.add_argument(
        "--train_portion", type=float, default=0.5, help="portion of training data"
    )
    parser.add_argument(
        "--unrolled",
        action="store_true",
        default=False,
        help="use one-step unrolled validation loss",
    )
    parser.add_argument(
        "--temperature", type=float, default=1, help="temperature of AGLayer"
    )
    parser.add_argument(
        "--arch_learning_rate",
        type=float,
        default=0.08,
        help="learning rate for arch encoding",
    )
    parser.add_argument(
        "--arch_learning_rate_min",
        type=float,
        default=0.0,
        help="minimum learning rate for arch encoding",
    )
    parser.add_argument(
        "--arch_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )
    parser.add_argument(
        "--gnn0_learning_rate",
        type=float,
        default=0.005,
        help="learning rate for gnn0 encoding",
    )
    parser.add_argument(
        "--gnn0_learning_rate_min",
        type=float,
        default=0.0,
        help="minimum learning rate for gnn0 encoding",
    )
    parser.add_argument(
        "--gnn0_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for gnn0 encoding",
    )
    parser.add_argument(
        "--causal_ratio",
        type=float,
        default=0.25,
        help="ratio for causal subgraph in causal encoding",
    )
    parser.add_argument(
        "--causal_learning_rate",
        type=float,
        default=0.005,
        help="learning rate for causal encoding",
    )
    parser.add_argument(
        "--causal_learning_rate_min",
        type=float,
        default=0.0,
        help="minimum learning rate for causal encoding",
    )
    parser.add_argument(
        "--causal_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for causal encoding",
    )
    parser.add_argument(
        "--num_sampled_intervs", type=int, default=10, help="num of sampled variant subgraphs as interventions"
    )
    parser.add_argument(
        "--pooling_ratio", type=float, default=0.5, help="global pooling ratio"
    )
    parser.add_argument("--beta", type=float, default=5e-3, help="global pooling ratio")
    parser.add_argument("--gamma", type=float, default=5.0, help="global pooling ratio")
    parser.add_argument("--eta", type=float, default=0.1, help="global pooling ratio")
    parser.add_argument(
        "--eta_max", type=float, default=0.5, help="global pooling ratio"
    )
    parser.add_argument(
        "--with_conv_linear",
        type=bool,
        default=False,
        help=" in NAMixOp with linear op",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="num of layers of GNN method."
    )
    parser.add_argument(
        "--withoutjk", action="store_true", default=False, help="remove la aggregtor"
    )
    parser.add_argument(
        "--search_act",
        action="store_true",
        default=False,
        help="search act in supernet.",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="default hidden_size in supernet"
    )
    parser.add_argument(
        "--BN", type=int, default=64, help="default hidden_size in supernet"
    )
    parser.add_argument(
        "--graph_dim", type=int, default=8, help="size of graph embedding results"
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.1,
        help="default hidden_size in supernet",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="default hidden_size in supernet"
    )
    parser.add_argument(
        "--num_sampled_archs", type=int, default=5, help="sample archs from supernet"
    )

    # for ablation stuty
    parser.add_argument(
        "--remove_varloss",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--remove_error0",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--remove_causalg_search",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--remove_causalnet",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_causal_x",
        action="store_true",
        default=False,
    )

    # in the stage of update theta.
    parser.add_argument(
        "--temp", type=float, default=0.2, help=" temperature in gumble softmax."
    )
    parser.add_argument(
        "--loc_mean",
        type=float,
        default=10.0,
        help="initial mean value to generate the location",
    )
    parser.add_argument(
        "--loc_std",
        type=float,
        default=0.01,
        help="initial std to generate the location",
    )
    parser.add_argument(
        "--lamda",
        type=int,
        default=2,
        help="sample lamda architectures in calculate natural policy gradient.",
    )
    parser.add_argument(
        "--adapt_delta",
        action="store_true",
        default=False,
        help="adaptive delta in update theta.",
    )
    parser.add_argument(
        "--delta", type=float, default=1.0, help="a fixed delta in update theta."
    )
    parser.add_argument(
        "--w_update_epoch", type=int, default=1, help="epoches in update W"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="darts",
        help="how to update alpha",
        choices=["mads", "darts", "snas"],
    )

    args = parser.parse_args()
    args.graph_classification_dataset = graph_classification_dataset
    torch.set_printoptions(precision=4)

    return args


    


def main(args):
    set_seed(args.seed)
    data, num_nodes = load_data(
        args.data, batch_size=args.batch_size, split_seed=args.seed
    )
    """
        data : list of data objects.
            [dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader]
        for SPMotif data[0] = train_dataset
    """

    num_features = data[0].num_features
    if 'SPMotif' in args.data:
        num_tasks = 1
    else:
        num_tasks = data[0].num_tasks
    num_classes = data[0].num_classes
    print("num_tasks",num_tasks,"num_classes",num_classes)

    if 'SPMotif' in args.data:
        estimator = OneShotOGBEstimator(
            loss_f="cross_entropy", evaluation=[Acc()]
        )
        args.eval = "acc"

        space = CarnasSpace(
        input_dim=num_features,
        output_dim=num_tasks,
        num_nodes=num_nodes,
        mol=False,  # true for ogbg (False, True)
        virtual=False,  # true for ogbg
        criterion=torch.nn.CrossEntropyLoss(),#torch.nn.BCEWithLogitsLoss(),  # for ogbg
        args=args,
        )

    else:
        estimator = OneShotOGBEstimator(
            loss_f="binary_cross_entropy_with_logits", evaluation=[Auc()]
        )
        args.eval = "auc"

        space = CarnasSpace(
            input_dim=num_features,
            output_dim=num_tasks,
            num_nodes=num_nodes,
            mol=True,  # true for ogbg (False, True)
            virtual=True,  # true for ogbg OGBG-MolHIV and OGBG-MolSIDER
            criterion=torch.nn.BCEWithLogitsLoss(),  # for ogbg
            args=args,
        )

    space.instantiate()
    algo = Carnas(num_epochs=args.epochs, args=args)
    perf, val_loss, trainauc, valauc, testauc, _ = algo.search(space, data, estimator)
    return val_loss, trainauc, valauc, testauc


if __name__ == "__main__":
    torch.cuda.set_device(5)
    other_hps = {

        "num_layers": 2, # 3,
        "batch_size": 512,
        "num_sampled_intervs": 20, # 128 for other, 16 for hiv
        "interv_ratio": 0.6, 
        "hidden_size": 128, # 64,
        "data": 'ogbg-molsider', #'ogbg-molhiv' 'ogbg-molbace' 'ogbg-molsider' 'SPMotif-0.7' 'SPMotif-0.8' 'SPMotif-0.9'
    }
    print(other_hps)
    args = parser_args()
    for k, v in other_hps.items():
        setattr(args, k, v)

    # hps = {'arch_learning_rate': 0.0006000687523802248, 
    #        'arch_weight_decay': 0.0024912502897298265, 
    #        'beta': 0.003,#0.003082730129094582, 
    #        'causal_learning_rate': 3.266242860604596e-05, 
    #        'causal_ratio': 0.494673329674935, 
    #        'causal_weight_decay': 0.0012185638705646145, 
    #        'dropout': 0.09355343834815459, 
    #        'eta': 0.1, #0.11375752369370261, 
    #        'eta_max': 0.7, #0.7001949585100775, 
    #        'gamma': 0.8, #0.8006187907131703, 
    #        'gnn0_learning_rate': 0.0006317080063955971, 
    #        'gnn0_weight_decay': 0.001746080542249526, 
    #        'interv_ratio': 0.5350571490803142, 
    #        'learning_rate': 0.0002147144898078729, 
    #        'learning_rate_min': 2.6409617989063522e-05, 
    #        'pooling_ratio': 0.9939881715473511, 
    #        'temperature': 2.660809182353149, 
    #        'weight_decay': 0.0009386781265614668}
    # hps = {'arch_learning_rate': 0.0006, 
    #        'arch_weight_decay': 0.0025, 
    #        'beta': 0.003082730129094582, 
    #        'causal_learning_rate': 3e-05, 
    #        'causal_ratio': 0.494673329674935, 
    #        'causal_weight_decay': 0.001, 
    #        'dropout': 0.09, 
    #        'eta': 0.1, #0.11375752369370261, 
    #        'eta_max': 0.7, #0.7001949585100775, 
    #        'gamma': 0.8006187907131703, 
    #        'gnn0_learning_rate': 0.0006, 
    #        'gnn0_weight_decay': 0.002, 
    #        'interv_ratio': 0.5350571490803142, 
    #        'learning_rate': 0.0002, 
    #        'learning_rate_min': 3e-05, 
    #        'pooling_ratio': 0.99, 
    #        'temperature': 2.7, 
    #        'weight_decay': 0.0009}
    hps = {
        "learning_rate": 0.0003,
        "learning_rate_min": 0.0002,
        "weight_decay": 0,
        "temperature": 4,
        "arch_learning_rate": 0.0004,
        "arch_weight_decay": 0.001,
        "gnn0_learning_rate": 0.03,
        "gnn0_weight_decay": 0,
        "causal_ratio": 0.4, #0.25,
        "causal_learning_rate": 0.03,
        "causal_weight_decay": 0,
        "pooling_ratio": 0.3,
        "dropout": 0.2,
        "beta": 0.005, # for cosloss
        "eta": 0.1,
        "eta_max": 0.7, 
        "gamma": 0.85, # for varloss
        }
    

    for k, v in hps.items():
        setattr(args, k, v)

    seeds_num = 5 # 10
    results_dir = "/villa/lpw/CARNAS/exp_sortcode/sider/interv_ratio/"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results-fg-mu07-{time.strftime('%d%H%M%S')}.csv")
    val_loss_res = []
    train_res = []
    val_res = []
    test_res = []

    with open(results_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Hyperparameters", str({**hps,**other_hps})])
        writer.writerow(["Seed", "Val Loss", "Train AUC", "Val AUC", "Test AUC"])

    for i in range(1, seeds_num+1):
        args.seed = random.randint(1, 10000)
        val_loss, trainauc, valauc, testauc = main(args)
        
        # Append results to the lists
        val_loss_res.append(val_loss)
        train_res.append(trainauc)
        val_res.append(valauc)
        test_res.append(testauc)

        # Write the results of this iteration to the CSV file
        with open(results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([args.seed, val_loss, trainauc, valauc, testauc])

    # Calculate mean and standard deviation
    men, st = np.mean(test_res), np.std(test_res)

    # Write the final results and hyperparameters to the CSV file
    with open(results_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Mean Test AUC", men, "STD Test AUC", st])
        # writer.writerow(["Hyperparameters", str(hps)])

    print(f"Results and hyperparameters saved to {results_file}")