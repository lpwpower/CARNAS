# finished sorting
import random
from autogllight.nas.space import BaseSpace
import torch
from .carnas_space import *
# from torch_geometric.data import Data, Dataset, InMemoryDataset, Batch

def find_sequence_indices(numbers, target):
    start_index = -1
    end_index = -1
    for i, num in enumerate(numbers):
        if num == target:
            if start_index == -1:
                start_index = i
            end_index = i
        elif start_index != -1:
            break
    return start_index, end_index

def intervene(causal_g, conf_g, interv):
    start, end = find_sequence_indices(conf_g[4], interv)
    batchsize = conf_g[4][len(conf_g[4])-1]
    num_attr = len(causal_g) # (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch)
    interv_g = [] # torch.tensor([]).to(causal_g[0].device)
    for j in range(num_attr):
        interv_g.append([])
        intervention = conf_g[j][start:end]
        for num in range(batchsize): # add intervention for each graph in batch
            causal_start,causal_end = find_sequence_indices(causal_g[4],num)
            interv_g[j].extend(causal_g[j][causal_start:causal_end])
            interv_g[j].extend(intervention)
        print("interv finished:",j)
    return interv_g


class CarnasSpace(BaseSpace):
    def __init__(self, input_dim, output_dim, num_nodes, mol, virtual, criterion, args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.mol = mol
        self.virtual = virtual
        self.criterion = criterion
        self.args = args
        self.use_forward = True

    def build_graph(self):
        self.causalnet = CausalAttNet(
            causal_ratio=self.args.causal_ratio, 
            in_channels=self.input_dim, 
            med_channels=self.args.hidden_size, # 32,
            use_causal_x=self.args.use_causal_x,
            ) #lpw
        if self.args.use_causal_x == True:
            self.supernet0 = GEncoder(
                criterion=self.criterion,
                in_dim=self.args.hidden_size, #lpw
                out_dim=self.output_dim,
                hidden_size=self.args.graph_dim,
                num_layers=2,
                dropout=0.5,
                epsilon=self.args.epsilon,
                args=self.args,
                with_conv_linear=self.args.with_conv_linear,
                num_nodes=self.num_nodes,
                mol=self.mol,
                virtual=self.virtual,
            )
        else:
            # print('self.args.use_causal_x',self.args.use_causal_x)
            self.supernet0 = GEncoder(
                criterion=self.criterion,
                in_dim=self.input_dim, #lpw
                out_dim=self.output_dim,
                hidden_size=self.args.graph_dim,
                num_layers=2,
                dropout=0.5,
                epsilon=self.args.epsilon,
                args=self.args,
                with_conv_linear=self.args.with_conv_linear,
                num_nodes=self.num_nodes,
                mol=self.mol,
                virtual=self.virtual,
            )
        self.supernet = Network(
            criterion=self.criterion,
            in_dim=self.input_dim, #lpw
            out_dim=self.output_dim,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout,
            epsilon=self.args.epsilon,
            args=self.args,
            with_conv_linear=self.args.with_conv_linear,
            num_nodes=self.num_nodes,
            mol=self.mol,
            virtual=self.virtual,
        )
        num_na_ops = len(NA_PRIMITIVES)
        num_pool_ops = len(POOL_PRIMITIVES)
        self.ag = AG(args=self.args, num_op=num_na_ops, num_pool=num_pool_ops)
        self.explore_num = 0


    def forward(self, data): # data has batchsize=512 graphs
        if not self.use_forward:
            return self.prediction
        
        if not self.args.remove_causalnet:
            self.causalnet.train()
            causal_g, conf_g, edge_score = self.causalnet(data)
            pred0, causal_graph_emb = self.supernet0(causal_g, mode="mixed")
            _, conf_graph_emb= self.supernet0(conf_g, mode="mixed")

            # ag
            causal_graph_alpha, cosloss = self.ag(causal_graph_emb)

            # interv varloss
            if not self.args.remove_varloss:
                varloss = torch.tensor(0,dtype=float).to(causal_g[0].device)
                batchsize = causal_graph_emb.shape[0]
                interv_size = conf_graph_emb.shape[0]
                sample_list = random.sample(range(interv_size),k = self.args.num_sampled_intervs)
                rand_conf_graph_emb = conf_graph_emb[sample_list,:]
                for i in range(batchsize):
                    interv_graph_emb = (1 - self.args.interv_ratio) * causal_graph_emb[i] + self.args.interv_ratio * rand_conf_graph_emb
                    # interv_graph_emb = (causal_graph_emb[i] + conf_graph_emb) / 2
                    interv_alpha, _ = self.ag(interv_graph_emb)
                    interv_alpha = torch.stack(interv_alpha,dim=1)
                    varloss += torch.sum(torch.var(interv_alpha,dim=0)) #torch.mean(torch.var(interv_alpha,dim=0))
                varloss = varloss / batchsize
            else:
                varloss = 0

            # final supernet
            if not self.args.remove_causalg_search:
                pred, _ = self.supernet(causal_g, mode="mads", graph_alpha=causal_graph_alpha)
            else:
                pred, _ = self.supernet(data, mode="mads", graph_alpha=causal_graph_alpha)

        else:
            pred0, graph_emb = self.supernet0(data, mode="mixed")
            graph_alpha, cosloss = self.ag(graph_emb)
            pred, _ = self.supernet(data, mode="mads", graph_alpha=graph_alpha)
            varloss = 0

        self.current_pred = pred
        return pred, cosloss, varloss, pred0 #, sslout

    def keep_prediction(self):
        self.prediction = self.current_pred

    def parse_model(self, selection):
        self.use_forward = False
        return self.wrap()
