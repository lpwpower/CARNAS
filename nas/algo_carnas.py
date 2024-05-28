# finished sorting
import csv
import os
import torch
# from torch_geometric.data import Data
from tqdm import trange

from autogllight.nas.estimator.base import BaseEstimator
from autogllight.nas.space import BaseSpace
from autogllight.nas.algorithm.base import BaseNAS


class Carnas(BaseNAS):
    """
    Carnas trainer.

    Parameters
    ----------
    num_epochs : int
        Number of epochs planned for training.
    device : str or torch.device
        The device of the whole process
    """

    def __init__(
        self,
        num_epochs=250,
        device="auto",
        disable_progress=False,
        args=None,
    ):
        super().__init__(device=device)
        self.num_epochs = num_epochs
        self.disable_progress = disable_progress
        self.args = args

    def train_graph(
        self,
        model_optimizer,
        arch_optimizer,
        gnn0_optimizer,
        causal_optimizer,
        eta,
    ):
        self.space.train()

        for id, train_data in enumerate(self.train_loader):
            model_optimizer.zero_grad()
            arch_optimizer.zero_grad()
            gnn0_optimizer.zero_grad()
            causal_optimizer.zero_grad()
            train_data = train_data.to(self.device)

            output, cosloss, varloss, output0 = self.space(train_data)
            # output0, output, cosloss, sslout = self.space(train_data)
            # output0 = output0.to(self.device)
            output = output.to(self.device)

            is_labeled = train_data.y == train_data.y
            
            # print("self.space.criterion",self.space.criterion)
            # print("output0",output0.shape,"train_data.y",train_data.y.shape)
            if 'SPMotif' in self.args.data:
                if not self.args.remove_error0:
                    error_loss0 = self.space.criterion(
                        output0[is_labeled],
                        train_data.y[is_labeled],
                    )
                error_loss = self.space.criterion(
                    output.to(torch.float32)[is_labeled],
                    train_data.y.to(torch.long)[is_labeled], # .reshape(output.shape)
                )
            else:
                if not self.args.remove_error0:
                    error_loss0 = self.space.criterion(
                        output0.to(torch.float32)[is_labeled],
                        train_data.y.to(torch.float32)[is_labeled],
                    )
                error_loss = self.space.criterion(
                    output.to(torch.float32)[is_labeled],
                    train_data.y.to(torch.float32)[is_labeled], # .reshape(output.shape)
                )
            
            if not self.args.remove_error0 and not self.args.remove_varloss:
                my_loss = (1 - eta) * (
                    error_loss0 + self.args.gamma * varloss + self.args.beta * cosloss
                    ) + eta * error_loss
            elif self.args.remove_error0 and not self.args.remove_varloss:
                my_loss = (1 - eta) * (
                    self.args.gamma * varloss + self.args.beta * cosloss
                    ) + eta * error_loss
            elif not self.args.remove_error0 and self.args.remove_varloss:
                my_loss = (1 - eta) * (
                    error_loss0 + self.args.beta * cosloss
                    ) + eta * error_loss
            elif self.args.remove_error0 and self.args.remove_varloss:
                my_loss = (1 - eta) * (
                    self.args.beta * cosloss
                    ) + eta * error_loss
                

            my_loss.backward()
            model_optimizer.step()
            gnn0_optimizer.step()
            causal_optimizer.step()
            arch_optimizer.step()

    def _infer(self, mask="train"):
        if mask == "train":
            dataloader = self.train_loader
        elif mask == "val":
            dataloader = self.val_loader
        elif mask == "test":
            dataloader = self.test_loader
        metric, loss = self.estimator.infer(self.space, dataloader)
        return metric, loss

    def prepare(self, data):
        """
        data : list of data objects.
            [dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader]
        """
        self.train_loader = data[4]
        self.val_loader = data[5]
        self.test_loader = data[6]

    def fit(self, record_file):
        causal_optimizer = torch.optim.Adam(
            self.space.causalnet.parameters(),
            self.args.causal_learning_rate,
            weight_decay=self.args.causal_weight_decay,
        )
        optimizer = torch.optim.Adam(
            self.space.supernet.parameters(),
            self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        arch_optimizer = torch.optim.Adam(
            self.space.ag.parameters(),
            self.args.arch_learning_rate,
            weight_decay=self.args.arch_weight_decay,
        )
        gnn0_optimizer = torch.optim.Adam(
            # self.space.graphembnet.parameters(),
            self.space.supernet0.parameters(),
            self.args.gnn0_learning_rate,
            weight_decay=self.args.gnn0_weight_decay,
        )
        scheduler_causal = torch.optim.lr_scheduler.CosineAnnealingLR(
            causal_optimizer,
            float(self.num_epochs),
            eta_min=self.args.causal_learning_rate_min,
        )
        scheduler_arch = torch.optim.lr_scheduler.CosineAnnealingLR(
            arch_optimizer,
            float(self.num_epochs),
            eta_min=self.args.arch_learning_rate_min,
        )
        scheduler_gnn0 = torch.optim.lr_scheduler.CosineAnnealingLR(
            gnn0_optimizer,
            float(self.num_epochs),
            eta_min=self.args.gnn0_learning_rate_min,
        )

        # self.criterion = torch.nn.BCEWithLogitsLoss()
        eta = self.args.eta
        best_performance = 0
        min_val_loss = float("inf")

        with trange(self.num_epochs, disable=self.disable_progress) as bar:
            for epoch in bar:
                """
                space training
                """
                self.space.train()
                eta = (
                    self.args.eta_max - self.args.eta
                ) * epoch / self.num_epochs + self.args.eta
                optimizer.zero_grad()
                arch_optimizer.zero_grad()
                gnn0_optimizer.zero_grad()
                causal_optimizer.zero_grad()

                self.train_graph(
                    optimizer,
                    arch_optimizer,
                    gnn0_optimizer,
                    causal_optimizer,
                    eta,
                )
                scheduler_arch.step()
                scheduler_gnn0.step()
                scheduler_causal.step()

                """
                space evaluation
                """
                self.space.eval()
                train_metric, train_loss = self._infer("train")
                val_metric, val_loss = self._infer("val")
                test_metric, test_loss = self._infer("test")

                if min_val_loss > val_loss:
                    min_val_loss, best_performance = val_loss, val_metric[self.args.eval]
                    self.space.keep_prediction()

                bar.set_postfix(
                    {
                        "train_auc": train_metric[self.args.eval],
                        "val_auc": val_metric[self.args.eval],
                        "test_auc": test_metric[self.args.eval],
                    }
                )
                
                if record_file!=None:
                    with open(record_file, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch, val_loss, train_metric[self.args.eval], val_metric[self.args.eval], test_metric[self.args.eval]])

        return best_performance, min_val_loss, train_metric[self.args.eval], val_metric[self.args.eval],test_metric[self.args.eval]

    def search(self, space: BaseSpace, dataset, estimator: BaseEstimator, record_file=None):
        self.estimator = estimator
        self.space = space.to(self.device)
        self.prepare(dataset)
        perf, val_loss, trainauc, valauc, testauc  = self.fit(record_file)
        return perf, val_loss, trainauc, valauc, testauc, space.parse_model(None)
