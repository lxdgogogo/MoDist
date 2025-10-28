import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv


class Distill_MOE_conf(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, teacher_logit,
                 device, lambda1=0.5, lambda2=0.3, tau=1):
        super(Distill_MOE_conf, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.dropout = 0.5
        self.tau = tau

        # single MLP
        self.single_MLP = nn.ModuleList()
        self.single_MLP.append(nn.Linear(input_size, hidden_size))
        self.single_MLP.append(torch.nn.ReLU())
        for _ in range(num_layers):
            self.single_MLP.append(nn.Linear(hidden_size, hidden_size))
            self.single_MLP.append(torch.nn.ReLU())
        self.single_MLP.append(nn.BatchNorm1d(hidden_size))
        self.single_MLP = nn.Sequential(*self.single_MLP)

        # homophilic MLP
        self.homophilic_MLP = nn.ModuleList()
        self.homophilic_MLP.append(nn.Linear(input_size, hidden_size))
        self.homophilic_MLP.append(torch.nn.ReLU())
        for _ in range(num_layers):
            self.homophilic_MLP.append(nn.Linear(hidden_size, hidden_size))
            self.homophilic_MLP.append(torch.nn.ReLU())
        self.homophilic_MLP.append(nn.BatchNorm1d(hidden_size))
        self.homophilic_MLP = nn.Sequential(*self.homophilic_MLP)

        # heterophilic MLP
        self.heterophilic_MLP = nn.ModuleList()
        self.heterophilic_MLP.append(nn.Linear(input_size, hidden_size))
        self.heterophilic_MLP.append(torch.nn.ReLU())
        for _ in range(num_layers):
            self.heterophilic_MLP.append(nn.Linear(hidden_size, hidden_size))
            self.heterophilic_MLP.append(torch.nn.ReLU())
        self.heterophilic_MLP.append(nn.BatchNorm1d(hidden_size))
        self.heterophilic_MLP = nn.Sequential(*self.heterophilic_MLP)

        self.gated = nn.Sequential(nn.Linear(input_size, 3), nn.Softmax(dim=1))
        self.final_layer = nn.Linear(hidden_size, output_size)
        self.batch_norm = torch.nn.BatchNorm1d(input_size)
        self.softmax = nn.Softmax(dim=1)
        self.teacher_logit = teacher_logit.detach()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss1 = nn.KLDivLoss(reduction="batchmean")  # KL divergence loss
        self.loss2 = nn.CrossEntropyLoss()

    def homophilic_embedding(self, graph, feature):
        graph.ndata['x'] = feature
        graph.update_all(fn.copy_src(src='x', out='m'), fn.mean(msg='m', out='h'))
        h = graph.ndata["h"].to(torch.float32)
        h = self.homophilic_MLP(h)
        return h

    def heterophilic_embedding(self, graph, feature):
        graph.ndata['x'] = feature
        graph.update_all(fn.copy_src(src='x', out='m'), fn.mean(msg='m', out='h'))
        # graph.ndata["h"] = torch.div(graph.ndata["h"], graph.in_degrees().unsqueeze(1))
        h = graph.ndata["feature"] - graph.ndata["h"]
        h = self.heterophilic_MLP(h)
        return h

    def gated_fusion(self, graph, h_homophilic, h_heterophilic, h_single, k=2):
        feature = graph.ndata['feature']
        confidence_ori = self.gated(feature)
        _, sort_idx = torch.sort(confidence_ori, dim=1, descending=True)
        sort_idx = sort_idx[:, :k]
        embedding_student = torch.stack([h_homophilic, h_heterophilic, h_single], dim=1)  # N * 3 * D
        mask = torch.ones_like(confidence_ori, dtype=torch.float) * (-1e5)  # N * 3
        mask.scatter_(1, sort_idx, 1)
        confidence_masked = confidence_ori * mask
        confidence = F.softmax(confidence_masked, dim=1)  # N * 3

        embedding_student = embedding_student * confidence.unsqueeze(-1)  # N * 3 * D
        embedding_student = embedding_student.sum(dim=1)  # N * D

        return embedding_student, confidence

    def auxilary_loss(self, confidence):
        penalty_each = torch.mean(confidence, dim=0)
        aux_loss = torch.sum(torch.abs(penalty_each - (1.0 / confidence.shape[1])))
        return aux_loss

    def forward(self, graph, feature, label, train_nodes, train=True):
        h = self.batch_norm(feature)
        h_homophilic = self.homophilic_embedding(graph, h)
        h_heterophilic = self.heterophilic_embedding(graph, h)
        h_single = self.single_MLP(h)
        embedding_student, confidence = self.gated_fusion(graph, h_homophilic, h_heterophilic, h_single)
        logit_student = self.final_layer(embedding_student)
        logit_student = self.softmax(logit_student)
        if not train:
            confidence_1 = confidence.clone()
            confidence_1[confidence != 0] = 1
            confidence_1[confidence == 0] = 0
            return logit_student, 0
        loss1 = self.loss1(self.teacher_logit / self.tau, logit_student / self.tau)
        loss2 = self.loss2(logit_student[train_nodes], label[train_nodes])
        confidence_penalty = self.auxilary_loss(confidence)
        loss = self.lambda1 * loss1 + (1 - self.lambda1) * loss2 + confidence_penalty * self.lambda2
        return logit_student, loss
