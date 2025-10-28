import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv


# add noise to the input feature
class Distill_MOE_noise(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, teacher_logit, 
                 teacher_embedding, graph, device, lambda1=0.5, lambda2=0.3):
        super(Distill_MOE_noise, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.graph = graph
        self.device = device

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
        self.final_layer = nn.ModuleList()
        for _ in range(3):
            self.final_layer.append(nn.Linear(hidden_size, output_size))
        self.batch_norm = torch.nn.BatchNorm1d(input_size)
        self.softmax = nn.Softmax(dim=1)
        self.teacher_logit = teacher_logit.detach()
        self.teacher_embedding = F.normalize(teacher_embedding, dim=1).detach()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss1 = nn.KLDivLoss(reduction="batchmean") # KL divergence loss
        self.loss2 = nn.CrossEntropyLoss()
        # self.loss3 = nn.KLDivLoss(reduction="batchmean")
    

    def homophilic_embedding(self, feature):
        self.graph.ndata['x'] = feature
        self.graph.update_all(fn.copy_src(src='x', out='m'), fn.mean(msg='m', out='h'))
        h = self.graph.ndata["h"].to(torch.float32)
        h = self.homophilic_MLP(h)
        return h

    def heterophilic_embedding(self, feature):
        self.graph.ndata['x'] = feature
        self.graph.update_all(fn.copy_src(src='x', out='m'), fn.mean(msg='m', out='h'))
        # self.graph.ndata["h"] = torch.div(self.graph.ndata["h"], self.graph.in_degrees().unsqueeze(1))
        h = self.graph.ndata["feature"] -  self.graph.ndata["h"]
        h = self.heterophilic_MLP(h)
        return h

    def gated_fusion(self, h_homophilic, h_heterophilic, h_single, k=3):
        feature = self.graph.ndata['feature']
        
        # confidence_ori = self.gated(h_single)
        confidence_ori = self.gated(feature)
        _, sort_idx = torch.sort(confidence_ori, dim=1, descending=True)
        sort_idx = sort_idx[:, :k]

        h_list = [h_homophilic, h_heterophilic, h_single]
        logit_list = []  # N * 3 * C
        for i, h in enumerate(h_list):
            h = self.final_layer[i](h)
            logit_single = self.softmax(h)
            logit_list.append(logit_single)
        logit_student = torch.stack(logit_list, dim=1)  # N * 3 * C
       
        # confidence_k = confidence.gather(1, sort_idx)  # N * k
        mask = torch.ones_like(confidence_ori, dtype=torch.float) * (-1e5) # N * 3
        mask.scatter_(1, sort_idx, 1)  # scatter 到第1维
        confidence_masked = confidence_ori * mask
        confidence = F.softmax(confidence_masked, dim=1)  # N * 3        
        
        logit_student = logit_student * confidence.unsqueeze(-1) # N * 3 * C
        logit_student = logit_student.sum(dim=1) # N * C

        return logit_student, confidence

    def auxilary_loss(self, confidence):
        penalty_each = torch.mean(confidence, dim=0)
        aux_loss = torch.sum(torch.abs(penalty_each - (1.0 / confidence.shape[1]) ) )
        # penalty = torch.abs(confidence - 1.0 / (confidence.shape[1]))
        # aux_loss = torch.mean(penalty)
        return aux_loss

    def forward(self, feature, label, train_nodes):
        with self.graph.local_scope():
            gaussian_noise = torch.normal(mean=0, std=0.1, size=feature.shape, device=self.device)
            h = self.batch_norm(feature)
            h_noise = h + gaussian_noise
            h_homophilic = self.homophilic_embedding(h_noise)
            h_heterophilic = self.heterophilic_embedding(h_noise)
            h_single = self.single_MLP(h_noise)
            logit_student, confidence = self.gated_fusion(h_homophilic, h_heterophilic, h_single)
            loss1 = self.loss1(self.teacher_logit, logit_student)
            loss2 = self.loss2(logit_student[train_nodes], label[train_nodes])
            confidence_penalty = self.auxilary_loss(confidence)
            loss = self.lambda1 * loss1 + (1 - self.lambda1) * loss2 + confidence_penalty * self.lambda2
            return logit_student, loss
        
    def eval(self, feature):
        with self.graph.local_scope():
            h = self.batch_norm(feature)
            h_homophilic = self.homophilic_embedding(h)
            h_heterophilic = self.heterophilic_embedding(h)
            h_single = self.single_MLP(h)
            logit_student= self.gated_fusion(h_homophilic, h_heterophilic, h_single)
            return logit_student

