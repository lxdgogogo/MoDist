import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

# define the teacher model as a GAT model
class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout, norm, activation, device):
        super(GAT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm = norm
        self.activation = torch.nn.ReLU()
        self.norm = torch.nn.BatchNorm1d(output_size)
        self.device = device
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats=input_size, out_feats=hidden_size, num_heads=num_heads, feat_drop=dropout, activation=self.activation))
        for _ in range(num_layers):
            self.layers.append(GATConv(in_feats=hidden_size * num_heads, out_feats=hidden_size, num_heads=num_heads, feat_drop=dropout, activation=self.activation))
        self.out = nn.Sequential(
                                nn.Linear(hidden_size * num_heads, hidden_size), 
                                self.activation
                                )
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, graph, feature, label, train_nodes):
        h = feature
        for layer in range(self.num_layers):
            h = self.layers[layer](graph, h)
            h = h.flatten(1)
        h = self.out(h)
        h = self.fc(h)
        h = self.norm(h)
        h = self.softmax(h)
        loss = F.cross_entropy(h[train_nodes], label[train_nodes])
        return h, loss
    
    def get_logits(self, graph, feature):
        h = feature
        for layer in range(self.num_layers):
            h = self.layers[layer](graph, h)
            h = h.flatten(1)
        h = self.out(h)
        h = self.fc(h)
        h = self.norm(h)
        h = self.softmax(h)
        return h

    def get_embedding(self, graph, feature):
        h = feature
        for layer in range(self.num_layers):
            h = self.layers[layer](graph, h)
            h = h.flatten(1)
        h = self.out(h)
        return h