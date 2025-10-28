import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, SAGEConv, GraphConv



class GAT(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            dropout_ratio,
            activation,
            num_heads=8,
            attn_drop=0.3,
            negative_slope=0.2,
            residual=False,
    ):
        super(GAT, self).__init__()
        # For GAT, the number of layers is required to be > 1
        assert num_layers > 1

        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation

        heads = ([num_heads] * num_layers) + [1]
        # input (no residual)
        self.layers.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads[0],
                dropout_ratio,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )

        for l in range(1, num_layers - 1):
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.layers.append(
                GATConv(
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )

        self.layers.append(
            GATConv(
                hidden_dim * heads[-2],
                output_dim,
                heads[-1],
                dropout_ratio,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

        self.final = nn.LogSoftmax(dim=1)

    def forward(self, g, feats, label, train_nodes):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            # [num_head, node_num, nclass] -> [num_head, node_num*nclass]
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
                h_list.append(h)
            else:
                h = h.mean(1)
        h = h.log_softmax(dim=1)
        loss = F.nll_loss(h[train_nodes], label[train_nodes])
        return h, loss

    def get_logits(self, g, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)
        h = self.final(h)
        return h


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_ratio, activation):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.activation = activation

        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, aggregator_type='gcn'))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, aggregator_type='gcn'))
            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type='gcn'))
            self.layers.append(SAGEConv(hidden_dim, output_dim, aggregator_type='gcn'))

    def forward(self, g, feats, label, train_nodes):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
            h_list.append(h)
        loss = F.cross_entropy(h[train_nodes], label[train_nodes])
        h = F.log_softmax(h, dim=1)
        return h, loss

    def get_logits(self, g, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
        h = F.log_softmax(h, dim=1)
        return h


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_ratio, activation):
        print("GCN model params: ")
        print(
            f"input_dim: {input_dim}, hidden_dim: {hidden_dim}, output_dim: {output_dim}, num_layers: {num_layers}, dropout_ratio: {dropout_ratio}")
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            for i in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats, label, train_nodes):
        h = feats
        h_list = []

        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.dropout(h)
            h_list.append(h)
        loss = F.cross_entropy(h[train_nodes], label[train_nodes])
        h = F.log_softmax(h, dim=1)
        return h, loss

    def get_logits(self, g, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
        h = F.log_softmax(h, dim=1)
        return h