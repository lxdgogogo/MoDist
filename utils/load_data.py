import dgl
import torch


def load_data(dataset, device):
    file_path = f"./dataset/{dataset}.dgl"
    graph = dgl.load_graphs(file_path)[0][0].to(device)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    return graph
