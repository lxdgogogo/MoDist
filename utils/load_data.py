import dgl
import torch


def load_data(dataset, device):
    file_path = f"./dataset/{dataset}"
    graph = dgl.load_graphs(file_path)[0][0].to(device)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    num_anomaly = torch.sum(graph.ndata["label"] == 1).item()
    print("num_anomaly: ", num_anomaly)
    return graph
