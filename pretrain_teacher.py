import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.teacher_model import GAT
from utils.load_data import load_data
from sklearn.model_selection import train_test_split
from pygod.metric import eval_roc_auc
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="weibo")
    return parser.parse_args()


def main(args):
    dataset = args.dataset
    num_layers = 4
    num_heads = 4
    dropout = 0.5
    hidden_size = 128
    lr = 1e-3
    epochs = 100
    weight_decay = 5e-4
    norm = "batchnorm"
    activation = "prelu"
    device = "cuda"
    output_size = 2
    print(os.getcwd())
    graph = load_data(dataset, device)
    label = graph.ndata["label"].to(torch.int64)
    feature = graph.ndata["feat"].to(torch.float32)
    train_nodes, test_nodes = train_test_split(graph.nodes().cpu().numpy(), train_size=0.2, test_size=0.8, random_state=42) 
    train_nodes = torch.tensor(train_nodes).to(device)
    test_nodes = torch.tensor(test_nodes).to(device)
    teacher = GAT(feature.shape[1], hidden_size, output_size, num_layers, num_heads, dropout, norm, activation, device).to(device)
    time_list = []
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    start = time.time()
    for epoch in range(epochs):
        embedding, loss = teacher(graph, feature, label, train_nodes)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    start = time.time()
    teacher.eval()
    logits = teacher.get_logits(graph, feature)
    end = time.time()
    label = label[test_nodes].detach().cpu()
    logits_anomaly = logits[test_nodes][:, 1].detach().cpu() 
    score = (logits_anomaly - torch.min(logits_anomaly)) / (torch.max(logits_anomaly) - torch.min(logits_anomaly))
    auc = eval_roc_auc(label, score)
    acc = (torch.argmax(logits[test_nodes].detach().cpu(), dim=1) == label).sum().item() / len(label)
    f = open(f"./results/{dataset}_teacher.txt", "a+")
    f.write(f"AUC: {auc}, Acc: {acc}, time: {end - start}\n")
    f.close()
    torch.save(teacher.state_dict(), f"./saved_model/{dataset}_teacher.pth")
    

if __name__ == "__main__":
    args = parse_args()
    # datasets = ['reddit', 'tfinance', 'elliptic', 'questions', 'tolokers', 'hetero_dataset/questions.dgl' hetero_dataset/roman_empire.dgl]
    datasets = ['hetero_dataset/roman_empire.dgl']
    for dataset in datasets:
        args.dataset = dataset
        main(args)