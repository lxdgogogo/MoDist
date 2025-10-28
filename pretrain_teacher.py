import copy
import time
import torch
import torch.nn.functional as F
from model.teacher_model import GAT, GraphSAGE
from utils.eval_model import eval_acc
from utils.load_data import load_data
from sklearn.model_selection import train_test_split
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon_ratings")
    parser.add_argument("--num_layers_teacher", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main(args):
    dataset = args.dataset
    num_layers = args.num_layers_teacher
    num_heads = args.num_heads
    dropout = args.dropout
    hidden_size = args.hidden_size
    lr = args.lr
    epochs = args.epochs
    weight_decay = args.weight_decay
    device = args.device

    print(os.getcwd())
    graph = load_data(dataset, device)
    label = graph.ndata["label"].to(torch.int64)
    if "feature" not in graph.ndata:
        graph.ndata["feature"] = graph.ndata["feat"].to(torch.float32)
    feature = graph.ndata["feature"].to(torch.float32)
    output_size = torch.unique(label).shape[0]
    if "train_mask" in graph.ndata:
        train_nodes = graph.ndata['train_mask'].nonzero(as_tuple=True)[0]
        val_nodes = graph.ndata['val_mask'].nonzero(as_tuple=True)[0]
        test_nodes = graph.ndata['test_mask'].nonzero(as_tuple=True)[0]
    else:
        train_nodes, test_nodes = train_test_split(graph.nodes().cpu().numpy(), train_size=0.05, test_size=0.95,
                                                   random_state=42)
        val_nodes, test_nodes = train_test_split(test_nodes, train_size=0.4, test_size=0.6, random_state=42)
        train_nodes = torch.tensor(train_nodes).to(device)
        val_nodes = torch.tensor(val_nodes).to(device)
        test_nodes = torch.tensor(test_nodes).to(device)
        print(f"Train nodes: {len(train_nodes)}, Val nodes: {len(val_nodes)}, Test nodes: {len(test_nodes)}")
    teacher = GAT(feature.shape[1], hidden_size, output_size, num_layers, dropout, F.relu, num_heads).to(device)
    print(feature.shape[1], hidden_size, output_size, num_layers, dropout, F.relu, num_heads)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(epochs):
        teacher.train()
        embedding, loss = teacher(graph, feature, label, train_nodes)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        teacher.eval()
        with torch.no_grad():
            embedding = teacher.get_logits(graph, feature)
        acc = eval_acc(embedding, label, val_nodes)
        if acc > best_val_acc:
            best_val_acc = acc
            best_test_acc = eval_acc(embedding, label, test_nodes)
            state = copy.deepcopy(teacher.state_dict())
    print(f"Best val acc: {best_val_acc}, best test acc: {best_test_acc}")
    teacher.load_state_dict(state)
    teacher.eval()
    start = time.time()
    logits = teacher.get_logits(graph, feature)
    acc = eval_acc(logits, label, test_nodes)
    end = time.time()
    torch.save(state, f"./saved_model/{dataset}_teacher.pth")
    f = open(f"./results/{dataset}_teacher.txt", "a+")
    f.write(f"Acc: {acc}, time: {end - start}\n")
    f.close()
    print("acc:", acc, best_test_acc)


if __name__ == "__main__":
    args = parse_args()
    main(args)