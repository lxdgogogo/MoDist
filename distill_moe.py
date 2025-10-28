import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.teacher_model import GAT
from model.student_moe import Distill_MOE_conf
from utils.load_data import load_data
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import dgl
from utils.eval_model import eval_acc, eval_f1

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--num_layers_teacher", type=int, default=2)
    parser.add_argument("--num_layers_student", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--lambda1", type=float, default=0.3)
    parser.add_argument("--lambda2", type=float, default=0.4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--file_name", type=str, default='')
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.6)
    return parser.parse_args()


def main(args):
    dataset = args.dataset
    num_layers_teacher = args.num_layers_teacher
    num_layers_student = args.num_layers_student
    num_heads = args.num_heads
    dropout = 0.6
    lr = 1e-3
    epochs = 100
    weight_decay = 5e-4
    hidden_size = args.hidden_size
    norm = "batchnorm"
    activation = "prelu"
    device = "cuda"
    trials = 1
    tau = 1
    if args.file_name != '':
        file_name = args.file_name
    else:
        file_name = f"./results/{dataset}_student.txt"
    print(os.getcwd())
    graph: dgl.DGLGraph = load_data(dataset, device)
    if "feature" not in graph.ndata:
        graph.ndata["feature"] = graph.ndata["feat"].to(torch.float32)
    feature = graph.ndata["feature"].to(torch.float32)
    label = graph.ndata["label"].to(torch.int64)
    output_size = torch.unique(label).shape[0]
    if "train_mask" in graph.ndata:
        train_nodes = graph.ndata['train_mask'].nonzero(as_tuple=True)[0]
        val_nodes = graph.ndata['val_mask'].nonzero(as_tuple=True)[0]
        test_nodes = graph.ndata['test_mask'].nonzero(as_tuple=True)[0]
    else:
        train_nodes, test_nodes = train_test_split(graph.nodes().cpu().numpy(), train_size=0.05, test_size=0.95, random_state=42)
        val_nodes, test_nodes = train_test_split(test_nodes, train_size=0.4, test_size=0.6, random_state=42)
        train_nodes = torch.tensor(train_nodes).to(device)
        val_nodes = torch.tensor(val_nodes).to(device)
        test_nodes = torch.tensor(test_nodes).to(device)

    teacher = GAT(feature.shape[1], hidden_size, output_size, num_layers_teacher, dropout, F.relu, num_heads).to(device)
    teacher.load_state_dict(torch.load(f"./saved_model/{dataset}_teacher.pth"))
    teacher.eval()
    teacher_logits = teacher.get_logits(graph, feature)
    acc_teacher = eval_acc(teacher_logits, label, test_nodes)
    f1 = eval_f1(teacher_logits, label, test_nodes)
    print(f"Teacher F1-macro: {f1}")
    print(f"Teacher ACC: {acc_teacher}")
    acc_list, time_list, f1_list = [], [], []
    for trial in range(trials):
        distill = Distill_MOE_conf(feature.shape[1], hidden_size, output_size, num_layers_student, teacher_logits,
                                   device, args.lambda1, args.lambda2, tau).to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, distill.parameters()), lr=lr, weight_decay=weight_decay)
        epoch_iter = tqdm(range(epochs), desc=f"Trial {trial}")
        best_val_acc = 0
        best_test_acc = 0
        best_test_f1 = 0
        for epoch in epoch_iter:
            student_logits, loss = distill(graph, feature, label, train_nodes)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            val_acc = eval_acc(student_logits, label, val_nodes)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                state = copy.deepcopy(distill.state_dict())
            epoch_iter.set_postfix(loss=loss.item(), val_acc=val_acc)
        print(f"Best val acc: {best_val_acc}, best test acc: {best_test_acc}")
        distill.load_state_dict(state)
        start = time.time()
        distill.eval()
        student_logits, _ = distill(graph, feature, label, train_nodes, train=False)
        end = time.time()
        test_acc = eval_acc(student_logits, label, test_nodes)
        print(f"Trial {trial}, ACC: {test_acc}, F1-macro: {best_test_f1}")
        acc_list.append(best_test_acc)
        time_list.append(end - start)
        f1_list.append(best_test_f1)
    f = open(file_name, "a+")
    f.write(f"ACC: {np.mean(acc_list)} ± {np.std(acc_list)}, , F1-macro: {np.mean(f1_list)}, time: {np.mean(time_list)}\n")
    f.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
