import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.teacher_model import GAT
from model.student_moe import Distill_MOE_2
from model.student_spectral import Distill_MOE_noise
from utils.load_data import load_data
from sklearn.model_selection import train_test_split
from pygod.metric import eval_roc_auc
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torchsummary import summary


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="weibo")
    parser.add_argument("--num_layers_teacher", type=int, default=2)
    parser.add_argument("--num_layers_student", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_hops", type=int, default=3)
    parser.add_argument("--lambda1", type=float, default=0.8)
    parser.add_argument("--lambda2", type=float, default=0.3)
    return parser.parse_args()


def main(args):
    dataset = args.dataset
    num_layers_teacher = 4
    num_layers_student = 4
    num_heads = 4
    dropout = 0.5
    lr = 1e-3
    epochs = 100
    weight_decay = 5e-4
    hidden_size = 128
    norm = "batchnorm"
    activation = "prelu"
    device = "cuda"
    output_size = 2
    trials = 1
    print(os.getcwd())
    graph = load_data(dataset, device)
    graph.ndata["feature"] = graph.ndata["feat"].to(torch.float32)
    feature = graph.ndata["feature"].to(torch.float32)
    label = graph.ndata["label"].to(torch.int64)
    train_nodes, test_nodes = train_test_split(graph.nodes().cpu().numpy(), train_size=0.05, test_size=0.95, random_state=42) 
    train_nodes = torch.tensor(train_nodes).to(device)
    test_nodes = torch.tensor(test_nodes).to(device)
    teacher = GAT(feature.shape[1], hidden_size, output_size, num_layers_teacher, num_heads, dropout, norm, activation, device).to(device)
    teacher.load_state_dict(torch.load(f"./saved_model/{dataset}_teacher.pth"))
    teacher.eval()
    teacher_logits = teacher.get_logits(graph, feature)
    auc = eval_roc_auc(label[test_nodes].cpu().detach(), teacher_logits[test_nodes][:, 1].detach().cpu())
    print(f"Teacher AUC: {auc}")
    teacher_embedding = teacher.get_embedding(graph, feature)
    auc_list, acc_list, time_list = [], [], []
    for trial in range(trials):
        distill = Distill_MOE_2(feature.shape[1], hidden_size, output_size, num_layers_student, teacher_logits, teacher_embedding, graph, device, args.lambda1, args.lambda2).to(device)
        # distill = Distill_MOE_noise(feature.shape[1], hidden_size, output_size, num_layers_student, teacher_logits, teacher_embedding, graph, device, args.lambda1, args.lambda2).to(device)
        # distill = Distill_MOE_3(feature.shape[1], hidden_size, output_size, num_layers_student, teacher_logits, teacher_embedding, graph, device, args.lambda1, args.lambda2).to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, distill.parameters()), lr=lr, weight_decay=weight_decay)
        epoch_iter = tqdm(range(epochs), desc=f"Trial {trial}")
        for epoch in epoch_iter:
            student_logits, loss = distill(feature, label, train_nodes)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iter.set_postfix(loss=loss.item())
        distill.eval(feature)
        start = time.time()
        student_logits = distill.eval(feature)
        end = time.time()
        auc = eval_roc_auc(label[test_nodes].cpu().detach(), student_logits[test_nodes][:, 1].detach().cpu())
        accuarcy = (torch.argmax(student_logits[test_nodes], dim=1) == label[test_nodes]).sum().item() / len(test_nodes)
        print(f"Trial {trial}, AUC: {auc}") 
        auc_list.append(auc)
        acc_list.append(accuarcy)
        time_list.append(end - start)
    print(f"AUC: {np.mean(auc_list)}, ACC: {np.mean(acc_list)}, time: {np.mean(time_list)}")
    torch.save(distill.state_dict(), f"./saved_model/{dataset}_student.pth")
    f = open(f"./results/{dataset}_student.txt", "a+")
    
    f.write(f"AUC: {np.mean(auc_list)} +- {np.std(auc_list)}, ACC: {np.mean(acc_list)} +- {np.std(acc_list)},  time: {np.mean(time_list)}\n")
    f.close()


if __name__ == "__main__":
    args = parse_args()

    datasets = ['hetero_dataset/roman_empire.dgl']  
    #'questions', 'tfinance' 'weibo', 'elliptic', 'reddit' 'tolokers' 'hetero_dataset/roman_empire.dgl' 'hetero_dataset/questions.dgl'
    for dataset in datasets:
        args.dataset = dataset
        main(args)
    # main(args)