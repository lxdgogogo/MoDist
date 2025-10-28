import torch
from sklearn.metrics import f1_score

def eval_acc(logit: torch.Tensor, label: torch.Tensor, mask: torch.Tensor):
    preds = logit.cpu().argmax(1)
    correct = (preds[mask.cpu()] == label.cpu()[mask.cpu()]).sum().item()
    return correct / len(label[mask])

def eval_f1(logit: torch.Tensor, label: torch.Tensor, mask: torch.Tensor):
    pred = torch.clone(logit[mask].detach())
    pred = torch.argmax(pred, dim=1)
    f1_macro = f1_score(label[mask].cpu().numpy(), pred.int().detach().cpu().numpy(), average='macro')
    return f1_macro