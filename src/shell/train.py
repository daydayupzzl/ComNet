import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from tabulate import tabulate
from src.tools.utils import AverageMeter, accuracy, evaluate
from src.tools.dataset import load_ddi_dataset
from src.models.ComNet import ComNet_DDI


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def initialize_model(device, node_dim, edge_dim, config):
    model_params = config['model']
    return ComNet_DDI(device, node_dim, edge_dim, hidden_dim=model_params['hidden_dim'],
                      dropout=model_params['dropout'], rmodule_dim=model_params['rmodule_dim']).to(device)


def initialize_optimizer(model, lr, weight_decay):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def log_metrics(epoch, train_loss, train_acc, val_metrics):
    headers = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Val AUROC", "Val F1 Score",
               "Val Precision", "Val Recall", "Val AP"]
    values = [epoch, f"{train_loss:.4f}", f"{train_acc:.4f}"] + [f"{metric:.4f}" for metric in val_metrics]
    print(tabulate([headers, values], tablefmt="grid"))


def save_best_model(save_path, model, iter_type, val_metrics, best_score, best_f1_score):
    val_f1_score, val_auroc, val_ap = val_metrics[3], val_metrics[2], val_metrics[6]
    if iter_type == 'f1' and val_f1_score > best_f1_score:
        torch.save(model.state_dict(), save_path)
        print(f'New best F1 score: {val_f1_score:.4f}. Model saved.')
        return val_f1_score, best_score
    elif iter_type == 'score':
        val_score = (val_auroc + val_ap) / 2
        if val_score > best_score:
            torch.save(model.state_dict(), save_path)
            print(f'New best (AUC+AP)/2 score: {val_score:.4f}. Model saved.')
            return best_f1_score, val_score
    return best_f1_score, best_score


def main():
    config = load_config()
    dataset_name = config['dataset']['name']
    data_root = config['dataset']['data_root']
    device = torch.device(f"cuda:{config['device']['gpu']}")

    training_params = config['training']
    save_dir = config['save_dir']
    epochs, batch_size = training_params['epochs'], training_params['batch_size']
    lr, weight_decay = training_params['lr'], training_params['weight_decay']
    iter_type = training_params['iter_metric']

    data_path = os.path.join(data_root, dataset_name)
    save_path = os.path.join(save_dir, f'{dataset_name}_best_model.pth')

    train_loader, val_loader, test_loader = load_ddi_dataset(root=data_path, batch_size=batch_size,
                                                             dataset=dataset_name)

    data = next(iter(train_loader))
    node_dim = data[0].x.size(-1)
    edge_dim = data[0].edge_attr.size(-1)
    model = initialize_model(device, node_dim, edge_dim, config)

    optimizer = initialize_optimizer(model, lr, weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    running_loss = AverageMeter()
    running_acc = AverageMeter()
    best_f1_score = best_score = 0.0

    for epoch in range(epochs):
        model.train()
        for data in tqdm(train_loader, desc="Training", unit="batch"):
            data = [d.to(device) for d in data]
            head_pairs, tail_pairs, head_fins, head_input_ids, head_mask, tail_fins, tail_input_ids, tail_mask, rel, head_smiles, tail_smiles, label = data
            pred = model((head_pairs, tail_pairs, head_fins, head_input_ids, head_mask, tail_fins, tail_input_ids,
                          tail_mask, rel, head_smiles, tail_smiles))
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_cls = (torch.sigmoid(pred) > 0.5).detach().cpu().numpy()
            acc = accuracy(label.detach().cpu().numpy(), pred_cls)
            running_acc.update(acc)
            running_loss.update(loss.item(), label.size(0))

        epoch_loss = running_loss.get_average()
        epoch_acc = running_acc.get_average()
        running_loss.reset()
        running_acc.reset()

        val_metrics = evaluate(model, criterion, val_loader, device)
        log_metrics(epoch, epoch_loss, epoch_acc, val_metrics)

        best_f1_score, best_score = save_best_model(save_path, model, iter_type, val_metrics, best_f1_score, best_score)


if __name__ == "__main__":
    main()
