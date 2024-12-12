import os
import torch
import csv
import numpy as np
from metrics import *
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import defaultdict
from torch_geometric.data import Data
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f'Input {x} not in allowable set {allowable_set}')
    return [x == s for s in allowable_set]


def encoding_unk(x, allowable_set):
    encoding = [False] * len(allowable_set)
    for atom in x:
        if atom in allowable_set:
            encoding[allowable_set.index(atom)] = True
    if any(encoding[:len(x)]):
        encoding[-1] = True
    return encoding


def one_hot_encoding_unk(x, allowable_set):
    x = x if x in allowable_set else allowable_set[-1]
    return [x == s for s in allowable_set]


class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)


def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss = AverageMeter()
    all_predictions = []
    all_labels = []
    for batch in dataloader:
        head_pairs, tail_pairs, head_fins, head_input_ids, head_mask, tail_fins, tail_input_ids, tail_mask, rel, head_smiles, tail_smiles, labels = [
            x.to(device) for x in batch]

        with torch.no_grad():
            predictions = model((head_pairs, tail_pairs, head_fins, head_input_ids, head_mask,
                                 tail_fins, tail_input_ids, tail_mask, rel, head_smiles, tail_smiles))
            loss = criterion(predictions, labels)
            predicted_probs = torch.sigmoid(predictions)

            all_predictions.append(predicted_probs.view(-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss.update(loss.item(), labels.size(0))
    predictions_concat = np.concatenate(all_predictions)
    labels_concat = np.concatenate(all_labels)
    accuracy, auroc, f1, precision, recall, avg_precision = compute_metrics(predictions_concat, labels_concat)
    average_loss = total_loss.get_average()
    total_loss.reset()
    model.train()
    return average_loss, accuracy, auroc, f1, precision, recall, avg_precision


def val_test(model, dataloader, device, result_path):
    model.eval()
    predictions = []
    labels = []
    for batch in dataloader:
        head_pairs, tail_pairs, head_fins, head_input_ids, head_mask, tail_fins, tail_input_ids, tail_mask, rel, head_smiles, tail_smiles, label = [
            x.to(device) for x in batch]
        with torch.no_grad():
            output = model((head_pairs, tail_pairs, head_fins, head_input_ids, head_mask,
                            tail_fins, tail_input_ids, tail_mask, rel, head_smiles, tail_smiles))
            predicted_probs = torch.sigmoid(output)

            predictions.append(predicted_probs.view(-1).cpu().numpy())
            labels.append(label.cpu().numpy())
    concatenated_predictions = np.concatenate(predictions)
    concatenated_labels = np.concatenate(labels)
    accuracy, auroc, f1, precision, recall, ap = compute_metrics(concatenated_predictions, concatenated_labels)
    results = {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auroc,
        'auc_prc': ap
    }
    display_metrics(results)
    save_results_to_file(results, result_path)
    return results


def display_metrics(results):
    metrics = ['accuracy', 'recall', 'f1_score', 'auc_roc', 'auc_prc']
    values = [results[metric] for metric in metrics]
    table = [metrics, values]
    print(tabulate(table, headers='firstrow', tablefmt='grid'))


def save_results_to_file(results, result_path):
    with open(result_path, 'w') as file:
        for metric, value in results.items():
            file.write(f'{metric}: {value:.4f}\n')

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        return self.sum / (self.count + 1e-12)