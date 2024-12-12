import os
import torch
import yaml
from src.tools.dataset import load_ddi_dataset
from src.models.ComNet import ComNet_DDI
from src.tools.utils import val_test


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def initialize_model(device, node_dim, edge_dim, config):
    model_params = config['model']
    return ComNet_DDI(device, node_dim, edge_dim, hidden_dim=model_params['hidden_dim'],
                      dropout=model_params['dropout'], rmodule_dim=model_params['rmodule_dim']).to(device)


def load_data(data_root, dataset_name, batch_size):
    """Load the dataset and return the train, validation, and test loaders."""
    data_path = os.path.join(data_root, dataset_name)
    return load_ddi_dataset(root=data_path, batch_size=batch_size, dataset=dataset_name)


def main():
    config = load_config()
    dataset_name = config['dataset']['name']
    data_root = config['dataset']['data_root']
    device = torch.device(f"cuda:{config['device']['gpu']}")
    paths = config['paths']
    batch_size = config['training']['batch_size']

    model_path = paths['model_path']
    result_path = paths['result_path']

    train_loader, val_loader, test_loader = load_data(data_root, dataset_name, batch_size)

    data = next(iter(train_loader))
    node_dim = data[0].x.size(-1)
    edge_dim = data[0].edge_attr.size(-1)

    model = initialize_model(device, node_dim, edge_dim, config)
    model.load_state_dict(torch.load(model_path))

    val_test(model, test_loader, device, result_path)


if __name__ == '__main__':
    main()
