import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset


from src.utils import output_to_label, label_to_output

class ISUPPredictor(nn.Module):
    def __init__(self, input_channels, n_classes=6):
        super(ISUPPredictor, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(input_channels)
        self.dense_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dense_layers.append(nn.Linear(input_channels, 32))
        self.batch_norms.append(nn.BatchNorm1d(32))
        self.output_layer = nn.Linear(32, n_classes - 1)
    
    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        for dense, bn in zip(self.dense_layers, self.batch_norms):
            x = F.relu(bn(dense(x)))
        x = torch.sigmoid(self.output_layer(x))
        return x


class ISUPDataset(Dataset):
    def __init__(self, df):
        self.image_id = df["image_id"].values
        self.X = df.drop(columns=["isup", "image_id"]).values
        self.y = df["isup"].values
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ISUPCCELoss(nn.Module):
    def __init__(self):
        super(ISUPCCELoss, self).__init__()

    def forward(self, y_pred, y_true):
        result = torch.where(y_true == 1.0, y_pred, 1.0 - y_pred)
        result_clipped = torch.clamp(result, min=1e-7, max=1 - 1e-3)
        log_result = torch.log(result_clipped)
        sum_log_result = -torch.sum(log_result, dim=1)
        return sum_log_result.sum()


class Client():
    def __init__(self, client_name, train_loader, val_loader, model, lr):
        self.client_name = client_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.n_train_data = len(self.train_loader.dataset)
        self.train_metrics = dict({"acc": list(), "loss": list(), "val_acc": list(), "val_loss": list()})

        self.epoch = 0
        self.results_folder = self.setup_folder()

        self.criterion = ISUPCCELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=lr)

    def setup_folder(self):
        """
        check if folder for storing exists and otherwise create it
        """
        path = os.path.join("/mnt", "output", self.client_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        return path
    

    def train(self):
        """
        
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            full_pred = outputs
            full_true = label_to_output(targets)
            scores_pred = output_to_label(outputs)
            scores_true = targets

            loss = self.criterion(full_pred, full_true)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total += targets.size(0)
            correct += (scores_pred == scores_true).sum().item()

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_accuracy = 100 * correct / total

        self.train_metrics["acc"].append(epoch_accuracy)
        self.train_metrics["loss"].append(epoch_loss)


    def validate(self):
        """
        
        """
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs, targets
                outputs = self.model(inputs)

                full_pred = outputs
                full_true = label_to_output(targets)
                scores_pred = output_to_label(outputs)
                scores_true = targets

                loss = self.criterion(full_pred, full_true)
                val_loss += loss.item()
                total += targets.size(0)
                correct += (scores_pred == scores_true).sum().item()

        val_loss = val_loss / len(self.val_loader.dataset)
        val_accuracy = 100 * correct / total
                
        self.train_metrics["val_acc"].append(val_accuracy)
        self.train_metrics["val_loss"].append(val_loss)
        
