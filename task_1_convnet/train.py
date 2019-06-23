import torch
import torch.nn as nn
import torch.optim as optim

from livelossplot import PlotLosses
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


class Trainer:
    def __init__(self, model, max_epochs, batch_size, learning_rate=1e-3):
        self.model = model
        self.model = self.model.to(DEVICE)
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        # Fix optimizer to Adam and loss to CrossEntropy
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

    @staticmethod
    def count_goods(y_pred, y_true):
        _, indices = y_pred.max(dim=1)
        return torch.sum(indices == y_true).item()

    def train(self, train_ds, valid_ds, plot_loss=True):
        # Initialize plotting
        if plot_loss:
            liveloss = PlotLosses()

        # Initialize DataLoaders
        tdl = DataLoader(train_ds, batch_size=self.batch_size, pin_memory=True)
        vdl = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False,
                         pin_memory=True)

        # Lists for losses
        train_losses, valid_losses = [], []
        # Lists for accuracies
        train_accs, valid_accs = [], []

        # Iterate over epochs
        for epoch in range(self.max_epochs):
            # Logs for livelossplot
            logs = {}

            batch_losses = []
            batch_count_goods = []
            # Iterate over batches
            for idx_batch, batch in enumerate(tdl):
                x = batch[0].to(DEVICE)
                y = batch[1].to(device=DEVICE, dtype=torch.long)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                batch_losses.append(loss.item())
                # Accuracy
                with torch.no_grad():
                    batch_count_goods.append(
                        self.count_goods(pred, y)
                    )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Save train loss and accuracy for the epoch
            train_losses.append(sum(batch_losses) / len(train_ds))
            train_accs.append(sum(batch_count_goods) / len(train_ds))

            # Compute and save validation loss and accuracy for the epoch
            with torch.no_grad():
                v_batch_losses, v_batch_count_goods = [], []
                for idx_batch, batch in enumerate(vdl):
                    x = batch[0].to(DEVICE)
                    y = batch[1].to(device=DEVICE, dtype=torch.long)
                    pred = self.model(x)
                    loss = self.loss_fn(pred, y)
                    v_batch_losses.append(loss.item())
                    v_batch_count_goods.append(self.count_goods(pred, y))
                valid_losses.append(sum(v_batch_losses) / len(valid_ds))
                valid_accs.append(sum(v_batch_count_goods) / len(valid_ds))

            if plot_loss:
                logs['log loss'] = train_losses[epoch]
                logs['val_log loss'] = valid_losses[epoch]
                logs['accuracy'] = train_accs[epoch]
                logs['val_accuracy'] = valid_accs[epoch]
                liveloss.update(logs)
                liveloss.draw()
