import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, max_epochs, batch_size):
        self.model = model
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        # Fix optimizer to Adam and loss to CrossEntropy
        self.optimizer = optim.Adam()
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, train_ds, valid_ds, plot_loss=True):
        # Initialize DataLoaders
        tdl = DataLoader(train_ds, batch_size=self.batch_size)
        vdl = DataLoader(valid_ds, batch_size=len(valid_ds))

        # Iterate over epochs
        for epoch in range(self.max_epochs):

            # Iterate over batches
            for idx_batch, batch in enumerate(tdl):
                pred = self.model(batch['image'])
                loss = self.loss_fn(pred, batch['y'])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

