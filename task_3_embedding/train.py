import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


class Trainer:
    def __init__(self, word_embedder, language_model, lr=1e-3):
        self.word_embedder = word_embedder.to(DEVICE)
        self.language_model = language_model.to(DEVICE)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.word_embedder.parameters()) +
            list(self.language_model.parameters()),
            lr=lr
        )

    def train(self, train_ds, valid_ds, epochs=200, batch_size=64):
        tdl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        vdl = DataLoader(valid_ds, batch_size=batch_size)
        for epoch in range(epochs):
            for idx_batch, batch in enumerate(tdl):
                pass
