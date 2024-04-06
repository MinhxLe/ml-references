from dataclasses import dataclass, asdict
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import wandb


# Set the random seed for reproducibility
torch.manual_seed(2049)

data_set_size = 2_000
# Creating the Gaussian blob dataset
data = torch.cat(
    (
        -0.5 + 0.2 * torch.randn(int(data_set_size * 1 / 5), 2),
        0.5 + 0.2 * torch.randn(int(data_set_size * 4 / 5), 2),
    ),
    dim=0,
)

print(data.shape)


# Plot the data points
def plot_data(data):
    plt.scatter(data[:, 0], data[:, 1], s=3, label="Data")
    plt.legend()
    plt.show()


def get_energy_map(model, device, density=100, val=1.6):
    xs = torch.linspace(-val, val, density, device=device)
    ys = torch.linspace(-val, val, density, device=device)
    X, Y = torch.meshgrid(xs, ys)
    XY = torch.stack([X, Y], dim=-1).view(-1, 2)

    Z = model(XY).detach().view(density, density)
    return X, Y, Z


def get_score_map(model, device, density=30, val=1.6):
    xs = torch.linspace(-val, val, density, device=device)
    ys = torch.linspace(-val, val, density, device=device)
    X, Y = torch.meshgrid(xs, ys)
    XY = torch.stack([X, Y], dim=-1).view(-1, 2)

    scores = -torch.autograd.grad(model(XY).sum(), XY, create_graph=True)[0]
    Z = scores.view(density, density, 2)
    return X, Y, Z


# Assuming 'ebm' is your energy-based model implemented in PyTorch
# and 'device' is your computation device, e.g., 'cuda' or 'cpu'


# Example plotting, assuming 'ebm' and 'device' are defined
def plot_score_map(ebm, device="cpu"):
    X, Y, Z = get_score_map(ebm, device)
    plt.figure(figsize=(10, 10))
    plt.quiver(X.cpu(), Y.cpu(), Z[:, :, 0].cpu(), Z[:, :, 1].cpu(), color="g")
    plt.show()


def plot_energy_map(ebm, device="cpu"):
    X, Y, Z = get_energy_map(ebm, device)
    plt.figure(figsize=(10, 10))
    plt.contourf(X.cpu(), Y.cpu(), Z.cpu(), 50, cmap="viridis")
    plt.colorbar()
    plt.show()


# model
class EnergyFunction(nn.Module):
    ...

    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.Sigmoid(),
            nn.Linear(input_dim * 4, 1),
        )

    def forward(self, x):
        return self.mlp(x)


model = EnergyFunction(data.shape[1])


# training
@dataclass
class TrainCfg:
    batch_size: int = 40
    num_epoch: int = 250
    sigma: float = 0.1  # noise sigma
    lr: float = 0.01


train_cfg = TrainCfg()
optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)


def loss_fn(estimate, target):
    delta = target - estimate
    return 0.5 * (delta * delta).sum(dim=0).mean()


# data
if not data.requires_grad:
    data = data.requires_grad_()  # require grad b/c we need it

dataset = TensorDataset(data)
dataloader = DataLoader(dataset, train_cfg.batch_size, shuffle=True)


# training
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="toy-ebm",
    # track hyperparameters and run metadata
    config=dict(train=asdict(train_cfg)),
)


for epoch in range(train_cfg.num_epoch):
    total_loss = 0
    for batch_num, (batch_data,) in enumerate(dataloader):
        optimizer.zero_grad()
        if batch_data.grad is not None:
            batch_data.grad.zero_()

        noise = (
            torch.randn_like(batch_data) * train_cfg.sigma
        )  # equiv to N(0, sigma**2)
        noised_data = batch_data + noise

        model_output = model(noised_data)

        (estimate_score,) = torch.autograd.grad(
            outputs=-model_output,  # negative to get -energy
            inputs=noised_data,
            # TODO learn what these 2 mean
            grad_outputs=torch.ones_like(model_output),
            create_graph=True,
        )

        target_score = -noise * ((train_cfg.sigma) ** (-2))
        loss = loss_fn(estimate_score, target_score)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        train_loss = total_loss / batch_num
        wandb.log(dict(epoch=epoch, train_loss=train_loss))
        print(f"epoch:{epoch}, loss: {train_loss:.4f}")

if data.requires_grad:
    data = data.requires_grad_(False)

wandb.finish()
