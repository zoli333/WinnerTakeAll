import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sparsity import init_weights, Sparsity, MaxpoolSparsity

out_features = 2000
in_features = 784
batch_size = 100
learning_rate = 0.0003
epochs = 1000


class Model(nn.Module):
    def __init__(self, out_features, use_tied_weights=True):
        super(Model, self).__init__()
        self.use_tied_weights = use_tied_weights
        last_encoder_linear = nn.Linear(in_features, out_features)
        first_decoder_linear = nn.Linear(out_features, in_features)
        if use_tied_weights:
            first_decoder_linear.weight = nn.Parameter(last_encoder_linear.weight.T)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            last_encoder_linear,
            nn.LeakyReLU(inplace=True)
        )
        # self.sparsity = MaxpoolSparsity(spatial_sparsity_amount=0, lifetime_sparsity_amount=5)
        self.sparsity = Sparsity(spatial_sparsity_amount=0, lifetime_sparsity_amount=5)
        self.decoder = nn.Sequential(
            first_decoder_linear
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.sparsity(x)
        x = self.decoder(x)
        return x


def train(model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.view(-1, 1, 28, 28)
        loss = criterion(output.view(-1, 1, 28, 28), data)
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        torch.save(model.state_dict(), "fc_wta_" + str(epoch) + ".pt")

    train_loss = train_loss / len(train_loader)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch,
        train_loss
    ))


if __name__ == "__main__":

    use_cuda = True
    torch.manual_seed(12345)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Model(out_features=out_features, use_tied_weights=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    model.apply(init_weights)

    for epoch in range(1, epochs + 1):
        train(model, criterion, device, train_loader, optimizer, epoch)
