import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sparsity import init_weights, Sparsity, MaxpoolSparsity
import os
from custom_optimizer import MomentumOptimizer

in_channels = 1
out_channels = 64
batch_size = 100
learning_rate = 0.0003
epochs = 1000


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_sparsity_amount=1, lifetime_sparsity_amount=20, use_tied_weights=False):
        super(Model, self).__init__()
        self.use_tied_weights = use_tied_weights
        last_encoder_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2, stride=1)
        first_decoder_conv = nn.ConvTranspose2d(out_channels, in_channels, kernel_size=11, padding=5, stride=1)
        if use_tied_weights:
            pass
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(),
            last_encoder_conv,
            nn.LeakyReLU(),
        )
        # self.sparsity = MaxpoolSparsity(spatial_sparsity_amount=spatial_sparsity_amount, lifetime_sparsity_amount=lifetime_sparsity_amount)
        self.sparsity = Sparsity(spatial_sparsity_amount=spatial_sparsity_amount, lifetime_sparsity_amount=lifetime_sparsity_amount)
        self.decoder = nn.Sequential(
            first_decoder_conv
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.sparsity(x)
        x = self.decoder(x)
        return x


def train(model, criterion, device, train_loader, optimizer, epoch, save_prefix=''):
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

    if epoch % 5 == 0:
        result_dir = "train_results"
        os.makedirs(result_dir, exist_ok=True)
        if save_prefix != '':
            torch.save(model.state_dict(), result_dir + "/" + save_prefix + "_conv_wta_" + str(epoch) + ".pt")
        else:
            torch.save(model.state_dict(), result_dir + "/" + "conv_wta_" + str(epoch) + ".pt")

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

    model = Model(in_channels=in_channels, out_channels=out_channels, spatial_sparsity_amount=1, lifetime_sparsity_amount=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    # f_init = lambda module: init_weights(module, 0.001)
    # model.apply(f_init)

    for epoch in range(1, epochs + 1):
        train(model, criterion, device, train_loader, optimizer, epoch)
        # scheduler.step()
