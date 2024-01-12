import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from fc_wta_train import Model
from torch.utils.data import Dataset
from sparsity import fc_wta_tie_weights_at_test_time, init_weights, LogisticRegression
from utils import FeatureDataset, test_classification_accuracy
import warnings

'''
Logistic regression on top of the trained features.
Tied weights applied to use the weights for encoding from the decoder layer
and then using the encoded features from forward pass the result passed to the final linear 
classification layer.
'''
if __name__ == "__main__":
    use_cuda = True
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_tied_weights_at_test_time = True
    out_features = 2000
    model = Model(out_features=out_features)
    model.load_state_dict(torch.load('train_results/adam_lr_0p0003_f_2000_bs_100_k5_mnist/fc_wta_1000.pt'))
    model.eval()

    if use_tied_weights_at_test_time is True and model.use_tied_weights is True:
        warnings.warn("The weights were already tied at training time")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=dataset1.data.shape[0], shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=dataset2.data.shape[0], shuffle=False)

    for train_data, train_labels in train_loader:
        x_train = train_data
        y_train = train_labels

    for test_data, test_labels in test_loader:
        x_test = test_data
        y_test = test_labels

    with torch.no_grad():
        if use_tied_weights_at_test_time:
            fc_wta_tie_weights_at_test_time(model)

        x_train = model.encoder(x_train)
        x_test = model.encoder(x_test)

    feature_dataset_train = FeatureDataset(x_train, y_train)
    feature_dataset_train_loader = torch.utils.data.DataLoader(feature_dataset_train, batch_size=100, shuffle=True)
    feature_dataset_test = FeatureDataset(x_test, y_test)
    feature_dataset_test_loader = torch.utils.data.DataLoader(feature_dataset_test, batch_size=100, shuffle=False)

    logistic_regression_model = LogisticRegression(out_features=out_features).to(device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(logistic_regression_model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 18], gamma=0.1)
    optimizer = optim.Adam(logistic_regression_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 33], gamma=0.1)

    logistic_regression_model.apply(init_weights)
    for epoch in range(40):
        train_loss = 0.0
        for idx, (x, y) in enumerate(feature_dataset_train_loader, 0):
            x = x.to(device)
            y = y.to(device)

            output = logistic_regression_model(x)
            loss = loss_fn(output, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(feature_dataset_train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))
        test_classification_accuracy(logistic_regression_model, device, feature_dataset_test_loader, loss_fn)
        scheduler.step()

