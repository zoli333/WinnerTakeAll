import torch
from torchvision import datasets, transforms
import sklearn.svm
from sklearn.metrics import accuracy_score
from fc_wta_train import Model
from sparsity import fc_wta_tie_weights_at_test_time
import warnings

'''
final accuracy: 98.75%
'''
if __name__ == "__main__":
    use_tied_weights_at_test_time = True

    model = Model(out_features=2000)
    # model.load_state_dict(torch.load('train_results/adam_lr_0p0003_f_2000_bs_100_k5_mnist/fc_wta_1000.pt'))
    model.load_state_dict(torch.load('train_results/adam_lr_0p0003_f_2000_bs_100_k5_mnist_tied_weights/fc_wta_1000.pt'))
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

    # get features by feedforward
    with torch.no_grad():
        if use_tied_weights_at_test_time:
            fc_wta_tie_weights_at_test_time(model)

        x_train = model.encoder(x_train)
        x_test = model.encoder(x_test)

    C = 0.1
    clf = sklearn.svm.LinearSVC(C=C, random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print('C={:.3f}, acc={:.4f}'.format(C, acc))


