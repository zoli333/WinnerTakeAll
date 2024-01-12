import torch
import matplotlib.pyplot as plt
from conv_wta_train import Model
import math
from utils import plot_weights


if __name__ == "__main__":
    model = Model(in_channels=1, out_channels=64)
    model_folder = 'train_results/'
    model_num = 'conv_wta_385'
    model.load_state_dict(torch.load(model_folder + model_num + '.pt'))
    model.eval()
    model_children = list(model.children())
    decoder_layer = model_children[-1][0]
    decoder_layer_data = decoder_layer.weight.data
    decoder_layer_data = decoder_layer_data.view(decoder_layer_data.size(0), -1)
    plot_weights(decoder_layer_data, model_folder, model_num, reshape_size=11, start_idx=0, end_idx=200, transpose=False)
