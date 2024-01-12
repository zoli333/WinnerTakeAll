import torch
from fc_wta_train import Model
from utils import plot_weights

if __name__ == "__main__":
    model = Model(out_features=2000)
    model_folder = ''
    model_num = 'fc_wta_600'
    model.load_state_dict(torch.load(model_folder + model_num + '.pt'))
    model.eval()
    model_children = list(model.children())
    decoder_layer = model_children[-1][0]
    # check whether the last encoder layer weights are tied
    # decoder_layer = model_children[0][1]
    decoder_layer_data = decoder_layer.weight.data
    plot_weights(decoder_layer_data, model_folder, model_num, start_idx=0, end_idx=200, transpose=True)
