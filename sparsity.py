import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, out_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(out_features, 10)

    def forward(self, x):
        return self.linear(x)


def init_weights(module, std_init_constant=0.01):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std_init_constant)
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)


def fc_wta_tie_weights_at_test_time(model):
    if model.use_tied_weights:
        # if the weights were already tied at training time, do nothing
        return
    # tie weights
    # -- get decoder weights
    decoder_weight = model.decoder[0].weight.data
    # -- copy to last encoder layer
    model.encoder[1].weight.data = torch.transpose(decoder_weight, 1, 0)


class CombinedSparsity(nn.Module):
    def __init__(self, spatial_sparsity=1, lifetime_sparsity=1):
        super(CombinedSparsity, self).__init__()
        self.spatial_sparsity = spatial_sparsity
        self.lifetime_sparsity = lifetime_sparsity
        self.pool = None
        self.unpool = None

    def _sparsity_mask(self, vals, k):
        vals_ = vals.view(vals.size(0), -1)
        values, indices = torch.topk(vals_, k=k, dim=0)
        t = torch.transpose(indices, 1, 0).flatten(), torch.repeat_interleave(torch.arange(vals_.size(1)), k)
        mask = torch.zeros_like(vals_, dtype=torch.float32)
        mask[t] = 1.0
        lifetime_sparsity_result = (vals_ * mask.squeeze().squeeze())
        lifetime_sparsity_result = lifetime_sparsity_result.unsqueeze(-1).unsqueeze(-1)
        return lifetime_sparsity_result

    def forward(self, activations):
        if not self.training:
            return activations

        if activations.ndim == 2:
            lifetime_sparsity_result = self._sparsity_mask(activations, self.lifetime_sparsity)
            return lifetime_sparsity_result

        if not self.pool and not self.unpool:
            assert activations.size(2) % self.spatial_sparsity == 0, "not divisible"
            pool_width = int(activations.size(2) / self.spatial_sparsity)
            self.pool = nn.MaxPool2d(kernel_size=pool_width, stride=pool_width, return_indices=True)
            self.unpool = nn.MaxUnpool2d(kernel_size=pool_width, stride=pool_width)

        spatial_sparsity_result, spatial_indices = self.pool(activations)
        lifetime_sparsity_result = self._sparsity_mask(spatial_sparsity_result, self.lifetime_sparsity)
        result = self.unpool(lifetime_sparsity_result, spatial_indices)
        return result


class MaxpoolSparsity(nn.Module):
    def __init__(self, spatial_sparsity_amount=1, lifetime_sparsity_amount=5):
        super(MaxpoolSparsity, self).__init__()
        self.spatial_sparsity_amount = spatial_sparsity_amount
        self.lifetime_sparsity_amount = lifetime_sparsity_amount

        self.sparsity = CombinedSparsity(spatial_sparsity=spatial_sparsity_amount, lifetime_sparsity=lifetime_sparsity_amount)

    def forward(self, x):
        x = self.sparsity(x)
        return x


# ----------------------------------------------------------------------

class SpatialSparsity(nn.Module):
    def __init__(self, k=1):
        super(SpatialSparsity, self).__init__()
        self.k = k

    def forward(self, activations):
        if not self.training:
            return activations

        b, c, w, h = activations.shape
        y = activations.view(b, c, w * h)
        winners, _ = torch.topk(y, k=self.k, dim=-1)
        mask = torch.where(y < winners, 0, 1)
        res = (y * mask).view_as(activations)
        return res, winners.squeeze() # b, c


class LifetimeSparsity(nn.Module):
    def __init__(self, k=1):
        super(LifetimeSparsity, self).__init__()
        self.k = k

    def forward(self, activations, winners):
        if not self.training:
            return activations

        if winners is None:
            # fully connected wta
            winners, _ = torch.topk(activations, k=self.k, dim=0)  # b, c
            mask = torch.where(activations < winners[self.k-1:self.k, :], 0, 1)
            return activations * mask
        winners = torch.transpose(winners, 1, 0)  # c, b
        new_winners, _ = torch.topk(winners, k=self.k, dim=-1)  # c, k
        mask = torch.where(winners < new_winners[:, self.k-1:self.k], 0, 1)
        mask = torch.transpose(mask, 1, 0)
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        return activations * mask


class Sparsity(nn.Module):
    def __init__(self, spatial_sparsity_amount=1, lifetime_sparsity_amount=5):
        super(Sparsity, self).__init__()
        self.spatial_sparsity_amount = spatial_sparsity_amount
        self.lifetime_sparsity_amount = lifetime_sparsity_amount

        self.spatial_sparsity = SpatialSparsity(k=spatial_sparsity_amount)
        self.lifetime_sparsity = LifetimeSparsity(k=lifetime_sparsity_amount)

    def forward(self, x):
        if not self.training:
            return x

        if self.spatial_sparsity_amount > 0 and self.lifetime_sparsity_amount > 0:
            x, winners = self.spatial_sparsity(x)
            x = self.lifetime_sparsity(x, winners)
            return x
        elif self.lifetime_sparsity_amount > 0:
            x = self.lifetime_sparsity(x, None)
            return x
        # spatial sparsity only
        x, _ = self.spatial_sparsity(x)
        return x


