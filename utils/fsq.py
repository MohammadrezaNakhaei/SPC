from torch.func import vmap
import torch.nn as nn 
from vector_quantize_pytorch import FSQ as _FSQ


class FSQ(nn.Module):
    """
    Finite Scalar Quantization
    """

    def __init__(self, levels):
        super().__init__()
        self.levels = levels
        self.num_channels = len(levels)
        self._fsq = _FSQ(levels)

    def forward(self, z):
        shp = z.shape
        z = z.view(*shp[:-1], -1, self.num_channels)
        if z.ndim > 3:  # TODO this might not work for CNN
            codes, indices = vmap(self._fsq)(z)
        else:
            codes, indices = self._fsq(z)
        return codes.flatten(-2), indices

    def __repr__(self):
        return f"FSQ(levels={self.levels})"