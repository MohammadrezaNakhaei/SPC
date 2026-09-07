
#!/usr/bin/env python3
import copy
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state


class Ensemble(nn.Module):
    """Vectorized ensemble of modules"""

    def __init__(self, modules, **kwargs):
        super().__init__()

        self.params_dict, self._buffers = stack_module_state(modules)
        self.params = nn.ParameterList([p for p in self.params_dict.values()])
        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        base_model = copy.deepcopy(modules[0])
        base_model = base_model.to("meta")

        def fmodel(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))

        self.vmap = torch.vmap(
            fmodel, in_dims=(0, 0, None), randomness="different", **kwargs
        )
        self._repr = str(modules)

    def forward(self, *args, **kwargs):
        return self.vmap(self._get_params_dict(), self._buffers, torch.cat(args, dim=-1), **kwargs)

    def _get_params_dict(self):
        params_dict = {}
        for key, value in zip(self.params_dict.keys(), self.params):
            params_dict.update({key: value})
        return params_dict

    def __repr__(self):
        return "Vectorized " + self._repr