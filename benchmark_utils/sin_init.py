import math
import torch


def sinusoidal_(tensor, *, mean, std, generator=None):
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            sinusoidal_, (tensor,), tensor=tensor, std=std, generator=generator
        )

    with torch.no_grad():
        # Flatten input channels for unified handling
        shape = tensor.shape
        if tensor.ndimension() == 2:
            n_out, n_in = shape
            n_flat = n_in
        elif tensor.ndimension() >= 3:
            n_out, n_in = shape[0], shape[1] * math.prod(shape[2:])
            n_flat = n_in
        else:
            raise ValueError("Tensor must have at least 2 dimensions")

        # Create sinusoidal pattern
        phase = (2 * math.pi / n_out)
        freq = (2 * math.pi / n_flat)
        position = torch.arange(n_flat, dtype=torch.float32) * freq + phase

        # Give each neuron a different wave
        neuron = torch.arange(n_out, dtype=torch.float32) + 1
        weights = torch.sin(position.unsqueeze(0) * neuron.unsqueeze(1))

        # Normalize amplitude
        amplitude = std / torch.var(weights, unbiased=True).item()
        weights = weights * amplitude

        # Reshape for conv layers if needed
        if tensor.ndimension() >= 3:
            weights = weights.view(shape[0], shape[1], *shape[2:])

        tensor.copy_(weights)
        return tensor
