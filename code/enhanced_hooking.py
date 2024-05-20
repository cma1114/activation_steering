### Flexible hooking at arbitrary layers and tokens
import torch
import torch.nn as nn
from collections import defaultdict

def attach_activation_hooks(model, layers_positions, activation_storage, get_at='end'):
    """
    Attach hooks to specified layers to capture activations at specified positions.
    """
    def capture_activations_hook(layer_idx, positions, get_at='end'):
        def hook(module, input, output):
            # output shape is (batch_size, sequence_length, hidden_size)
            if isinstance(output, tuple): output = output[0]
###            for i, pos_list in enumerate(positions):
###                activation_storage[layer_idx].extend(output[i, pos_list, :].detach().cpu())
            for batch_idx in range(len(positions)):
                for pos_idx, seq_pos in enumerate(positions[batch_idx]):
                    activation_storage[layer_idx][pos_idx].append(output[batch_idx, seq_pos, :].detach().cpu())
        def pre_hook(module, input):
            for i, pos_list in enumerate(positions):
                selected_activations = input[0][i, pos_list, :]
                #print(f"Type of selected_activations: {type(selected_activations)}, size: {selected_activations.size()}")
                activation_storage[layer_idx].extend(selected_activations.detach().cpu())
        return hook if get_at == 'end' else pre_hook

    # Clear previous storage
    activation_storage.clear()

    # Access transformer blocks and attach hooks
    transformer_blocks = get_blocks(model)
    for idx, block in enumerate(transformer_blocks):
        if idx in layers_positions:
            hook = capture_activations_hook(idx, layers_positions[idx], get_at)
            if get_at=='end': block.register_forward_hook(hook)
            else: block.register_forward_pre_hook(hook)


def get_activations(model, tokens, layers_positions, get_at='end'):
    """
    Get activations from specific layers and positions.
    """
    # Prepare storage for activations
    activation_storage = defaultdict(lambda: defaultdict(list))###defaultdict(list)

    # Attach hooks to the model
    attach_activation_hooks(model, layers_positions, activation_storage, get_at)

    # Ensure the model is in eval mode
    model.eval()

    # Run the model with the tokens
    with torch.no_grad():
        model(tokens.to(next(model.parameters()).device))

    # Remove hooks after use (to avoid memory leak)
    for block in get_blocks(model):
        if get_at == 'end': block._forward_hooks.clear()
        else: block._forward_pre_hooks.clear()

###    return dict(activation_storage)
    return {layer: {pos: torch.stack(tensors) for pos, tensors in pos_dict.items()} 
            for layer, pos_dict in activation_storage.items()}   


def create_add_activations_hook(layers_activations, add_at='end'):
    """
    Create a hook to add activation vectors at specified positions within specified layers.
    layers_activations: A dictionary where keys are layer indices and values are tuples of (positions, activation_vectors).
    add_at: 'start' to add at the beginning of a block, 'end' to add at the end of a block.
    """
    def hook(module, inputs, outputs):
        layer_idx = getattr(module, 'layer_idx', None)
        if layer_idx in layers_activations:
            activation_info = layers_activations[layer_idx]
            if isinstance(outputs, tuple): 
                print(f"outputs is a tuple of length {len(outputs)}")
                main_tensor = outputs[0]  
            else: main_tensor = outputs
            if main_tensor.shape[1] == 1: return outputs#hack to turn this off during generation

            for position, activation_vector in activation_info.items():
                # Check if the position is valid for the current output tensor shape
                if position < main_tensor.shape[1]:
                    print(f"Adding activations at layer {layer_idx} at position {position}")
                    print(f"outputs.size={main_tensor.size()}, activation_vector.size={activation_vector.size()}")
                    main_tensor[:, position, :] += activation_vector
                else:
                    print(f"Position {position} is out of bounds for the current sequence ({main_tensor.shape[1]}).")
        
        return (main_tensor,) + outputs[1:] if isinstance(outputs, tuple) else main_tensor
    return hook


def create_add_activations_pre_hook(layers_activations, add_at='end'):
    """
    Create a hook to add activation vectors at specified positions within specified layers.
    layers_activations: A dictionary where keys are layer indices and values are tuples of (positions, activation_vectors).
    add_at: 'start' to add at the beginning of a block, 'end' to add at the end of a block.
    """
    def hook(module, inputs):
        if inputs[0].shape[1] == 1: return inputs #hack to turn this off during generation (same as returning None)
        layer_idx = getattr(module, 'layer_idx', None)
        if layer_idx in layers_activations:
            activation_info = layers_activations[layer_idx]
            for position, activation_vector in activation_info.items():
                # Check if the position is valid for the current input tensor shape
                if position < inputs[0].shape[1]:
                    print(f"Adding activations at layer {layer_idx} at position {position}")
                    print(f"inputs[0].size={inputs[0].size()}, activation_vector.size={activation_vector.size()}")
                    inputs = list(inputs)  # Convert tuple to list for mutability
                    inputs[0][:, position, :] += activation_vector
                    inputs = tuple(inputs)  # Convert back to tuple to maintain integrity
                else:
                    print(f"Position {position} is out of bounds for the current sequence ({inputs[0].shape[1]}).")
        
        return inputs
    return hook


def create_continuous_activation_hook(continuouspos_layer_activations):
    def hook(module, inputs, outputs):
        current_layer_idx = getattr(module, 'layer_idx', None)
        if current_layer_idx in continuouspos_layer_activations:
            activation_vector = continuouspos_layer_activations[current_layer_idx]
            main_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
            # Always add to the last position in the sequence
            #print(f"outputs.size={main_tensor.size()}, activation_vector.size={activation_vector.size()}")
            #print(f"Adding continuous activations at layer {current_layer_idx} at position {main_tensor.size(1) - 1} to value {main_tensor[:, main_tensor.size(1) - 1, :5]}")
#            main_tensor[:, main_tensor.size(1) - 1, :] += activation_vector
            main_tensor += activation_vector #add to every token in the prompt at the first pass, and then adds to each new token (takes a long time to run with use_cache=False, since then it adds to whole prompt at every pass)
        return (main_tensor,) + outputs[1:] if isinstance(outputs, tuple) else main_tensor
    return hook


def create_continuous_activation_pre_hook(continuous_layers_activations):
    def pre_hook(module, inputs):
        current_layer_idx = getattr(module, 'layer_idx', None)
        if current_layer_idx in continuous_layers_activations:
            activation_vector = continuous_layers_activations[current_layer_idx]
            inputs = list(inputs)  # Convert tuple to list for mutability
#            inputs[0][:, inputs[0].size(1) - 1, :] += activation_vector
            inputs[0] += activation_vector
            inputs = tuple(inputs)  # Convert back to tuple to maintain integrity
        return inputs
    return pre_hook


def add_activations_and_generate(model, tokens, specificpos_layer_activations, continuouspos_layer_activations, sampling_kwargs, add_at='end'):
    transformer_blocks = get_blocks(model)
    
    # Attach hooks for specific initial positions
    for idx, block in enumerate(transformer_blocks):
        setattr(block, 'layer_idx', idx)
        if idx in specificpos_layer_activations:
            if add_at == 'end':
                hook = create_add_activations_hook(specificpos_layer_activations)
                block.register_forward_hook(hook)
            else:
                hook = create_add_activations_pre_hook(specificpos_layer_activations)
                block.register_forward_pre_hook(hook)

    # Attach hooks for multiple continuous activations across different layers
    for idx, block in enumerate(transformer_blocks):
        setattr(block, 'layer_idx', idx)
        if idx in continuouspos_layer_activations:
            if add_at == 'end':
                continuous_hook = create_continuous_activation_hook(continuouspos_layer_activations)
                block.register_forward_hook(continuous_hook)
            else:
                continuous_hook = create_continuous_activation_pre_hook(continuouspos_layer_activations)
                block.register_forward_pre_hook(continuous_hook)

    # Generate tokens
    tokens = {k: v.to(next(model.parameters()).device) for k, v in tokens.items()}
    generated_ids = model.generate(**tokens, **sampling_kwargs)

    # Cleanup hooks
    for block in transformer_blocks:
        if add_at == 'end':
            block._forward_hooks.clear()
        else:
            block._forward_pre_hooks.clear()

    return generated_ids


def get_blocks(model: nn.Module) -> nn.ModuleList:
    """ Get the ModuleList containing the transformer blocks from a model. """
    def numel_(mod):
###        return sum(p.numel() for p in mod.parameters())
        if isinstance(mod, nn.Module):
            num_elements = sum(p.numel() for p in mod.parameters())
            return num_elements
        else:
            print(f"Non-module object encountered: {mod}")
            return 0
    model_numel = numel_(model)
    candidates = [mod for mod in model.modules() if isinstance(mod, nn.ModuleList) and numel_(mod) > .5 * model_numel]
    assert len(candidates) == 1, f'Found {len(candidates)} ModuleLists with >50% of model params.'
    return candidates[0]


def clear_hooks(model):
    transformer_blocks = get_blocks(model)
    for block in transformer_blocks:
        block._forward_hooks.clear()
        block._forward_pre_hooks.clear()