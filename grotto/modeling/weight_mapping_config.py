"""
Centralized weight mapping configurations.
Simple key-to-key mapping with optional value transformations.
"""
import torch


def squeeze_gamma(tensor):
    """Squeeze RmsNorm gamma from [C,1,1] to [C]"""
    return tensor.squeeze()


# Mapping table: old_key -> (new_key, transform_fn or None)
VAE_DECODER_MAPPING = {
    # Conv2 - direct copy
    'conv2.weight': ('conv2.weight', None),
    'conv2.bias': ('conv2.bias', None),

    # Decoder conv1
    'decoder.conv1.weight': ('decoder.conv1.weight', None),
    'decoder.conv1.bias': ('decoder.conv1.bias', None),

    # Decoder head
    'decoder.head.0.gamma': ('decoder.head_norm.gamma', squeeze_gamma),
    'decoder.head.2.weight': ('decoder.head_conv.weight', None),
    'decoder.head.2.bias': ('decoder.head_conv.bias', None),

    # Middle blocks (residual -> flat structure)
    'decoder.middle.0.residual.0.gamma': ('decoder.middle.0.norm1.gamma', squeeze_gamma),
    'decoder.middle.0.residual.2.weight': ('decoder.middle.0.conv1.weight', None),
    'decoder.middle.0.residual.2.bias': ('decoder.middle.0.conv1.bias', None),
    'decoder.middle.0.residual.3.gamma': ('decoder.middle.0.norm2.gamma', squeeze_gamma),
    'decoder.middle.0.residual.6.weight': ('decoder.middle.0.conv2.weight', None),
    'decoder.middle.0.residual.6.bias': ('decoder.middle.0.conv2.bias', None),
    'decoder.middle.0.shortcut.weight': ('decoder.middle.0.shortcut.weight', None),

    # Middle attention block
    'decoder.middle.1.norm.gamma': ('decoder.middle.1.norm.gamma', squeeze_gamma),
    'decoder.middle.1.to_qkv.weight': ('decoder.middle.1.to_qkv.weight', None),
    'decoder.middle.1.to_qkv.bias': ('decoder.middle.1.to_qkv.bias', None),
    'decoder.middle.1.proj.weight': ('decoder.middle.1.proj_out.weight', None),
    'decoder.middle.1.proj.bias': ('decoder.middle.1.proj_out.bias', None),

    'decoder.middle.2.residual.0.gamma': ('decoder.middle.2.norm1.gamma', squeeze_gamma),
    'decoder.middle.2.residual.2.weight': ('decoder.middle.2.conv1.weight', None),
    'decoder.middle.2.residual.2.bias': ('decoder.middle.2.conv1.bias', None),
    'decoder.middle.2.residual.3.gamma': ('decoder.middle.2.norm2.gamma', squeeze_gamma),
    'decoder.middle.2.residual.6.weight': ('decoder.middle.2.conv2.weight', None),
    'decoder.middle.2.residual.6.bias': ('decoder.middle.2.conv2.bias', None),
}

# Generate upsample mappings (they follow the same pattern)
for i in range(20):  # Enough for most architectures
    VAE_DECODER_MAPPING.update({
        f'decoder.upsamples.{i}.residual.0.gamma': (f'decoder.upsamples.{i}.norm1.gamma', squeeze_gamma),
        f'decoder.upsamples.{i}.residual.2.weight': (f'decoder.upsamples.{i}.conv1.weight', None),
        f'decoder.upsamples.{i}.residual.2.bias': (f'decoder.upsamples.{i}.conv1.bias', None),
        f'decoder.upsamples.{i}.residual.3.gamma': (f'decoder.upsamples.{i}.norm2.gamma', squeeze_gamma),
        f'decoder.upsamples.{i}.residual.6.weight': (f'decoder.upsamples.{i}.conv2.weight', None),
        f'decoder.upsamples.{i}.residual.6.bias': (f'decoder.upsamples.{i}.conv2.bias', None),
        f'decoder.upsamples.{i}.shortcut.weight': (f'decoder.upsamples.{i}.shortcut.weight', None),
        f'decoder.upsamples.{i}.resample.0.weight': (f'decoder.upsamples.{i}.resample.0.weight', None),
        f'decoder.upsamples.{i}.resample.0.bias': (f'decoder.upsamples.{i}.resample.0.bias', None),
        f'decoder.upsamples.{i}.resample.1.weight': (f'decoder.upsamples.{i}.resample.1.weight', None),
        f'decoder.upsamples.{i}.resample.1.bias': (f'decoder.upsamples.{i}.resample.1.bias', None),
        f'decoder.upsamples.{i}.time_conv.weight': (f'decoder.upsamples.{i}.time_conv.weight', None),
        f'decoder.upsamples.{i}.time_conv.bias': (f'decoder.upsamples.{i}.time_conv.bias', None),
    })


def apply_mapping(state_dict, mapping_table):
    """
    Apply weight mapping based on mapping table.

    Args:
        state_dict: Original state dict
        mapping_table: Dict mapping old_key -> (new_key, transform_fn)

    Returns:
        Mapped state dict
    """
    new_state_dict = {}

    for old_key, value in state_dict.items():
        if old_key in mapping_table:
            new_key, transform_fn = mapping_table[old_key]
            new_value = transform_fn(value) if transform_fn else value
            new_state_dict[new_key] = new_value
        else:
            # Key not in mapping, copy as-is (for backward compatibility)
            new_state_dict[old_key] = value

    return new_state_dict


def detect_old_vae_format(state_dict):
    """Check if state_dict is from old VAEDecoderWrapper"""
    return any('.residual.' in key for key in state_dict.keys())


# Predictor/Model weight mapping for img_emb
PREDICTOR_IMG_EMB_MAPPING = {
    'model.img_emb.proj.0.weight': ('model.img_emb.0.weight', None),
    'model.img_emb.proj.0.bias': ('model.img_emb.0.bias', None),
    'model.img_emb.proj.1.weight': ('model.img_emb.1.weight', None),
    'model.img_emb.proj.1.bias': ('model.img_emb.1.bias', None),
    'model.img_emb.proj.3.weight': ('model.img_emb.3.weight', None),
    'model.img_emb.proj.3.bias': ('model.img_emb.3.bias', None),
    'model.img_emb.proj.4.weight': ('model.img_emb.4.weight', None),
    'model.img_emb.proj.4.bias': ('model.img_emb.4.bias', None),
}


def detect_old_predictor_format(state_dict):
    """Check if state_dict has old img_emb.proj format"""
    return any('model.img_emb.proj.' in key for key in state_dict.keys())
