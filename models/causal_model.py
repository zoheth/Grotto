from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin

class CausalWanModel(ModelMixin, ConfigMixin):
    pass