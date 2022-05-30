from jax.lib import xla_client
from .build import _impl_jax
for _name, _value in _impl_jax.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

from .ops_rast import rasterize
from .ops_interp import interpolate
from .ops_anti import antialias

__all__ = ["rasterize", "interpolate", "antialias"]
