# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray

# If the GPU version exists, also register those
from .build import _impl_jax
for _name, _value in _impl_jax.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

xops = xla_client.ops

w = 512
h = 512


# This function exposes the primitive to user code and this is the only
# public-facing function in this module
def rasterize(pos, tri):
    return _rasterize_prim.bind(pos, tri)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _rasterize_abstract(pos, tri):
    shape = pos.shape
    dtype = dtypes.canonicalize_dtype(pos.dtype)
    N = shape[0]
    return ShapedArray((N, w, h, 4), dtype)


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _rasterize_translation(c, pos, tri, *, platform="gpu"):
    dtype = c.get_shape(pos).element_type()
    dims_pos = c.get_shape(pos).dimensions()
    dims_tri = c.get_shape(tri).dimensions()
    N = dims_pos[0]
    V = dims_pos[1]
    F = dims_tri[0]
    assert dims_pos[2] == 4
    assert dims_tri[1] == 3
    out_shape = xla_client.Shape.array_shape(dtype, [N, w, h, 4], [3, 2, 1, 0])

    # On the GPU, we do things a little differently and encapsulate the
    # dimension using the 'opaque' parameter
    opaque = _impl_jax.build_descriptor(w, h, N, V, F)

    return xops.CustomCallWithLayout(
        c,
        b"rasterize_fwd",
        operands=(pos, tri),
        operand_shapes_with_layout=(c.get_shape(pos), c.get_shape(tri)),
        shape_with_layout=out_shape,
        opaque=opaque,
    )


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# Here we define the differentiation rules using a JVP derived using implicit
# differentiation of Kepler's equation:
#
#  M = E - e * sin(E)
#  -> dM = dE * (1 - e * cos(E)) - de * sin(E)
#  -> dE/dM = 1 / (1 - e * cos(E))  and  de/dM = sin(E) / (1 - e * cos(E))
#
# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.
def _kepler_jvp(args, tangents):
    mean_anom, ecc = args
    d_mean_anom, d_ecc = tangents

    # We use "bind" here because we don't want to mod the mean anomaly again
    sin_ecc_anom, cos_ecc_anom = _kepler_prim.bind(mean_anom, ecc)

    def zero_tangent(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

    # Propagate the derivatives
    d_ecc_anom = (
        zero_tangent(d_mean_anom, mean_anom)
        + zero_tangent(d_ecc, ecc) * sin_ecc_anom
    ) / (1 - ecc * cos_ecc_anom)

    return (sin_ecc_anom, cos_ecc_anom), (
        cos_ecc_anom * d_ecc_anom,
        -sin_ecc_anom * d_ecc_anom,
    )


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************

# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _kepler_batch(args, axes):
    assert axes[0] == axes[1]
    return kepler(*args), axes


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_rasterize_prim = core.Primitive("rasterize")
_rasterize_prim.multiple_results = False
_rasterize_prim.def_impl(partial(xla.apply_primitive, _rasterize_prim))
_rasterize_prim.def_abstract_eval(_rasterize_abstract)

# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["gpu"][_rasterize_prim] = partial(
    _rasterize_translation, platform="gpu"
)

# # Connect the JVP and batching rules
# ad.primitive_jvps[_kepler_prim] = _kepler_jvp
# batching.primitive_batchers[_kepler_prim] = _kepler_batch
