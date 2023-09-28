from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.lib import xla_client
from jax.interpreters import xla
from jaxlib.xla_extension import XlaBuilder

from .build import _impl_jax  # TODO: setup.py changes dir
from .utils import check_array

# **********************
# *  USER'S INTERFACE  *
# **********************

@jax.custom_vjp
def antialias(color, rast, pos, tri, ev_hash=None):
    if ev_hash is None:
        ev_hash = _ev_hash_prim.bind(tri)
    out, work_buffer = _antialias_prim.bind(color, rast, pos, tri, ev_hash)
    return out


def antialias_fwd(color, rast, pos, tri, ev_hash):
    if ev_hash is None:
        ev_hash = _ev_hash_prim.bind(tri)
    out, work_buffer = _antialias_prim.bind(color, rast, pos, tri, ev_hash)
    return out, (color, rast, pos, tri, work_buffer)  # output, 'res' for bwd


def antialias_bwd(fwd_res, dy):
    color, rast, pos, tri, work_buffer = fwd_res
    grad = _antialias_grad_prim.bind(color, rast, pos, tri, dy, work_buffer)
    return (grad[0], None, grad[1], None, None)


antialias.defvjp(antialias_fwd, antialias_bwd)


def get_ev_hash(tri):
    return _ev_hash_prim.bind(tri)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _antialias_prim_abstract(color, rast, pos, tri, ev_hash):
    # check
    check_array("color", color, shapes=[(None, None, None, None)], dtype=jnp.float32)
    check_array("rast", rast, shapes=[(None, None, None, 4)], dtype=jnp.float32)
    check_array("pos", pos, shapes=[(None, None, 4), (None, 4)], dtype=jnp.float32)
    check_array("tri", tri, shapes=[(None, 3)], dtype=jnp.int32)
    # return abstract array
    dtype = jax.dtypes.canonicalize_dtype(color.dtype)
    N, H, W, _ = color.shape
    return (
        ShapedArray(color.shape, dtype),
        ShapedArray((N*H*W*8+4,), jax.dtypes.canonicalize_dtype(jnp.int32))
    )


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _antialias_prim_translation_gpu(c: XlaBuilder, color, rast, pos, tri, ev_hash, *args):
    dtype = c.get_shape(color).element_type()
    itype = c.get_shape(tri).element_type()
    dims_color = c.get_shape(color).dimensions()
    dims_pos = c.get_shape(pos).dimensions()
    dims_tri = c.get_shape(tri).dimensions()

    # get mode booleans
    instance_mode = len(dims_pos) > 2
    # get output shape
    N, H, W, C = dims_color
    out_shape = xla_client.Shape.array_shape(dtype, [N, H, W, C], [3, 2, 1, 0])
    buf_shape = xla_client.Shape.array_shape(itype, [N*H*W*8+4], [0])

    # Encapsulate the information using the 'opaque' parameter
    opaque = _impl_jax.build_antialias_descriptor(
        num_vertices=dims_pos[-2], num_triangles=dims_tri[-2],
        n=N, width=W, height=H, channels=C,
        instance_mode=instance_mode,
        alloc_triangles=_get_n_alloc_triangles(dims_tri[0]),
    )

    operands = (color, rast, pos, tri, ev_hash)
    return xla_client.ops.CustomCallWithLayout(
        c,
        b"antialias_fwd",
        operands=operands,
        operand_shapes_with_layout=tuple([c.get_shape(x) for x in operands]),
        shape_with_layout=xla_client.Shape.tuple_shape((out_shape, buf_shape)),
        opaque=opaque,
    )


def _antialias_grad_prim_abstract(color, rast, pos, tri, dy, work_buffer):
    # check
    check_array("dy", dy, shapes=[color.shape], dtype=color.dtype)
    # return
    return (
        ShapedArray(color.shape, jax.dtypes.canonicalize_dtype(color.dtype)),
        ShapedArray(pos.shape,   jax.dtypes.canonicalize_dtype(pos.dtype))
    )


def _antialias_grad_prim_translation_gpu(c: XlaBuilder, color, rast, pos, tri, dy, work_buffer, *args):
    dims_color = c.get_shape(color).dimensions()
    dims_pos = c.get_shape(pos).dimensions()
    dims_tri = c.get_shape(tri).dimensions()

    # get mode booleans
    instance_mode = len(dims_pos) > 2
    # get output shape
    N, H, W, C = dims_color

    # Encapsulate the information using the 'opaque' parameter
    opaque = _impl_jax.build_antialias_descriptor(
        num_vertices=dims_pos[-2], num_triangles=dims_tri[-2],
        n=N, width=W, height=H, channels=C,
        instance_mode=instance_mode,
        alloc_triangles=_get_n_alloc_triangles(dims_tri[0])
    )

    operands = (color, rast, pos, tri, dy, work_buffer)
    return xla_client.ops.CustomCallWithLayout(
        c,
        b"antialias_bwd",
        operands=operands,
        operand_shapes_with_layout=tuple([c.get_shape(x) for x in operands]),
        shape_with_layout=xla_client.Shape.tuple_shape((c.get_shape(color), c.get_shape(pos))),
        opaque=opaque,
    )


def _ev_hash_prim_abstract(tri):
    n_alloc = _get_n_alloc_triangles(tri.shape[0])
    n_elements = n_alloc * _impl_jax.get_aa_hash_elements_per_triangle() * 4
    return ShapedArray((n_elements,), jax.dtypes.canonicalize_dtype(jnp.uint32))


def _ev_hash_prim_translation_gpu(c: XlaBuilder, tri, *args):
    shape_tri = c.get_shape(tri)
    dims_tri = shape_tri.dimensions()
    # get output shape
    F = dims_tri[0]
    n_alloc = _get_n_alloc_triangles(F)
    n_elements = n_alloc * _impl_jax.get_aa_hash_elements_per_triangle() * 4
    out_shape = xla_client.Shape.array_shape(jax.dtypes.canonicalize_dtype(jnp.uint32), [n_elements], [0])

    # Encapsulate the information using the 'opaque' parameter
    opaque = _impl_jax.build_antialias_descriptor(
        num_vertices=0, num_triangles=n_alloc,
        n=0, width=0, height=0, channels=0,
        instance_mode=True,
        alloc_triangles=_get_n_alloc_triangles(dims_tri[0]),
    )

    operands = (tri,)
    return xla_client.ops.CustomCallWithLayout(
        c,
        b"antialias_get_ev_hash",
        operands=operands,
        operand_shapes_with_layout=tuple([c.get_shape(x) for x in operands]),
        shape_with_layout=out_shape,
        opaque=opaque,
    )


def _get_n_alloc_triangles(n_tris):
    n_alloc = 64
    while n_alloc < n_tris:
        n_alloc *= 2
    return n_alloc


# *****************************
# *  PRIMITIVE REGISTERATION  *
# *****************************

_antialias_prim = jax.core.Primitive("antialias")
_antialias_prim.multiple_results = True
_antialias_prim.def_impl(partial(xla.apply_primitive, _antialias_prim))
_antialias_prim.def_abstract_eval(_antialias_prim_abstract)
xla.backend_specific_translations["gpu"][_antialias_prim] = _antialias_prim_translation_gpu  # for JIT compilation

_antialias_grad_prim = jax.core.Primitive("antialias_grad")
_antialias_grad_prim.multiple_results = True
_antialias_grad_prim.def_impl(partial(xla.apply_primitive, _antialias_grad_prim))
_antialias_grad_prim.def_abstract_eval(_antialias_grad_prim_abstract)
xla.backend_specific_translations["gpu"][_antialias_grad_prim] = _antialias_grad_prim_translation_gpu  # for JIT compilation

_ev_hash_prim = jax.core.Primitive("get_ev_hash")
_ev_hash_prim.multiple_results = False
_ev_hash_prim.def_impl(partial(xla.apply_primitive, _ev_hash_prim))
_ev_hash_prim.def_abstract_eval(_ev_hash_prim_abstract)
xla.backend_specific_translations["gpu"][_ev_hash_prim] = _ev_hash_prim_translation_gpu  # for JIT compilation
