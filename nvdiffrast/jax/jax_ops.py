from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.lib import xla_client
from jax.interpreters import xla
from jax.abstract_arrays import ShapedArray
from jaxlib.xla_extension import XlaBuilder

from .build import _impl_jax  # TODO: setup.py changes dir

for _name, _value in _impl_jax.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


# **********************
# *  USER'S INTERFACE  *
# **********************

@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def rasterize(pos: jnp.ndarray, tri: jnp.ndarray, w: int, h: int, enable_db: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return _rasterize_prim.bind(pos, tri, w=w, h=h, enable_db=enable_db)


def rasterize_fwd(pos, tri, w, h, enable_db):
    rast_out, db_out = rasterize(pos, tri, w, h, enable_db)
    return (rast_out, db_out), (pos, rast_out)  # output, 'res' for bwd


# nondiff_argnums 1, 2, 3, 4 start the arguments list
def rasterize_bwd(tri, w, h, enable_db, fwd_res, d_out):
    pos, out = fwd_res
    dy, ddb = d_out
    grad = _rasterize_grad_prim.bind(pos, tri, out, dy, ddb, w=w, h=h, enable_db=enable_db)
    return tuple(grad)


rasterize.defvjp(rasterize_fwd, rasterize_bwd)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _rasterize_prim_abstract(pos, tri, w, h, enable_db):
    dtype = jax.dtypes.canonicalize_dtype(pos.dtype)
    shape = pos.shape
    return (
        ShapedArray((shape[0], h, w, 4), dtype),
        ShapedArray((shape[0], h, w, 4 if enable_db else 0), dtype)
    )


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _rasterize_prim_translation_gpu(c: XlaBuilder, pos, tri, w, h, enable_db, *args):
    dtype = c.get_shape(pos).element_type()
    shape_pos = c.get_shape(pos)
    shape_tri = c.get_shape(tri)
    dims_pos = shape_pos.dimensions()
    dims_tri = shape_tri.dimensions()
    # we are in instance-mode
    assert len(dims_pos) == 3 and dims_pos[-1] == 4, "[nvdiffrast]: instance mode, pos must have shape [>0, >0, 4]"
    assert len(dims_tri) == 2 and dims_tri[-1] == 3, "[nvdiffrast]: instance mode, tri must have shape [>0, 3]"
    N, V = dims_pos[:2]
    F = dims_tri[0]
    # get output shape
    out_shape = xla_client.Shape.array_shape(dtype, [N, h, w, 4], [3, 2, 1, 0])
    odb_shape = xla_client.Shape.array_shape(dtype, [N, h, w, 4 if enable_db else 0], [3, 2, 1, 0])

    # Encapsulate the information using the 'opaque' parameter
    opaque = _impl_jax.build_rasterize_descriptor(
        num_vertices=V, num_triangles=F, width=w, height=h, enable_db=enable_db,
        instance_mode=True, pos_count=N*V*4, tri_count=F*3, vtx_per_instance=V, depth=N,
    )

    return xla_client.ops.CustomCallWithLayout(
        c,
        b"rasterize_fwd",
        operands=(pos, tri),
        operand_shapes_with_layout=(c.get_shape(pos), c.get_shape(tri)),
        shape_with_layout=xla_client.Shape.tuple_shape((out_shape, odb_shape)),
        opaque=opaque,
    )


def _rasterize_grad_prim_abstract(pos, tri, out, dy, ddb, w, h, enable_db):
    dtype = jax.dtypes.canonicalize_dtype(pos.dtype)
    N, V, _4 = pos.shape
    return (ShapedArray((N, V, _4), dtype),)


def _rasterize_grad_prim_translation_gpu(c: XlaBuilder, pos, tri, out, dy, ddb, w, h, enable_db, *args):
    shape_pos = c.get_shape(pos)
    shape_tri = c.get_shape(tri)
    dims_pos = shape_pos.dimensions()
    dims_tri = shape_tri.dimensions()
    # we are in instance-mode
    assert len(dims_pos) == 3 and dims_pos[-1] == 4, "[nvdiffrast]: instance mode, pos must have shape [>0, >0, 4]"
    assert len(dims_tri) == 2 and dims_tri[-1] == 3, "[nvdiffrast]: instance mode, tri must have shape [>0, 3]"
    N, V = dims_pos[:2]
    F = dims_tri[0]

    # Encapsulate the information using the 'opaque' parameter
    opaque = _impl_jax.build_rasterize_descriptor(
        num_vertices=V, num_triangles=F, width=w, height=h, enable_db=enable_db,
        instance_mode=True, pos_count=N*V*4, tri_count=F*3, vtx_per_instance=V, depth=N,
    )

    return xla_client.ops.CustomCallWithLayout(
        c,
        b"rasterize_bwd",
        operands=(pos, tri, out, dy, ddb),
        operand_shapes_with_layout=(c.get_shape(pos), c.get_shape(tri), c.get_shape(out), c.get_shape(dy), c.get_shape(ddb)),
        shape_with_layout=xla_client.Shape.tuple_shape((shape_pos,)),
        opaque=opaque,
    )


# *****************************
# *  PRIMITIVE REGISTERATION  *
# *****************************

_rasterize_prim = jax.core.Primitive("rasterize")
_rasterize_prim.multiple_results = True
_rasterize_prim.def_impl(partial(xla.apply_primitive, _rasterize_prim))
_rasterize_prim.def_abstract_eval(_rasterize_prim_abstract)
xla.backend_specific_translations["gpu"][_rasterize_prim] = _rasterize_prim_translation_gpu  # for JIT compilation

_rasterize_grad_prim = jax.core.Primitive("rasterize_grad")
_rasterize_grad_prim.multiple_results = True
_rasterize_grad_prim.def_impl(partial(xla.apply_primitive, _rasterize_grad_prim))
_rasterize_grad_prim.def_abstract_eval(_rasterize_grad_prim_abstract)
xla.backend_specific_translations["gpu"][_rasterize_grad_prim] = _rasterize_grad_prim_translation_gpu  # for JIT compilation
