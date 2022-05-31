from functools import partial
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jax.lib import xla_client
from jax.interpreters import xla
from jax.abstract_arrays import ShapedArray
from jaxlib.xla_extension import XlaBuilder

from .build import _impl_jax  # TODO: setup.py changes dir
from .utils import check_array


# **********************
# *  USER'S INTERFACE  *
# **********************

@partial(jax.custom_vjp, nondiff_argnums=(2, 4))
def rasterize(
    pos: jnp.ndarray,
    tri: jnp.ndarray,
    resolution: Tuple[int, int],
    ranges: Optional[jnp.ndarray] = None,
    grad_db: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    w, h = resolution
    _ranges = ranges if ranges is not None else jnp.empty((0, 2), dtype=jnp.int32)
    return _rasterize_prim.bind(pos, tri, _ranges, w=w, h=h, enable_db=grad_db)


def rasterize_fwd(pos, tri, resolution, ranges, grad_db):
    w, h = resolution
    _ranges = ranges if ranges is not None else jnp.empty((0, 2), dtype=jnp.int32)
    rast_out, db_out = _rasterize_prim.bind(pos, tri, _ranges, w=w, h=h, enable_db=grad_db)
    return (rast_out, db_out), (pos, tri, rast_out)  # output, 'res' for bwd


# nondiff_argnums 2, 3 start the arguments list
def rasterize_bwd(resolution, grad_db, fwd_res, d_out):
    pos, tri, out = fwd_res
    dy, ddb = d_out
    w, h = resolution
    grad = _rasterize_grad_prim.bind(pos, tri, out, dy, ddb, w=w, h=h, enable_db=grad_db)
    return (grad[0], None, None)


rasterize.defvjp(rasterize_fwd, rasterize_bwd)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# Abstract Evaluation function for JIT. We also check shapes and dtypes here.
def _rasterize_prim_abstract(pos, tri, ranges, w, h, enable_db):
    # sanity check
    check_array("pos", pos, shapes=[(None, None, 4), (None, 4)], dtype=jnp.float32)
    check_array("tri", tri, shapes=[(None, 3)], dtype=jnp.int32)
    if len(pos.shape) == 2:
        # ranges mode
        check_array("ranges", ranges, shapes=[(None, 2)], dtype=jnp.int32)
    # return
    dtype = jax.dtypes.canonicalize_dtype(pos.dtype)
    n = pos.shape[0] if len(pos.shape) > 2 else ranges.shape[0]
    assert n > 0
    return (
        ShapedArray((n, h, w, 4), dtype),
        ShapedArray((n, h, w, 4 if enable_db else 0), dtype)
    )


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _rasterize_prim_translation_gpu(c: XlaBuilder, pos, tri, ranges, w, h, enable_db, *args):
    dtype = c.get_shape(pos).element_type()
    shape_pos = c.get_shape(pos)
    shape_tri = c.get_shape(tri)
    dims_pos = shape_pos.dimensions()
    dims_tri = shape_tri.dimensions()
    dims_rng = c.get_shape(ranges).dimensions()

    instance_mode = len(dims_pos) > 2
    if instance_mode:
        N, V, F = dims_pos[0], dims_pos[1], dims_tri[0]
        pos_count = N * V * 4
        vtx_per_instance = V
    else:
        N, V, F = dims_rng[0], dims_pos[0], dims_tri[0]
        pos_count = V * 4
        vtx_per_instance = 0
    # get output shape
    out_shape = xla_client.Shape.array_shape(dtype, [N, h, w, 4], [3, 2, 1, 0])
    odb_shape = xla_client.Shape.array_shape(dtype, [N, h, w, 4 if enable_db else 0], [3, 2, 1, 0])

    # Encapsulate the information using the 'opaque' parameter
    opaque = _impl_jax.build_rasterize_descriptor(
        num_vertices=V, num_triangles=F,
        width=w, height=h, depth=N,
        enable_db=enable_db,
        instance_mode=instance_mode,
        pos_count=pos_count,
        tri_count=F*3,
        vtx_per_instance=vtx_per_instance,
    )

    operands = (pos, tri, ranges)
    return xla_client.ops.CustomCallWithLayout(
        c,
        b"rasterize_fwd",
        operands=operands,
        operand_shapes_with_layout=tuple([c.get_shape(x) for x in operands]),
        shape_with_layout=xla_client.Shape.tuple_shape((out_shape, odb_shape)),
        opaque=opaque,
    )


def _rasterize_grad_prim_abstract(pos, tri, out, dy, ddb, w, h, enable_db):
    # check gradients
    check_array("dy", dy, shapes=[out.shape], dtype=out.dtype)
    check_array("ddb", ddb, shapes=[out.shape[:3] + ((0, 4),)], dtype=out.dtype)
    # return abstract array
    dtype = jax.dtypes.canonicalize_dtype(pos.dtype)
    N, V, _4 = pos.shape
    return (ShapedArray((N, V, _4), dtype),)


def _rasterize_grad_prim_translation_gpu(c: XlaBuilder, pos, tri, out, dy, ddb, w, h, enable_db, *args):
    shape_pos = c.get_shape(pos)
    shape_tri = c.get_shape(tri)
    dims_pos = shape_pos.dimensions()
    dims_tri = shape_tri.dimensions()

    instance_mode = len(dims_pos) > 2
    N = c.get_shape(out).dimensions()[0]
    if instance_mode:
        V, F = dims_pos[1], dims_tri[0]
        pos_count = N * V * 4
        vtx_per_instance = V
    else:
        V, F = dims_pos[0], dims_tri[0]
        pos_count = V * 4
        vtx_per_instance = 0

    # Encapsulate the information using the 'opaque' parameter
    opaque = _impl_jax.build_rasterize_descriptor(
        num_vertices=V, num_triangles=F,
        width=w, height=h, depth=N,
        enable_db=enable_db,
        instance_mode=instance_mode,
        pos_count=pos_count,
        tri_count=F*3,
        vtx_per_instance=vtx_per_instance,
    )

    operands = (pos, tri, out, dy, ddb)
    return xla_client.ops.CustomCallWithLayout(
        c,
        b"rasterize_bwd",
        operands=operands,
        operand_shapes_with_layout=tuple([c.get_shape(x) for x in operands]),
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
