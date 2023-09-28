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

@partial(jax.custom_vjp, nondiff_argnums=(4,))
def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    _rast_db, diff_attrs = _parse_none_inputs(attr, rast, rast_db, diff_attrs)
    return _interpolate_prim.bind(attr, rast, tri, _rast_db, diff_attrs=diff_attrs)


def interpolate_fwd(attr, rast, tri, rast_db, diff_attrs):
    _rast_db, diff_attrs = _parse_none_inputs(attr, rast, rast_db, diff_attrs)
    pix_attr, pix_attr_db = _interpolate_prim.bind(attr, rast, tri, _rast_db, diff_attrs=diff_attrs)
    return (pix_attr, pix_attr_db), (attr, rast, tri, rast_db)  # output, 'res' for bwd


# nondiff_argnums 4 start the arguments list
def interpolate_bwd(diff_attrs, fwd_res, d_out):
    attr, rast, tri, rast_db = fwd_res
    dy, dda = d_out
    _rast_db, diff_attrs = _parse_none_inputs(attr, rast, rast_db, diff_attrs)
    grad = _interpolate_grad_prim.bind(attr, rast, tri, dy, _rast_db, dda, diff_attrs=diff_attrs)
    return (grad[0], grad[1], None, (None if rast_db is None else grad[2]))


interpolate.defvjp(interpolate_fwd, interpolate_bwd)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

def _parse_none_inputs(attr, rast, rast_db=None, diff_attrs=None):
    if rast_db is None:
        N, H, W, _ = rast.shape
        rast_db = jnp.empty((N, H, W, 0), dtype=rast.dtype)
        if diff_attrs is not None:
            diff_attrs = None
    diff_attrs = _parse_diff_attrs(diff_attrs, attr.shape[-1])
    return rast_db, diff_attrs


def _parse_diff_attrs(diff_attrs, A):
    if diff_attrs is None:
        diff_attrs = tuple([])
    elif diff_attrs == "all":
        diff_attrs = tuple(range(A))
    assert isinstance(diff_attrs, (list, tuple))
    assert all(x < A for x in diff_attrs)
    return diff_attrs


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _interpolate_prim_abstract(attr, rast, tri, rast_db, diff_attrs):
    # check inputs
    check_array("attr", attr, shapes=[(None, None, None), (None, None)], dtype=jnp.float32)
    check_array("rast", rast, shapes=[(None, None, None, 4)], dtype=jnp.float32)
    check_array("tri", tri, shapes=[(None, 3)], dtype=jnp.int32)
    check_array("rast_db", rast_db, shapes=[(None, None, None, (0, 4))], dtype=jnp.float32)
    assert isinstance(diff_attrs, (tuple, list)), "invalid type: {}".format(type(diff_attrs))
    # return abstract array
    dtype = jax.dtypes.canonicalize_dtype(attr.dtype)
    N, H, W = rast.shape[:3]
    A = attr.shape[-1]
    D = len(diff_attrs)
    return (ShapedArray((N, H, W, A), dtype), ShapedArray((N, H, W, D*2), dtype))


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _interpolate_prim_translation_gpu(c: XlaBuilder, attr, rast, tri, rast_db, diff_attrs, *args):
    dtype = c.get_shape(attr).element_type()
    dims_attr = c.get_shape(attr).dimensions()
    dims_rast = c.get_shape(rast).dimensions()
    dims_tri = c.get_shape(tri).dimensions()
    dims_rast_db = c.get_shape(rast_db).dimensions()

    # get mode booleans
    instance_mode = len(dims_attr) == 3
    enable_db = dims_rast_db[-1] != 0
    # get output shape
    N, H, W, _ = dims_rast
    A = dims_attr[-1]
    D = len(diff_attrs)
    out_shape = xla_client.Shape.array_shape(dtype, [N, H, W, A], [3, 2, 1, 0])
    odb_shape = xla_client.Shape.array_shape(dtype, [N, H, W, D*2], [3, 2, 1, 0])

    # Encapsulate the information using the 'opaque' parameter
    attr_depth = 1
    if instance_mode:
        attr_depth = dims_attr[0] if len(dims_attr) > 1 else 0
    opaque = _impl_jax.build_interpolate_descriptor(
        num_vertices=dims_attr[-2], num_triangles=dims_tri[-2], num_attrs=A,
        attr_bc=1 if (instance_mode and dims_attr[0] == 1) else 0,
        attr_depth=attr_depth,
        width=W, height=H, depth=N, enable_db=enable_db, instance_mode=instance_mode,
        diff_attrs_all=(D == A), diff_attrs_list=diff_attrs,
    )

    operands = (attr, rast, tri, rast_db)
    return xla_client.ops.CustomCallWithLayout(
        c,
        b"interpolate_fwd",
        operands=operands,
        operand_shapes_with_layout=tuple([c.get_shape(x) for x in operands]),
        shape_with_layout=xla_client.Shape.tuple_shape((out_shape, odb_shape)),
        opaque=opaque,
    )


def _interpolate_grad_prim_abstract(attr, rast, tri, dy, rast_db, dda, diff_attrs):
    # check dy, dda shape
    check_array("dy", dy, shapes=[rast.shape[:3] + (attr.shape[-1],)], dtype=attr.dtype)
    check_array("dda", dda, shapes=[rast.shape[:3] + ((0, len(diff_attrs)*2),)], dtype=attr.dtype)
    # return abstract array
    dtype = jax.dtypes.canonicalize_dtype(attr.dtype)
    return (
        ShapedArray(attr.shape, dtype),
        ShapedArray(rast.shape, dtype),
        ShapedArray(rast_db.shape, dtype),
    )


def _interpolate_grad_prim_translation_gpu(c: XlaBuilder, attr, rast, tri, dy, rast_db, dda, diff_attrs, *args):
    dims_attr = c.get_shape(attr).dimensions()
    dims_rast = c.get_shape(rast).dimensions()
    dims_tri = c.get_shape(tri).dimensions()
    dims_rast_db = c.get_shape(rast_db).dimensions()

    # get mode booleans
    instance_mode = len(dims_attr) == 3
    enable_db = dims_rast_db[-1] != 0
    # get output shape
    N, H, W, _ = dims_rast
    A = dims_attr[-1]
    D = len(diff_attrs)

    # Encapsulate the information using the 'opaque' parameter
    attr_depth = 1
    if instance_mode:
        attr_depth = dims_attr[0] if len(dims_attr) > 1 else 0
    attr_bc = 1 if instance_mode and attr_depth < N else 0
    opaque = _impl_jax.build_interpolate_descriptor(
        num_vertices=dims_attr[-2], num_triangles=dims_tri[-2], num_attrs=A,
        attr_bc=attr_bc, attr_depth=attr_depth,
        width=W, height=H, depth=N,
        enable_db=enable_db, instance_mode=instance_mode,
        diff_attrs_all=(D == A), diff_attrs_list=diff_attrs,
    )

    operands = (attr, rast, tri, dy, rast_db, dda)
    return xla_client.ops.CustomCallWithLayout(
        c,
        b"interpolate_bwd",
        operands=operands,
        operand_shapes_with_layout=tuple([c.get_shape(x) for x in operands]),
        shape_with_layout=xla_client.Shape.tuple_shape((c.get_shape(attr), c.get_shape(rast), c.get_shape(rast_db))),
        opaque=opaque,
    )


# *****************************
# *  PRIMITIVE REGISTERATION  *
# *****************************

_interpolate_prim = jax.core.Primitive("interpolate")
_interpolate_prim.multiple_results = True
_interpolate_prim.def_impl(partial(xla.apply_primitive, _interpolate_prim))
_interpolate_prim.def_abstract_eval(_interpolate_prim_abstract)
xla.backend_specific_translations["gpu"][_interpolate_prim] = _interpolate_prim_translation_gpu  # for JIT compilation

_interpolate_grad_prim = jax.core.Primitive("interpolate_grad")
_interpolate_grad_prim.multiple_results = True
_interpolate_grad_prim.def_impl(partial(xla.apply_primitive, _interpolate_grad_prim))
_interpolate_grad_prim.def_abstract_eval(_interpolate_grad_prim_abstract)
xla.backend_specific_translations["gpu"][_interpolate_grad_prim] = _interpolate_grad_prim_translation_gpu  # for JIT compilation
