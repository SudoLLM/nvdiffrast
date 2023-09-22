from functools import partial
from typing import Any, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.core import ShapedArray, Primitive
from jax.lib import xla_client
from jax.interpreters import xla
from jaxlib.xla_extension import XlaBuilder, XlaOp

from .build import _impl_jax  # TODO: setup.py changes dir
from .utils import check_array


# **********************
# *  USER'S INTERFACE  *
# **********************

filter_mode_dict = {
    'nearest': 0,
    'linear': 1,
    'linear-mipmap-nearest': 2,
    'linear-mipmap-linear': 3,
}
boundary_mode_dict = {
    'cube': 0,
    'wrap': 1,
    'clamp': 2,
    'zero': 3,
}


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def texture(
    tex: Array,
    uv: Array,
    filter_mode: str = 'auto',
    boundary_mode: str = 'wrap',
) -> Tuple[Array, Array]:
    w, h = resolution
    _ranges = ranges if ranges is not None else jnp.empty((0, 2), dtype=jnp.int32)
    return _texture_prim.bind(pos, tri, _ranges, w=w, h=h, enable_db=grad_db)  # type: ignore


def texture_fwd(
    tex: Array,
    uv: Array,
    filter_mode: str = 'auto',
    boundary_mode: str = 'wrap',
):
    if filter_mode == 'auto':
        filter_mode = 'linear'
    # Convert filter mode to internal enumeration.
    filter_mode_enum = filter_mode_dict[filter_mode]
    # Convert boundary mode to internal enumeration.
    boundary_mode_enum = boundary_mode_dict[boundary_mode]

    out = _texture_prim.bind(tex, uv, filter_mode_enum, boundary_mode_enum)
    return out, (tex, uv)  # output, 'res' for bwd


# nondiff_argnums 2, 3 start the arguments list
def texture_bwd(
    filter_mode: str,
    boundary_mode: str,
    fwd_res: Tuple[Array, Array],
    d_out: Array,
):
    if filter_mode == 'auto':
        filter_mode = 'linear'
    # Convert filter mode to internal enumeration.
    filter_mode_enum = filter_mode_dict[filter_mode]
    # Convert boundary mode to internal enumeration.
    boundary_mode_enum = boundary_mode_dict[boundary_mode]

    tex, uv = fwd_res
    dy = d_out
    
    assert filter_mode == 'linear'
    g_tex, g_uv = _texture_grad_prim.bind(tex, uv, dy, filter_mode_enum, boundary_mode_enum)
    return (g_tex, g_uv)


texture.defvjp(texture_fwd, texture_bwd)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# Abstract Evaluation function for JIT. We also check shapes and dtypes here.
def _texture_prim_abstract(
    tex: ShapedArray,
    uv: ShapedArray,
    filter_mode_enum: int,
    boundary_mode_enum: int,
):
    # sanity check
    check_array("tex", tex, shapes=[(None, None, None, None)], dtype=jnp.float32)
    check_array("uv", uv, shapes=[(None, None, None, 2)], dtype=jnp.float32)
    
    # return
    dtype = jax.dtypes.canonicalize_dtype(tex.dtype)
    n, c = tex.shape[0], tex.shape[-1]
    h, w = uv.shape[1], uv.shape[2]
    return ShapedArray((n, h, w, c), dtype)


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _texture_prim_translation_gpu(
    builder: XlaBuilder,
    tex: XlaOp,
    uv: XlaOp,
    filter_mode_enum: int,
    boundary_mode_enum: int,
    *args: Any
):
    dtype = builder.get_shape(tex).element_type()
    shape_tex = builder.get_shape(tex)
    shape_uv = builder.get_shape(uv)
    dims_tex = shape_tex.dimensions()
    dims_uv = shape_uv.dimensions()
    
    n, c = dims_tex[0], dims_tex[-1]
    h, w = dims_uv[1], dims_uv[2]

    # get output shape
    out_shape = xla_client.Shape.array_shape(dtype, [n, h, w, c], [3, 2, 1, 0])

    # Encapsulate the information using the 'opaque' parameter
    opaque = _impl_jax.build_texture_descriptor(
        filter_mode=filter_mode_enum,
        boundary_mode=boundary_mode_enum,
        tex_n=dims_tex[0],
        tex_h=dims_tex[1],
        tex_w=dims_tex[2],
        tex_c=dims_tex[3],
        uv_n=dims_uv[0],
        uv_h=dims_uv[1],
        uv_w=dims_uv[2],
    )

    operands = (tex, uv)
    return xla_client.ops.CustomCallWithLayout(
        builder,
        b"texture_fwd",
        operands=operands,
        operand_shapes_with_layout=tuple([builder.get_shape(x) for x in operands]),
        shape_with_layout=xla_client.Shape.tuple_shape((out_shape,)),
        opaque=opaque,
    )


def _texture_grad_prim_abstract(
    tex: ShapedArray,
    uv: ShapedArray,
    dy: ShapedArray,
    filter_mode_enum: int,
    boundary_mode_enum: int,
):
    # check gradients
    check_array("dy", dy, shapes=[(uv.shape[0], uv.shape[1], uv.shape[2], tex.shape[3])], dtype=tex.dtype)
    # return abstract array
    dtype = jax.dtypes.canonicalize_dtype(tex.dtype)
    return (
        ShapedArray(tex.shape, dtype),
        ShapedArray(uv.shape, dtype),
    )


def _texture_grad_prim_translation_gpu(
    builder: XlaBuilder,
    tex: XlaOp,
    uv: XlaOp,
    dy: XlaOp,
    filter_mode_enum: int,
    boundary_mode_enum: int,
    *args: Any
):
    shape_tex = builder.get_shape(tex)
    shape_uv = builder.get_shape(uv)
    shape_dy = builder.get_shape(dy)
    dims_tex = shape_tex.dimensions()
    dims_uv = shape_uv.dimensions()
    dims_dy = shape_dy.dimensions()

    # Encapsulate the information using the 'opaque' parameter
    opaque = _impl_jax.build_texture_descriptor(
        filter_mode=filter_mode_enum,
        boundary_mode=boundary_mode_enum,
        tex_n=dims_tex[0],
        tex_h=dims_tex[1],
        tex_w=dims_tex[2],
        tex_c=dims_tex[3],
        uv_n=dims_uv[0],
        uv_h=dims_uv[1],
        uv_w=dims_uv[2],
    )

    operands = (tex, uv, dy)
    return xla_client.ops.CustomCallWithLayout(
        builder,
        b"texture_bwd",
        operands=operands,
        operand_shapes_with_layout=tuple([builder.get_shape(x) for x in operands]),
        shape_with_layout=xla_client.Shape.tuple_shape((shape_tex, shape_uv)),
        opaque=opaque,
    )


# *****************************
# *  PRIMITIVE REGISTERATION  *
# *****************************

_texture_prim = Primitive("texture")
_texture_prim.multiple_results = True
_texture_prim.def_impl(partial(xla.apply_primitive, _texture_prim))
_texture_prim.def_abstract_eval(_texture_prim_abstract)
xla.backend_specific_translations["gpu"][_texture_prim] = _texture_prim_translation_gpu  # for JIT compilation

_texture_grad_prim = Primitive("texture_grad")
_texture_grad_prim.multiple_results = True
_texture_grad_prim.def_impl(partial(xla.apply_primitive, _texture_grad_prim))
_texture_grad_prim.def_abstract_eval(_texture_grad_prim_abstract)
xla.backend_specific_translations["gpu"][_texture_grad_prim] = _texture_grad_prim_translation_gpu  # for JIT compilation

