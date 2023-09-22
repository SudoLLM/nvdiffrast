import os
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import nvdiffrast.jax as ops
from .meshio import load_mesh

ROOT = os.path.dirname(os.path.abspath(__file__))
TDIR = os.path.join(ROOT, "test_texutre")
os.makedirs(TDIR, exist_ok=True)

texture = cv2.imread(os.path.join(ROOT, "data/cow_mesh/cow_texture.png"))
texture = cv2.resize(texture, (256, 256)).astype(np.float32) / 255.0
verts, tris, aux = load_mesh(os.path.join(ROOT, "data/cow_mesh/cow.obj"))
# verts -= verts.mean(axis=0)[None]  # move to center.
# verts[:, 2] *= -1
# verts[:, 0] *= -1
verts[:, 2] -= 1  # move to camera

# vertices -> triangles.
verts = verts[tris.flatten()]
tris = np.arange(verts.shape[0], dtype=tris.dtype).reshape(-1, 3)
uvs = aux["verts_uv"][aux["faces_tex"].flatten()]

# homo
verts = np.pad(verts, [[0, 0], [0, 1]], "constant", constant_values=1)

# batch
verts = verts[np.newaxis].repeat(4, axis=0)
uvs   = uvs[np.newaxis].repeat(4, axis=0)
texs  = texture[np.newaxis].repeat(4, axis=0)
viz_bi = 1

A = 256


def try_torch(verts, tris):
    import torch
    import torch.autograd
    import nvdiffrast.torch as dr
    print(">>> Run PyTorch version...")

    glctx = dr.RasterizeCudaContext("cuda:0")
    pos = torch.tensor(verts, dtype=torch.float32, device="cuda:0")
    tri = torch.tensor(tris, dtype=torch.int32, device="cuda:0")
    tex = torch.tensor(texs, dtype=torch.float32, device="cuda:0", requires_grad=True)
    uv = torch.tensor(uvs, dtype=torch.float32, device="cuda:0", requires_grad=True)

    rast_out, _ = dr.rasterize(glctx, pos, tri, (A, A))
    pix_uv, _ = dr.interpolate(uv, rast_out, tri)
    color = dr.texture(tex, pix_uv)
    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.

    loss = color.mean()
    grad_tex, grad_uv = torch.autograd.grad(loss, [tex, uv])[:4]
    # grad_pos = torch.autograd.grad(loss, pos)[0]
    # print("PyTorch: Grad shape is", grad_rast.shape, ", range is", grad_rast.min(), grad_rast.max())
    # print("PyTorch: Grad shape is", grad_pos.shape, ", range is", grad_pos.min(), grad_pos.max())

    rast_out = rast_out.detach().cpu().numpy()
    pix_uv = pix_uv.detach().cpu().numpy()
    pix_uv = np.pad(pix_uv, [(0, 0)] * 3 + [(0, 1)])
    color = color.detach().cpu().numpy()
    cv2.imwrite(f"{TDIR}/torch_rast.png", np.clip(rast_out[viz_bi, ..., :3] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/torch_uv.png", np.clip(pix_uv[viz_bi, ..., :] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/torch_color.png", np.clip(color[viz_bi, ..., :] * 255, 0, 255).astype(np.uint8))

    grad_tex = grad_tex.detach().cpu().numpy()
    grad_uv = grad_uv.detach().cpu().numpy()
    np.save(f"{TDIR}/torch_rast.npy", rast_out)
    np.save(f"{TDIR}/torch_uv.npy", pix_uv)
    np.save(f"{TDIR}/torch_color.npy", color)
    np.save(f"{TDIR}/torch_grad_tex.npy", grad_tex)
    np.save(f"{TDIR}/torch_grad_uv.npy", grad_uv)


def try_jax(verts, tris):
    print(">>> Run Jax version...")

    pos = jnp.asarray(verts, dtype=jnp.float32)
    tri = jnp.asarray(tris, dtype=jnp.int32)
    tex = jnp.asarray(texs, dtype=jnp.float32)
    uv = jnp.asarray(uvs, dtype=jnp.float32)

    def loss_fn(tex, uv, pos, tri):
        rast_out, _ = ops.rasterize(None, pos, tri, (A, A), grad_db=True)
        pix_uv, _ = ops.interpolate(uv, rast_out, tri)
        color = ops.texture(tex, pix_uv)
        color = color * jnp.clip(rast_out[..., -1:], 0, 1) # Mask out background.
        return color.mean(), (rast_out, pix_uv, color)

    (loss, (rast_out, pix_uv, color)), (grad_tex, grad_uv) = \
        jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(tex, uv, pos, tri)

    # print('Jax: Grad shape is', grad_rast.shape, ", range is: ", grad_rast.min(), grad_rast.max())
    # print('Jax: Grad shape is', grad_pos.shape, ", range is: ", grad_pos.min(), grad_pos.max())

    rast_out = jax.device_get(rast_out)
    pix_uv = jax.device_get(pix_uv)
    pix_uv = np.pad(pix_uv, [(0, 0)] * 3 + [(0, 1)])
    color = jax.device_get(color)
    grad_tex = jax.device_get(grad_tex)
    grad_uv = jax.device_get(grad_uv)
    cv2.imwrite(f"{TDIR}/jax_rast.png", np.clip(rast_out[viz_bi, ..., :3] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/jax_uv.png", np.clip(pix_uv[viz_bi, ..., :] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/jax_color.png", np.clip(color[viz_bi, ..., :] * 255, 0, 255).astype(np.uint8))
    np.save(f"{TDIR}/jax_rast.npy", rast_out)
    np.save(f"{TDIR}/jax_uv.npy", pix_uv)
    np.save(f"{TDIR}/jax_color.npy", color)
    np.save(f"{TDIR}/jax_grad_tex.npy", grad_tex)
    np.save(f"{TDIR}/jax_grad_uv.npy", grad_uv)


try_torch(verts, tris)
try_jax(verts, tris)

cmp_names = ["rast", "uv", "color", "grad_tex", "grad_uv"]
for name in cmp_names:
    out_trh = np.load(f"{TDIR}/torch_{name}.npy")
    out_jax = np.load(f"{TDIR}/jax_{name}.npy")
    print(f"> Compare '{name} ({out_jax.shape})'...")
    print(f"  torch: {out_trh.min():.10f}~{out_trh.max():.10f}")
    print(f"  jax:   {out_jax.min():.10f}~{out_jax.max():.10f}")
    if not np.all(out_trh == out_jax):
        max_delta = np.abs(out_trh - out_jax).max()
        print(f"! Jax outputs different '{name}' from torch's. Max delta is {max_delta}")
        print(f"! Is close? {np.all(np.isclose(out_trh, out_jax))}")
    else:
        print("+ Pass!")
