import os
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import nvdiffrast.jax as ops
from .meshio import load_mesh

ROOT = os.path.dirname(os.path.abspath(__file__))
TDIR = os.path.join(ROOT, "test_rast")
os.makedirs(TDIR, exist_ok=True)

verts, tris, _ = load_mesh(os.path.join(ROOT, "data/cow_mesh/cow.obj"))
verts -= verts.mean(axis=0)[None]
verts[:, 2] -= 1
verts = np.pad(verts, [[0, 0], [0, 1]], "constant", constant_values=1)
verts = verts[np.newaxis].repeat(4, axis=0)
print(verts.shape, np.prod(verts.shape))
print(tris.shape, np.prod(tris.shape))
A = 256
enable_db = True


def try_torch(verts, tris):
    import torch
    import torch.autograd
    import nvdiffrast.torch as dr
    print(">>> Run PyTorch version...")

    glctx = dr.RasterizeGLContext(enable_db)
    pos = torch.tensor(verts, dtype=torch.float32, device="cuda:0", requires_grad=True)
    tri = torch.tensor(tris, dtype=torch.int32, device="cuda:0")
    rast_out, out_db = dr.rasterize(glctx, pos, tri, (A, A))
    loss = rast_out.mean()
    print(loss)
    grad = torch.autograd.grad(loss, pos)[0]
    print("PyTorch grad: ", grad.shape, grad.min(), grad.max())

    rast_out = rast_out.detach().cpu().numpy()
    out_db = out_db.detach().cpu().numpy()
    cv2.imwrite(f"{TDIR}/torch_rast.png", np.clip(rast_out[0, ..., :3] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/torch_rast_db.png", np.clip(out_db[0, ..., :3] * 255, 0, 255).astype(np.uint8))
    np.save(f"{TDIR}/torch_rast.npy", rast_out)
    np.save(f"{TDIR}/torch_rast_db.npy", out_db)
    np.save(f"{TDIR}/torch_grad_pos.npy", grad.detach().cpu().numpy())


def try_jax(verts, tris):
    print(">>> Run Jax version...")

    pos = jnp.asarray(verts, dtype=jnp.float32)
    tri = jnp.asarray(tris, dtype=jnp.int32)

    def loss_fn(pos, tri):
        rast_out, out_db = ops.rasterize(pos, tri, (A, A), grad_db=enable_db)
        return rast_out.mean(), (rast_out, out_db)

    (loss, (rast_out, out_db)), grad_pos = jax.value_and_grad(loss_fn, has_aux=True)(pos, tri)
    print(rast_out.shape, out_db.shape, rast_out.mean())
    print('loss', loss)
    print('grad shape', grad_pos.shape, grad_pos.min(), grad_pos.max())

    rast_out = jax.device_get(rast_out)
    out_db = jax.device_get(out_db)
    grad_pos = jax.device_get(grad_pos)
    print(rast_out.dtype, type(rast_out))
    print(rast_out.shape)
    print(rast_out.min(), rast_out.max())
    cv2.imwrite(f"{TDIR}/jax_rast.png", np.clip(rast_out[0, ..., :3] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/jax_rast_db.png", np.clip(out_db[0, ..., :3] * 255, 0, 255).astype(np.uint8))
    np.save(f"{TDIR}/jax_rast.npy", rast_out)
    np.save(f"{TDIR}/jax_rast_db.npy", out_db)
    np.save(f"{TDIR}/jax_grad_pos.npy", grad_pos)


try_torch(verts, tris)
try_jax(verts, tris)

cmp_names = ["rast", "rast_db", "grad_pos"]
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
