import os
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import nvdiffrast.jax as ops
from .meshio import load_mesh

ROOT = os.path.dirname(os.path.abspath(__file__))
TDIR = os.path.join(ROOT, "test_rast")
os.makedirs(TDIR, exist_ok=True)

verts, tris, _ = load_mesh(os.path.join(ROOT, "data/cow_mesh/cow.obj"))
verts -= verts.mean(axis=0)[None]
verts[:, 2] -= 1

# vertices -> triangles.
verts = verts[tris.flatten()]
tris = np.arange(verts.shape[0], dtype=tris.dtype).reshape(-1, 3)

verts = np.pad(verts, [[0, 0], [0, 1]], mode="constant", constant_values=1)  # type: ignore
verts = verts[np.newaxis].repeat(4, axis=0)
viz_bi = 1
print(verts.shape, np.prod(verts.shape))
print(tris.shape, np.prod(tris.shape))
A = 256
enable_db = True


def try_tch(verts: npt.NDArray[np.float32], tris: npt.NDArray[np.int32]):
    import torch
    import torch.autograd
    import nvdiffrast.torch as dr
    print(">>> Run PyTorch version...")

    glctx = dr.RasterizeCudaContext(device="cuda:0")
    pos = torch.tensor(verts, dtype=torch.float32, device="cuda:0", requires_grad=True)
    tri = torch.tensor(tris, dtype=torch.int32, device="cuda:0")
    rast_out, out_db = dr.rasterize(glctx, pos, tri, (A, A))
    loss = rast_out.mean()
    print(loss)
    grad = torch.autograd.grad(loss, pos)[0]
    print("PyTorch grad: ", grad.shape, grad.min(), grad.max())

    rast_out = rast_out.detach().cpu().numpy()
    out_db = out_db.detach().cpu().numpy()
    cv2.imwrite(f"{TDIR}/tch_rast.png", np.clip(rast_out[viz_bi, ..., :3] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/tch_rast_db.png", np.clip(out_db[viz_bi, ..., :3] * 255, 0, 255).astype(np.uint8))
    np.save(f"{TDIR}/tch_rast.npy", rast_out)
    np.save(f"{TDIR}/tch_rast_db.npy", out_db)
    np.save(f"{TDIR}/tch_grad_pos.npy", grad.detach().cpu().numpy())


def try_jax(verts: npt.NDArray[np.float32], tris: npt.NDArray[np.int32]):
    print(">>> Run Jax version...")

    pos = jnp.asarray(verts, dtype=jnp.float32)
    tri = jnp.asarray(tris, dtype=jnp.int32)

    def loss_fn(pos: jax.Array, tri: jax.Array):
        rast_out, out_db = ops.rasterize(None, pos, tri, (A, A), grad_db=enable_db)
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
    cv2.imwrite(f"{TDIR}/jax_rast.png", np.clip(rast_out[viz_bi, ..., :3] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/jax_rast_db.png", np.clip(out_db[viz_bi, ..., :3] * 255, 0, 255).astype(np.uint8))
    np.save(f"{TDIR}/jax_rast.npy", rast_out)
    np.save(f"{TDIR}/jax_rast_db.npy", out_db)
    np.save(f"{TDIR}/jax_grad_pos.npy", grad_pos)


try_tch(verts, tris)
try_jax(verts, tris)

cmp_names = ["rast", "rast_db", "grad_pos"]
for name in cmp_names:
    out_tch = np.load(f"{TDIR}/tch_{name}.npy")
    out_jax = np.load(f"{TDIR}/jax_{name}.npy")
    if name.startswith("rast"):
        delta = out_tch - out_jax
        print(delta.shape)
        cv2.imwrite(f"{TDIR}/delta_{name}.png", np.clip(np.abs(delta[viz_bi, ..., :3]) * 255, 0, 255).astype(np.uint8))
    print(f"> Compare '{name} ({out_jax.shape})'...")
    print(f"  tch: {out_tch.min():.10f}~{out_tch.max():.10f}")
    print(f"  jax: {out_jax.min():.10f}~{out_jax.max():.10f}")
    if not np.all(out_tch == out_jax):
        max_delta = np.abs(out_tch - out_jax).max()
        print(f"! Jax outputs different '{name}' from torch's. Max delta is {max_delta}")
        print(f"! Is close? {np.all(np.isclose(out_tch, out_jax))}")
    else:
        print("+ Pass!")
