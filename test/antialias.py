import os
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import nvdiffrast.jax as ops
from .meshio import load_mesh

ROOT = os.path.dirname(os.path.abspath(__file__))
TDIR = os.path.join(ROOT, "test_antialias")
os.makedirs(TDIR, exist_ok=True)

verts, tris, _ = load_mesh(os.path.join(ROOT, "data/cow_mesh/cow.obj"))
# depth (scaled)
depth = verts[:, 2:3].copy()
depth = (depth - depth.min()) / (depth.max() - depth.min())
# move to center
verts -= verts.mean(axis=0)[None]
# move to camera
verts[:, 2] -= 1
# homo
verts = np.pad(verts, [[0, 0], [0, 1]], "constant", constant_values=1)
# batch
verts = verts[np.newaxis].repeat(4, axis=0)
depth = depth[np.newaxis].repeat(4, axis=0)
viz_bi = 1

A = 256
enable_db = True
diff_attrs = 'all' if enable_db else None


def try_torch(verts, tris):
    import torch
    import torch.autograd
    import nvdiffrast.torch as dr
    print(">>> Run PyTorch version...")

    glctx = dr.RasterizeCudaContext("cuda:0")
    pos = torch.tensor(verts, dtype=torch.float32, device="cuda:0", requires_grad=True)
    tri = torch.tensor(tris, dtype=torch.int32, device="cuda:0")
    attr = torch.tensor(depth, dtype=torch.float32, device="cuda:0", requires_grad=True)

    rast_out, rast_db = dr.rasterize(glctx, pos, tri, (A, A))
    pix_depth, pix_depth_db = dr.interpolate(attr, rast_out, tri, rast_db, diff_attrs=diff_attrs)
    # stop pix_depth's grad flow back to pos, only check this function
    pos = torch.tensor(verts, dtype=torch.float32, device="cuda:0", requires_grad=True)
    pix_depth_aa = dr.antialias(pix_depth, rast_out, pos, tri)

    loss = pix_depth_aa.mean()
    grad_col, grad_pos = torch.autograd.grad(loss, (pix_depth, pos))[:2]
    print(grad_col.shape)
    print(grad_pos.shape)

    rast_out = rast_out.detach().cpu().numpy()
    pix_depth = pix_depth.detach().cpu().numpy()
    pix_depth_aa = pix_depth_aa.detach().cpu().numpy()
    grad_col = grad_col.detach().cpu().numpy()
    grad_pos = grad_pos.detach().cpu().numpy()
    cv2.imwrite(f"{TDIR}/torch_rast.png", np.clip(rast_out[viz_bi, ..., :3] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/torch_depth.png", np.clip(pix_depth[viz_bi, ..., :] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/torch_depth_aa.png", np.clip(pix_depth_aa[viz_bi, ..., :] * 255, 0, 255).astype(np.uint8))
    np.save(f"{TDIR}/torch_rast.npy", rast_out)
    np.save(f"{TDIR}/torch_depth.npy", pix_depth)
    np.save(f"{TDIR}/torch_depth_aa.npy", pix_depth_aa)
    np.save(f"{TDIR}/torch_grad_col.npy", grad_col)
    np.save(f"{TDIR}/torch_grad_pos.npy", grad_pos)
    print("Max delta between non-aa and aa", np.abs(pix_depth - pix_depth_aa).max())
    if enable_db:
        rast_db = rast_db.detach().cpu().numpy()
        pix_depth_db = pix_depth_db.detach().cpu().numpy()
        np.save(f"{TDIR}/torch_rast_db.npy", rast_db)
        np.save(f"{TDIR}/torch_depth_db.npy", pix_depth)


def try_jax(verts, tris):
    print(">>> Run Jax version...")

    pos = jnp.asarray(verts, dtype=jnp.float32)
    tri = jnp.asarray(tris, dtype=jnp.int32)
    attr = jnp.asarray(depth, dtype=jnp.float32)

    rast_out, rast_db = ops.rasterize(pos, tri, (A, A), grad_db=enable_db)
    pix_depth, pix_depth_db = ops.interpolate(attr, rast_out, tri, rast_db, diff_attrs=diff_attrs)

    ev_hash = ops.get_ev_hash(tri)
    print(ev_hash.shape)

    def loss_fn(pix_depth, pos):
        pix_depth_aa = ops.antialias(pix_depth, rast_out, pos, tri, ev_hash=ev_hash)
        return pix_depth_aa.mean(), (pix_depth_aa, )
    
    (loss, (pix_depth_aa,)), (grad_col, grad_pos) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
        pix_depth, pos
    )
    print(grad_col.shape)
    print(grad_pos.shape)

    rast_out = jax.device_get(rast_out)
    pix_depth = jax.device_get(pix_depth)
    pix_depth_aa = jax.device_get(pix_depth_aa)
    grad_col = jax.device_get(grad_col)
    grad_pos = jax.device_get(grad_pos)
    cv2.imwrite(f"{TDIR}/jax_rast.png", np.clip(rast_out[viz_bi, ..., :3] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/jax_depth.png", np.clip(pix_depth[viz_bi, ..., :] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/jax_depth_aa.png", np.clip(pix_depth_aa[viz_bi, ..., :] * 255, 0, 255).astype(np.uint8))
    np.save(f"{TDIR}/jax_rast.npy", rast_out)
    np.save(f"{TDIR}/jax_depth.npy", pix_depth)
    np.save(f"{TDIR}/jax_depth_aa.npy", pix_depth_aa)
    np.save(f"{TDIR}/jax_grad_col.npy", grad_col)
    np.save(f"{TDIR}/jax_grad_pos.npy", grad_pos)
    if enable_db:
        rast_db = jax.device_get(rast_db)
        pix_depth_db = jax.device_get(pix_depth_db)
        np.save(f"{TDIR}/jax_rast_db.npy", rast_db)
        np.save(f"{TDIR}/jax_depth_db.npy", pix_depth)


try_torch(verts, tris)
try_jax(verts, tris)

cmp_names = ["rast", "depth", "depth_aa", "grad_col", "grad_pos"]
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
