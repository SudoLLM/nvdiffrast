import os
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import nvdiffrast.jax as ops
from .meshio import load_mesh

ROOT = os.path.dirname(os.path.abspath(__file__))
TDIR = os.path.join(ROOT, "test_interp")
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

A = 256
enable_db = True
diff_attrs = 'all' if enable_db else None


def try_torch(verts, tris):
    import torch
    import torch.autograd
    import nvdiffrast.torch as dr
    print(">>> Run PyTorch version...")

    glctx = dr.RasterizeGLContext(enable_db)
    pos = torch.tensor(verts, dtype=torch.float32, device="cuda:0", requires_grad=True)
    tri = torch.tensor(tris, dtype=torch.int32, device="cuda:0")
    attr = torch.tensor(depth, dtype=torch.float32, device="cuda:0", requires_grad=True)

    rast_out, rast_db = dr.rasterize(glctx, pos, tri, (A, A))
    pix_depth, pix_depth_db = dr.interpolate(attr, rast_out, tri, rast_db, diff_attrs=diff_attrs)

    loss = pix_depth.mean() + pix_depth_db.mean()
    grad_rast, grad_rast_db, grad_attr, grad_pos = torch.autograd.grad(loss, [rast_out, rast_db, attr, pos])[:4]
    # grad_pos = torch.autograd.grad(loss, pos)[0]
    # print("PyTorch: Grad shape is", grad_rast.shape, ", range is", grad_rast.min(), grad_rast.max())
    # print("PyTorch: Grad shape is", grad_pos.shape, ", range is", grad_pos.min(), grad_pos.max())

    rast_out = rast_out.detach().cpu().numpy()
    pix_depth = pix_depth.detach().cpu().numpy()
    grad_rast = grad_rast.detach().cpu().numpy()
    grad_rast_db = grad_rast_db.detach().cpu().numpy()
    grad_attr = grad_attr.detach().cpu().numpy()
    grad_pos = grad_pos.detach().cpu().numpy()
    cv2.imwrite(f"{TDIR}/torch_rast.png", np.clip(rast_out[0, ..., :3] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/torch_depth.png", np.clip(pix_depth[0, ..., :] * 255, 0, 255).astype(np.uint8))
    np.save(f"{TDIR}/torch_rast.npy", rast_out)
    np.save(f"{TDIR}/torch_depth.npy", pix_depth)
    np.save(f"{TDIR}/torch_grad_rast.npy", grad_rast)
    np.save(f"{TDIR}/torch_grad_rast_db.npy", grad_rast_db)
    np.save(f"{TDIR}/torch_grad_attr.npy", grad_attr)
    np.save(f"{TDIR}/torch_grad_pos.npy", grad_pos)
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

    def loss_fn(rast_out, rast_db, attr, tri):
        pix_depth, pix_depth_db = ops.interpolate(attr, rast_out, tri, rast_db, diff_attrs=diff_attrs)
        return (pix_depth.mean() + pix_depth_db.mean()), (pix_depth, pix_depth_db)
    rast_out, rast_db = ops.rasterize(pos, tri, (A, A), enable_db=enable_db)
    (loss, (pix_depth, pix_depth_db)), (grad_rast, grad_rast_db, grad_attr) = \
        jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)(rast_out, rast_db, attr, tri)

    def loss_fn(pos, attr, tri):
        rast_out, rast_db = ops.rasterize(pos, tri, (A, A), enable_db=enable_db)
        pix_depth, pix_depth_db = ops.interpolate(attr, rast_out, tri, rast_db, diff_attrs=diff_attrs)
        return (pix_depth.mean() + pix_depth_db.mean()), (rast_out, rast_db, pix_depth, pix_depth_db)
    (loss, (rast_out, rast_db, pix_depth, pix_depth_db)), (grad_pos, grad_attr) = \
        jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(pos, attr, tri)

    # print('Jax: Grad shape is', grad_rast.shape, ", range is: ", grad_rast.min(), grad_rast.max())
    # print('Jax: Grad shape is', grad_pos.shape, ", range is: ", grad_pos.min(), grad_pos.max())

    rast_out = jax.device_get(rast_out)
    pix_depth = jax.device_get(pix_depth)
    grad_rast = jax.device_get(grad_rast)
    grad_rast_db = jax.device_get(grad_rast_db)
    grad_attr = jax.device_get(grad_attr)
    grad_pos = jax.device_get(grad_pos)
    cv2.imwrite(f"{TDIR}/jax_rast.png", np.clip(rast_out[0, ..., :3] * 255, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{TDIR}/jax_depth.png", np.clip(pix_depth[0, ..., :] * 255, 0, 255).astype(np.uint8))
    np.save(f"{TDIR}/jax_rast.npy", rast_out)
    np.save(f"{TDIR}/jax_depth.npy", pix_depth)
    np.save(f"{TDIR}/jax_grad_rast.npy", grad_rast)
    np.save(f"{TDIR}/jax_grad_rast_db.npy", grad_rast_db)
    np.save(f"{TDIR}/jax_grad_attr.npy", grad_attr)
    np.save(f"{TDIR}/jax_grad_pos.npy", grad_pos)
    if enable_db:
        rast_db = jax.device_get(rast_db)
        pix_depth_db = jax.device_get(pix_depth_db)
        np.save(f"{TDIR}/jax_rast_db.npy", rast_db)
        np.save(f"{TDIR}/jax_depth_db.npy", pix_depth)


try_torch(verts, tris)
try_jax(verts, tris)

cmp_names = ["rast", "depth"]
if enable_db:
    cmp_names.extend(["rast_db", "depth_db"])
cmp_names.append("grad_rast")
cmp_names.append("grad_rast_db")
cmp_names.append("grad_attr")
cmp_names.append("grad_pos")
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
