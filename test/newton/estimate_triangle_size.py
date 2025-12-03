import os
import numpy as np

def main():
    try:
        import warp.examples
        from pxr import Usd, UsdGeom
    except Exception as e:
        print(e)
        return

    usd_path = os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd")
    stage = Usd.Stage.Open(usd_path)
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/cloth/cloth"))

    points = np.array(geom.GetPointsAttr().Get(), dtype=np.float64)
    indices = np.array(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    tris = indices.reshape(-1, 3)

    P = points[tris]
    e01 = np.linalg.norm(P[:, 1] - P[:, 0], axis=1)
    e12 = np.linalg.norm(P[:, 2] - P[:, 1], axis=1)
    e20 = np.linalg.norm(P[:, 0] - P[:, 2], axis=1)
    edges = np.concatenate([e01, e12, e20])

    areas = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)

    xs = np.unique(points[:, 0])
    zs = np.unique(points[:, 2])
    dx = np.min(np.diff(xs)) if xs.size > 1 else 0.0
    dz = np.min(np.diff(zs)) if zs.size > 1 else 0.0

    scale = 0.01

    print("vertex_count:", points.shape[0])
    print("triangle_count:", tris.shape[0])
    print("grid_dx:", float(dx))
    print("grid_dz:", float(dz))
    print("edge_len_min:", float(edges.min()))
    print("edge_len_max:", float(edges.max()))
    print("edge_len_mean:", float(edges.mean()))
    print("area_min:", float(areas.min()))
    print("area_max:", float(areas.max()))
    print("area_mean:", float(areas.mean()))
    print("scaled_edge_len_mean:", float(edges.mean() * scale))
    print("scaled_area_mean:", float(areas.mean() * scale * scale))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triangle_size.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"vertex_count: {points.shape[0]}\n")
        f.write(f"triangle_count: {tris.shape[0]}\n")
        f.write(f"grid_dx: {float(dx)}\n")
        f.write(f"grid_dz: {float(dz)}\n")
        f.write(f"edge_len_min: {float(edges.min())}\n")
        f.write(f"edge_len_max: {float(edges.max())}\n")
        f.write(f"edge_len_mean: {float(edges.mean())}\n")
        f.write(f"area_min: {float(areas.min())}\n")
        f.write(f"area_max: {float(areas.max())}\n")
        f.write(f"area_mean: {float(areas.mean())}\n")
        f.write(f"scaled_edge_len_mean: {float(edges.mean() * scale)}\n")
        f.write(f"scaled_area_mean: {float(areas.mean() * scale * scale)}\n")
    print(out_path)

if __name__ == "__main__":
    main()

