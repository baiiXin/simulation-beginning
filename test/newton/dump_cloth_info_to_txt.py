import os
import numpy as np

def main():
    try:
        import warp.examples
        from pxr import Usd, UsdGeom
    except Exception as e:
        print(e)
        return

    asset_dir = warp.examples.get_asset_directory()
    usd_path = os.path.join(asset_dir, "square_cloth.usd")
    stage = Usd.Stage.Open(usd_path)
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/cloth/cloth"))

    points = np.array(geom.GetPointsAttr().Get())
    indices = np.array(geom.GetFaceVertexIndicesAttr().Get())
    faces = indices.reshape(-1, 3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "cloth_info.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"vertex_count: {points.shape[0]}\n")
        f.write("vertices:\n")
        for p in points:
            f.write(f"{float(p[0])} {float(p[1])} {float(p[2])}\n")
        f.write(f"triangle_count: {faces.shape[0]}\n")
        f.write("indices:\n")
        for tri in faces:
            f.write(f"{int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

    print(out_path)

if __name__ == "__main__":
    main()

