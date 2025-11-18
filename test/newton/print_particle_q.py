import numpy as np
import os


class FakeViewer:
    def set_model(self, model):
        pass

    def apply_forces(self, state):
        pass

    def begin_frame(self, t):
        pass

    def log_state(self, state):
        pass

    def end_frame(self):
        pass


def main():
    try:
        import warp as wp
        import warp.examples
        from pxr import Usd, UsdGeom
        from newton.examples.cloth.example_cloth_twist import Example
    except Exception as e:
        print("依赖缺失:", e)
        return

    class ExampleNoCapture(Example):
        def capture(self):
            self.graph = None

    example = ExampleNoCapture(FakeViewer())
    arr_wp = example.state_0.particle_q
    arr_np = arr_wp.numpy()
    print("type:", type(arr_wp))
    print("dtype_wp:", arr_wp.dtype)
    print("shape_wp:", arr_wp.shape)
    print("dtype_np:", arr_np.dtype)
    print("shape_np:", arr_np.shape)
    print("head_np:", arr_np[:10])
    print("min:", arr_np.min(axis=0))
    print("max:", arr_np.max(axis=0))

    usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth/cloth"))
    mesh_points = np.array(usd_geom.GetPointsAttr().Get())
    mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
    vertices = [wp.vec3(v) for v in mesh_points]

    print("mesh_points dtype:", mesh_points.dtype)
    print("mesh_points shape:", mesh_points.shape)
    print("mesh_indices dtype:", mesh_indices.dtype)
    print("mesh_indices shape:", mesh_indices.shape)
    print("vertices type:", type(vertices))
    print("vertices len:", len(vertices))
    print("vertices head:", vertices[:5])
    print("faces dtype:", example.faces.dtype)
    print("faces shape:", example.faces.shape)
    print("faces head:", example.faces[:5])


if __name__ == "__main__":
    main()