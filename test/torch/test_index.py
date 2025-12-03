import torch

def main():
    Jacobi = torch.tensor([
        [
            [
                [100.0, 101.0, 102.0],
                [110.0, 111.0, 112.0],
            ],
            [
                [200.0, 201.0, 202.0],
                [210.0, 211.0, 212.0],
            ],
            [
                [300.0, 301.0, 302.0],
                [310.0, 311.0, 312.0],
            ],
        ],
        [
            [
                [400.0, 401.0, 402.0],
                [410.0, 411.0, 412.0],
            ],
            [
                [500.0, 501.0, 502.0],
                [510.0, 511.0, 512.0],
            ],
            [
                [600.0, 601.0, 602.0],
                [610.0, 611.0, 612.0],
            ],
        ],
    ], dtype=torch.float32)

    jdx = 1
    space_jdx = 2
    print("Jacobi_shape", Jacobi.shape)
    print("Jacobi_values\n", Jacobi)
    out = Jacobi[:, jdx, :, space_jdx]
    print(f"index [:, {jdx}, :, {space_jdx}]")
    print("slice_shape", out.shape)
    print("slice_values\n", out)

    # assign
    out *= -1.0
    print("assign_values\n", out)
    Jacobi[:, jdx, :, space_jdx] = out
    print("assign_values\n", Jacobi)

if __name__ == "__main__":
    main()