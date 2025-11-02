import warp as wp

BLOCKSIZE = wp.constant(2)


# Dummy version of Cute's layout
@wp.struct
class layout:
    shape: wp.array(dtype=int)
    stride: wp.array(dtype=int)
    tile_shape: wp.array(dtype=int)


@wp.func
def apply_layout(L: layout, block_idx: int):
    subpart_shape, subpart_stride = L.shape, L.stride

    local = int(0)
    remaining_block = block_idx
    for i in range(len(subpart_shape) - 1, -1, -1):
        coord = remaining_block % subpart_shape[i]
        remaining_block = remaining_block // subpart_shape[i]
        local += coord * subpart_stride[i]

    return local


@wp.func
def index_to_grid(grid_shape: wp.array(dtype=int), idx: int):
    return idx % grid_shape[0], idx // grid_shape[0]


@wp.func
def coord_to_index(grid_shape: wp.array(dtype=int), x: int, y: int):
    return x + y * grid_shape[0]


@wp.func
def apply_2d_layout(L: layout, x: int, y: int):
    """Apply a 2D layout transformation to (x,y) coordinates.

    Returns the transformed (x', y') coordinates according to the layout.
    """
    # Convert (x,y) to linear index
    linear_idx = coord_to_index(L.shape, x, y)

    # Apply layout transformation
    transformed_idx = apply_layout(L, linear_idx)

    # Convert back to 2D coordinates
    return index_to_grid(L.shape, transformed_idx)


@wp.kernel
def transpose(A: wp.array2d(dtype=float), layoutC: layout, C: wp.array2d(dtype=float)):
    x, y = wp.tid()

    # Load tile from A and transpose it
    a = wp.tile_load(A, shape=(BLOCKSIZE, BLOCKSIZE), offset=(BLOCKSIZE * x, BLOCKSIZE * y))
    at = wp.tile_transpose(a)

    # Apply layout to determine output position (swap x,y for transpose)
    ltx, lty = apply_2d_layout(layoutC, y, x)

    # Store to output
    wp.tile_store(C, at, offset=(BLOCKSIZE * ltx, BLOCKSIZE * lty))


@wp.kernel
def copy(SRC: wp.array2d(dtype=float), layoutSRC: layout, DST: wp.array2d(dtype=float)):
    x, y = wp.tid()

    # Apply layout to find source position
    ltx, lty = apply_2d_layout(layoutSRC, x, y)

    # Load from transformed position, store to regular position
    t = wp.tile_load(SRC, shape=(BLOCKSIZE, BLOCKSIZE), offset=(BLOCKSIZE * ltx, BLOCKSIZE * lty))
    wp.tile_store(DST, t, offset=(BLOCKSIZE * x, BLOCKSIZE * y))


A = wp.empty((10, 10), dtype=float)
C = wp.empty((10, 10), dtype=float)

# Initialize layout struct with arrays
shape_arr = wp.array([5, 5], dtype=int)
stride_arr = wp.array([1, 5], dtype=int)
tile_shape_arr = wp.array([2, 2], dtype=int)
layoutC = layout()
layoutC.shape = shape_arr
layoutC.stride = stride_arr
layoutC.tile_shape = tile_shape_arr


@wp.kernel
def range_fill_kernel(out: wp.array2d(dtype=float)):
    i = wp.tid() // 10
    j = wp.tid() % 10
    out[i, j] = wp.float(wp.tid())


wp.launch(range_fill_kernel, dim=10 * 10, outputs=[A])
print("Input array A:")
print(A.numpy())

wp.launch_tiled(transpose, dim=(5, 5), inputs=[A, layoutC], outputs=[C], device="cuda:0", block_dim=32)
print("\nTransposed array C (with special layout):")
print(C.numpy())

D = wp.empty((10, 10), dtype=float)
wp.launch_tiled(copy, dim=(5, 5), inputs=[C, layoutC], outputs=[D], device="cuda:0", block_dim=32)
print("\nCopied back to regular layout:")
print(D.numpy())
