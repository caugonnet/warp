"""
Utilities for NUMBA-based STF operations.
"""

from numba import cuda


def init_logical_data(ctx, ld, value, data_place=None, exec_place=None):
    """
    Initialize a logical data with a constant value.

    Uses CuPy's optimized fill if available, otherwise falls back to a Numba kernel.

    Parameters
    ----------
    ctx : context
        STF context
    ld : logical_data
        Logical data to initialize
    value : scalar
        Value to fill the array with
    data_place : data_place, optional
        Data place for the initialization task
    exec_place : exec_place, optional
        Execution place for the fill operation
    """
    # Create write dependency with optional data place
    dep_arg = ld.write(data_place) if data_place else ld.write()

    # Create task arguments - include exec_place if provided
    task_args = []
    if exec_place is not None:
        task_args.append(exec_place)
    task_args.append(dep_arg)

    with ctx.task(*task_args) as t:
        # Get the array as a numba device array
        nb_stream = cuda.external_stream(t.stream_ptr())
        nb_array = t.numba_arguments()

        try:
            # Use CuPy's optimized fill operation (much faster than custom kernels)
            import cupy as cp

            with cp.cuda.Stream(nb_stream):
                cp_view = cp.asarray(nb_array)
                cp_view.fill(value)
        except ImportError:
            # Fallback to Numba kernel if CuPy not available
            _fill_with_fallback_kernel(nb_array, value, nb_stream)


@cuda.jit
def _fill_kernel_fallback(array, value):
    """Fallback kernel for filling arrays when CuPy is not available."""
    idx = cuda.grid(1)
    if idx < array.size:
        array.flat[idx] = value


def _fill_with_fallback_kernel(array, value, stream):
    """Fill array using Numba kernel when CuPy is unavailable."""
    total_size = array.size
    threads_per_block = 256
    blocks_per_grid = (total_size + threads_per_block - 1) // threads_per_block

    # Convert value to array's dtype
    typed_value = array.dtype.type(value)
    _fill_kernel_fallback[blocks_per_grid, threads_per_block, stream](
        array, typed_value
    )
