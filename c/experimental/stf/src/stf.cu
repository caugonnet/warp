//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cccl/c/experimental/stf/stf.h>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

extern "C" {

void stf_ctx_create(stf_ctx_handle* ctx)
{
  assert(ctx);
  *ctx = new context{};
}

void stf_ctx_create_graph(stf_ctx_handle* ctx)
{
  assert(ctx);
  *ctx = new context{graph_ctx()};
}

void stf_ctx_finalize(stf_ctx_handle ctx)
{
  assert(ctx);
  auto* context_ptr = static_cast<context*>(ctx);
  context_ptr->finalize();
  delete context_ptr;
}

cudaStream_t stf_fence(stf_ctx_handle ctx)
{
  assert(ctx);
  auto* context_ptr = static_cast<context*>(ctx);
  return context_ptr->fence();
}

void stf_logical_data(stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz)
{
  // Convenience wrapper: assume host memory
  stf_logical_data_with_place(ctx, ld, addr, sz, make_host_data_place());
}

/* Convert the C-API stf_data_place to a C++ data_place object */
data_place to_data_place(const stf_data_place* data_p)
{
  assert(data_p);

  switch (data_p->kind)
  {
    case STF_DATA_PLACE_HOST:
      return data_place::host();

    case STF_DATA_PLACE_MANAGED:
      return data_place::managed();

    case STF_DATA_PLACE_AFFINE:
      return data_place::affine();

    case STF_DATA_PLACE_DEVICE:
      return data_place::device(data_p->u.device.dev_id);

    default:
      assert(!"Invalid data place kind");
      return data_place::invalid(); // invalid data_place
  }
}

/* Convert the C-API stf_exec_place to a C++ exec_place object */
exec_place to_exec_place(const stf_exec_place* exec_p)
{
  assert(exec_p);

  switch (exec_p->kind)
  {
    case STF_EXEC_PLACE_HOST:
      return exec_place::host();

    case STF_EXEC_PLACE_DEVICE:
      return exec_place::device(exec_p->u.device.dev_id);

    default:
      assert(!"Invalid execution place kind");
      return exec_place{}; // invalid exec_place
  }
}

void stf_logical_data_with_place(
  stf_ctx_handle ctx, stf_logical_data_handle* ld, void* addr, size_t sz, stf_data_place dplace)
{
  assert(ctx);
  assert(ld);

  auto* context_ptr = static_cast<context*>(ctx);

  // Convert C data_place to C++ data_place
  data_place cpp_dplace = to_data_place(&dplace);

  // Create logical data with the specified data place
  auto ld_typed = context_ptr->logical_data(make_slice(static_cast<char*>(addr), sz), mv(cpp_dplace));

  // Store the logical_data_untyped directly as opaque pointer
  *ld = new logical_data_untyped{mv(ld_typed)};
}

void stf_logical_data_set_symbol(stf_logical_data_handle ld, const char* symbol)
{
  assert(ld);
  assert(symbol);

  auto* ld_ptr = static_cast<logical_data_untyped*>(ld);
  ld_ptr->set_symbol(symbol);
}

void stf_logical_data_destroy(stf_logical_data_handle ld)
{
  assert(ld);

  auto* ld_ptr = static_cast<logical_data_untyped*>(ld);
  delete ld_ptr;
}

void stf_logical_data_empty(stf_ctx_handle ctx, size_t length, stf_logical_data_handle* to)
{
  assert(ctx);
  assert(to);

  auto* context_ptr = static_cast<context*>(ctx);
  auto ld_typed     = context_ptr->logical_data(shape_of<slice<char>>(length));
  *to               = new logical_data_untyped{mv(ld_typed)};
}

void stf_token(stf_ctx_handle ctx, stf_logical_data_handle* ld)
{
  assert(ctx);
  assert(ld);

  auto* context_ptr = static_cast<context*>(ctx);
  *ld               = new logical_data_untyped{context_ptr->token()};
}

void stf_task_create(stf_ctx_handle ctx, stf_task_handle* t)
{
  assert(ctx);
  assert(t);

  auto* context_ptr = static_cast<context*>(ctx);
  *t                = new context::unified_task<>{context_ptr->task()};
}

void stf_task_set_exec_place(stf_task_handle t, const stf_exec_place* exec_p)
{
  assert(t);
  assert(exec_p);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  task_ptr->set_exec_place(to_exec_place(exec_p));
}

void stf_task_set_symbol(stf_task_handle t, const char* symbol)
{
  assert(t);
  assert(symbol);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  task_ptr->set_symbol(symbol);
}

void stf_task_add_dep(stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m)
{
  assert(t);
  assert(ld);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  auto* ld_ptr   = static_cast<logical_data_untyped*>(ld);
  task_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
}

void stf_task_add_dep_with_dplace(
  stf_task_handle t, stf_logical_data_handle ld, stf_access_mode m, const stf_data_place* data_p)
{
  assert(t);
  assert(ld);
  assert(data_p);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  auto* ld_ptr   = static_cast<logical_data_untyped*>(ld);
  task_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m), to_data_place(data_p)));
}

void* stf_task_get(stf_task_handle t, int index)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  auto s         = task_ptr->template get<slice<const char>>(index);
  return const_cast<void*>(static_cast<const void*>(s.data_handle()));
}

void stf_task_start(stf_task_handle t)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  task_ptr->start();
}

void stf_task_end(stf_task_handle t)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  task_ptr->end();
}

void stf_task_enable_capture(stf_task_handle t)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  task_ptr->enable_capture();
}

CUstream stf_task_get_custream(stf_task_handle t)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  return static_cast<CUstream>(task_ptr->get_stream());
}

void stf_task_destroy(stf_task_handle t)
{
  assert(t);

  auto* task_ptr = static_cast<context::unified_task<>*>(t);
  delete task_ptr;
}

// Type alias for cuda_kernel return type (used in multiple cuda_kernel functions below)
using kernel_type = decltype(::std::declval<context>().cuda_kernel());

void stf_cuda_kernel_create(stf_ctx_handle ctx, stf_cuda_kernel_handle* k)
{
  assert(ctx);
  assert(k);

  auto* context_ptr = static_cast<context*>(ctx);
  *k                = new kernel_type{context_ptr->cuda_kernel()};
}

void stf_cuda_kernel_set_exec_place(stf_cuda_kernel_handle k, const stf_exec_place* exec_p)
{
  assert(k);
  assert(exec_p);

  auto* kernel_ptr = static_cast<kernel_type*>(k);
  kernel_ptr->set_exec_place(to_exec_place(exec_p));
}

void stf_cuda_kernel_set_symbol(stf_cuda_kernel_handle k, const char* symbol)
{
  assert(k);
  assert(symbol);

  auto* kernel_ptr = static_cast<kernel_type*>(k);
  kernel_ptr->set_symbol(symbol);
}

void stf_cuda_kernel_add_dep(stf_cuda_kernel_handle k, stf_logical_data_handle ld, stf_access_mode m)
{
  assert(k);
  assert(ld);

  auto* kernel_ptr = static_cast<kernel_type*>(k);
  auto* ld_ptr     = static_cast<logical_data_untyped*>(ld);
  kernel_ptr->add_deps(task_dep_untyped(*ld_ptr, access_mode(m)));
}

void stf_cuda_kernel_start(stf_cuda_kernel_handle k)
{
  assert(k);

  auto* kernel_ptr = static_cast<kernel_type*>(k);
  kernel_ptr->start();
}

void stf_cuda_kernel_add_desc_cufunc(
  stf_cuda_kernel_handle k,
  CUfunction cufunc,
  dim3 grid_dim_,
  dim3 block_dim_,
  size_t shared_mem_,
  int arg_cnt,
  const void** args)
{
  assert(k);

  auto* kernel_ptr = static_cast<kernel_type*>(k);

  cuda_kernel_desc desc;
  desc.configure_raw(cufunc, grid_dim_, block_dim_, shared_mem_, arg_cnt, args);
  kernel_ptr->add_kernel_desc(mv(desc));
}

void* stf_cuda_kernel_get_arg(stf_cuda_kernel_handle k, int index)
{
  assert(k);

  auto* kernel_ptr = static_cast<kernel_type*>(k);
  auto s           = kernel_ptr->template get<slice<const char>>(index);
  return const_cast<void*>(static_cast<const void*>(s.data_handle()));
}

void stf_cuda_kernel_end(stf_cuda_kernel_handle k)
{
  assert(k);

  auto* kernel_ptr = static_cast<kernel_type*>(k);
  kernel_ptr->end();
}

void stf_cuda_kernel_destroy(stf_cuda_kernel_handle t)
{
  assert(t);

  auto* kernel_ptr = static_cast<kernel_type*>(t);
  delete kernel_ptr;
}

} // extern "C"
