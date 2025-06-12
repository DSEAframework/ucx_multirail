/** @file ucx_mr_bench_types.h
 *  @brief Functions for handling the memory needed to perform the benchmarks
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef MR_BENCHMARK_UCX_MR_BENCH_MEMORY_H
#define MR_BENCHMARK_UCX_MR_BENCH_MEMORY_H


/** @brief Free one memory area
 *
 *  @param buffer pointer to the memory area
 *  @param mem_type type of memory (Cuda/CPU)
 *  @return void
 */
void 
ucx_mr_free(void *buffer, ucs_memory_type_t mem_type)
{
  if (mem_type == UCS_MEMORY_TYPE_HOST)
  {
    free(buffer);
  }
  else if (mem_type == UCS_MEMORY_TYPE_CUDA)
  {
    cudaFree(buffer);
  }
}


/** @brief Free all memory of benchmark context
 *
 *  @param mr_ctx multi rail benchmark context
 *  @return void
 */
void 
ucx_mr_free_mem(ucx_mr_bench_context_t *mr_ctx)
{
  DEBUG_PRINT("Free Memory.\n");
  for (int i = 0; i < mr_ctx->recv_buffer_count; ++i)
  {
    ucx_mr_free(mr_ctx->recv_buffer[i], mr_ctx->mem_type);
  }
  for (int i = 0; i < mr_ctx->send_buffer_count; ++i)
  {
    ucx_mr_free(mr_ctx->send_buffer[i], mr_ctx->mem_type);
  }
}


/** @brief Allocate one memory area
 *
 *  @param buffer (output) pointer to the allocated memory
 *  @param length number of bytes to be allocated
 *  @param mem_type type of memory to be allocated (Cuda/CPU)
 *  @param device possible device index if GPU memory
 *  @return ucs_status_t with possible error code
 */
ucs_status_t
ucx_mr_alloc(void **buffer, size_t length, ucs_memory_type_t mem_type, int device)
{
  ucs_status_t status = UCS_OK;
  if (mem_type == UCS_MEMORY_TYPE_HOST)
  {
    *buffer = malloc(length);
    if (*buffer == NULL)
    {
      ERROR_PRINT("failed to allocate host memory");
      status = UCS_ERR_NO_MEMORY;
    }
  }
  else if (mem_type == UCS_MEMORY_TYPE_CUDA)
  {
    cudaSetDevice(device);
    cudaError_t e = cudaMalloc(buffer, length);
    if (e != cudaSuccess)
    {
      printf("Error in CudaMalloc.\n\n");
      status = UCS_ERR_NO_MEMORY;
    }
  }

  return status;
}


/** @brief Allocate all memory for benchmark context
 *
 *  @param mr_ctx multirail benchmark context
 *  @return ucs_status_t with possible error code
 */
ucs_status_t
ucx_mr_alloc_mem(ucx_mr_bench_context_t *mr_ctx)
{
  ucs_status_t status;

  DEBUG_PRINT("Allocate Memory of size: %ld\n", mr_ctx->msg_size);

  for (int i = 0; i < NOF_RAILS; ++i)
  {
    int device = i % NOF_RAILS;
    status = ucx_mr_alloc(&mr_ctx->recv_buffer[i], mr_ctx->msg_size, mr_ctx->mem_type, device);
    if (status != UCS_OK)
    {
      ucx_mr_free_mem(mr_ctx);
      return status;
    }
    mr_ctx->recv_buffer_count = i + 1;
  }

  for (int i = 0; i < NOF_RAILS; ++i)
  {
    int device = i % NOF_RAILS;
    status = ucx_mr_alloc(&mr_ctx->send_buffer[i], mr_ctx->msg_size, mr_ctx->mem_type, device);
    if (status != UCS_OK)
    {
      ucx_mr_free_mem(mr_ctx);
      return status;
    }
    mr_ctx->send_buffer_count = i + 1;
  }

  return status;
}

#endif // MR_BENCHMARK_UCX_MR_BENCH_MEMORY_H
