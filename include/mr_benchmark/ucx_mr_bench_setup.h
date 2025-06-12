/** @file ucx_mr_bench_setup.h
 *  @brief Functions for setting up ucx multirail enviroment inside the benchmark enviroment.
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef MR_BENCHMARK_UCX_MR_BENCH_SETUP_H
#define MR_BENCHMARK_UCX_MR_BENCH_SETUP_H

#include <ucp/api/ucp.h>
#include "ucx_mr_bench_types.h"
#include "ucx_mr_bench_memory.h"

#include "../ucx_mr_aux.h"


/** @brief Parse command line arguments to set basic enviroment variables and select benchmark
 *
 *  General ucx_multirail arguments
 *  P:  communication port
 *  R:  Net devices to be used
 *  A:  Server Address
 *
 *  Benchmark settings.
 *  T:  Test case
 *  M:  Memory type
 *  s:  message size
 *  p:  repeated tests and number of repetitions
 *
 *  @param mr_bench_ctx empty multi-rail benchmark context
 *  @return ucs_status_t possible error status
 */
ucs_status_t
parse_bench_opts(ucx_mr_bench_context_t *mr_bench_ctx, int argc, char **argv)
{
  int c;
  char *ptr;

  mr_bench_ctx->repeated_test = 0;

  optind = 1;
  while ((c = getopt(argc, argv, "P:R:A:T:m:s:p:n:r:k:")) != -1)
  {
    switch (c)
    {
    // Command line parameters for benchmark enviroment
    case 'T':
      DEBUG_PRINT("Got Test %s\n", optarg);
      if (!strcmp(optarg, "MR"))
      {
        mr_bench_ctx->test_type = MR;
      }
      else if (!strcmp(optarg, "SINGLE"))
      {
        mr_bench_ctx->test_type = SINGLE;
      }
      else if (!strcmp(optarg, "SPLIT"))
      {
        mr_bench_ctx->test_type = SPLIT;
      }
      else if (!strcmp(optarg, "TEST_SPLIT"))
      {
        mr_bench_ctx->test_type = TEST_SPLIT;
      }
      else
      {
        return UCS_ERR_INVALID_PARAM;
      }
      break;
    case 'm':
      DEBUG_PRINT("Got Memory %s\n", optarg);
      if (!strcmp(optarg, "CUDA"))
      {
        mr_bench_ctx->mem_type = UCS_MEMORY_TYPE_CUDA;
      }
      else if (!strcmp(optarg, "HOST"))
      {
        mr_bench_ctx->mem_type = UCS_MEMORY_TYPE_HOST;
      }
      else
      {
        return UCS_ERR_INVALID_PARAM;
      }
      break;
    case 's':
      DEBUG_PRINT("Got Message size %s\n", optarg);
      mr_bench_ctx->msg_size = atoi(optarg);
      break;
    case 'p':
      DEBUG_PRINT("Got Repeated test with %s repetitions per msg size\n", optarg);
      mr_bench_ctx->repetitions = atoi(optarg);
      mr_bench_ctx->repeated_test = 1;
      break;
    case 'n':
      DEBUG_PRINT("Got number of rails for test %s\n", optarg);
      mr_bench_ctx->nof_rails = atoi(optarg);
      break;
    case 'r':
      DEBUG_PRINT("Got split ratio %s percent\n", optarg);
      mr_bench_ctx->ratio = atoi(optarg);
      break;
    case 'k':
      DEBUG_PRINT("Got number of pipeline stages %s\n", optarg);
      mr_bench_ctx->stages = atoi(optarg);
      break;


    // Command line parameters for general ucx multirail
    case 'P':
      DEBUG_PRINT("Got a port %s\n", optarg);
      mr_bench_ctx->mr_ctx.port = atoi(optarg);
      break;
    case 'R':
      for (int i = 0; i < NOF_RAILS; ++i)
      {
        if (i == 0)
        {
          ptr = strtok(optarg, ",");
        }
        else
        {
          ptr = strtok(NULL, ",");
        }
        mr_bench_ctx->mr_ctx.rails[i] = ptr;
        DEBUG_PRINT("Got Rail %s\n", ptr);
      }
      break;
    case 'A':
      DEBUG_PRINT("Got Server Address %s\n", optarg);
      mr_bench_ctx->mr_ctx.server_addr = optarg;
      break;
    default:
      DEBUG_PRINT("Default\n");
      break;
    }
  }

  return UCS_OK;
}


/** @brief Setting up multi-rail ucx context and benchmark enviroment
 *
 *  Additional to the general ucx_multirail setup memory buffers for the messages is allocated
 *
 *  @param mr_ctx empty multirail benchmark context
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_bench_setup(ucx_mr_bench_context_t *mr_bench_ctx)
{
  ucs_status_t status;
  mr_bench_ctx->recv_buffer_count = 0;
  mr_bench_ctx->send_buffer_count = 0;

  DEBUG_PRINT("Setup UCX!\n");
  status = ucx_mr_setup(&mr_bench_ctx->mr_ctx);
  if (status != UCS_OK)
  {
    DEBUG_PRINT("Something went wrong!\n");
    return status;
  }

  DEBUG_PRINT("Allocate Memory !\n");
  status = ucx_mr_alloc_mem(mr_bench_ctx);

  return status;
}

#endif // MR_BENCHMARK_UCX_MR_BENCH_SETUP_H
