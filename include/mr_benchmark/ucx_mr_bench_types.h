/** @file ucx_mr_bench_types.h
 *  @brief Collection of data objects to organize multi-railed ucx and benchmark enviroment.
 *
 *  Data structs and enums which help to organize the benchmark enviroment. As central instance there is the ucx multi rail benchmark
 *  context, which holds the ucx multirail context and additional information about the benchmark. It holds pointers to the memory
 *  buffers from which messages are sent.
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef MR_BENCHMARK_UCX_MR_BENCH_TYPES_H
#define MR_BENCHMARK_UCX_MR_BENCH_TYPES_H

#include "../ucx_multirail.h"


/** @brief Enum for selecting correct benchmark
 *
 */
typedef enum MR_Benchmarks
{
  SINGLE,
  MR,
  SPLIT,
  TEST_SPLIT,
} MR_Benchmarks;


/** @brief UCX multi rail benchmark context
 *
 *  Data structure for organizing the genral ucx multirail context and additional information
 *  about the benchmarks to be performed. It holds also pointers to the memory buffers so messages
 *  can be sent.
 *
 */
typedef struct ucx_mr_bench_context
{

  // Multirail context
  ucx_mr_context_t mr_ctx;
  // Memory
  size_t msg_size;
  int recv_buffer_count;
  int send_buffer_count;
  void *recv_buffer[NOF_RAILS];
  void *send_buffer[NOF_RAILS];

  ucs_memory_type_t mem_type;

  MR_Benchmarks test_type;
  int repetitions;
  int repeated_test;
  unsigned nof_rails; // nof rails used in test
  unsigned ratio; // split ratio used in test
  unsigned stages;  // pipeline stages used in test

} ucx_mr_bench_context_t;

#endif // MR_BENCHMARK_UCX_MR_BENCH_TYPES_H