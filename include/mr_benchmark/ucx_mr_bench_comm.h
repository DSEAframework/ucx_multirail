/** @file ucx_mr_bench_comm.h
 *  @brief Functions for specific benchmark tests
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef UCX_MR_BENCH_COMM_H
#define UCX_MR_BENCH_COMM_H

#include <sys/times.h>
#include <sys/time.h>
#include <unistd.h>

#include <ucp/api/ucp.h>

#include "ucx_mr_bench_types.h"
#include "../ucx_multirail.h"

#ifndef MAX_RUNS
#define MAX_RUNS 50 // repetitions for calculating mean
#endif


/** @brief Test case with data splitting (sender side)
 *
 *  Send one message from one memory location with 2/4 rails in parallel
 *
 *  @param mr_bench_ctx multirail benchmark context
 *  @param split_ratio ratio how much memory is not sent from rail0
 *  @param pipeline_stages number of stages on rail 1 / 1-3
 *  @return void
 */
void 
ucx_mr_bench_test_send_split(ucx_mr_bench_context_t *mr_bench_ctx, float split_ratio, int pipeline_stages)
{
  ucx_mr_context_t *mr_ctx = &mr_bench_ctx->mr_ctx;
  ucp_tag_t tag = 0x33;

  void **buffer = mr_bench_ctx->send_buffer;
  ucs_memory_type_t mem_type = mr_bench_ctx->mem_type;
  size_t length = mr_bench_ctx->msg_size;

  int element_size = 4;

  printf("\n\n\n\n=== TEST SPLIT ===\n");

  printf("Start to send!\n");
  ucx_mr_create_test_message(buffer[0], length / element_size, 1, mem_type);
  printf("Message: ");
  ucx_mr_read_test_message(buffer[0], length / element_size, mem_type);

  if (mr_bench_ctx->nof_rails == 2)
  {
    DEBUG_PRINT("DualRail\n");
    ucx_mr_dual_split_send(mr_ctx, tag, split_ratio, element_size,
                           buffer[0], length, mem_type, 0,
                           buffer[1], mem_type, 1, pipeline_stages);
  }
  else if (mr_bench_ctx->nof_rails == 4)
  {
    DEBUG_PRINT("QuadRail\n");
    ucx_mr_quad_split_send(mr_ctx, tag, split_ratio, element_size,
                           buffer[0], length, mem_type, 0,
                           buffer[1], mem_type, 1,
                           buffer[2], mem_type, 2,
                           buffer[3], mem_type, 3, pipeline_stages);
  }
}


/** @brief Test case with data splitting (receiver side)
 *
 *  Receive one message from one memory location with 2/4 rails in parallel
 *
 *  @param mr_bench_ctx multirail benchmark context
 *  @param split_ratio ratio how much memory is not sent from rail0
 *  @param pipeline_stages number of stages on rail 1 / 1-3
 *  @return void
 */
void 
ucx_mr_bench_test_recv_split(ucx_mr_bench_context_t *mr_bench_ctx, float split_ratio, int pipeline_stages)
{
  ucx_mr_context_t *mr_ctx = &mr_bench_ctx->mr_ctx;
  ucp_tag_t tag = 0x33;

  void **buffer = mr_bench_ctx->recv_buffer;
  ucs_memory_type_t mem_type = mr_bench_ctx->mem_type;
  size_t length = mr_bench_ctx->msg_size;

  int element_size = 4;

  printf("\n\n\n\n=== TEST SPLIT ===\n");

  printf("Before Reiving: ");
  ucx_mr_read_test_message(buffer[0], length / element_size, mem_type);

  if (mr_bench_ctx->nof_rails == 2)
  {
    DEBUG_PRINT("DualRail\n");
    ucx_mr_dual_split_recv(mr_ctx, tag, split_ratio, element_size,
                           buffer[0], length, mem_type, 0,
                           buffer[1], mem_type, 1, pipeline_stages);
  }
  else if (mr_bench_ctx->nof_rails == 4)
  {
    DEBUG_PRINT("QuadRail\n");
    ucx_mr_quad_split_recv(mr_ctx, tag, split_ratio, element_size,
                           buffer[0], length, mem_type, 0,
                           buffer[1], mem_type, 1,
                           buffer[2], mem_type, 2,
                           buffer[3], mem_type, 3, pipeline_stages);
  }

  printf("Received Message: ");
  ucx_mr_read_test_message(buffer[0], length / element_size, mem_type);
}


/** @brief Benchmark sending with data splitting
 *
 *  Send one message from one memory location with 2/4 rails in parallel and measure bandwidth
 *
 *  @param mr_bench_ctx multirail benchmark context
 *  @param split_ratio ratio how much memory is not sent from rail0
 *  @param pipeline_stages number of stages on rail 1 / 1-3
 *  @return double measured bandwidth
 */
double 
ucx_mr_bench_send_split(ucx_mr_bench_context_t *mr_bench_ctx, float split_ratio, int pipeline_stages)
{
  ucx_mr_context_t *mr_ctx = &mr_bench_ctx->mr_ctx;
  ucp_tag_t tag = 0x33;
  void **buffer = mr_bench_ctx->send_buffer;
  ucs_memory_type_t mem_type = mr_bench_ctx->mem_type;
  size_t length = mr_bench_ctx->msg_size;

  int element_size = 4;

  // synchronizing
  if (mr_bench_ctx->nof_rails == 2)
  {
    printf("Synchronizing Dual Rail\n");
    ucx_mr_dual_split_send(mr_ctx, tag, split_ratio, element_size,
                            buffer[0], length, mem_type, 0,
                            buffer[1], mem_type, 1, pipeline_stages);
  }
  else if (mr_bench_ctx->nof_rails == 4)
  {
    printf("Synchronizing Quad Rail\n");
    ucx_mr_quad_split_send(mr_ctx, tag, split_ratio, element_size,
                            buffer[0], length, mem_type, 0,
                            buffer[1], mem_type, 1,
                            buffer[2], mem_type, 2,
                            buffer[3], mem_type, 3, pipeline_stages);
  }
  else
  {
    printf("Test not implemented!\n");
  }

  int runs;

  struct timeval t0;
  struct timeval t1;
  gettimeofday(&t0, NULL);

  for (runs = 0; runs < MAX_RUNS; ++runs)
  {
    if (mr_bench_ctx->nof_rails == 2)
    {
      DEBUG_PRINT("DualRail\n");
      ucx_mr_dual_split_send(mr_ctx, tag, split_ratio, element_size,
                             buffer[0], length, mem_type, 0,
                             buffer[1], mem_type, 1, pipeline_stages);
    }
    else if (mr_bench_ctx->nof_rails == 4)
    {
      DEBUG_PRINT("QuadRail\n");
      ucx_mr_quad_split_send(mr_ctx, tag, split_ratio, element_size,
                             buffer[0], length, mem_type, 0,
                             buffer[1], mem_type, 1,
                             buffer[2], mem_type, 2,
                             buffer[3], mem_type, 3, pipeline_stages);
    }
    else
    {
      printf("Test not implemented!\n");
    }

    DEBUG_PRINT("Run Finished!\n\n\n");
  }

  gettimeofday(&t1, NULL);

  double tu = (double)(t1.tv_sec - t0.tv_sec) * 1000000;
  tu += (double)(t1.tv_usec - t0.tv_usec);
  tu /= 1000000;

  printf("\n\n\n====================\n");

  printf("%d runs took %lfs to send.\n", runs, tu);

  double msg_size = (double)length / 1024 / 1024;
  double t_per_run = tu / runs;
  printf("Msg size: %lf MB, Bandwidth: %lf MB/s\n", msg_size, msg_size / t_per_run);

  printf("====================\n\n\n");

  return msg_size / t_per_run;
}


/** @brief Benchmark receiving with data splitting
 *
 *  Send one message from one memory location with 2/4 rails in parallel and measure bandwidth
 *
 *  @param mr_bench_ctx multirail benchmark context
 *  @param split_ratio ratio how much memory is not sent from rail0
 *  @param pipeline_stages number of stages on rail 1 / 1-3
 *  @return double measured bandwidth
 */
double 
ucx_mr_bench_recv_split(ucx_mr_bench_context_t *mr_bench_ctx, float split_ratio,  int pipeline_stages)
{
  ucx_mr_context_t *mr_ctx = &mr_bench_ctx->mr_ctx;
  ucp_tag_t tag = 0x33;
  void **buffer = mr_bench_ctx->recv_buffer;
  ucs_memory_type_t mem_type = mr_bench_ctx->mem_type;
  size_t length = mr_bench_ctx->msg_size;

  int element_size = 4;

  int runs;

  // synchronizing
  if (mr_bench_ctx->nof_rails == 2)
  {
    printf("Synchronizing Dual Rail\n");
    ucx_mr_dual_split_recv(mr_ctx, tag, split_ratio, element_size,
                            buffer[0], length, mem_type, 0,
                            buffer[1], mem_type, 1, pipeline_stages);
  }
  else if (mr_bench_ctx->nof_rails == 4)
  {
    printf("Synchronizing Quad Rail\n");
    ucx_mr_quad_split_recv(mr_ctx, tag, split_ratio, element_size,
                            buffer[0], length, mem_type, 0,
                            buffer[1], mem_type, 1,
                            buffer[2], mem_type, 2,
                            buffer[3], mem_type, 3, pipeline_stages);
  }

  struct timeval t0;
  struct timeval t1;
  gettimeofday(&t0, NULL);

  for (runs = 0; runs < MAX_RUNS; ++runs)
  {
    DEBUG_PRINT("Run: %d\n", runs);

    if (mr_bench_ctx->nof_rails == 2)
    {
      DEBUG_PRINT("DualRail\n");
      ucx_mr_dual_split_recv(mr_ctx, tag, split_ratio, element_size,
                             buffer[0], length, mem_type, 0,
                             buffer[1], mem_type, 1, pipeline_stages);
    }
    else if (mr_bench_ctx->nof_rails == 4)
    {
      DEBUG_PRINT("QuadRail\n");
      ucx_mr_quad_split_recv(mr_ctx, tag, split_ratio, element_size,
                             buffer[0], length, mem_type, 0,
                             buffer[1], mem_type, 1,
                             buffer[2], mem_type, 2,
                             buffer[3], mem_type, 3, pipeline_stages);
    }
    else
    {
      printf("Test not implemented!\n");
    }

    DEBUG_PRINT("Run Finished!\n\n\n");
  }

  gettimeofday(&t1, NULL);

  double tu = (double)(t1.tv_sec - t0.tv_sec) * 1000000;
  tu += (double)(t1.tv_usec - t0.tv_usec);
  tu /= 1000000;

  printf("\n\n\n====================\n");

  printf("%d runs took %lfs to send.\n", runs, tu);

  double msg_size = (double)length / 1024 / 1024;
  double t_per_run = tu / runs;
  printf("Msg size: %lf MB, Bandwidth: %lf MB/s\n", msg_size, msg_size / t_per_run);

  printf("====================\n\n\n");

  return msg_size / t_per_run;
}


/** @brief Benchmark sending without data splitting
 *
 *  Send one message for each rail
 *
 *  @param mr_bench_ctx multirail benchmark context
 *  @return double measured bandwidth
 */
double 
ucx_mr_bench_send_mr(ucx_mr_bench_context_t *mr_bench_ctx)
{
  ucx_mr_context_t *mr_ctx = &mr_bench_ctx->mr_ctx;
  ucp_tag_t tag = 0x33;
  void **buffer = mr_bench_ctx->send_buffer;
  ucs_memory_type_t mem_type = mr_bench_ctx->mem_type;
  size_t length = mr_bench_ctx->msg_size;

  int runs;

  struct timeval t0;
  struct timeval t1;
  gettimeofday(&t0, NULL);

  // double t0 = wallTime();
  for (runs = 0; runs < MAX_RUNS; ++runs)
  {
    if (mr_bench_ctx->nof_rails == 2)
    {
      DEBUG_PRINT("DualRail\n");
      ucx_mr_dual_send(mr_ctx, tag,
                       buffer[0], length, mem_type, 0,
                       buffer[1], length, mem_type, 1);
    }
    else if (mr_bench_ctx->nof_rails == 4)
    {
      DEBUG_PRINT("QuadRail\n");
      ucx_mr_quad_send(mr_ctx, tag,
                       buffer[0], length, mem_type, 0,
                       buffer[1], length, mem_type, 1,
                       buffer[2], length, mem_type, 2,
                       buffer[3], length, mem_type, 3);
    }
    else
    {
      printf("Test not implemented!\n");
    }

    DEBUG_PRINT("Run Finished!\n\n\n");
  }
  // double t1 = wallTime() - t0;
  gettimeofday(&t1, NULL);

  double tu = (double)(t1.tv_sec - t0.tv_sec) * 1000000;
  tu += (double)(t1.tv_usec - t0.tv_usec);
  tu /= 1000000;

  printf("\n\n\n====================\n");

  printf("%d runs took %lfs to send.\n", runs, tu);

  double msg_size = (double)length / 1024 / 1024;
  double t_per_run = tu / runs;
  printf("Msg size: %lf MB, Bandwidth: %d * %lf MB/s\n", msg_size, mr_bench_ctx->nof_rails, msg_size / t_per_run);

  printf("====================\n\n\n");

  return mr_bench_ctx->nof_rails * msg_size / t_per_run;
}


/** @brief Benchmark receiving without data splitting
 *
 *  Receive one message for each rail
 *
 *  @param mr_bench_ctx multirail benchmark context
 *  @return double measured bandwidth
 */
double 
ucx_mr_bench_recv_mr(ucx_mr_bench_context_t *mr_bench_ctx)
{
  ucx_mr_context_t *mr_ctx = &mr_bench_ctx->mr_ctx;
  ucp_tag_t tag = 0x33;
  void **buffer = mr_bench_ctx->recv_buffer;
  ucs_memory_type_t mem_type = mr_bench_ctx->mem_type;
  size_t length = mr_bench_ctx->msg_size;

  size_t recv_sizes[mr_bench_ctx->nof_rails];

  int runs;

  struct timeval t0;
  struct timeval t1;
  gettimeofday(&t0, NULL);

  // double t0 = wallTime();
  for (runs = 0; runs < MAX_RUNS; ++runs)
  {

    DEBUG_PRINT("Run: %d\n", runs);

    if (mr_bench_ctx->nof_rails == 2)
    {
      DEBUG_PRINT("DualRail\n");
      ucx_mr_dual_recv(mr_ctx, tag,
                       buffer[0], length, mem_type, 0,
                       buffer[1], length, mem_type, 1,
                       recv_sizes);
    }
    else if (mr_bench_ctx->nof_rails == 4)
    {
      DEBUG_PRINT("QuadRail\n");
      ucx_mr_quad_recv(mr_ctx, tag,
                       buffer[0], length, mem_type, 0,
                       buffer[1], length, mem_type, 1,
                       buffer[2], length, mem_type, 2,
                       buffer[3], length, mem_type, 3,
                       recv_sizes);
    }
    else
    {
      printf("Test not implemented!\n");
    }

    DEBUG_PRINT("Run Finished!\n\n\n");
  }

  gettimeofday(&t1, NULL);

  double tu = (double)(t1.tv_sec - t0.tv_sec) * 1000000;
  tu += (double)(t1.tv_usec - t0.tv_usec);
  tu /= 1000000;

  printf("\n\n\n====================\n");

  printf("%d runs took %lfs to send.\n", runs, tu);

  double msg_size = (double)length / 1024 / 1024;
  double t_per_run = tu / runs;
  printf("Msg size: %lf MB, Bandwidth: %d * %lf MB/s\n", msg_size, mr_bench_ctx->nof_rails, msg_size / t_per_run);

  printf("====================\n\n\n");

  return mr_bench_ctx->nof_rails * msg_size / t_per_run;
}


/** @brief Benchmark one single rail
 *
 *  Receive one message on one rail
 *
 *  @param mr_bench_ctx multirail benchmark context
 *  @param rail rail to be tested
 *  @return double measured bandwidth
 */
double 
ucx_mr_bench_recv_single(ucx_mr_bench_context_t *mr_bench_ctx, int rail)
{
  cudaSetDevice(rail);

  ucx_mr_context_t *mr_ctx = &mr_bench_ctx->mr_ctx;
  ucp_tag_t tag = 0x33;
  void *buffer = mr_bench_ctx->recv_buffer[rail];
  ucs_memory_type_t mem_type = mr_bench_ctx->mem_type;
  size_t length = mr_bench_ctx->msg_size;

  int runs;

  struct timeval t0;
  struct timeval t1;
  gettimeofday(&t0, NULL);

  for (runs = 0; runs < MAX_RUNS; ++runs)
  {

    DEBUG_PRINT("Run: %d\n", runs);

    ucx_mr_single_recv(mr_ctx, rail, tag, buffer, length, mem_type, rail);
  }

  gettimeofday(&t1, NULL);

  double tu = (double)(t1.tv_sec - t0.tv_sec) * 1000000;
  tu += (double)(t1.tv_usec - t0.tv_usec);
  tu /= 1000000;
  printf("\n\n\n====================\n");

  printf("%d runs took %lfs to send.\n", runs, tu);

  double msg_size = (double)length / 1024 / 1024;
  double t_per_run = tu / runs;
  printf("Msg size: %lf MB, Bandwidth: %lf MB/s\n", msg_size, msg_size / t_per_run);

  printf("====================\n\n\n");

  return msg_size / t_per_run;
}


/** @brief Benchmark one single rail
 *
 *  Send one message on one rail
 *
 *  @param mr_bench_ctx multirail benchmark context
 *  @param rail rail to be tested
 *  @return double measured bandwidth
 */
double 
ucx_mr_bench_send_single(ucx_mr_bench_context_t *mr_bench_ctx, int rail)
{
  cudaSetDevice(rail);

  ucx_mr_context_t *mr_ctx = &mr_bench_ctx->mr_ctx;
  ucp_tag_t tag = 0x33;
  void *buffer = mr_bench_ctx->send_buffer[rail];
  ucs_memory_type_t mem_type = mr_bench_ctx->mem_type;
  size_t length = mr_bench_ctx->msg_size;

  int runs;

  struct timeval t0;
  struct timeval t1;
  gettimeofday(&t0, NULL);

  for (runs = 0; runs < MAX_RUNS; ++runs)
  {
    DEBUG_PRINT("Run: %d Send message with tag: %llx\n", runs, (long long)tag + runs);

    ucx_mr_single_send(mr_ctx, rail, tag, buffer, length, mem_type, rail);

    DEBUG_PRINT("Run Finished!\n\n\n");
  }

  gettimeofday(&t1, NULL);

  double tu = (double)(t1.tv_sec - t0.tv_sec) * 1000000;
  tu += (double)(t1.tv_usec - t0.tv_usec);
  tu /= 1000000;

  printf("\n\n\n====================\n");

  printf("%d runs took %lfs to send.\n", runs, tu);

  double msg_size = (double)length / 1024 / 1024;
  double t_per_run = tu / runs;
  printf("Msg size: %lf MB, Bandwidth: %lf MB/s\n", msg_size, msg_size / t_per_run);

  printf("====================\n\n\n");

  return msg_size / t_per_run;
}

#endif // UCX_MR_BENCH_COMM_H