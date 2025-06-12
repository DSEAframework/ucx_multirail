/** @file ucx_mr_comm_test.h
 *  @brief Functions for testing established connection.
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef UCX_MR_COMM_TEST_H
#define UCX_MR_COMM_TEST_H

#include <cuda_runtime.h>
#include <cuda.h>

#include <ucp/api/ucp.h>

#include "../ucx_mr_comm.h"
#include "../ucx_mr_types.h"
#include "../ucx_mr_wait.h"
#include "ucx_mr_comm_single.h"


/** @brief Initializes a testing message with a given value.
 *
 *  @param buffer pointer to memory of test message
 *  @param write_length number of integers which are written into the message
 *  @param init_val initializing value
 *  @param mem_type memory type of buffer (Cuda/CPU)
 *  @return void
 */
static void
ucx_mr_init_test_message(void *buffer, size_t write_length, int init_val, ucs_memory_type_t mem_type)
{

  int arr[write_length];
  for (int i = 0; i < write_length; ++i)
  {
    arr[i] = init_val;
  }

  if (mem_type == UCS_MEMORY_TYPE_CUDA)
  {
    cudaMemcpy(buffer, arr, write_length * sizeof(int), cudaMemcpyHostToDevice);
  }
  else if (mem_type == UCS_MEMORY_TYPE_HOST)
  {
    for (int i = 0; i < write_length; ++i)
    {
      ((int *)buffer)[i] = arr[i];
    }
  }
}


/** @brief Creates a test message with a series of increasing numbers
 *
 *  @param buffer pointer to memory of test message
 *  @param write_length number of integers which are written into the message
 *  @param start_val value of first number
 *  @param mem_type memory type of buffer (Cuda/CPU)
 *  @return void
 */
static void
ucx_mr_create_test_message(void *buffer, size_t write_length, int start_val, ucs_memory_type_t mem_type)
{

  int arr[write_length];
  for (int i = 0; i < write_length; ++i)
  {
    arr[i] = start_val + i;
  }

  if (mem_type == UCS_MEMORY_TYPE_CUDA)
  {
    cudaMemcpy(buffer, arr, write_length * sizeof(int), cudaMemcpyHostToDevice);
  }
  else if (mem_type == UCS_MEMORY_TYPE_HOST)
  {
    for (int i = 0; i < write_length; ++i)
    {
      ((int *)buffer)[i] = arr[i];
    }
  }
}


/** @brief Reads and prints out a buffer with a given length
 *
 *  @param buffer pointer to memory of test message
 *  @param read_length number of integers to be read
 *  @param mem_type memory type of buffer (Cuda/CPU)
 *  @return void
 */
static void
ucx_mr_read_test_message(void *buffer, size_t read_length, ucs_memory_type_t mem_type)
{
  int arr[read_length];

  if (mem_type == UCS_MEMORY_TYPE_CUDA)
  {
    cudaMemcpy(arr, buffer, read_length * sizeof(int), cudaMemcpyDeviceToHost);
  }
  else if (mem_type == UCS_MEMORY_TYPE_HOST)
  {
    for (int i = 0; i < read_length; ++i)
    {
      arr[i] = ((int *)buffer)[i];
    }
  }

  for (int i = 0; i < read_length; ++i)
  {
    printf("%d, ", arr[i]);
  }
  printf("\n");
}


/** @brief Sends a specific test message to peer to check connection of given rail
 *
 *  @param mr_ctx multirail context
 *  @param rail rail which should be tested
 *  @param tag tag of sent message
 *  @return void
 */
ucs_status_t
ucx_mr_test_send(ucx_mr_context_t *mr_ctx, int rail, ucp_tag_t tag)
{
  ucs_status_t status;

  struct ucx_context *request = NULL;
  ucp_request_param_t send_param;

  ucp_ep_h ep = mr_ctx->ep[rail];
  ucp_worker_h worker = mr_ctx->worker[rail];

  size_t length = 10;

  void *buffer = malloc(length * sizeof(int));
  ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;

  printf("Start to send!\n");
  ucx_mr_create_test_message(buffer, length, length * rail, mem_type);
  printf("Message: ");
  ucx_mr_read_test_message(buffer, length, mem_type);

  ucx_mr_write_send_param(&send_param, mem_type);

  request = (ucx_context *) ucp_tag_send_nbx(ep, buffer, length * sizeof(int), tag, &send_param);
  status = ucx_wait(worker, request, "send", data_msg_str);

  free(buffer);

  return status;
}


/** @brief Receive and prints out test message to check connection
 *
 *  @param mr_ctx multirail context
 *  @param rail rail which should be tested
 *  @param tag tag of received message
 *  @return void
 */
ucs_status_t
ucx_mr_test_recv(ucx_mr_context_t *mr_ctx, int rail, ucp_tag_t tag)
{
  ucs_status_t status;

  struct ucx_context *request = NULL;
  ucp_tag_message_h msg_tag;
  ucp_tag_recv_info_t info_tag;

  ucp_request_param_t recv_param;

  ucp_worker_h worker = mr_ctx->worker[rail];

  size_t length = 10;

  void *buffer = malloc(length * sizeof(int));
  ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST;

  ucx_mr_write_recv_param(&recv_param, mem_type);

  printf("Start to receive!\n");
  ucx_mr_init_test_message(buffer, length, 0, mem_type);
  printf("Message previous: ");
  ucx_mr_read_test_message(buffer, length, mem_type);

  for (;;)
  {
    msg_tag = ucp_tag_probe_nb(worker, tag, tag_mask, 1, &info_tag);

    if (msg_tag != NULL)
    {
      break;
    }
    ucp_worker_progress(worker);
  }

  printf("Message is here! \n");

  request = (ucx_context *) ucp_tag_msg_recv_nbx(worker, buffer, length * sizeof(int), msg_tag, &recv_param);

  status = ucx_wait(worker, request, "receive", data_msg_str);

  printf("Message received: ");
  ucx_mr_read_test_message(buffer, length, mem_type);

  free(buffer);

  return status;
}


/** @brief Tests connection for each rail with a increasing series of numbers.
 *
 *  Sends and receives a testing message which is a series of increasing numbers. This is done for
 *  all rails on its own. The send and received messages are printed to be checked.
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @return void
 */
void 
ucx_mr_test_connection(ucx_mr_context_t *mr_ctx)
{
  ucp_tag_t tag = 0x12;

  for (int i = 0; i < NOF_RAILS; ++i)
  {
    if (mr_ctx->role == SENDER)
    {
      ucx_mr_test_send(mr_ctx, i, tag);
    }
    else
    {
      ucx_mr_test_recv(mr_ctx, i, tag);
    }
  }
}

#endif // UCX_MR_COMM_TEST_H