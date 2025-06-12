/** @file ucx_mr_comm_single.h
 *  @brief Point to point communication with ucx
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef UCX_MR_COMM_SINGLE_H
#define UCX_MR_COMM_SINGLE_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <driver_types.h>

#include "../ucx_mr_comm.h"
#include "../ucx_mr_types.h"
#include "../ucx_mr_wait.h"


/** @brief Send a message on one specific rail.
 *
 *  Blocking implementation.
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @param tag tag for communication
 *  @param buffer pointer to the send buffer
 *  @param msg_size message size in bytes
 *  @param mem_type memory type of buffer (Cuda/CPU)
 *  @param device (0-3) possible device index if memory type is Cuda
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_single_send(ucx_mr_context_t *mr_ctx, int rail, ucp_tag_t tag,
                   void *buffer, size_t msg_size, ucs_memory_type_t mem_type, int device)
{
  ucs_status_t status;

  struct ucx_context *request = NULL;
  ucp_request_param_t send_param;

  ucp_ep_h *ep = mr_ctx->ep;
  ucp_worker_h *worker = mr_ctx->worker;

  ucx_mr_write_send_param(&send_param, mem_type);

  cudaSetDevice(rail);

  request = (ucx_context *) ucp_tag_send_nbx(ep[rail], buffer, msg_size, tag, &send_param);
  status = ucx_wait(worker[rail], request, "send", data_msg_str);

  return status;
}


/** @brief Receive a message on one specific rail.
 *
 *  Blocking implementation.
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @param tag tag for communication
 *  @param buffer pointer to the receiving buffer
 *  @param buffer_size msg_size to be received in bytes
 *  @param mem_type memory type of the buffer (Cuda/CPU)
 *  @param device possible device index if memory type is Cuda
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_single_recv(ucx_mr_context_t *mr_ctx, int rail, ucp_tag_t tag,
                   void *buffer, size_t msg_size, ucs_memory_type_t mem_type, int device)
{
  ucs_status_t status;

  struct ucx_context *request = NULL;
  ucp_tag_message_h msg_tag = NULL;
  ucp_tag_recv_info_t info_tag;

  ucp_request_param_t recv_param;

  ucp_worker_h *worker = mr_ctx->worker;

  ucx_mr_write_recv_param(&recv_param, mem_type);

  cudaSetDevice(rail);

  for (;;)
  {
    msg_tag = ucp_tag_probe_nb(worker[rail], tag, tag_mask, 1, &info_tag);
    if (msg_tag != NULL)
    {
      break;
    }

    ucp_worker_progress(worker[rail]);
  }

  request = (ucx_context *) ucp_tag_msg_recv_nbx(worker[rail], buffer, msg_size, msg_tag, &recv_param);

  status = ucx_wait(worker[rail], request, "receive", data_msg_str);

  return status;
}

#endif // UCX_MR_COMM_SINGLE_H