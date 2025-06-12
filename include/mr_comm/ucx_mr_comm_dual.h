#ifndef UCX_MR_COMM_DUAL_H
#define UCX_MR_COMM_DUAL_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <driver_types.h>

#include <ucp/api/ucp.h>

#include "../ucx_mr_comm.h"
#include "../ucx_mr_types.h"
#include "../ucx_mr_wait.h"
#include "ucx_mr_comm_single.h"

ucs_status_t
ucx_mr_dual_recv(ucx_mr_context_t *mr_ctx, ucp_tag_t tag,
                 void *buffer0, size_t msg_size0, ucs_memory_type_t mem_type0, int device0,
                 void *buffer1, size_t msg_size1, ucs_memory_type_t mem_type1, int device1,
                 size_t *recv_sizes)
{
  ucs_status_t status0;
  ucs_status_t status1;

  struct ucx_context *request0 = NULL;
  struct ucx_context *request1 = NULL;
  ucp_tag_message_h msg_tag0 = NULL;
  ucp_tag_message_h msg_tag1 = NULL;

  ucp_tag_recv_info_t info_tag0;
  ucp_tag_recv_info_t info_tag1;

  ucp_request_param_t recv_param0;
  ucp_request_param_t recv_param1;

  ucp_worker_h *worker = mr_ctx->worker;

  ucx_mr_write_recv_param(&recv_param0, mem_type0);
  ucx_mr_write_recv_param(&recv_param1, mem_type1);

  for (;;)
  {
    if (msg_tag0 == NULL)
    {
      cudaSetDevice(device0);
      msg_tag0 = ucp_tag_probe_nb(worker[0], tag, tag_mask, 1, &info_tag0);

      if (msg_tag0 != NULL)
      {
        request0 = (ucx_context *) ucp_tag_msg_recv_nbx(worker[0], buffer0, msg_size0, msg_tag0, &recv_param0);
        recv_sizes[0] = info_tag0.length;
      }
      ucp_worker_progress(worker[0]);
    }

    if (msg_tag1 == NULL)
    {
      cudaSetDevice(device1);
      msg_tag1 = ucp_tag_probe_nb(worker[1], tag, tag_mask, 1, &info_tag1);

      if (msg_tag1 != NULL)
      {
        request1 = (ucx_context *) ucp_tag_msg_recv_nbx(worker[1], buffer1, msg_size1, msg_tag1, &recv_param1);
        recv_sizes[1] = info_tag1.length;
      }
      ucp_worker_progress(worker[1]);
    }

    if (msg_tag0 != NULL && msg_tag1 != NULL)
      break;
  }

  cudaSetDevice(device0);
  status0 = ucx_wait(worker[0], request0, "receive", data_msg_str);
  cudaSetDevice(device1);
  status1 = ucx_wait(worker[1], request1, "receive", data_msg_str);

  if (status0 != UCS_OK)
    return status0;
  if (status1 != UCS_OK)
    return status1;

  return UCS_OK;
}

ucs_status_t
ucx_mr_dual_send(ucx_mr_context_t *mr_ctx, ucp_tag_t tag,
                 void *buffer0, size_t msg_size0, ucs_memory_type_t mem_type0, int device0,
                 void *buffer1, size_t msg_size1, ucs_memory_type_t mem_type1, int device1)
{
  ucs_status_t status0;
  ucs_status_t status1;

  struct ucx_context *request0 = NULL;
  struct ucx_context *request1 = NULL;

  ucp_request_param_t send_param0;
  ucp_request_param_t send_param1;

  ucp_ep_h *ep = mr_ctx->ep;
  ucp_worker_h *worker = mr_ctx->worker;

  ucx_mr_write_send_param(&send_param0, mem_type0);
  ucx_mr_write_send_param(&send_param1, mem_type1);

  cudaSetDevice(device0);
  request0 = (ucx_context *) ucp_tag_send_nbx(ep[0], buffer0, msg_size0, tag, &send_param0);
  cudaSetDevice(device1);
  request1 = (ucx_context *) ucp_tag_send_nbx(ep[1], buffer1, msg_size1, tag, &send_param1);

  cudaSetDevice(device0);
  status0 = ucx_wait(worker[0], request0, "send", data_msg_str);
  cudaSetDevice(device1);
  status1 = ucx_wait(worker[1], request1, "send", data_msg_str);

  if (status0 != UCS_OK)
    return status0;
  if (status1 != UCS_OK)
    return status1;

  return UCS_OK;
}

ucs_status_t
ucx_mr_dual_split_recv(ucx_mr_context_t *mr_ctx, ucp_tag_t tag, float split_ratio, int element_size,
                       void *buffer0, size_t buffer_size, ucs_memory_type_t mem_type0, int device0,
                       void *buffer1, ucs_memory_type_t mem_type1, int device1, int pipeline_stages)
{
  // 0 is the collecting device
  int RAILS = 2;
  int NOF_MESSAGES = pipeline_stages*(RAILS-1)+1;

  cudaError_t c_status;
  ucs_status_t status[NOF_MESSAGES];
  for (int i = 0; i < NOF_MESSAGES; ++i) {
    status[i] = UCS_ERR_NO_MESSAGE;
  }

  struct ucx_context *request[NOF_MESSAGES];
  for (int i = 0; i < NOF_MESSAGES; ++i) {
    request[i] = NULL;
  }
  ucp_request_param_t recv_param[2];
  
  ucp_tag_recv_info_t info_tag[2];
  ucp_tag_message_h msg_tag[NOF_MESSAGES];
  for (int i = 0; i < NOF_MESSAGES; ++i) {
    msg_tag[i] = NULL;
  }

  size_t msg_sizes[2] = {0,0};

  int devices[2]                        = {device0, device1};
  ucs_memory_type_t mem_types[2]        = {mem_type0, mem_type1};
  void *buffer[2]                       = {buffer0, buffer1};
  size_t buffer_sizes[2]                = {buffer_size, buffer_size};

  ucp_worker_h *worker = mr_ctx->worker;

  cudaStream_t cuda_stream = mr_ctx->cuda_stream;
  cudaEvent_t *cuda_events = mr_ctx->cuda_events;

  int finished = 0;
  size_t offset = 0;
  size_t offset_p = 0;

  finished = 0;
  for (int i = 0; i < 2; ++i)
  {
    ucx_mr_write_recv_param(&recv_param[i], mem_types[i]);
  }

  // Check for all incoming messages
  for (;;) {
    // Not pipelined message
    if (msg_tag[0] == NULL)
    {
      cudaSetDevice(devices[0]);
      msg_tag[0] = ucp_tag_probe_nb(worker[0], tag, tag_mask, 1, &info_tag[0]);

      if (msg_tag[0] != NULL)
      {
        DEBUG_PRINT("Message 0 found!\n");
        request[0] = (ucx_context *) ucp_tag_msg_recv_nbx(worker[0], buffer[0], buffer_sizes[0], msg_tag[0], &recv_param[0]);
        //msg_sizes[0] = info_tag[0].length;
      }
      ucx_mr_worker_progress(mr_ctx, 0);
    } 

    // Pipelined messages
    for (int p = 0; p < pipeline_stages; ++p) {
      for (int i = 0; i < 1; ++i)
      {
        if (msg_tag[i + p*(RAILS-1)+1] == NULL)
        {
          ucx_mr_worker_progress(mr_ctx, 1);
          cudaSetDevice(devices[i+1]);
          msg_tag[i + p*(RAILS-1)+1] = ucp_tag_probe_nb(worker[i+1], tag+p, tag_mask, 1, &info_tag[i+1]);

          if (msg_tag[i + p*(RAILS-1)+1] != NULL)
          {
            DEBUG_PRINT("Message found on rail %d, stage %d\n",i+1, p);
            DEBUG_PRINT("Idx: %d\n",i + p*(RAILS-1));
            
            msg_sizes[i+1] = info_tag[i+1].length;
            msg_sizes[0] = info_tag[i+1].sender_tag >> 32;
            DEBUG_PRINT("Message Size Recv: %ld\n", msg_sizes[i+1]);
            DEBUG_PRINT("Offset_p: %ld\n", p*msg_sizes[i+1]);
            request[i + p*(RAILS-1)+1] = (ucx_context *) ucp_tag_msg_recv_nbx(worker[i+1], buffer[i+1] + p * msg_sizes[i+1], buffer_sizes[i+1], msg_tag[i + p*(RAILS-1)+1], &recv_param[i+1]);   
            DEBUG_PRINT("Request Stored\n");
          }
        }
      }
    }


    if (status[0] != UCS_OK && request[0] != NULL) {
      
      cudaSetDevice(devices[0]);
      status[0] = ucx_poll(worker[0], request[0]);
      if (status[0] == UCS_OK) {
        DEBUG_PRINT("Message 0 finished!\n");
        finished++;
      } else if (status[0] != UCS_INPROGRESS) {
        return status[0];
      }
    }

    
    for (int p = 0; p < pipeline_stages; ++p) {
      offset = 0;
      offset_p = 0;
      for (int i = 0; i < 1; ++i)
      {
        if (status[i + p*(RAILS-1)+1] != UCS_OK && request[i + p*(RAILS-1)+1] != NULL) {
          //DEBUG_PRINT("Check for message on rail %d stage %d\n", i, p);
          cudaSetDevice(devices[i+1]);
          status[i + p*(RAILS-1)+1] = ucx_poll(worker[i+1], request[i + p*(RAILS-1)+1]);

          if (status[i + p*(RAILS-1)+1] == UCS_OK) {
            DEBUG_PRINT("Message on rail %d stage %d ready\n", i, p);
            finished++;

            offset += msg_sizes[i];
            offset_p = msg_sizes[i + p*(RAILS-1)+1] * p;
            
            c_status = cudaMemcpyPeerAsync(
              buffer[0] + offset + offset_p, devices[0], buffer[i+1] + offset_p, 
              devices[i+1], msg_sizes[i + p*(RAILS-1)+1], cuda_stream
            );
            DEBUG_PRINT("Cuda Copy Started on rail %d stage %d!\n", i, p);
            if (c_status != cudaSuccess)
            {
              ERROR_PRINT("cudaMemcpyPeerAsync Failed! (device%d to device0), c_status code: %d!\n", i + 1, c_status);
              return UCS_ERR_REJECTED;
            }
            c_status = cudaEventRecord(cuda_events[i + p*(RAILS-1)], cuda_stream);
            if (c_status != cudaSuccess)
            {
              DEBUG_PRINT("cudaEventRecord Failed!, c_status code: %d!\n", c_status);
              return UCS_ERR_REJECTED;
            }     
          } else if (status[i + p*(RAILS-1)+1] != UCS_INPROGRESS) {
              return status[i + p*(RAILS-1)+1];
          } 
        }

      }
    }

    if (finished == pipeline_stages*(RAILS-1)+1) break;

  }

  // Final synchronization of cudaEvents
  for (int i = 0; i < (RAILS - 1) * pipeline_stages; ++i)
  {
    c_status = cudaEventSynchronize(cuda_events[i]);
    if (c_status != cudaSuccess)
    {
      ERROR_PRINT("cudaEventSynchronize Failed! (cuda_events[%d]), c_status code: %d\n", i, c_status);
      return UCS_ERR_REJECTED;
    }
  }

  return UCS_OK;
}

ucs_status_t
ucx_mr_dual_split_send(ucx_mr_context_t *mr_ctx, ucp_tag_t tag, float split_ratio, int element_size,
                       void *buffer0, size_t msg_size0, ucs_memory_type_t mem_type0, int device0,
                       void *buffer1, ucs_memory_type_t mem_type1, int device1, int pipeline_stages)
{
  // split_ratio : factor how much is sent to peer (0.39 -> 39% is sent to peer device)
  // element_size: nof bytes of splitted memory needs to be multiple of element_size
  // buf_size1   : maximum possible splittable memory
  int RAILS = 2;
  int NOF_MESSAGES = pipeline_stages*(RAILS-1)+1;

  if (msg_size0 % element_size != 0) {
    ERROR_PRINT("Msg Size must be multiple of Element Size!\n");
    return UCS_ERR_INVALID_PARAM; 
  }

  cudaError_t c_status;
  ucs_status_t status;

  struct ucx_context *request[NOF_MESSAGES];
  for (int i = 0; i < NOF_MESSAGES; ++i) {
    request[i] = NULL;
  }

  ucp_request_param_t send_param[2];
  
  size_t msg_sizes[2];

  int devices[2]                  = {device0, device1};
  ucs_memory_type_t mem_types[2]  = {mem_type0, mem_type1};
  void *buffer[2]                 = {buffer0, buffer1};

  ucp_ep_h *ep          = mr_ctx->ep;
  ucp_worker_h *worker  = mr_ctx->worker;

  cudaStream_t cuda_stream  = mr_ctx->cuda_stream;
  cudaEvent_t *cuda_events    = mr_ctx->cuda_events;

  //int finished = 0;
  size_t offset = 0;
  size_t offset_p = 0;


  for (int i = 0; i < 2; ++i)
  {
    ucx_mr_write_send_param(&send_param[i], mem_types[i]);
  }

  calculate_msg_sizes_dual_pipeline(msg_size0, split_ratio, element_size, msg_sizes, pipeline_stages);

  // TODO: Check for msg size > 2^32!!!
  tag += msg_sizes[0] << 32;

  // Start sending staying message part
  cudaSetDevice(devices[0]);
  DEBUG_PRINT("Send Message0\n");
  request[0] = (ucx_context *) ucp_tag_send_nbx(ep[0], buffer[0], msg_sizes[0], tag, &send_param[0]);

  // Sending Data to Peer GPU
  for (int p = 0; p < pipeline_stages; ++p) {
    offset = 0;
    offset_p = 0;
    for (int i = 0; i < 1; ++i)
    {
      offset += msg_sizes[i];
      offset_p = msg_sizes[i+1] / pipeline_stages * p;

      DEBUG_PRINT("Offset: %ld, Offset_p: %ld\n", offset, offset_p);
      
      DEBUG_PRINT("Cuda Copy Started\n");
      c_status = cudaMemcpyPeerAsync(
        buffer[i+1]+offset_p, devices[i+1], buffer[0] + offset + offset_p, device0,
         msg_sizes[i+1] / pipeline_stages, cuda_stream
      );

      //DEBUG_PRINT("Sent to PEER:\n");
      //ucx_mr_read_test_message(buffer[0] + offset + offset_p, msg_sizes[i+1] / NOF_PIPELINE_STAGES / element_size, UCS_MEMORY_TYPE_CUDA);
      
      if (c_status != cudaSuccess)
      {
        ERROR_PRINT("cudaMemcpyPeerAsync Failed! (device0 to device%d), c_status code: %d!\n", i + 1, c_status);
        return UCS_ERR_REJECTED;
      }

      DEBUG_PRINT("Event Record with ix: %d for rail %d and Stage %d\n", i + p*(RAILS-1), i+1, p);
      c_status = cudaEventRecord(cuda_events[i + p*(RAILS-1)], cuda_stream);
      if (c_status != cudaSuccess)
      {
        ERROR_PRINT("cudaEventRecord Failed!, c_status code: %d!\n", c_status);
        return UCS_ERR_REJECTED;
      }
      DEBUG_PRINT("Cuda Copy Success\n");
    }
  }

  // Secondary Possible Synchronization Method without polling, if ordering of cuda stream is guaranteed
  for (int p = 0; p < pipeline_stages; ++p) {
    for (int i = 0; i < 1; ++i)
    {
      offset_p = msg_sizes[i+1] / pipeline_stages * p;
      cudaEventSynchronize(cuda_events[i + p*(RAILS-1)]);
      cudaSetDevice(i+1);
      request[i + p*(RAILS-1)+1] = (ucx_context *) ucp_tag_send_nbx(ep[i+1], buffer[i+1] + offset_p, msg_sizes[i+1] / pipeline_stages, tag+p, &send_param[i+1]);
      //status = ucx_wait(worker[i+1], request[i + p*(RAILS-1)+1], "send", data_msg_str);
    }
  }

  DEBUG_PRINT("All Sends started!\n");
  
  // Final synchronization of ucx communication
  for (int p = 0; p < pipeline_stages; ++p) 
  {
    for (int i = 0; i < 1; ++i)
    {
      for (;;) {
        cudaSetDevice(devices[i+1]);
        status = ucx_poll(worker[i+1], request[i + p*(RAILS-1)+1]);
        if (status == UCS_OK) {
          break;
        } else if (status != UCS_INPROGRESS) {
          return status;
        }  
      }
    }
  }
  
  status = ucx_wait(worker[0], request[0], "send", data_msg_str);
  return status; 
}


#endif // UCX_MR_COMM_DUAL_H