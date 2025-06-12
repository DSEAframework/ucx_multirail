/** @file ucx_mr_comm_quad.h
 *  @brief Quad railed communication methods
 *
 *  Communciation methods which use 4 rails in parallel. Either data can be send from 4 four buffers
 *  on seperated rails in parallel or one message is split and then sent with all 4 available rails.
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 *  @todo Remove quad_recv/send_async and check if only piped version works for pipelinestages=1
 */
#ifndef UCX_MR_COMM_QUAD_H
#define UCX_MR_COMM_QUAD_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <driver_types.h>

#include <ucp/api/ucp.h>

#include "../ucx_mr_comm.h"
#include "../ucx_mr_types.h"
#include "../ucx_mr_wait.h"


/** @brief Receive a message on each rail in parallel.
 *
 *  Seperated data is received on each rail in parallel
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @param tag tag for communication
 *  @param buffer (0-3) pointer to the receiving buffer for each rail
 *  @param buffer_size (0-3) buffer size for each rail in bytes
 *  @param mem_type (0-3) memory type of each buffer (Cuda/CPU)
 *  @param device (0-3) possible device index if memory type is Cuda
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_quad_recv(ucx_mr_context_t *mr_ctx, ucp_tag_t tag,
                 void *buffer0, size_t buffer_size0, ucs_memory_type_t mem_type0, int device0,
                 void *buffer1, size_t buffer_size1, ucs_memory_type_t mem_type1, int device1,
                 void *buffer2, size_t buffer_size2, ucs_memory_type_t mem_type2, int device2,
                 void *buffer3, size_t buffer_size3, ucs_memory_type_t mem_type3, int device3,
                 size_t *recv_sizes)
{
  ucs_status_t status[4];
  struct ucx_context *request[4];
  ucp_tag_message_h msg_tag[] = {NULL, NULL, NULL, NULL};
  ucp_tag_recv_info_t info_tag[4];

  ucp_request_param_t recv_param[4];

  ucp_worker_h *worker = mr_ctx->worker;

  int devices[] = {device0, device1, device2, device3};
  ucs_memory_type_t mem_types[] = {mem_type0, mem_type1, mem_type2, mem_type3};
  void *buffer[] = {buffer0, buffer1, buffer2, buffer3};
  size_t buffer_sizes[] = {buffer_size0, buffer_size1, buffer_size2, buffer_size3};

  for (int i = 0; i < 4; ++i)
  {
    ucx_mr_write_recv_param(&recv_param[i], mem_types[i]);
  }

  for (;;)
  {
    for (int i = 0; i < 4; ++i)
    {
      if (msg_tag[i] == NULL)
      {
        cudaSetDevice(devices[i]);
        msg_tag[i] = ucp_tag_probe_nb(worker[i], tag, tag_mask, 1, &info_tag[i]);

        if (msg_tag[i] != NULL)
        {
          request[i] = (ucx_context *) ucp_tag_msg_recv_nbx(worker[i], buffer[i], buffer_sizes[i], msg_tag[i], &recv_param[i]);
          recv_sizes[i] = info_tag[i].length;
        }
        ucp_worker_progress(worker[i]);
      }
    }

    if (msg_tag[0] != NULL && msg_tag[1] != NULL && msg_tag[2] != NULL && msg_tag[3] != NULL)
      break;
  }

  for (int i = 0; i < 4; ++i)
  {
    cudaSetDevice(devices[i]);
    status[i] = ucx_wait(worker[i], request[i], "receive", data_msg_str);
    if (status[i] != UCS_OK)
      return status[i];
  }

  return UCS_OK;
}


/** @brief Send a message on each rail in parallel.
 *
 *  Seperated data is sent on each rail in parallel
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @param tag tag for communication
 *  @param buffer (0-3) pointer to the send buffer for each rail
 *  @param msg_size (0-3) message size for each rail in bytes
 *  @param mem_type (0-3) memory type of each buffer (Cuda/CPU)
 *  @param device (0-3) possible device index if memory type is Cuda
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_quad_send(ucx_mr_context_t *mr_ctx, ucp_tag_t tag,
                 void *buffer0, size_t msg_size0, ucs_memory_type_t mem_type0, int device0,
                 void *buffer1, size_t msg_size1, ucs_memory_type_t mem_type1, int device1,
                 void *buffer2, size_t msg_size2, ucs_memory_type_t mem_type2, int device2,
                 void *buffer3, size_t msg_size3, ucs_memory_type_t mem_type3, int device3)
{
  ucs_status_t status[4];
  struct ucx_context *request[4];

  ucp_request_param_t send_param[4];

  ucp_ep_h *ep = mr_ctx->ep;
  ucp_worker_h *worker = mr_ctx->worker;

  int devices[] = {device0, device1, device2, device3};
  ucs_memory_type_t mem_types[] = {mem_type0, mem_type1, mem_type2, mem_type3};
  void *buffer[] = {buffer0, buffer1, buffer2, buffer3};
  size_t msg_sizes[] = {msg_size0, msg_size1, msg_size2, msg_size3};

  for (int i = 0; i < 4; ++i)
  {
    ucx_mr_write_send_param(&send_param[i], mem_types[i]);
    cudaSetDevice(devices[i]);
    request[i] = (ucx_context *) ucp_tag_send_nbx(ep[i], buffer[i], msg_sizes[i], tag, &send_param[i]);
  }

  for (int i = 0; i < 4; ++i)
  {
    cudaSetDevice(devices[i]);
    status[i] = ucx_wait(worker[i], request[i], "send", data_msg_str);
    if (status[i] != UCS_OK)
      return status[i];
  }

  return UCS_OK;
}


ucs_status_t
ucx_mr_quad_split_recv_async(ucx_mr_context_t *mr_ctx, ucp_tag_t tag, float split_ratio, int element_size,
                       void *buffer0, size_t buffer_size, ucs_memory_type_t mem_type0, int device0,
                       void *buffer1, ucs_memory_type_t mem_type1, int device1,
                       void *buffer2, ucs_memory_type_t mem_type2, int device2,
                       void *buffer3, ucs_memory_type_t mem_type3, int device3)
{
  // 0 is the collecting device
  cudaError_t c_status;
  ucs_status_t status[4] = {UCS_ERR_NO_MESSAGE, UCS_ERR_NO_MESSAGE, UCS_ERR_NO_MESSAGE, UCS_ERR_NO_MESSAGE};

  struct ucx_context *request[4] = {NULL, NULL, NULL, NULL};
  ucp_request_param_t recv_param[4];

  ucp_tag_recv_info_t info_tag[4];
  ucp_tag_message_h msg_tag[4] = {NULL, NULL, NULL, NULL};
  size_t msg_sizes[4] = {0};


  int devices[4]                  = {device0, device1, device2, device3};
  ucs_memory_type_t mem_types[4]  = {mem_type0, mem_type1, mem_type2, mem_type3};
  void *buffer[4]                 = {buffer0, buffer1, buffer2, buffer3};
  size_t buffer_sizes[4]          = {buffer_size, buffer_size, buffer_size, buffer_size};

  ucp_worker_h *worker = mr_ctx->worker;

  cudaStream_t *cuda_streams = mr_ctx->cuda_streams;
  //cudaEvent_t *cuda_events = mr_ctx->cuda_events;

  for (int i = 0; i < 4; ++i)
  {
    ucx_mr_write_recv_param(&recv_param[i], mem_types[i]);
  }

  for (;;)
  {
    for (int i = 0; i < 4; ++i)
    {
      if (msg_tag[i] == NULL)
      {
        cudaSetDevice(devices[i]);
        msg_tag[i] = ucp_tag_probe_nb(worker[i], tag, tag_mask, 1, &info_tag[i]);

        if (msg_tag[i] != NULL)
        {
          request[i] = (ucx_context *) ucp_tag_msg_recv_nbx(worker[i], buffer[i], buffer_sizes[i], msg_tag[i], &recv_param[i]);
          msg_sizes[i] = info_tag[i].length;
        }
        ucp_worker_progress(worker[i]);
      }
    }

    if (msg_tag[0] != NULL && msg_tag[1] != NULL && msg_tag[2] != NULL && msg_tag[3] != NULL)
      break;
  }

  DEBUG_PRINT("Now polling for cuda \n");

  size_t offset;
  for (;;) {
    offset = 0;
    for (int i = 0; i < 4; ++i)
    {
      if (status[i] != UCS_OK) {
        cudaSetDevice(devices[i]);
        status[i] = ucx_poll(worker[i], request[i]);
        if (status[i] != UCS_OK && status[i] != UCS_INPROGRESS)
          return status[i];

        if (i != 0 && status[i] == UCS_OK) {
          offset += msg_sizes[i-1];
          
          c_status = cudaMemcpyPeerAsync((buffer0 + offset), devices[0], buffer[i], devices[i], msg_sizes[i], cuda_streams[i-1]);
          if (c_status != cudaSuccess)
          {
            ERROR_PRINT("cudaMemcpyPeer Failed!\n");
            return UCS_ERR_REJECTED;
          }
          
        }
      }
    }

    if (status[0] == UCS_OK && status[1] == UCS_OK && status[2] == UCS_OK && status[3] == UCS_OK)
      break;
  }

  DEBUG_PRINT("Synchronizing Cuda \n");

  for (int i = 0; i < 3; ++i)
  {
    DEBUG_PRINT("Synchronize Streams %d\n", i);
    c_status = cudaStreamSynchronize(cuda_streams[i]);
    if (c_status != cudaSuccess)
    {
      ERROR_PRINT("cudaStreamSynchronize Failed!\n");
      return UCS_ERR_REJECTED;
    }
  }

  return UCS_OK;
}

ucs_status_t
ucx_mr_quad_split_send_async(ucx_mr_context_t *mr_ctx, ucp_tag_t tag, float split_ratio, int element_size,
                       void *buffer0, size_t msg_size0, ucs_memory_type_t mem_type0, int device0,
                       void *buffer1, ucs_memory_type_t mem_type1, int device1,
                       void *buffer2, ucs_memory_type_t mem_type2, int device2,
                       void *buffer3, ucs_memory_type_t mem_type3, int device3)
{
  // split_ratio : factor how much is sent to peer (0.39 -> 39% is sent to peer devices)
  // element_size: nof bytes of splitted memory needs to be multiple of element_size
  // buf_size1   : maximum possible splittable memory
  cudaError_t c_status;
  ucs_status_t status[4] = {UCS_ERR_NO_MESSAGE, UCS_ERR_NO_MESSAGE, UCS_ERR_NO_MESSAGE, UCS_ERR_NO_MESSAGE};

  struct ucx_context *request[4] = {NULL};
  ucp_request_param_t send_param[4];

  size_t msg_sizes[4];

  int devices[4]                     = {device0, device1, device2, device3};
  ucs_memory_type_t mem_types[4]     = {mem_type0, mem_type1, mem_type2, mem_type3};
  void *buffer[4]                    = {buffer0, buffer1, buffer2, buffer3};

  ucp_ep_h *ep              = mr_ctx->ep;
  ucp_worker_h *worker      = mr_ctx->worker;

  cudaStream_t *cuda_streams  = mr_ctx->cuda_streams;
  cudaEvent_t *cuda_events    = mr_ctx->cuda_events;

  for (int i = 0; i < 4; ++i)
  {
    ucx_mr_write_send_param(&send_param[i], mem_types[i]);
  }

  calculate_msg_sizes_quad(msg_size0, split_ratio, element_size, msg_sizes);
  //calculate_msg_sizes_quad_alt(msg_size0, split_ratio, element_size, msg_sizes);

  cudaSetDevice(devices[0]);
  request[0] = (ucx_context *) ucp_tag_send_nbx(ep[0], buffer[0], msg_sizes[0], tag, &send_param[0]);

  
  size_t offset = 0;
  for (int i = 0; i < 3; ++i)
  {

    offset += msg_sizes[i];

    DEBUG_PRINT("Offset: %ld\n", offset);

    c_status = cudaMemcpyPeerAsync(buffer[i + 1], devices[i + 1], buffer0 + offset, device0, msg_sizes[i + 1], cuda_streams[i]);
    if (c_status != cudaSuccess)
    {
      DEBUG_PRINT("CudaMemcpy Peer Error!\n");
      return UCS_ERR_REJECTED;
    }
    c_status = cudaEventRecord(cuda_events[i], cuda_streams[i]);
    if (c_status != cudaSuccess)
    {
      DEBUG_PRINT("cudaEvent Record Error!\n");
      return UCS_ERR_REJECTED;
    }
  }

  /*
  for (int i = 0; i < 3; ++i)
  {
    cudaEventSynchronize(cuda_events[i]);
    cudaSetDevice(i+1);
    request[i+1] = ucp_tag_send_nbx(ep[i+1], buffer[i+1], msg_sizes[i+1], tag, &send_param[i+1]);

  }
  */


  int finished = 0;
  int events_finished[3] = {0, 0, 0};

  while (finished != 3) {
    for (int i = 1; i < 4; ++i)
    {
      if (events_finished[i-1] == 0) {
        c_status = cudaEventQuery(cuda_events[i-1]);
        if (c_status == cudaSuccess) {
          finished++;
          events_finished[i-1]++;

          cudaSetDevice(i);
          request[i] = (ucx_context *) ucp_tag_send_nbx(ep[i], buffer[i], msg_sizes[i], tag, &send_param[i]);


        }
      }
    }
  }
  


  DEBUG_PRINT("CudaMemcpy Peer finished!\n");

  for (;;) {
    for (int i = 0; i < 4; ++i)
    {
      if (status[i] != UCS_OK) {
        cudaSetDevice(devices[i]);
        status[i] = ucx_poll(worker[i], request[i]);
        if (status[i] != UCS_OK && status[i] != UCS_INPROGRESS)
          return status[i];
      }
    }

    if (status[0] == UCS_OK && status[1] == UCS_OK && status[2] == UCS_OK && status[3] == UCS_OK)
      break;
  }
  
  return UCS_OK;
}


/** @brief Receive one message through 4 rails. (GPU memory only)
 *
 *  One message is partly received on 4 rails and afterwards collected on device0
 *  The receiving can be done pipelined where the messages on rail 1-3 come in stages.
 *  This is done for better performance. A split ratio is given which says how much memory
 *  is transferred by rail 1-3 and not received by device0 directly.
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @param tag tag for communication
 *  @param split_ratio amount of memory sent by rail 1-3
 *  @param buffer (0-3) pointer to the receiving buffer for each rail
 *  @param buffer_size buffer size on each device in bytes (TODO: device1-3 do not need full buffer size!)
 *  @param mem_type (0-3) memory type of each buffer (TODO: remove, split only implemented for GPU memory)
 *  @param device (0-3) device index of each rail
 *  @param pipeline_stages number of pipeline_stages for messages on rail 1-3
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_quad_split_recv_piped(ucx_mr_context_t *mr_ctx, ucp_tag_t tag, float split_ratio, int element_size,
                       void *buffer0, size_t buffer_size, ucs_memory_type_t mem_type0, int device0,
                       void *buffer1, ucs_memory_type_t mem_type1, int device1,
                       void *buffer2, ucs_memory_type_t mem_type2, int device2,
                       void *buffer3, ucs_memory_type_t mem_type3, int device3, int pipeline_stages)
{
  // 0 is the collecting device
  int RAILS = 4;
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
  ucp_request_param_t recv_param[4];
  
  ucp_tag_recv_info_t info_tag[4];
  ucp_tag_message_h msg_tag[NOF_MESSAGES];
  for (int i = 0; i < NOF_MESSAGES; ++i) {
    msg_tag[i] = NULL;
  }

  size_t msg_sizes[4] = {0};

  int devices[4]                  = {device0, device1, device2, device3};
  ucs_memory_type_t mem_types[4]  = {mem_type0, mem_type1, mem_type2, mem_type3};
  void *buffer[4]                 = {buffer0, buffer1, buffer2, buffer3};
  size_t buffer_sizes[4]          = {buffer_size, buffer_size, buffer_size, buffer_size};


  ucp_worker_h *worker = mr_ctx->worker;

  cudaStream_t cuda_stream = mr_ctx->cuda_stream;
  cudaEvent_t *cuda_events = mr_ctx->cuda_events;

  int finished = 0;
  size_t offset = 0;
  size_t offset_p = 0;

  finished = 0;
  for (int i = 0; i < 4; ++i)
  {
    ucx_mr_write_recv_param(&recv_param[i], mem_types[i]);
  }

  // Check for all incoming messages
  for (;;) {
    // Not pipelined message
    if (msg_tag[0] == NULL)
    {
      cudaSetDevice(devices[0]);
      // Check for message on rail0
      msg_tag[0] = ucp_tag_probe_nb(worker[0], tag, tag_mask, 1, &info_tag[0]);

      if (msg_tag[0] != NULL)
      {
        DEBUG_PRINT("Message 0 found!\n");
        // Receive single message on rail0
        request[0] = (ucx_context *) ucp_tag_msg_recv_nbx(worker[0], buffer[0], buffer_sizes[0], msg_tag[0], &recv_param[0]);
      }
      ucx_mr_worker_progress(mr_ctx, 0);
    } 

    // Pipelined messages
    for (int p = 0; p < pipeline_stages; ++p) {
      for (int i = 0; i < 3; ++i)
      {
        if (msg_tag[i + p*(RAILS-1)+1] == NULL)
        {
          ucx_mr_worker_progress(mr_ctx, i+1);
          cudaSetDevice(devices[i+1]);
          // Check for message on rail i+1 and stage p
          msg_tag[i + p*(RAILS-1)+1] = ucp_tag_probe_nb(worker[i+1], tag+p, tag_mask, 1, &info_tag[i+1]);

          if (msg_tag[i + p*(RAILS-1)+1] != NULL)
          {
            DEBUG_PRINT("Message found on rail %d, stage %d\n",i+1, p);
            DEBUG_PRINT("Idx: %d\n",i + p*(RAILS-1));
            
            msg_sizes[i+1] = info_tag[i+1].length;
            // Extract message size on rail0 from tag, to make sure the offset is known even if message on rail0 is not received yet
            msg_sizes[0] = info_tag[i+1].sender_tag >> 32;
            DEBUG_PRINT("Message Size Recv: %ld\n", msg_sizes[i+1]);
            DEBUG_PRINT("Offset_p: %ld\n", p*msg_sizes[i+1]);
            // Start receiving message on rail i+1 and stage p
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
      offset = msg_sizes[0];
      offset_p = 0;
      for (int i = 0; i < 3; ++i)
      {
        
        offset_p = msg_sizes[i+1] * p;

        if (status[i + p*(RAILS-1)+1] != UCS_OK && request[i + p*(RAILS-1)+1] != NULL) {
          cudaSetDevice(devices[i+1]);
          // Check if receiving is finished of message on rail i+1 and stage p
          status[i + p*(RAILS-1)+1] = ucx_poll(worker[i+1], request[i + p*(RAILS-1)+1]);

          if (status[i + p*(RAILS-1)+1] == UCS_OK) {
            DEBUG_PRINT("Message on rail %d stage %d ready\n", i+1, p);
            finished++;     
            DEBUG_PRINT("Copy to location: offset: %ld, offset_p: %ld\n", offset, offset_p);
            
            // Start transfer of message to correct location on device0
            c_status = cudaMemcpyPeerAsync(
              buffer[0] + offset + offset_p, devices[0], buffer[i+1] + offset_p, 
              devices[i+1], msg_sizes[i+1], cuda_stream
            );
            DEBUG_PRINT("Cuda Copy Started on rail %d stage %d!\n", i+1, p);
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
        // Update memory offset for next rail
        offset +=  msg_sizes[i+1] * pipeline_stages;
      }
    }

    // Check if a memory transfers are started
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


/** @brief Receive one message through 4 rails. (GPU memory only)
 *
 *  One message is partly received on 4 rails and afterwards collected on device0
 *  The receiving can be done pipelined where the messages on rail 1-3 come in stages.
 *  This is done for better performance. A split ratio is given which says how much memory
 *  is transferred by rail 1-3 and not received by device0 directly.
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @param tag tag for communication
 *  @param split_ratio amount of memory sent by rail 1-3
 *  @param buffer (0-3) pointer to the receiving buffer for each rail
 *  @param buffer_size buffer size on each device in bytes (TODO: device1-3 do not need full buffer size!)
 *  @param mem_type (0-3) memory type of each buffer (TODO: remove, split only implemented for GPU memory)
 *  @param device (0-3) device index of each rail
 *  @param pipeline_stages number of pipeline_stages for messages on rail 1-3
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_quad_split_send_piped(ucx_mr_context_t *mr_ctx, ucp_tag_t tag, float split_ratio, int element_size,
                       void *buffer0, size_t msg_size0, ucs_memory_type_t mem_type0, int device0,
                       void *buffer1, ucs_memory_type_t mem_type1, int device1,
                       void *buffer2, ucs_memory_type_t mem_type2, int device2,
                       void *buffer3, ucs_memory_type_t mem_type3, int device3, int pipeline_stages)
{
  // split_ratio : factor how much is sent to peer (0.39 -> 39% is sent to peer devices)
  // element_size: nof bytes of splitted memory needs to be multiple of element_size
  // buf_size1   : maximum possible splittable memory
  int RAILS = 4;
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


  ucp_request_param_t send_param[4];
  
  size_t msg_sizes[4];

  int devices[4]                     = {device0, device1, device2, device3};
  ucs_memory_type_t mem_types[4]     = {mem_type0, mem_type1, mem_type2, mem_type3};
  void *buffer[4]                    = {buffer0, buffer1, buffer2, buffer3};

  ucp_ep_h *ep          = mr_ctx->ep;
  ucp_worker_h *worker  = mr_ctx->worker;

  cudaStream_t cuda_stream  = mr_ctx->cuda_stream;
  cudaEvent_t *cuda_events    = mr_ctx->cuda_events;

  //int finished = 0;
  size_t offset = 0;
  size_t offset_p = 0;


  for (int i = 0; i < 4; ++i)
  {
    ucx_mr_write_send_param(&send_param[i], mem_types[i]);
  }

  calculate_msg_sizes_quad_pipeline(msg_size0, split_ratio, element_size, msg_sizes, pipeline_stages);

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
    for (int i = 0; i < 3; ++i)
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
    for (int i = 0; i < 3; ++i)
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
    for (int i = 0; i < 3; ++i)
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

ucs_status_t
ucx_mr_quad_split_recv(ucx_mr_context_t *mr_ctx, ucp_tag_t tag, float split_ratio, int element_size,
                       void *buffer0, size_t msg_size0, ucs_memory_type_t mem_type0, int device0,
                       void *buffer1, ucs_memory_type_t mem_type1, int device1,
                       void *buffer2, ucs_memory_type_t mem_type2, int device2,
                       void *buffer3, ucs_memory_type_t mem_type3, int device3, int pipeline_stages)
{
  if (pipeline_stages == 1) {
    DEBUG_PRINT("Use async method!\n");
    return ucx_mr_quad_split_recv_async(mr_ctx, tag, split_ratio, element_size,
                       buffer0, msg_size0, mem_type0, device0,
                       buffer1, mem_type1, device1,
                       buffer2, mem_type2, device2,
                       buffer3, mem_type3, device3);
  } else {
    DEBUG_PRINT("Use pipelined method!\n");
    return ucx_mr_quad_split_recv_piped(mr_ctx, tag, split_ratio, element_size,
                       buffer0, msg_size0, mem_type0, device0,
                       buffer1, mem_type1, device1,
                       buffer2, mem_type2, device2,
                       buffer3, mem_type3, device3, pipeline_stages);
  }  
}

ucs_status_t
ucx_mr_quad_split_send(ucx_mr_context_t *mr_ctx, ucp_tag_t tag, float split_ratio, int element_size,
                       void *buffer0, size_t msg_size0, ucs_memory_type_t mem_type0, int device0,
                       void *buffer1, ucs_memory_type_t mem_type1, int device1,
                       void *buffer2, ucs_memory_type_t mem_type2, int device2,
                       void *buffer3, ucs_memory_type_t mem_type3, int device3, int pipeline_stages)
{
  if (pipeline_stages == 1) {
    DEBUG_PRINT("Use async method!\n");
    return ucx_mr_quad_split_send_async(mr_ctx, tag, split_ratio, element_size,
                        buffer0, msg_size0, mem_type0, device0,
                        buffer1, mem_type1, device1,
                        buffer2, mem_type2, device2,
                        buffer3, mem_type3, device3);
  } else {
    DEBUG_PRINT("Use pipelined method!\n");
    return ucx_mr_quad_split_send_piped(mr_ctx, tag, split_ratio, element_size,
                        buffer0, msg_size0, mem_type0, device0,
                        buffer1, mem_type1, device1,
                        buffer2, mem_type2, device2,
                        buffer3, mem_type3, device3, pipeline_stages);
  }
}



#endif // UCX_MR_COMM_QUAD_H