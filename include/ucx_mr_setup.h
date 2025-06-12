/** @file ucx_mr_setup.h
 *  @brief Functions for setting up ucx multirail enviroment.
 *
 *  Functions for generating a ucx multi-rail context (ucx_mr_context) with
 *  all its ucx objects like ucp_context, ucp_worker and ucp_endpoints. Additionally
 *  the Cuda enviroment is setup and objects like streams and events are created for
 *  asynchronous Cuda Methods. Methods are adapted from the ucx perftest.
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 *  @todo Use only single cuda stream throughout the project
 */
#ifndef UCX_MR_SETUP_H
#define UCX_MR_SETUP_H

#include <cuda_runtime.h>
#include <cuda.h>

#include <ucp/api/ucp.h>

#include "ucx_mr_types.h"
#include "ucx_mr_aux.h"
#include "ucx_mr_sock_comm.h"
#include "ucx_mr_cleanup.h"

#include "ucx_mr_comm.h"

/** @brief Parse command line arguments to set basic enviroment variables 
 *
 *  Communication port, network devices and server address can be set from command line arguments.
 *  P:  communication port
 *  R:  Net devices to be used
 *  A:  Server Address
 *
 *  @param mr_ctx empty multi-rail context
 *  @return ucs_status_t possible error status
 */
ucs_status_t
parse_opts(ucx_mr_context_t *mr_ctx, int argc, char **argv)
{
  int c;
  char *ptr;

  optind = 1;
  while ((c = getopt(argc, argv, "P:R:A:")) != -1)
  {
    switch (c)
    {
    case 'P':
      DEBUG_PRINT("Got a port %s\n", optarg);
      mr_ctx->port = atoi(optarg);
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
        mr_ctx->rails[i] = ptr;
        DEBUG_PRINT("Got Rail %s\n", ptr);
      }
      break;
    case 'A':
      DEBUG_PRINT("Got Server Address %s\n", optarg);
      mr_ctx->server_addr = optarg;
      break;
    default:
      DEBUG_PRINT("Default\n");
      break;
    }
  }

  return UCS_OK;
}

/** @brief Create ucp_endpoint and connect with corresponding node
 *
 *  Fully adapted by ucx perftest and adjusted to fit in this multi-rail approach.
 *  Helper functions are all adapted from perftest. (Adapted from ucx perftest)
 *
 *  @param mr_ctx multi-rail context with created ucp_context's and ucp_worker's
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_create_endpoint(ucx_mr_context_t *mr_ctx)
{
  ucs_status_t status;

  unsigned group_size = mr_ctx->sock_rte_group.size;
  unsigned group_index = mr_ctx->sock_rte_group.is_server ? 0 : 1;
  unsigned peer_index = rte_peer_index(group_size, group_index);

  // printf("group_size %u, group_index %u, peer_index %u\n", group_size, group_index, peer_index);

  int idx = mr_ctx->ep_count;

  if (idx < NOF_RAILS)
  {

    status = ucx_mr_send_local_data(mr_ctx, idx);
    if (status != UCS_OK)
    {
      return status;
    }

    /* receive remote peer's endpoints' data and connect to them */
    status = ucx_mr_recv_remote_data(mr_ctx, peer_index, idx);
    if (status != UCS_OK)
    {
      (void)ucx_mr_exchange_status(mr_ctx, status);
      return status;
    }

    mr_ctx->ep_count = idx + 1;
  }

  /* sync status across all processes */
  status = ucx_mr_exchange_status(mr_ctx, UCS_OK);
  if (status != UCS_OK)
  {
    ucx_mr_destroy_endpoint(mr_ctx);
    (void)ucx_mr_exchange_status(mr_ctx, status);
    return status;
  }

  /* force wireup completion */
  return ucx_mr_flush_workers(mr_ctx);
}

/** @brief Create ucp_worker for next rail
 *
 *  Check for the next rail which is processed. With the corresponding
 *  ucp_contex a ucp_worker is created and stored in the multi-rail ucx context
 *  at the correct index. (Adapted from ucx perftest)
 *
 *  @param mr_ctx multi-rail context with created ucp_context's
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_create_worker(ucx_mr_context_t *mr_ctx)
{
  ucs_status_t status = UCS_OK;
  ucp_worker_params_t *worker_params = (ucp_worker_params_t *)malloc(sizeof(worker_params));
  if (worker_params == NULL)
  {
    ERROR_PRINT("failed to allocate memory for worker params");
    status = UCS_ERR_NO_MEMORY;
    return status;
  }

  worker_params->field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params->thread_mode = UCS_THREAD_MODE_SINGLE;

  int idx = mr_ctx->worker_count;

  if (idx < NOF_RAILS)
  {
    status = ucp_worker_create(mr_ctx->ctx[idx], worker_params, &mr_ctx->worker[idx]);
    if (status == UCS_OK)
    {
      mr_ctx->worker_count = idx + 1;
    }
  }
  else
  {
    printf("Maximum number of workers reached!\n");
  }

  free(worker_params);
  return status;
}


/** @brief Setting up ucp_contexts for each rail.
 *
 *  Reading in the ucp config and modify it so that for each rail a specific
 *  network device is used. Afterwards the ucp_context is initialized and stored
 *  in the multi-rail ucx context.
 *
 *  @param mr_ctx empty multi-rail context with all enviroment data
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_init(ucx_mr_context_t *mr_ctx)
{
  ucp_params_t ucp_params;
  ucs_status_t status;
  ucp_config_t *config;

  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_REQUEST_SIZE |
                          UCP_PARAM_FIELD_REQUEST_INIT |
                          UCP_PARAM_FIELD_NAME;
  ucp_params.features = 0;
  ucp_params.features |= UCP_FEATURE_TAG;
  ucp_params.request_size = sizeof(struct ucx_context);
  ucp_params.request_init = request_init;

  // Set up enviroment and create UCP_Context
  status = ucp_config_read(NULL, NULL, &config);
  if (status != UCS_OK)
  {
    return status;
  }

  int idx = mr_ctx->ctx_count;

  // Set specific net device for this rail
  status = ucp_config_modify(config, "NET_DEVICES", mr_ctx->rails[idx]);

  if (status != UCS_OK)
  {
    ucp_config_release(config);
    return status;
  }

  if (idx < NOF_RAILS)
  {
    status = ucp_init(&ucp_params, config, &mr_ctx->ctx[idx]);
    if (status == UCS_OK)
    {
      mr_ctx->ctx_count = idx + 1;
    }
  }
  else
  {
    printf("Maximum number of contexts reached!\n");
  }

  return status;
}

/** @brief Setting up multi-rail ucx context with all needed information
 *
 *  Communication is initated through sockets. Afterwards all ucx objects are created.
 *  For each rail a context a worker and a endpoint is created and connected. The Cuda
 *  enviroment is setup so that the peer communication is enabled between all gpu's which
 *  are used for the multi-railed communication. Streams and events are created so that
 *  asynchronous communication can be used.
 *
 *  @param mr_ctx empty multi-rail context with all enviroment data
 *  @return ucs_status_t possible error status
 */
ucs_status_t
ucx_mr_setup(ucx_mr_context_t *mr_ctx)
{
  ucs_status_t status;
  cudaError_t c_status;
  mr_ctx->ctx_count = 0;
  mr_ctx->worker_count = 0;
  mr_ctx->ep_count = 0;

  // Connect Communication
  status = ucx_mr_connect_comm(mr_ctx);
  if (status != UCS_OK)
  {
    ERROR_PRINT("Connection cannot be established!\n");
    return status;
  }
  DEBUG_PRINT("Connection established!\n");

  // Setup UCX Contexts
  for (int i = 0; i < NOF_RAILS; ++i)
  {
    status = ucx_mr_init(mr_ctx);
    if (status != UCS_OK)
    {
      ERROR_PRINT("Error in ucp_context initialization (Rail:%d)!\n", i);
      ucx_mr_cleanup(mr_ctx, CTX);
      return status;
    }
  }
  DEBUG_PRINT("Contexts created!\n");

  // Setup UCX Worker
  for (int i = 0; i < NOF_RAILS; ++i)
  {
    status = ucx_mr_create_worker(mr_ctx);
    if (status != UCS_OK)
    {
      ERROR_PRINT("Error in ucp_worker creation (Rail:%d)!\n", i);
      ucx_mr_cleanup(mr_ctx, WORKER);
      return status;
    }
  }
  DEBUG_PRINT("Workers created!\n");

  // Setup UCX Endpoints
  for (int i = 0; i < NOF_RAILS; ++i)
  {
    status = ucx_mr_create_endpoint(mr_ctx);
    if (status != UCS_OK)
    {
      ERROR_PRINT("Error in ucp_endpoint creation (Rail:%d)!\n", i);
      ucx_mr_cleanup(mr_ctx, EP);
      return status;
    }
  }
  DEBUG_PRINT("Endpoints created!\n");

  // Setup Cuda Peer Access
  for (int i = 0; i < NOF_RAILS; ++i)
  {
    for (int j = 0; j < NOF_RAILS; ++j)
    {
      if (i != j)
      {
        cudaSetDevice(i);
        cudaDeviceEnablePeerAccess(j, i);
      }
    }
  }

  // Check Cuda Peer Access
  int canAccess;
  for (int i = 0; i < NOF_RAILS; ++i)
  {
    for (int j = 0; j < NOF_RAILS; ++j)
    {
      if (i != j)
      {
        cudaDeviceCanAccessPeer(&canAccess, i, j);
        if (canAccess != 1) {
          ERROR_PRINT("Error in Peer Access: Can Access %d -> %d: %d\n", i, j, canAccess);
          ucx_mr_cleanup(mr_ctx, EP);
          return UCS_ERR_NOT_CONNECTED;
        }
      }
    }
  }

  DEBUG_PRINT("Cuda Peer Access enabled!\n");

  // Create Cuda Streams for Ansychronous Splitting
  // TODO: Use only single cuda stream if performance does not decrease!
  mr_ctx->cuda_stream = NULL;
  cudaStreamCreate(&mr_ctx->cuda_stream);

  for (int i = 0; i < NOF_RAILS - 1; ++i)
  {
    mr_ctx->cuda_streams[i] = NULL;
    c_status = cudaStreamCreate(&mr_ctx->cuda_streams[i]);
    if (c_status != cudaSuccess)
    {
      ERROR_PRINT("Error while creating CudaStreams!\n");
      ucx_mr_cleanup(mr_ctx, STREAMS);
      return UCS_ERR_REJECTED;
    }
  }

  DEBUG_PRINT("Pipeline Stream Enabled!\n");

  // Create Cuda Events for Ansychronous Synchronization
  for (int i = 0; i < (NOF_RAILS - 1) * MAX_PIPELINE_STAGES; ++i)
  {
    mr_ctx->cuda_events[i] = NULL;
    c_status = cudaEventCreate(&mr_ctx->cuda_events[i]);
    if (c_status != cudaSuccess)
    {
      ERROR_PRINT("Error while creating CudaEvents!\n");
      ucx_mr_cleanup(mr_ctx, EVENTS);
      return UCS_ERR_REJECTED;
    }
  }

  DEBUG_PRINT("Synchronization Events created!\n");

  return status;
}

#endif // UCX_MR_SETUP_H