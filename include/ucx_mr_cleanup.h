/** @file ucx_mr_cleanup.h
 *  @brief Collection of clean up function for stopping the programm.
 *
 *  Cleanup functions for giving free resources of specific ucp objects, like endpoints,
 *  workers and contexts. One collective cleanup funcion for error handling and cleaning up
 *  from the correct position.
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef UCX_MR_CLEANUP
#define UCX_MR_CLEANUP

#include <ucp/api/ucp.h>

#include "ucx_mr_types.h"
#include "ucx_mr_sock_comm.h"

/** @brief Destroy all exisiting ucp endpoints in given multi-rail context
 *
 *  Close all existing endpoint connections and handle still open communication and requests.
 *  (Adapted from ucx perftest)
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @return void
 */
void 
ucx_mr_destroy_endpoint(ucx_mr_context_t *mr_ctx)
{
  for (int i = 0; i < mr_ctx->ep_count; ++i)
  {
    // Not ready
    ucs_status_ptr_t *req = NULL;
    ucs_status_t status;

    ucx_mr_worker_progress(mr_ctx, i);

    if (mr_ctx->ep[i] != NULL)
    {
      req = (ucs_status_ptr_t *) ucp_ep_close_nb(mr_ctx->ep[i], UCP_EP_CLOSE_MODE_FLUSH);
    }
    if (!UCS_PTR_IS_PTR(req) && req != NULL)
    {
      WARN_PRINT("failed to close ep %p: %s\n",
                 mr_ctx->ep[i],
                 ucs_status_string(UCS_PTR_STATUS(req)));
    }

    ucx_mr_worker_progress(mr_ctx, i);
    if (req)
    {
      status = ucp_request_check_status(req);
      while (status == UCS_INPROGRESS)
      {
        ucx_mr_worker_progress(mr_ctx, i);
        status = ucp_request_check_status(req);
      }
      ucp_request_free(req);
    }
  }
}

/** @brief Destroy all exisiting ucp workers in given multi-rail context
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @return void
 */
void 
ucx_mr_destroy_worker(ucx_mr_context_t *mr_ctx)
{
  for (int i = 0; i < mr_ctx->worker_count; ++i)
  {
    ucp_worker_destroy(mr_ctx->worker[i]);
  }
}

/** @brief Destroy all exisiting ucp contexts in given multi-rail context
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @return void
 */
void 
ucx_mr_destroy_ctx(ucx_mr_context_t *mr_ctx)
{
  for (int i = 0; i < mr_ctx->ctx_count; ++i)
  {
    ucp_cleanup(mr_ctx->ctx[i]);
  };
}

/** @brief Collected cleanup function for full multi-rail context
 *  
 *  Collected cleanup functions with possiblity to cleanup only from a specific position
 *  so this function can be used for error handling in the setup functions. All the ucp
 *  instances are cleaned up and open Cuda Streams and Events are synchronized and destroyed.
 *   
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @param pos position from which the cleanup should be started
 *  @return void
 */
void 
ucx_mr_cleanup(ucx_mr_context_t *mr_ctx, CleanUp pos)
{
  switch (pos)
  {
  case FULL:
  case EVENTS:
    for (int i = 0; i < NOF_RAILS - 1; ++i)
    {
      if (mr_ctx->cuda_events[i] != NULL)
      {
        cudaEventSynchronize(mr_ctx->cuda_events[i]);
        cudaEventDestroy(mr_ctx->cuda_events[i]);
      }
    }
  case STREAMS:
    for (int i = 0; i < NOF_RAILS - 1; ++i)
    {
      if (mr_ctx->cuda_streams[i] != NULL)
      {
        cudaStreamSynchronize(mr_ctx->cuda_streams[i]);
        cudaStreamDestroy(mr_ctx->cuda_streams[i]);
      }
    }
  case EP:
    ucx_mr_destroy_endpoint(mr_ctx);
    DEBUG_PRINT("Endpoints destroyed.\n");
  case WORKER:
    ucx_mr_destroy_worker(mr_ctx);
    DEBUG_PRINT("Worker destroyed.\n");
  case CTX:
    ucx_mr_destroy_ctx(mr_ctx);
    DEBUG_PRINT("UCP Cleanup.\n");
  case COMM:
    ucx_mr_cleanup_comm(mr_ctx);
    DEBUG_PRINT("Comm cleanup.\n");
  default:;
  }
}

#endif // UCX_MR_CLEANUP