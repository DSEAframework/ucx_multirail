/** @file ucx_mr_aux.h
 *  @brief Wrapped auxiliary ucx functions.
 *
 *  Auxiliary functions for processing ucx workers and generate progress on communication
 *  lanes. Functions are adapted from the ucx perftest and slightly modified to fit this
 *  multi-rail approach.
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef UCX_MR_AUX_H
#define UCX_MR_AUX_H

#include <cuda_runtime.h>
#include <cuda.h>

#include <ucp/api/ucp.h>

#include <string.h>

#include <sys/times.h>
#include <unistd.h>

#include "ucx_multirail.h"
#include "ucx_mr_types.h"

/** @brief Callback function for flushing workers.
 *
 *  (Adapted from ucx perftest).
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @return void
 */
static void
ucp_mr_worker_flush_callback(void *request, ucs_status_t status, void *user_data)
{
  ucp_mr_flush_context_t *ctx = (ucp_mr_flush_context_t *) user_data;

  --ctx->num_outstanding;
  if (status != UCS_OK)
  {
    ERROR_PRINT("worker flush callback got error status: %d", status);
    ctx->status = status;
  }
  ucp_request_free(request);
}

/** @brief Flush all outstanding communications on all existing workers.
 *
 *  This routine flushes all outstanding AMO and RMA communications on the worker. 
 *  All the AMO and RMA operations issued on the worker prior to this call are completed 
 *  both at the origin and at the target when this call returns. Used to wireup created
 *  endpoints. (Adapted from ucx perftest)
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @return ucs_status_t possible error status
 */
static ucs_status_t
ucx_mr_flush_workers(ucx_mr_context_t *mr_ctx)
{
  ucp_mr_flush_context_t ctx = {
      .num_outstanding = 0,
      .status = UCS_OK};
  ucp_request_param_t param;
  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  param.cb.send = ucp_mr_worker_flush_callback;
  param.user_data = &ctx;
  void *flush_req;
  unsigned i;

  for (i = 0; i < mr_ctx->worker_count; ++i)
  {
    flush_req = ucp_worker_flush_nbx(mr_ctx->worker[i], &param);
    if (UCS_PTR_IS_ERR(flush_req))
    {
      ctx.status = UCS_PTR_STATUS(flush_req);
      ERROR_PRINT("ucp_worker_flush_nbx failed on rail %d with status: %d", i, ctx.status);
    }

    if (UCS_PTR_IS_PTR(flush_req))
    {
      ++ctx.num_outstanding;
    }
  }

  /* Progress all workers in parallel to avoid deadlocks */
  while (ctx.num_outstanding > 0)
  {
    ucx_mr_workers_progress(mr_ctx);
  }

  return ctx.status;
}

/** @brief Progress all communication operations on specific worker.
 *
 *  Progress outstanding communication operations on worker with index idx when waiting for
 *  communications requests.
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @param idx index of worker which should be processed
 *  @return void
 */
static void
ucx_mr_worker_progress(ucx_mr_context_t *mr_ctx, int idx)
{
  ucp_worker_progress(mr_ctx->worker[idx]);
}

/** @brief Function for initialization of a request memory.
 *
 *  This function will be called only on the very first time a request memory is initialized, 
 *  and may not be called again if a request is reused. If a request should be reset before 
 *  the next reuse, it can be done before calling ucp_request_free.
 *
 *  @param request pointer to a request object
 *  @return void
 */
static void
request_init(void *request)
{
  struct ucx_context *contex = (struct ucx_context *)request;

  contex->completed = 0;
}

#endif // UCX_MR_AUX_H
