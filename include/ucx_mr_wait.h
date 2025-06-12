/** @file ucx_mr_wait.h
 *  @brief Collection of functions for checking for communication
 *
 *  ***
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef UCX_MR_WAIT_H
#define UCX_MR_WAIT_H

#include <ucp/api/ucp.h>

/** @brief Progress all communication on all existing worker.
 *
 *  @param mr_ctx multi-rail contex with all enviroment data
 *  @return void
 */
static void 
ucx_mr_workers_progress(ucx_mr_context_t *mr_ctx) {
  for (int i = 0; i < mr_ctx->worker_count; ++i) {
    ucp_worker_progress(mr_ctx->worker[i]);
  }
}


/** @brief Checking once if communication on worker is completed
 *
 *  @param ucp_worker worker to be processed
 *  @param request request of checked message
 *  @return status ucs_status with possible error code
 */
static ucs_status_t 
ucx_poll(ucp_worker_h ucp_worker, struct ucx_context *request)
{
  ucs_status_t status;

  if (UCS_PTR_IS_PTR(request)) {

    ucp_worker_progress(ucp_worker);

    if (request->completed) {
      request->completed = 0;
      status = ucp_request_check_status(request);
      ucp_request_free(request);
      return status;
    } else {
      return UCS_INPROGRESS; 
    }
  }

  return UCS_INPROGRESS;
}


/** @brief Wait till communication on worker is completed.
 *
 *  Blocking routine till communication on worker is completed. In between communication on worker needs
 *  to be progressed for further updates
 *
 *  @param ucp_worker worker to be processed
 *  @param request request of checked message
 *  @param op_str string for better debug output
 *  @param data_str string for better debug output
 *  @return status ucs_status with possible error code
 */
static ucs_status_t 
ucx_wait(ucp_worker_h ucp_worker, struct ucx_context *request, const char *op_str, const char *data_str)
{
  ucs_status_t status;
  if (UCS_PTR_IS_ERR(request))
  {
    status = UCS_PTR_STATUS(request);
  }
  else if (UCS_PTR_IS_PTR(request))
  {
    while (!request->completed)
    {
      ucp_worker_progress(ucp_worker);
    }

    request->completed = 0;
    status = ucp_request_check_status(request);
    ucp_request_free(request);
  }
  else
  {
    status = UCS_OK;
  }

  if (status != UCS_OK)
  {
    ERROR_PRINT("unable to %s %s (%s)\n", op_str, data_str,
            ucs_status_string(status));
  }
  else
  {
    DEBUG_PRINT("finish to %s %s\n", op_str, data_str);
  }

  return status;
}

#endif // UCX_MR_WAIT_H