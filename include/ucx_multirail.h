#ifndef UCX_MULTIRAIL_H
#define UCX_MULTIRAIL_H

#if DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stdout, "DEBUG: " fmt, ##args)
#else
#define DEBUG_PRINT(fmt, args...) /* Don't do anything in release builds */
#endif

#define ERROR_PRINT(fmt, args...) fprintf(stdout, "ERROR: " fmt, ##args)
#define WARN_PRINT(fmt, args...) fprintf(stdout, "WARNING: " fmt, ##args)

#include <cuda_runtime.h>
#include <cuda.h>
#include <driver_types.h>

#include <ucp/api/ucp.h>

#include "ucx_mr_types.h"
#include "ucx_mr_wait.h"
#include "ucx_mr_aux.h"
#include "ucx_mr_cleanup.h"
#include "ucx_mr_comm.h"
#include "ucx_mr_setup.h"
#include "ucx_mr_sock_comm.h"

#endif // UCX_MULTIRAIL_H