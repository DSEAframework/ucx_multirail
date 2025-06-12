/** @file ucx_mr_types.h
 *  @brief Collection of data objects to organize multi-railed ucx.
 *
 *  Data structs and enums which help to organize multi-rail ucx. As central instance there is the ucx multi rail
 *  context, which collects all ucp objects and further information like Cuda Streams for async communication.
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 *  @todo Use only single cuda stream throughout the project, replace arrays in contex with pointers, so number of rails
 *      can be changed dynamically when setting up the context
 */
#ifndef UCX_MR_TYPES_H
#define UCX_MR_TYPES_H

#include <ucp/api/ucp.h>

#define ADDR_BUF_SIZE 4096

#define NOF_RAILS 4  /* Maximum number usable rails */
#define MAX_PIPELINE_STAGES 8  /* Maximum number of pipeline stages per rail */
static const ucp_tag_t tag_mask = 0x00000000ffffffff;   /* 32 bit for sending additional information with messages */
static const char *data_msg_str = "UCX data message";

typedef struct
{
  int num_outstanding; /* Number of outstanding flush operations */
  ucs_status_t status; /* Cumulative status of all flush operations */
} ucp_mr_flush_context_t;

struct ucx_context
{
  int completed;
};

typedef enum CommRole
{
  SENDER,
  RECEIVER
} CommRole;

/** @brief Enum for jumping to the right position in the clean up routine
 *
 */
typedef enum CleanUp
{
  FULL,
  CTX,
  COMM,
  WORKER,
  EP,
  STREAMS,
  EVENTS
} CleanUp;

/** @brief Communication group for socket communication
 *
 */
typedef struct sock_rte_group
{
  int sendfd;
  int recvfd;
  int is_server;
  int size;
  int peer;
} sock_rte_group_t;

/** @brief UCX multi rail context
 *
 *  Data structure for organizing all ucp objects and further auxillary for the multi railed communciation.
 *  For each rail a seperated ucp_context, ucp_worker and ucp_endpoint is created and stored. Cuda Streams
 *  and events are stored for asynchronous communication when splitting and sending data to the other rails.
 *
 */
typedef struct ucx_mr_context
{
  // ucx objects
  ucp_context_h ctx[NOF_RAILS];
  ucp_worker_h worker[NOF_RAILS];
  ucp_ep_h ep[NOF_RAILS];

  // Auxiliary
  int ctx_count;
  int worker_count;
  int ep_count;
  CommRole role;

  const char *rails[NOF_RAILS];

  // socket communictation
  const char *server_addr;
  sock_rte_group_t sock_rte_group;
  int port;

  // Cuda Streams
  cudaStream_t cuda_streams[NOF_RAILS - 1];
  cudaEvent_t cuda_events[(NOF_RAILS - 1) * MAX_PIPELINE_STAGES];

  cudaStream_t cuda_stream;

} ucx_mr_context_t;

#endif // UCX_MR_TYPES_H
