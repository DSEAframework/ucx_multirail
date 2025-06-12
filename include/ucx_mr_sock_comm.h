/** @file ucx_mr_sock_comm.h
 *  @brief Low level functions for socket communication.
 *
 *  Collection of low level functions for setting up communication with sockets. Mainly
 *  adapted from ucx perftest.
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef UCX_MR_SOCK_COMM_H
#define UCX_MR_SOCK_COMM_H

#include <ucp/api/ucp.h>

#include "ucx_mr_types.h"
#include "ucx_mr_sock_rte.h"

/** @brief Exchange status via socket communication 
 *
 *  Exchange ucs_status during wireup. (Adapted from ucx_perftest)
 *
 *  @param mr_ctx multi-rail context
 *  @param status ucs status to be exchanged
 *  @return ucs_status_t possible error status
 */
static ucs_status_t
ucx_mr_exchange_status(ucx_mr_context_t *mr_ctx, ucs_status_t status)
{
  unsigned group_size = mr_ctx->sock_rte_group.size;
  unsigned group_index = mr_ctx->sock_rte_group.is_server ? 0 : 1;
  unsigned peer_index = rte_peer_index(group_size, group_index);
  ucs_status_t collective_status = status;
  struct iovec vec;
  void *req = NULL;

  vec.iov_base = &status;
  vec.iov_len = sizeof(status);

  sock_rte_post_vec(&mr_ctx->sock_rte_group, &vec, 1, &req);

  sock_rte_recv(&mr_ctx->sock_rte_group, peer_index, &status, sizeof(status), req);

  if (status != UCS_OK)
  {
    collective_status = status;
  }

  return collective_status;
}

/** @brief Receive data from peer for endpoint creation
 *
 *  (Adapted from ucx perftest)
 *
 *  @param mr_ctx multi-rail context
 *  @param peer_index index of peer in rte socket group
 *  @param idx rail index
 *  @return ucs_status_t possible error status
 */
static ucs_status_t
ucx_mr_recv_remote_data(ucx_mr_context_t *mr_ctx, unsigned peer_index, int idx)
{
  void *req = NULL;
  ucp_ep_params_t ep_params;
  ucp_address_t *address;
  ucs_status_t status;
  size_t buffer_size;
  void *buffer = 0;

  buffer_size = ADDR_BUF_SIZE;

  buffer = malloc(buffer_size);
  if (buffer == NULL)
  {
    ERROR_PRINT("failed to allocate RTE receive buffer");
    status = UCS_ERR_NO_MEMORY;
    return status;
  }

  /* Initialize all endpoints and rkeys to NULL to handle error flow */
  mr_ctx->ep[idx] = NULL;

  /* receive the data from the remote peer, extract the address from it
   * (along with additional wireup info) and create an endpoint to the peer */
  sock_rte_recv(&mr_ctx->sock_rte_group, peer_index, buffer, buffer_size, req);

  address = (ucp_address_t *)buffer;

  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address = address;

  status = ucp_ep_create(mr_ctx->worker[idx], &ep_params, &mr_ctx->ep[idx]);

  ucp_ep_attr_t ep_attrs;

  ep_attrs.field_mask = UCP_EP_ATTR_FIELD_TRANSPORTS;
  ep_attrs.transports.entries = (ucp_transport_entry_t *)
      malloc(10 * sizeof(ucp_transport_entry_t));
  ep_attrs.transports.num_entries = 10;
  ep_attrs.transports.entry_size = sizeof(ucp_transport_entry_t);

  status = ucp_ep_query(mr_ctx->ep[idx], &ep_attrs);

  if (status != UCS_OK)
  {
    free(buffer);
    return status;
  }
  return status;
}

/** @brief Send local data to peer for endpoint creation
 *
 *  (Adapted from ucx perftest)
 *
 *  @param mr_ctx multi-rail context
 *  @param idx rail index
 *  @return ucs_status_t possible error status
 */
static ucs_status_t
ucx_mr_send_local_data(ucx_mr_context_t *mr_ctx, int idx)
{
  size_t address_length = 0;
  void *req = NULL;
  ucp_address_t *address;
  ucs_status_t status;
  struct iovec *vec;

  vec = (struct iovec *)calloc(1, sizeof(struct iovec));
  if (vec == NULL)
  {
    ERROR_PRINT("failed to allocate iovec");
    status = UCS_ERR_NO_MEMORY;
    return status;
  }

  status = ucp_worker_get_address(mr_ctx->worker[idx], &address, &address_length);
  if (status != UCS_OK)
  {
    ucp_worker_destroy(mr_ctx->worker[idx]);
    free(vec);
    return status;
  }

  vec->iov_base = address;
  vec->iov_len = address_length;

  sock_rte_post_vec(&mr_ctx->sock_rte_group, vec, 1, &req);

  free(vec);

  return UCS_OK;
}

/** @brief Close low level socket communication
 *
 *  Close open channel for socket communication at the end of the program (Adapted from ucx perftest)
 *
 *  @param mr_ctx multi-rail context
 *  @return ucs_status_t possible error status
 */
static ucs_status_t
ucx_mr_cleanup_comm(ucx_mr_context_t *mr_ctx)
{
  sock_rte_group_t *rte_group = &mr_ctx->sock_rte_group;

  close(rte_group->sendfd);

  if (rte_group->sendfd != rte_group->recvfd)
  {
    close(rte_group->recvfd);
  }

  return UCS_OK;
}

/** @brief Establish socket communication
 *
 *  Setup communication via sockets before ucx communication can be established (Adapted from ucx perftest)
 *
 *  @param mr_ctx multi-rail context
 *  @return ucs_status_t possible error status
 */
static ucs_status_t
ucx_mr_connect_comm(ucx_mr_context_t *mr_ctx)
{
  int optval = 1;
  int sockfd = -1;
  struct sockaddr_storage client_addr;
  socklen_t client_addr_len;
  int connfd = -1;
  struct addrinfo hints, *res, *t;
  ucs_status_t status;
  int ret;
  char service[8];
  char err_str[64];

  snprintf(service, sizeof(service), "%u", mr_ctx->port);
  memset(&hints, 0, sizeof(hints));
  hints.ai_flags = (mr_ctx->server_addr == NULL) ? AI_PASSIVE : 0;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  ret = getaddrinfo(mr_ctx->server_addr, service, &hints, &res);
  if (ret < 0)
  {
    ERROR_PRINT("getaddrinfo(server:%s, port:%s) error: [%s]",
                mr_ctx->server_addr, service, gai_strerror(ret));
    status = UCS_ERR_IO_ERROR;
    return status;
  }

  if (res == NULL)
  {
    snprintf(err_str, 64, "getaddrinfo() returned empty list");
  }

  for (t = res; t != NULL; t = t->ai_next)
  {
    sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
    if (sockfd < 0)
    {
      snprintf(err_str, 64, "socket() failed: %m");
      continue;
    }

    if (mr_ctx->server_addr != NULL)
    {
      if (connect(sockfd, t->ai_addr, t->ai_addrlen) == 0)
      {
        break;
      }
      snprintf(err_str, 64, "connect() failed: %m");
    }
    else
    {
      status = (ucs_status_t) setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR,
                          &optval, sizeof(optval));
      if (status != UCS_OK)
      {
        status = UCS_ERR_IO_ERROR;
        close(sockfd);
        freeaddrinfo(res);
        return status;
      }

      if (bind(sockfd, t->ai_addr, t->ai_addrlen) == 0)
      {
        ret = listen(sockfd, 10);
        if (ret < 0)
        {
          ERROR_PRINT("listen() failed: %m");
          status = UCS_ERR_IO_ERROR;
          close(sockfd);
          freeaddrinfo(res);
          return status;
        }

        printf("Waiting for connection...\n");

        /* Accept next connection */
        client_addr_len = sizeof(client_addr);
        connfd = accept(sockfd, (struct sockaddr *)&client_addr,
                        &client_addr_len);
        if (connfd < 0)
        {
          ERROR_PRINT("accept() failed: %m");
          status = UCS_ERR_IO_ERROR;
          close(sockfd);
          freeaddrinfo(res);
          return status;
        }

        printf("Accepted connection\n");
        close(sockfd);
        break;
      }
      snprintf(err_str, 64, "bind() failed: %m");
    }
    close(sockfd);
    sockfd = -1;
  }

  if (sockfd < 0)
  {
    ERROR_PRINT("%s failed. %s",
                (mr_ctx->server_addr != NULL) ? "client" : "server", err_str);
    status = UCS_ERR_IO_ERROR;
    freeaddrinfo(res);
    return status;
  }

  if (mr_ctx->server_addr == NULL)
  {
    /*Receiver Side*/
    if (ret)
    {
      status = UCS_ERR_IO_ERROR;
      close(connfd);
      freeaddrinfo(res);
      return status;
    }

    mr_ctx->sock_rte_group.sendfd = connfd;
    mr_ctx->sock_rte_group.recvfd = connfd;
    mr_ctx->sock_rte_group.peer = 1;
    mr_ctx->sock_rte_group.is_server = 1;
    mr_ctx->role = RECEIVER;
  }
  else
  {
    /*Sender Side*/
    mr_ctx->sock_rte_group.sendfd = sockfd;
    mr_ctx->sock_rte_group.recvfd = sockfd;
    mr_ctx->sock_rte_group.peer = 0;
    mr_ctx->sock_rte_group.is_server = 0;
    mr_ctx->role = SENDER;
  }

  mr_ctx->sock_rte_group.size = 2;

  status = UCS_OK;
  freeaddrinfo(res);
  return status;
}

#endif // UCX_MR_SOCK_COMM_H