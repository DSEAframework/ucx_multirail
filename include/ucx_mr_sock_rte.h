/** @file ucx_mr_sock_rte.h
 *  @brief Helper function for low level socket communication.
 *
 *  Collection of low level functions for setting up communication with sockets. Mainly
 *  adapted from ucx perftest.
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef UCX_MR_SOCK_RTE_H
#define UCX_MR_SOCK_RTE_H

#include <errno.h>

#include <assert.h>
#include <sys/uio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/poll.h>

unsigned
rte_peer_index(unsigned group_size, unsigned group_index)
{
  unsigned peer_index = group_size - 1 - group_index;

  assert(group_index < group_size);
  return peer_index;
}

static int
sock_io(int sock, ssize_t (*sock_call)(int, void *, size_t, int),
        int poll_events, void *data, size_t size,
        void (*progress)(void *arg), void *arg, const char *name)
{
  size_t total = 0;
  struct pollfd pfd;
  int ret;

  while (total < size)
  {
    pfd.fd = sock;
    pfd.events = poll_events;
    pfd.revents = 0;

    ret = poll(&pfd, 1, 1); /* poll for 1ms */
    if (ret > 0)
    {
      assert(ret == 1);
      assert(pfd.revents & poll_events);

      ret = sock_call(sock, (char *)data + total, size - total, 0);
      if (ret < 0)
      {
        ERROR_PRINT("%s() failed: %m", name);
        return -1;
      }
      total += ret;
    }
    else if ((ret < 0) && (errno != EINTR))
    {
      ERROR_PRINT("poll(fd=%d) failed: %m", sock);
      return -1;
    }

    /* progress user context */
    if (progress != NULL)
    {
      progress(arg);
    }
  }
  return 0;
}

static int
safe_send(int sock, void *data, size_t size, void (*progress)(void *arg), void *arg)
{
  typedef ssize_t (*sock_call)(int, void *, size_t, int);

  assert(sock >= 0);
  return sock_io(sock, (sock_call)send, POLLOUT, data, size, progress, arg, "send");
}

static int
safe_recv(int sock, void *data, size_t size, void (*progress)(void *arg), void *arg)
{
  typedef ssize_t (*sock_call)(int, void *, size_t, int);

  assert(sock >= 0);
  return sock_io(sock, (sock_call)recv, POLLIN, data, size, progress, arg, "recv");
}

static void
sock_rte_post_vec(sock_rte_group_t *rte_group, const struct iovec *iovec, int iovcnt, void **req)
{
  size_t size;
  int i;

  size = 0;
  for (i = 0; i < iovcnt; ++i)
  {
    size += iovec[i].iov_len;
  }

  safe_send(rte_group->sendfd, &size, sizeof(size), NULL, NULL);
  for (i = 0; i < iovcnt; ++i)
  {
    safe_send(rte_group->sendfd, iovec[i].iov_base, iovec[i].iov_len, NULL,
              NULL);
  }
}

static void
sock_rte_recv(sock_rte_group_t *rte_group, unsigned src, void *buffer, size_t max, void *req)
{
  size_t size;

  if (src != rte_group->peer)
  {
    return;
  }

  safe_recv(rte_group->recvfd, &size, sizeof(size), NULL, NULL);
  assert(size <= max);
  safe_recv(rte_group->recvfd, buffer, size, NULL, NULL);
}

#endif // UCX_MR_SOCK_RTE_H