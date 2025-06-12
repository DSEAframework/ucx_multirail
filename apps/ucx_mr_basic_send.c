/** @file ucx_mr_basic_send.h
 *  @brief Example file for setting up ucx_multirail with one testing message (sender side)
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#include "ucx_multirail.h"

int main(int argc, char **argv)
{
  cudaSetDevice(0);

  ucs_status_t status;

  ucx_mr_context_t mr_ctx;
  mr_ctx.server_addr = NULL;

  parse_opts(&mr_ctx, argc, argv);

  status = ucx_mr_setup(&mr_ctx);
  if (status != UCS_OK)
  {
    printf("There was a problem!\n");
  }

  ucx_mr_test_connection(&mr_ctx);

  ucx_mr_cleanup(&mr_ctx, FULL);
}