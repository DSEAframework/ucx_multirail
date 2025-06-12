/** @file ucx_mr_comm_aux.h
 *  @brief Auxilliary functions for multi railed communication
 *
 *  Calculation of message sizes for data splitting and creation of send/recv parameter
 *
 *  @author Lukas Ramsperger
 *  @bug No known bugs.
 */
#ifndef UCX_MR_COMM_AUX_H
#define UCX_MR_COMM_AUX_H

#include <math.h>


/** @brief Calculated message sizes for sending with 4 rails pipelined.
 *
 *  The message size for all 4 rails is calculated from the full message size. A split ratio is given which
 *  says which percentage of the full message size is distributed on to rail 1-3. The calculation is done such that
 *  all messages sizes are dividable by the element size so no value is cut in between. The message sizes for the rails
 *  1-3 are also dividable by the pipeline stages. This is done that the pipelined communication is done more easily.
 *
 *  @param msg_size full message size in bytes which needs to be split onto 4 rails
 *  @param split_ratio percentage how much of the message should be distributed onto rail 1-3 (0 < r < 1)
 *  @param element_size size of elements in bytes which are transferred
 *  @param msg_sizes  (output) array where the message size of all 4 rails are stored in bytes
 *  @param pipeline_stages number of pipeline stages on rail 1-3
 *  @return void
 */
void calculate_msg_sizes_quad_pipeline(size_t msg_size, float split_ratio, unsigned int element_size, size_t *msg_sizes, unsigned int pipeline_stages)
{
  size_t nof_e;
  size_t nof_e_split;
  size_t nof_e_stay;

  split_ratio = 1 - split_ratio;
  
  // Calculate message sizes as number of elements so no value is cut in between
  nof_e = msg_size / element_size;
  nof_e_split = (size_t)roundf(nof_e * (1 - split_ratio));

  // Number of elements which are transferred to rail 1-3 need to be dividable by 3 and number of pipelinestages
  nof_e_split /= (pipeline_stages * 3);
  nof_e_split *= (pipeline_stages * 3);

  DEBUG_PRINT("Full message size: %ld Bytes, nof_e: %ld\n", msg_size, nof_e);
  DEBUG_PRINT("Nof elements per peer: %ld \n", nof_e_split / 3);

  nof_e_stay = nof_e - nof_e_split;

  msg_sizes[0] = nof_e_stay * element_size;
  msg_sizes[1] = nof_e_split / 3 * element_size;
  msg_sizes[2] = nof_e_split / 3 * element_size;
  msg_sizes[3] = nof_e_split / 3 * element_size;

  DEBUG_PRINT(
      "Calculated: \n msg_sizes0: %ld Bytes, \n msg_sizes1: %ld Bytes, \n msg_sizes2: %ld Bytes, \n msg_sizes3: %ld Bytes, \n Sum: %ld Bytes\n",
      msg_sizes[0], msg_sizes[1], msg_sizes[2], msg_sizes[3], msg_sizes[0] + msg_sizes[1] + msg_sizes[2] + msg_sizes[3]);
}

void calculate_msg_sizes_quad(size_t msg_size, float split_ratio, unsigned int element_size, size_t *msg_sizes)
{
  size_t nof_e, nof_e_stay, nof_e_split, nof_e_rest;

  nof_e = (size_t)roundf((msg_size / element_size));

  DEBUG_PRINT("Full message size: %ld Bytes, nof_e: %ld\n", msg_size, nof_e);

  nof_e_stay = (size_t)roundf((msg_size / element_size) * (1 - split_ratio));
  nof_e_split = (nof_e - nof_e_stay) / 3;
  nof_e_rest = (nof_e - nof_e_stay) % 3;

  msg_sizes[0] = nof_e_stay * element_size;
  msg_sizes[1] = nof_e_split * element_size;
  msg_sizes[2] = nof_e_split * element_size;
  msg_sizes[3] = (nof_e_split + nof_e_rest) * element_size;

  DEBUG_PRINT(
      "Calculated: \n msg_sizes0: %ld Bytes, \n msg_sizes1: %ld Bytes, \n msg_sizes2: %ld Bytes, \n msg_sizes3: %ld Bytes, \n Sum: %ld Bytes\n",
      msg_sizes[0], msg_sizes[1], msg_sizes[2], msg_sizes[3], msg_sizes[0] + msg_sizes[1] + msg_sizes[2] + msg_sizes[3]);
}

void calculate_msg_sizes_dual(size_t msg_size, float split_ratio, unsigned int element_size, size_t *msg_sizes)
{

  size_t tmp;
  size_t nof_e;

  nof_e = (size_t)roundf((msg_size / element_size) * (1 - split_ratio));

  DEBUG_PRINT("Full message size: %ld Bytes, nof_e: %ld\n", msg_size, nof_e);

  tmp = nof_e * element_size;

  msg_sizes[1] = msg_size - tmp;
  msg_sizes[0] = tmp;

  DEBUG_PRINT("Calculated msg_size0: %ld Bytes, msg_size_1: %ld Bytes, Sum: %ld Bytes\n", msg_sizes[0], msg_sizes[1], msg_sizes[0] + msg_sizes[1]);
}


/** @brief Calculated message sizes for sending with 2 rails pipelined.
 *
 *  The message size for both rails is calculated from the full message size. A split ratio is given which
 *  says which percentage of the full message size is sent to rail 1. The calculation is done such that
 *  all messages sizes are dividable by the element size so no value is cut in between. The message sizes for the rail
 *  1 is also dividable by the pipeline stages. This is done that the pipelined communication is done more easily.
 *
 *  @param msg_size full message size in bytes which needs to be split onto 2 rails
 *  @param split_ratio percentage how much of the message should be distributed onto rail 1 (0 < r < 1)
 *  @param element_size size of elements in bytes which are transferred
 *  @param msg_sizes  (output) array where the message size of both rails are stored in bytes
 *  @param pipeline_stages number of pipeline stages on rail 1
 *  @return void
 */
void calculate_msg_sizes_dual_pipeline(size_t msg_size, float split_ratio, unsigned int element_size, size_t *msg_sizes, unsigned int pipeline_stages)
{

  size_t tmp;
  size_t nof_e;
  size_t nof_e_split;

  nof_e = msg_size / element_size;
  nof_e_split = (size_t)roundf(nof_e * (1 - split_ratio));

  // Message size on rail 1 needs to be dividable by the pipeline stages
  nof_e_split /= pipeline_stages;
  nof_e_split *= pipeline_stages;

  DEBUG_PRINT("Full message size: %ld Bytes, nof_e: %ld\n", msg_size, nof_e);

  tmp = nof_e_split * element_size;

  msg_sizes[1] = tmp;
  msg_sizes[0] = msg_size - tmp;

  DEBUG_PRINT("Calculated msg_size0: %ld Bytes, msg_size_1: %ld Bytes, Sum: %ld Bytes\n", msg_sizes[0], msg_sizes[1], msg_sizes[0] + msg_sizes[1]);
}


/** @brief Callback function when receiving messages
 *
 *  (Adapted from ucx perftest)
 *
 *  @param request request which is set to completed when function is called
 *  @param status
 *  @param info
 *  @param user_data
 *  @return void
 */
void recv_handler(void *request, ucs_status_t status,
                  const ucp_tag_recv_info_t *info, void *user_data)
{
  struct ucx_context *context = (struct ucx_context *)request;

  context->completed = 1;
}


/** @brief Prepare receive parameter for ucx communication
 *
 *  @param recv_param (output) data structure to be written with receive parameters
 *  @param mem_type memory type of receiving buffer (Cuda/CPU)
 *  @return void
 */
void ucx_mr_write_recv_param(ucp_request_param_t *recv_param, ucs_memory_type_t mem_type)
{
  recv_param->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                             UCP_OP_ATTR_FIELD_DATATYPE |
                             UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                             UCP_OP_ATTR_FLAG_NO_IMM_CMPL;

  recv_param->datatype = ucp_dt_make_contig(1);
  recv_param->memory_type = mem_type;
  recv_param->cb.recv = recv_handler;
}


/** @brief Callback function when sending messages
 *
 *  (Adapted from ucx perftest)
 *
 *  @param request request which is set to completed when function is called
 *  @param status
 *  @param ctx
 *  @return void
 */
void send_handler(void *request, ucs_status_t status, void *ctx)
{
  struct ucx_context *context = (struct ucx_context *)request;
  const char *str = (const char *)ctx;
  context->completed = 1;
  DEBUG_PRINT("send handler called for \"%s\" with status %d\n",
              str, status);
}


/** @brief Prepare send parameter for ucx communication
 *
 *  @param send_param (output) data structure to be written with send parameters
 *  @param mem_type memory type of sending buffer (Cuda/CPU)
 *  @return void
 */
void ucx_mr_write_send_param(ucp_request_param_t *send_param, ucs_memory_type_t mem_type)
{
  send_param->op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                             UCP_OP_ATTR_FIELD_USER_DATA |
                             UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  send_param->cb.send = send_handler;
  send_param->user_data = (void *)data_msg_str;
  send_param->memory_type = mem_type;
}

#endif // UCX_MR_COMM_AUX_H
