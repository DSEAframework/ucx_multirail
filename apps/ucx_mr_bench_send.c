#include <stdio.h>
#include <time.h>
#include <sys/times.h>
#include <unistd.h>

#include "ucx_multirail.h"
#include "mr_benchmark/ucx_mr_bench.h"

char filename[128];
FILE *fptr;

struct tm *timenow;

int main(int argc, char **argv)
{
  cudaSetDevice(0);
  ucs_status_t status;

  ucx_mr_bench_context_t mr_bench_ctx;
  mr_bench_ctx.mr_ctx.port = 13337;

  parse_bench_opts(&mr_bench_ctx, argc, argv);

  status = ucx_mr_bench_setup(&mr_bench_ctx);
  if (status != UCS_OK)
  {
    printf("There was a problem!\n");
  }

  //ucx_mr_test_connection(&mr_bench_ctx.mr_ctx);

  if (mr_bench_ctx.repeated_test == 0)
  {
    float split_ratio = (float) mr_bench_ctx.ratio / 100;
    unsigned pipeline_stages = mr_bench_ctx.stages;
    switch (mr_bench_ctx.test_type)
    {
    case MR:
      ucx_mr_bench_send_mr(&mr_bench_ctx);
      break;
    case SINGLE:
      for (int i = 0; i < NOF_RAILS; ++i)
      {
        ucx_mr_bench_send_single(&mr_bench_ctx, i);
      }
      break;
    case SPLIT:;
      printf("Run split bench with %f split ratio, %d stages and %d rails\n", split_ratio, pipeline_stages, mr_bench_ctx.nof_rails);
      ucx_mr_bench_send_split(&mr_bench_ctx, split_ratio, pipeline_stages);
      break;
    case TEST_SPLIT:
      ucx_mr_bench_test_send_split(&mr_bench_ctx, split_ratio, pipeline_stages);
      break;
    default:
      printf("No Test selected\n");
    }
  }
  else
  {
    // Repeated test with exported data to plot.
    printf("Repeated test selected.\n");
    time_t now = time(NULL);
    timenow = gmtime(&now);

    strftime(filename, sizeof(filename), "./export/export_send_%Y-%m-%d_%H:%M.txt", timenow);

    fptr = fopen(filename, "w");

    fprintf(fptr, "-----Repeated Benchmarks on Sender Side-----\n");
    fprintf(fptr, "%d runs per measurement.\n", MAX_RUNS);

    fprintf(fptr, "\n");
    fprintf(fptr, "+----------------------------------------------------------------------+\n");
    fprintf(fptr, "|   Type  | Memory |       Msg_size         | Bandwidth | Split | Rail/|\n");
    fprintf(fptr, "|         |        |         [MB]           |   [MB/s]  |       | Stage|\n");
    fprintf(fptr, "+---------+--------+------------------------+-----------+-------+------+\n");

    float split_ratio = (float) mr_bench_ctx.ratio / 100;
    unsigned pipeline_stages = mr_bench_ctx.stages;
    unsigned nof_rails = mr_bench_ctx.nof_rails;

    long msg_size = 16384;
    // Set i < 34
    for (int k = 0; k < 10; ++k)
    {
      ucx_mr_free_mem(&mr_bench_ctx);
      mr_bench_ctx.msg_size = msg_size;
      status = ucx_mr_alloc_mem(&mr_bench_ctx);

      double bw = 0;

      float center;

      double ms = (double)msg_size / 1024 / 1024;

      char *mem;
      if (mr_bench_ctx.mem_type == UCS_MEMORY_TYPE_CUDA)
      {
        mem = "CUDA";
      }
      else
      {
        mem = "HOST";
      }

      for (int j = 0; j <= mr_bench_ctx.repetitions; ++j)
      {
        switch (mr_bench_ctx.test_type)
        {
        case MR:
          bw = ucx_mr_bench_send_mr(&mr_bench_ctx);
          if (j != 0)
            fprintf(fptr, "|   MR%d   |  %s  |% 24e|% 11.2lf|   -   |   -  |\n", nof_rails, mem, nof_rails * ms, bw);
          break;
        case SINGLE:
          for (int i = 0; i < NOF_RAILS; ++i) {
            bw = ucx_mr_bench_send_single(&mr_bench_ctx, i);
            if (j != 0)
              fprintf(fptr, "| Single  |  %s  |% 24e|% 11.2lf|   -   |   %d  |\n", mem, ms, bw, i);
          }
          break;
        case SPLIT:;
          center = split_ratio;
          for (int p = 1; p <= MAX_PIPELINE_STAGES; ++p) {
          for (double i = center - 0.1; i <= center + 0.1; i += 0.02)
          //for (double i = center - 0.0; i <= center + 0.0; i += 0.01)
          {
            bw = ucx_mr_bench_send_split(&mr_bench_ctx, i, p);
            if (j != 0)
              fprintf(fptr, "|  Split  |  %s  |% 24e|% 11.2lf| % 3.2lf |  %2d  |\n", mem, ms, bw, i, p);
            fflush(fptr);
          }
          }
          break;
        default:
          printf("No Test selected\n");
        }
        fflush(fptr);
      }
      msg_size *= 2;
    }

    fprintf(fptr, "+---------+--------+------------------------+-----------+-------+------+\n");
    fflush(fptr);
    // Close the file
    fclose(fptr);
  }

  ucx_mr_cleanup(&mr_bench_ctx.mr_ctx, FULL);
  ucx_mr_free_mem(&mr_bench_ctx);
}