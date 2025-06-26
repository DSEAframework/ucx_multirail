# UCX_MULTIRAIL - Proof of Concept

## Overview

**UCX_MULTIRAIL** is a proof-of-concept demonstrating multi-rail communication using the Unified Communication X (UCX) framework across multiple GPUs and nodes. It showcases how a single message can be split and transmitted over multiple communication rails to increase effective bandwidth and overall throughput.

## Concept

Multi-rail communication is achieved by dividing a message buffer into multiple chunks and distributing them across multiple GPUs using `cudaMemcpy()`. Each chunk is transmitted via a distinct UCX endpoint, enabling parallel communication. On the receiving side, chunks are gathered from the respective GPUs and reassembled into the original message buffer.

The project supports:

- Configurable number of communication rails (1, 2, 4)
- Pipelined communication to enhance overlap and throughput
- Benchmarking modes for evaluating bandwidth and scalability

Key parameters:

- **Split ratio**: Defines how the message is divided among communication rails
- **Pipeline stages**: Controls the number of overlapping communication steps

> **Note:** Optimal settings depend on the message size and hardware. A parameter sweep is recommended to identify the best configuration.

---

## Build Instructions

### Prerequisites

Ensure the following dependencies are installed:

- CMake >= 3.21  
- CUDA >= 12.0  
- UCX >= 1.17  

### Build

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Executables

- `run_basic_`: Basic test to validate correctness of multi-rail communication. Sends a single message and prints the result.
- `run_bench_`: Executes a single benchmark with configurable parameters.

### Example CLI Options

Each benchmark supports the following flags:

- `-T`: Select test type  
  - `TEST`: Basic communication test  
  - `SPLIT`: Message split across multiple rails  
  - `PROFILE`: Enables profiling  
  - `MR`: Multi-rail parallel send  
  - `SINGLE`: Single-rail performance

- `-n`: Number of communication rails (e.g., 1, 2, 4)
- `-k`: Number of pipeline stages
- `-r`: Split ratio (percentage of message assigned to each rail)

> **Note:** Update the receiver's address in the sender script before running any test.

## Benchmark Results

Benchmarks were conducted on **Hawk-AI** at the High-Performance Computing Center Stuttgart (HLRS).

| Configuration          | Message Size | Rails | Pipeline Stages | Split Ratio (%) | Observed Bandwidth |
|------------------------|--------------|-------|------------------|------------------|--------------------|
| Baseline (Single-Rail) | 10 MB        | 1     | -                | -                | ~20 GB/s           |
| Two-Rail               | 20 MB        | 2     | 1                | 50               | ~38 GB/s           |
| Four-Rail              | 40 MB        | 4     | 1                | 75               | ~63 GB/s           |

## Citation

If you use this work in academic or scientific contexts, please cite:

> M. Rose, S. Homes, L. Ramsperger, J. Gracia, C. Niethammer, and J. Vrabec.  
> *Cyclic Data Streaming on GPUs for Short Range Stencils Applied to Molecular Dynamics*.  
> Submitted to Euro-Par 2025.
