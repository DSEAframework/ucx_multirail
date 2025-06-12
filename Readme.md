# UCX_MULTIRAIL - Proof of Concept
## Introduction
Proof of concept for multi railed communication leveraging UCX.

## Implementation
A multi railed communication is implemented leveraging UCX as communication API. A message in a single buffer is distributed across multiple buffers on GPUs in a single node via `cudaMemcpy()`. From each buffer a UCX communication is initiated in parallel. On receiving side the message is received into multiple buffers across the GPUs on the receiving node. The message is then collected via `cudaMemcpy()` into a single buffer.

## Getting Started
### Prerequisites
The following tools are required to build dsea
* cmake 3.21 or newer
* CUDA version 12.0 or newer
* UCX 1.17


### Installation
```shell
mkdir build
cd build
cmake ..
make
```


### Usage
`run_basic_` used for general testing of multi railed communication. One test message is sent from sender to receiver. Messages is printed to be checked.
`run_bench_` used to perform a single benchmark.
`run_survey_` used to perform a broad benchmark study across a wide range of message sizes, pipeline stages and split ratios.

For usage adjust in sender script adress of receiver. 
For benchmarks following configurations are possible:
`-T` select test (TEST, SPLIT, PROFILE, MR, SINGLE)
`-n` select number of communications rails
`-k` select number of pipeline stages
`-r` select split ratio in percent

Survey studies:
- SINGLE: Measures for each communication rail bandwidth seperately
- MR: Measures bandwidth of synchronous send instructions across multiple communication rails
- SPLIT: Takes single message distributes across the communication rails and then sends the message and collects it on receiving end.



## Citation
Please cite this work as:

"M. Rose, S. Homes, L. Ramsperger, J. Gracia, C. Niethammer, and J. Vrabec. Cyclic Data Streaming on GPUs for Short Range
Stencils Applied to Molecular Dynamics. Submitted to Euro-Par 2025."
