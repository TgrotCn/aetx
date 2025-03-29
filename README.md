# aetx

**aetx** is an advanced inference framework designed to evolve from a simple engine into a mainstream, production-grade inference solution. Currently, the framework supports the Qwen2-1.5B model and is actively being enhanced with features such as quantization, multi-card (multi-GPU) inference, and a host of performance optimizations.

---

## Overview

Infersys aims to provide a modular, scalable, and high-performance inference engine for AI practitioners and developers. With a clean and extensible architecture, it facilitates rapid integration of new models and optimizations, enabling state-of-the-art inference capabilities in real-world applications.

---

## Key Features

- **Model Support**: 
  - Out-of-the-box support for the Qwen2-1.5B model.
- **Modular Architecture**: 
  - Easily extendable design for incorporating additional models and functionalities.
- **Quantization**:
  - Planned support to reduce memory footprint and boost inference throughput.
- **Multi-Card Inference**:
  - Future capability for leveraging multiple GPUs for high-scale inference.
- **Performance Optimization**:
  - Engineered for low latency and high throughput, with advanced optimizations.
- **Cross-Platform Compatibility**:
  - Designed to work on multiple operating systems and hardware configurations.

---

Latest benchmarks with Mistral-7B-Instruct-v0.2 in FP16 with 4k context, on RTX 4090 + EPYC 7702P:

| Engine      | Avg. throughput (~120 tokens) tok/s | Avg. throughput (~4800 tokens) tok/s |
| ----------- | ----------- | ----------- |
| huggingface transformers, GPU | 25.9 | 25.7 |
| llama.cpp, GPU | 61.0 | 58.8 |
| calm, GPU | 66.0 | 65.7 |
| aetx, GPU | 63.8 | 58.7 |

## Getting Started

### Prerequisites

Before you begin, make sure you have the following installed:
- A C++ compiler with support for C++17 or later.
- CUDA Toolkit (for GPU acceleration).
- CMake (version 3.15 or higher).
- [Any additional dependencies, e.g., Boost, OpenMP, etc.]

### Installation

Clone the repository and build the project:

```bash
git clone https://github.com/tgrotcn/aetx.git
cd aetx
mkdir build && cd build
cmake ..
make -j$(nproc)
