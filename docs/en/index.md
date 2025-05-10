---
hide:
  - navigation
  - toc
---

# FORCE AI

**FORCE AI** (Fast Optimization for Resource-Constrained and Efficient AI Inference) is an open-source framework designed for educational purposes, demonstrating how to build an optimized AI accelerator for a specific model from scratch. This project covers the full development workflow, including model training and quantization, performance profiling, hardware architecture design, compiler construction, and runtime integration, helping learners understand and implement end-to-end AI inference acceleration.

The framework is designed to be educational and extensible, with a clean and modular structure. It is suitable for course projects, self-learning, or as a foundation for advanced research. All tools and modules are based on open-source technologies, including PyTorch, TVM, Verilator, and ONNX. It also features a custom-designed power-of-2 symmetric quantization algorithm and a Python-based performance model, combining RTL simulation and hardware design space exploration to support a design-driven development approach for AI accelerators.

## :mortar_board: Who is this for?

- Beginners, graduate students, and senior undergraduates in the field of AI accelerators
- Developers interested in bridging AI inference and hardware design
- Learners who want to build their own accelerator environment at low cost

## :wrench: What you will learn

- How to train and quantize neural network models for hardware deployment
- How to design and simulate a CNN accelerator based on the Eyeriss architecture
- How to perform RTL-level simulation and verification using Verilator
- How to build a simple inference runtime and use the TVM compiler for code generation
- How to derive hardware design specifications from front-end model requirements
