

## The Official Implementation of **XQueryer: An Intelligent Crystal Structure Identifier for Powder X-ray Diffraction**

## [Paper](https://doi.org/10.1093/nsr/nwaf421) | [Website](https://example.com) | [YouTube](https://www.youtube.com/watch?v=OYPoh7K5uM0) | [Benchmarks](https://github.com/WPEM/XqueryerBench)

Our system revolutionizes PXRD-based crystal identification through high-fidelity data synthesis and the cutting-edge **XQueryer** model. Seamlessly integrated with diffractometers, it enables precise, AI-driven material discovery and extends its capabilities to broader chemical applications. **XQueryer** comprises **1.03 B** parameters.

## Overview
- **Source Code**: Available in the [./src](./src) directory.

- **Dataset**: [OneDrive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy81ZDg2MjYyMzg0NzBiNDllL0V1d09VMTNQM2JoSHNiU2lEMTRON3hZQmZCTEdCYTFjX0VhVkhrbGZUajRxZXc%5FZT0xa3liaFg&id=5D8626238470B49E%21s5d530eecddcf47b8b1b4a20f5e0def16&cid=5D8626238470B49E)
- **Benchmarks**: Access the benchmark code at repo [XqueryerBench](https://github.com/WPEM/XqueryerBench).
- **Simulation Code**: Available in the [./sim](./sim) directory.
- **RRUFF–MP ID Matching**: Available in the [./match](./match) directory.

## Tutorials
- **Training/Val/Testing**: [model_tutorial](./src/Tutorial.ipynb)
- **Simulation**: [sim_tutorial](./sim/XRD.ipynb)
- **High-throughput simulation**: [HTsim_tutorial](./sim/tutorial_sim.ipynb)
## About
Maintained by project contributors. Please use GitHub issues for questions.

---

## Multi-phase adaptation summary

This baseline is adapted to the XDecomposer multi-phase separation task while keeping the runtime interface unchanged.

### Unchanged input / output contract
- Input `intensity`: `[B, 3500]`
- Input `element`: `[B, 92]`
- Output `xrds`: `[B, S, 3500]`
- Output `ratios`: `[B, S]`
- Output `features`: `[B, S, feature_dim]`
- Output `feat_logits`: `[B, S, num_classes]`
