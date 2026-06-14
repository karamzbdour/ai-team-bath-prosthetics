# Prosthetic Sensor Autoencoder Benchmark

## Description
This repository serves as a research-oriented benchmarking framework designed to evaluate the efficacy of autoencoder architectures when applied to prosthetic sensor data. The project provides a structured pipeline for the replication of established research results and the systematic development of novel neural network architectures for prosthetic feedback systems.

## Architectural Objectives
The project is designed to address the following research requirements:
*   **Reproducibility:** Facilitating deterministic training pipelines through fixed seed management and configuration-driven experimentation.
*   **Modularity:** Decoupling the data processing pipeline from model architecture definitions to allow for rapid iterative testing.
*   **Standardisation:** Providing a uniform evaluation methodology to compare performance across diverse autoencoder topologies.

## Repository Structure
```text
prosthetics-autoencoder-eval/
├── data/               # Raw sensor data and processed feature sets
├── conf/               # Hydra-based configuration files for hyperparameters
├── src/                # Core implementation
│   ├── models/         # Implementation of autoencoder architectures
│   ├── pipeline/       # Training, validation, and evaluation routines
│   └── utils/          # Preprocessing and diagnostic utilities
├── notebooks/          # Exploratory analysis and statistical validation
├── tests/              # Unit tests for data pipeline and model shapes
└── pyproject.toml      # Dependency management and project metadata
