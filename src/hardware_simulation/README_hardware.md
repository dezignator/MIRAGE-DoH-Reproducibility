# Hardware Simulation Documentation

This document explains what the MIRAGE-DoH artifact simulates, what it does not, and why. It directly addresses the reviewer's requirement: *"Where hardware or data cannot be shared, the candidate should provide simulation scripts or synthetic substitutes that reproduce the same trends, and should state explicitly which claims rest on non-shareable resources."*

## 1. Overview

Chapter 3 of the thesis reports measurements taken on:
- **Intel Loihi 2** neuromorphic chip (78 µs latency, 0.12 mJ energy).
- **NVIDIA A100** GPU (15.3 M QPS throughput, 2.1 ms latency).

Neither is available on a typical reviewer laptop or in a standard cloud container.

## 2. What We Simulate (and What We Do Not)

### Intel Loihi 2 (`loihi_mock.py`)
- **We DO simulate**: The event-driven latency trend (78 µs) and energy efficiency (0.12 mJ) using a calibrated scaling factor on CPU time.
- **We DO NOT simulate**: Silicon-specific timing, on-chip communication fabric, thermal throttling, or exact power metering.

### NVIDIA A100 (`a100_emulation.py`)
- **We DO simulate**: The throughput scaling law (Eq. 31) to predict 15.3 M QPS, and the 2.1 ms latency.
- **We DO NOT simulate**: SM scheduling, tensor core internals, or NVLink topology.

## 3. Honesty Tags

Every output from these simulators is tagged with:
```json
{
  "simulated": true,
  "source": "loihi_mock_v1.0.0" (or "a100_emulation_v1.0.0")
}
