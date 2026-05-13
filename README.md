# RFEvolve

> Genetic algorithm–based optimization of RF communication networks for electric utility SCADA systems.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

## Overview

**RFEvolve** is a research-oriented toolkit for optimizing radio-frequency (RF) network topologies in geographically distributed industrial communication systems, with a focus on electric distribution utilities and their SCADA/telecontrol infrastructure. The tool applies a **genetic algorithm (GA)** to jointly optimize repeater placement, link assignments, and frequency planning, subject to terrain, line-of-sight, and link-budget constraints.

Outputs are exported in **KML** format for direct visualization and validation in [Radio Mobile](https://www.ve2dbe.com/english1.html) and standard GIS tools.

## Key features

- **GA-based optimization** with configurable population, selection, crossover, and mutation operators.
- **Multi-objective formulation**: minimize repeater count and total path loss while maximizing coverage and redundancy.
- **Geographic-aware fitness**: integrates DEM/terrain data for line-of-sight and Fresnel-zone checks.
- **KML export pipeline** for visual inspection of candidate solutions in Google Earth and Radio Mobile.
- **Reproducible experiments**: seeded runs, configuration files, and logged convergence metrics.

## Repository structure

```
rf-evolve/
├── src/
│   ├── ga/              # Genetic algorithm core (operators, fitness, population)
│   ├── rf/              # Link-budget, path-loss, and propagation models
│   ├── geo/             # Coordinate transforms and KML I/O
│   └── io/              # Configuration loading and result serialization
├── data/                # Sample inputs (DEM, site lists, constraints)
├── notebooks/           # Exploratory analysis and result post-processing
├── tests/               # Unit tests
├── configs/             # YAML experiment configurations
└── results/             # GA logs, KML outputs, and figures
```

## Requirements

- Python 3.10+
- NumPy, SciPy, pandas
- `simplekml` (KML generation)
- `pyproj` (geodetic transforms)
- `matplotlib` (optional, for plots)

## Installation

```bash
git clone https://github.com/<your-username>/rf-evolve.git
cd rf-evolve
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
python -m rfevolve.run --config configs/example.yaml
```

This runs the GA with the example configuration and writes:

- `results/<run-id>/solution.kml` — best topology, ready for Radio Mobile.
- `results/<run-id>/convergence.csv` — fitness history per generation.
- `results/<run-id>/summary.json` — final metrics and configuration snapshot.

## Methodology

The optimization problem is formulated as:

$$
\min_{\mathbf{x} \in \mathcal{X}} \; f(\mathbf{x}) = w_1 \cdot N_{\text{rep}}(\mathbf{x}) + w_2 \cdot \overline{L}(\mathbf{x}) - w_3 \cdot C(\mathbf{x})
$$

where $\mathbf{x}$ encodes the candidate repeater set and link assignments, $N_{\text{rep}}$ is the repeater count, $\overline{L}$ is the average path loss, and $C$ is the fraction of remote sites with viable connectivity. Constraints enforce minimum SNR margins and terrain-aware line of sight.

Details on the encoding, operators, and convergence analysis are documented in the accompanying paper (see *Citation* below).

## Citation

If you use this software in academic work, please cite:

```bibtex
@article{rfevolve2026,
  title   = {Genetic Algorithm–Based Optimization of RF Communication Networks for Electric Distribution SCADA},
  author  = {<Author names>},
  journal = {<Venue>},
  year    = {2026}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Contact

**Josué <Apellido>** — Electrical Engineer / OT-SCADA Analyst
For questions or collaboration: `<email or GitHub profile>`
