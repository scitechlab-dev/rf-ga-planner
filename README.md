# RFEvolve

<<<<<<< HEAD
> Genetic-algorithm placement of RF concentrators over real terrain, for utility SCADA/telecontrol networks.
=======
> Genetic algorithm–based optimization of RF communication networks for electric utility SCADA systems.
>>>>>>> ee5501030d4a17c8f37b79b5c2a0ddcdea1b5e38

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

## Overview

<<<<<<< HEAD
**RFEvolve** selects the minimum set of RF concentrators needed to cover a population of IEDs (reclosers, switches, sensors, etc.) in an electric distribution network, subject to **terrain-aware line-of-sight and Fresnel-zone constraints**. Elevation is taken from SRTM tiles; the optimizer is a binary-encoded genetic algorithm with support for **fixed concentrators** (existing repeater sites that must be preserved).

Designed and validated on the DELSUR distribution network (El Salvador), but the pipeline generalizes to any region with SRTM coverage.

## Features

- **Physical RF model**: Haversine geometry, Earth-curvature correction, 1st-Fresnel-zone radius, configurable clearance criterion (default: 60% Fresnel free at 900 MHz).
- **SRTM-based terrain**: loads HGT tiles (30 m or 90 m) directly or from a zipped archive; combines adjacent tiles into a single elevation map.
- **Genetic optimizer** with two-regime fitness: drives coverage to 100% first, then minimizes concentrator count while rewarding Fresnel margin.
- **Fixed-concentrator support**: lock existing repeater sites so the GA can only add complementary concentrators.
- **Interactive CLI**: select device types to treat as IEDs, anchor existing sites, set tower heights and run budget.
- **Outputs**: grouped KML (one folder per concentrator, ready for Google Earth / Radio Mobile), per-link terrain plots (PNG), and a coverage-detail CSV.
=======
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
>>>>>>> ee5501030d4a17c8f37b79b5c2a0ddcdea1b5e38

## Requirements

- Python 3.10+
<<<<<<< HEAD
- `numpy`, `pandas`, `matplotlib`
- SRTM tiles covering the area of interest (`.hgt` or `SRTM.zip`)

```bash
pip install numpy pandas matplotlib
```

## Input data

**1. Device CSV** (`datos.csv` by default) with at least:

| Column | Description |
|---|---|
| `Origen_DataFrame` | Device class (e.g. `RECLOSERS`, `SITIO DE REPETICION`) |
| `Placa` | Unique device identifier |
| `Latitud`, `Longitud` | WGS-84 coordinates |

**2. SRTM elevation** in the project root, as either:
- Individual `.hgt` files (e.g. `N13W089.hgt`), or
- A single `SRTM.zip` archive containing them.

## Usage

```bash
python main.py
```

The CLI then prompts for:

- Antenna heights (IED, concentrator).
- Mode: `1` single-link analysis, or `2` full GA optimization.
- For mode 2: which device types act as IEDs, which existing repeater sites to lock as fixed, GA generations, and an optional cap on total concentrators.

Outputs are written to the working directory: `red_rf_optimizada.kml`, `cobertura_detalle.csv`, and per-link `enlace_*.png` plots.

## Project structure

```
rf-evolve/
├── main.py                 # Interactive CLI / orchestrator
├── linea_vista_sv.py       # SRTM loader, LOS + Fresnel model, KML export
├── algoritmo_genetico.py   # Binary GA (selection, crossover, mutation, fitness)
├── datos.csv               # Device inventory (user-provided)
└── SRTM.zip                # Elevation tiles (user-provided)
```

## Default parameters

| Parameter | Default |
|---|---|
| Frequency | 900 MHz |
| Max link distance | 30 km |
| IED antenna height | 8 m |
| Concentrator tower height | 30 m |
| Clearance criterion | 60% of 1st Fresnel zone |
| GA population | 100 |
| GA generations | 200 |
=======
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
>>>>>>> ee5501030d4a17c8f37b79b5c2a0ddcdea1b5e38

## Citation

If you use this software in academic work, please cite:

```bibtex
@article{rfevolve2026,
<<<<<<< HEAD
  title   = {Genetic Algorithm--Based Placement of RF Concentrators with
             Terrain-Aware Fresnel Constraints for Electric Distribution SCADA},
=======
  title   = {Genetic Algorithm–Based Optimization of RF Communication Networks for Electric Distribution SCADA},
>>>>>>> ee5501030d4a17c8f37b79b5c2a0ddcdea1b5e38
  author  = {<Author names>},
  journal = {<Venue>},
  year    = {2026}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Contact

**Josué <Apellido>** — Electrical Engineer / OT-SCADA Analyst
<<<<<<< HEAD
`<email or GitHub profile>`
=======
For questions or collaboration: `<email or GitHub profile>`
>>>>>>> ee5501030d4a17c8f37b79b5c2a0ddcdea1b5e38
