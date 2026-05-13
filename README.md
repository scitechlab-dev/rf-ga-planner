# RFEvolve

> Genetic-algorithm placement of RF concentrators over real terrain, for utility SCADA/telecontrol networks.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

## Overview

**RFEvolve** selects the minimum set of RF concentrators needed to cover a population of IEDs (reclosers, switches, sensors, etc.) in an electric distribution network, subject to **terrain-aware line-of-sight and Fresnel-zone constraints**. Elevation is taken from SRTM tiles; the optimizer is a binary-encoded genetic algorithm with support for **fixed concentrators** (existing repeater sites that must be preserved).

Designed and validated on the DELSUR distribution network (El Salvador), but the pipeline generalizes to any region with SRTM coverage.

## Features

- **Physical RF model**: Haversine geometry, Earth-curvature correction, 1st-Fresnel-zone radius, configurable clearance criterion (default: 60% Fresnel free at 900 MHz).
- **SRTM-based terrain**: loads HGT tiles (30 m or 90 m) directly or from a zipped archive; combines adjacent tiles into a single elevation map.
- **Genetic optimizer** with two-regime fitness: drives coverage to 100% first, then minimizes concentrator count while rewarding Fresnel margin.
- **Fixed-concentrator support**: lock existing repeater sites so the GA can only add complementary concentrators.
- **Interactive CLI**: select device types to treat as IEDs, anchor existing sites, set tower heights and run budget.
- **Outputs**: grouped KML (one folder per concentrator, ready for Google Earth / Radio Mobile), per-link terrain plots (PNG), and a coverage-detail CSV.

## Requirements

- Python 3.10+
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

## Citation

If you use this software in academic work, please cite:

```bibtex
@article{rfevolve2026,
  title   = {Genetic Algorithm--Based Placement of RF Concentrators with
             Terrain-Aware Fresnel Constraints for Electric Distribution SCADA},
  author  = {<Author names>},
  journal = {<Venue>},
  year    = {2026}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Contact

**Josué <Apellido>** — Electrical Engineer / OT-SCADA Analyst
`<email or GitHub profile>`