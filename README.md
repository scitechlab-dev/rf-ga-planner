# RFEvolve

Genetic-algorithm placement of RF concentrators over real terrain, for utility SCADA/telecontrol networks.

## Overview

RFEvolve selects the minimum set of RF concentrators needed to cover a population of IEDs (reclosers, switches, sensors) in an electric distribution network, subject to terrain-aware line-of-sight and Fresnel-zone constraints. Elevation comes from SRTM tiles; the optimizer is a binary-encoded genetic algorithm with optional fixed concentrators for existing repeater sites.

Developed against the DELSUR distribution network (El Salvador); generalizes to any region with SRTM coverage.

## Features

- Physical RF model: Haversine geometry, Earth-curvature correction, 1st-Fresnel-zone radius, 60% clearance criterion.
- SRTM terrain loader (30 m or 90 m HGT tiles, plain or zipped).
- Genetic optimizer with two-regime fitness: coverage first, minimization second.
- Fixed-concentrator support to lock existing repeater sites.
- Outputs: KML grouped by concentrator, per-link terrain plots, coverage CSV.

## Requirements

Python 3.10+, NumPy, pandas, matplotlib, and SRTM tiles for the area of interest.

```bash
pip install numpy pandas matplotlib
```

## Input data

A device CSV with columns `Origen_DataFrame`, `Placa`, `Latitud`, `Longitud`, and SRTM elevation in the project root, either as individual `.hgt` files or a single `SRTM.zip`.

## Usage

```bash
python main.py
```

The CLI prompts for antenna heights, mode (single link or GA optimization), IED device types, fixed sites, and GA budget. Results are written to the working directory.

## Project structure

```
main.py                 Interactive CLI / orchestrator
linea_vista_sv.py       SRTM loader, LOS + Fresnel model, KML export
algoritmo_genetico.py   Binary GA (selection, crossover, mutation, fitness)
```

## Default parameters

Frequency 900 MHz, max link distance 30 km, IED antenna 8 m, concentrator tower 30 m, GA population 100, generations 200.

## License

MIT.