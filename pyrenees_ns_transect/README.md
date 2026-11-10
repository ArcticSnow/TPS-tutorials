# TopoPyScale in the Pyrenees: a North–South transect

`TopoPyScale_Pyrenees_transect.ipynb` downscales ERA5 (2019–2020, 6-hourly) with
[TopoPyScale](https://github.com/ArcticSnow/TopoPyScale) along a 30 km wide transect that crosses the whole
Pyrenees, from Huesca to Pau, using a 30 m Copernicus DEM. It shows:

- the **N–S climate gradient**;
- the **elevation gradient** that ERA5 cannot resolve (highest-elevation ERA5 cell is at ~1400 m);
- **slope/aspect/shading effects on shortwave radiation**;
- a **validation at the Izas AWS** (2113 m, IPE-CSIC);
- a **snow simulation** with FSM, compared with the station snow depth.

## Setup

```bash
bash setup_env.sh                # conda env 'tps' + topocalc + patched TopoPyScale + FSM + Jupyter kernel
cp data_links.example.yml data_links.yml   # then paste the data links you received for the workshop
conda activate tps
jupyter lab                      # open the notebook, kernel "Python (tps, TopoPyScale)"
```

- A recent conda or mamba is recommended; the libmamba solver is much faster.
- FSM is compiled with `gfortran`.
- The data are downloaded on first use into `./data`, or into `data_root` / `$TPS_PYR_DATA`. They are:
  - the ERA5 zip;
  - the Izas AWS file;
  - two Copernicus GLO-30 tiles.
- The data links are private and are not part of this repository.

## Switches (top of the notebook, or environment variables)

| variable | default | meaning |
|---|---|---|
| `TPS_MODE` | `light` | `light` = 300 clusters, `full` = 1500 clusters (both on the 30 m DEM) |
| `TPS_CORES` | `min(8, n_cpu)` | CPU cores |
| `TPS_PYR_DATA` | `data_root` in `data_links.yml`, else `./data` | data + outputs folder |

Results already on disk are re-used. Delete `<data>/transect_<mode>/outputs` to recompute.

## Run time (light mode, 8 cores)

The expensive part is the terrain analysis of the 30 m DEM (8.8 Mpx including a 10 km horizon buffer). It is done
once, and it does not depend on the number of clusters:

| step | time |
|---|---|
| sky-view factor / DEM parameters (single core) | ~48 min |
| TopoSUB clustering (4.6 Mpx → 300 clusters) | ~7 min |
| horizon angles (36 azimuths) | ~26 min |
| downscaling, ERA5 read from **zarr** | **~1 min** (0.16 s per cluster) |
| FSM, 300 clusters | < 1 min |
| Izas points project (24 km window) | ~1 min |

Outputs:

- `ds_param.nc` is about 0.8 GB and `da_horizon.nc` about 2.5 GB.
- Sharing these two files lets participants skip the terrain step.


- surface: one request per month;
- pressure levels: 7-day chunks.
