# TopoPyScale in the Pyrenees: a North–South transect

Two notebooks downscale ERA5 (2019–2020, 6-hourly) with [TopoPyScale](https://github.com/ArcticSnow/TopoPyScale)
along a 30 km wide transect that crosses the whole Pyrenees, from Huesca to Pau, using a 30 m Copernicus DEM.
Run them in this order:

1. **`1_downscaling.ipynb`** runs TopoPyScale step by step, for readers who have never used it. Its sections follow
   the workflow figure: Inputs, 1. Terrain, 2. TopoSUB (or a list of points), 3. Solar geometry, 4. Horizons,
   5. TopoScale, Results. It shows:
   - the terrain parameters and the clusters, with 300 and 3000 clusters;
   - the points around the Izas AWS and the station horizon vs the sun path;
   - a downscaled field and the partial results of the downscaling around Izas;
   - the summary and the compute times.
2. **`2_analysis.ipynb`** loads the outputs of notebook 1 and shows:
   - the **N–S climate gradient**;
   - the **elevation gradient** that ERA5 cannot resolve (highest-elevation ERA5 cell is at ~1400 m);
   - **slope/aspect/shading effects on shortwave radiation**;
   - a **validation at the Izas AWS** (2113 m, IPE-CSIC);
   - a **snow simulation** with FSM, compared with the station snow depth.

Notebook 2 does not run the downscaling; it stops with an error if the outputs of notebook 1 are missing.

## Setup

```bash
bash setup_env.sh                # conda env 'tps' + topocalc + patched TopoPyScale + FSM + Jupyter kernel
cp data_links.example.yml data_links.yml   # then paste the data links you received for the workshop
conda activate tps
jupyter lab                      # open 1_downscaling.ipynb, then 2_analysis.ipynb (kernel "Python (tps, TopoPyScale)")
```

- A recent conda or mamba is recommended; the libmamba solver is much faster.
- FSM is compiled with `gfortran`.
- The data are downloaded on first use into `./data`, or into `data_root` / `$TPS_PYR_DATA`. They are:
  - the ERA5 zip;
  - the Izas AWS file;
  - two Copernicus GLO-30 tiles.
- The data links are private and are not part of this repository.

## Switches (top of both notebooks)

| switch | default | meaning |
|---|---|---|
| `MODE` | `"full"` | `"light"`: the transect with 300 clusters (fine on a laptop); `"full"`: also with 3000 clusters, which notebook 2 then analyses |
| `ERA5_TIMESTEP` | `"6h"` | `"6h"`: the ERA5 provided for the workshop (6-hourly, 2019–2020); `"1h"`: hourly ERA5 for 1 Nov 2019 – 30 Apr 2020 (see "Hourly ERA5" below), in projects ending in `_1h` that re-use the terrain, clusters and horizons of the 6-hourly ones. Notebook 2 analyses the 6-hourly run |
| `CPU_CORES` | `min(20, n_cpu)` | at most 20 on the shared machine; worker processes are limited to one thread each |

The data folder is `data_root` in `data_links.yml` (or `$TPS_PYR_DATA`), else `./data`. Results already on disk are
re-used; delete `<data>/transect_light/outputs` or `<data>/transect_full/outputs` to recompute.

## Run time

The expensive part is the terrain analysis of the 30 m DEM (8.8 million pixels, including a 10 km horizon buffer). It is done
once, and it does not depend on the number of clusters:

| step | time |
|---|---|
| 1. Terrain: sky-view factor / DEM parameters (single core) | ~48 min |
| 4. Horizons (36 azimuths, every pixel) | ~26 min |
| 300 clusters (8 cores): 2. TopoSUB / 5. TopoScale, ERA5 from **zarr** / FSM | ~7 / **~1** / < 1 min |
| 3000 clusters (20 cores, terrain re-used): 2. TopoSUB / 3. Solar geometry / 5. TopoScale / FSM | ~3 / ~1.5 / **~4.5** / ~3.5 min |
| Izas points project (24 km window), steps 1–5 | ~1 min |

A first run in full mode, with the terrain files already there, takes ~25 min. With all outputs on disk, each notebook
re-runs in a few minutes.

Outputs:

- `ds_param.nc` is about 0.8 GB and `da_horizon.nc` about 2.5 GB.
- Sharing these two files lets participants skip the terrain step. Notebook 1 opens an existing `ds_param.nc`
  with `pyr_utils.dem_param()` instead of `compute_dem_param()`. The TopoPyScale call rewrites the whole file
  (~8 min), and the cluster label maps come out scrambled when a `ds_param.nc` read from disk is clustered.

## Hourly ERA5

With `ERA5_TIMESTEP = "1h"`, notebook 1 expects hourly ERA5 in `<data>/era5_1h/monthly`, downloaded with
`download_era5_1h.py` (as few CDS requests as the cost limits allow: surface one request per month, pressure levels in
7-day chunks). It prepares them in `<data>/era5_1h/topopyscale` and checks them against the 6-hourly files at
00/06/12/18 UTC.
