"""
Download hourly ERA5 (surface + pressure levels) with AS FEW CDS REQUESTS AS THE CDS COST LIMITS ALLOW.

The CDS queue slows down after the first few requests, so the number of requests matters more than their size.
TopoPyScale's fetch_era5.retrieve_era5 makes one request per day and dataset (Nov 2019 - Apr 2020: 364 requests).
Here: surface = one request per month; pressure levels = chunks of PLEV_DAYS days within each month
(a month, 15 or 8 days exceed the CDS cost limit for 6 variables x 9 levels; 7 days pass) -> 6 + 30 = 36 requests.
Variables, pressure levels, grid and request format are the same as TopoPyScale's era5_request_surf /
era5_request_plev, and the area is the same as the workshop 6-hourly files (same grid points).

Files: <era_dir>/monthly/mSURF_YYYYMM.nc and mPLEV_YYYYMMDD-DD.nc (existing files are skipped -> restartable).
SURF zip archives sent by the new CDS (instant + accumulated streams) are merged with fetch_era5.unzip_file.

usage: python download_era5_1h.py ERA_DIR START_MONTH END_MONTH [N_THREADS] [SURF,PLEV] [PLEV_DAYS] [MAX_JOBS]
       e.g. python download_era5_1h.py data/era5_1h 2019-11 2020-04 6 SURF,PLEV 7
"""
import calendar
import sys
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cdsapi
import pandas as pd
from TopoPyScale import fetch_era5 as fe

AREA = [44.62, -3.561, 40.619, 3.94]                 # N, W, S, E: same as the 6-hourly workshop files
HOURS = [f"{h:02d}:00" for h in range(24)]
PLEVELS = [600, 650, 700, 750, 800, 850, 900, 950, 1000]
SURF_VARS = ["geopotential", "2m_dewpoint_temperature", "surface_thermal_radiation_downwards",
             "surface_solar_radiation_downwards", "surface_pressure", "total_precipitation", "2m_temperature"]
PLEV_VARS = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind",
             "specific_humidity", "relative_humidity"]


def request(kind, month, target, day_range=None):
    last = calendar.monthrange(month.year, month.month)[1]
    d0, d1 = day_range or (1, last)
    days = [f"{d:02d}" for d in range(d0, d1 + 1)]
    req = {"product_type": ["reanalysis"], "variable": SURF_VARS if kind == "SURF" else PLEV_VARS,
           "year": str(month.year), "month": f"{month.month:02d}", "day": days, "time": HOURS,
           "area": AREA, "grid": "0.25/0.25", "data_format": "netcdf", "download_format": "unarchived"}
    if kind == "PLEV":
        req["pressure_level"] = [str(p) for p in PLEVELS]
    dataset = "reanalysis-era5-single-levels" if kind == "SURF" else "reanalysis-era5-pressure-levels"
    part = target.with_suffix(".part.nc")
    cdsapi.Client().retrieve(dataset, req, str(part))
    fe.unzip_file(str(part))          # SURF may come as a zip of two netcdf files -> merged in place
    part.rename(target)


def run(job):
    kind, month, target, day_range = job
    err = None
    for attempt in range(3):
        try:
            t0 = time.time()
            request(kind, month, target, day_range)
            return f"OK   {target.name} ({(time.time() - t0) / 60:.1f} min)"
        except Exception as e:      # retry transient CDS errors
            err = e
            time.sleep(60)
    return f"FAIL {target.name}: {err}"


if __name__ == "__main__":
    era_dir, start, end = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    n_threads = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    kinds = sys.argv[5].split(",") if len(sys.argv) > 5 else ["SURF", "PLEV"]
    PLEV_DAYS = int(sys.argv[6]) if len(sys.argv) > 6 else 7          # ~9,100 fields: largest accepted by CDS
    MAX_JOBS = int(sys.argv[7]) if len(sys.argv) > 7 else 0           # >0: only the first N requests (probing)
    out = era_dir / "monthly"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for m in pd.date_range(start, end, freq="MS"):
        last = calendar.monthrange(m.year, m.month)[1]
        if "SURF" in kinds:                       # surface: one request per month
            jobs.append(("SURF", m, out / f"mSURF_{m:%Y%m}.nc", None))
        if "PLEV" in kinds:                       # pressure levels: chunks of PLEV_DAYS days (CDS cost limit)
            for d0 in range(1, last + 1, PLEV_DAYS):
                d1 = min(d0 + PLEV_DAYS - 1, last)
                jobs.append(("PLEV", m, out / f"mPLEV_{m:%Y%m}{d0:02d}-{d1:02d}.nc", (d0, d1)))
    if MAX_JOBS:
        jobs = jobs[:MAX_JOBS]
    jobs = [j for j in jobs if not j[2].exists()]
    print(f"{len(jobs)} requests to do ({n_threads} threads, PLEV chunks of {PLEV_DAYS} days)", flush=True)
    with ThreadPool(n_threads) as pool:
        for i, msg in enumerate(pool.imap_unordered(run, jobs), 1):
            print(f"[{i}/{len(jobs)}] {time.strftime('%H:%M:%S')} {msg}", flush=True)
    print("DOWNLOAD FINISHED", flush=True)
