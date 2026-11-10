"""
Helpers for the TopoPyScale Pyrenees N-S transect workshop notebook.

Plumbing only (downloads, ERA5 re-formatting, DEM preparation, station QC, FSM runs, small
plotting utilities). Everything TopoPyScale-specific happens in the notebook itself.
"""
from __future__ import annotations

import contextlib
import glob
import os
import string
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)
from pyproj import Transformer

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------
# The workshop data links are private: they are read from data_links.yml (git-ignored), see data_links.example.yml
LINKS_FILE = Path(__file__).with_name("data_links.yml")
COP30_URL =("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{t}_DEM/"
             "Copernicus_DSM_COG_10_{t}_DEM.tif")
DEM_TILES = ["N42_00_W001_00", "N43_00_W001_00"]

EPSG = 32630                      # UTM 30N
ERA5_COLS = (-0.561, -0.311)      # the two ERA5 grid columns inside the transect (deg E)
ERA5_BBOX = (41.4, 44.1, -1.6, 0.8)   # S, N, W, E crop of the ERA5 files (keeps 3x3 stencil interior)
TRANSECT = dict(lat_s=42.10, lat_n=43.35, lon_c=float(np.mean(ERA5_COLS)), width_km=30.0)

# Izas AWS (IPE-CSIC) position and elevation from the station metadata. (The rounded literature value
# 42°44'N 0°25'W is ~1 km off and falls on a 25° SE-facing slope at 1900 m in the DEM.)
IZAS = dict(name="Izas_AWS", lat=42.74268, lon=-0.41912, elev=2113.0)

PLACES = {  # for map annotations (lat, lon)
    "Huesca": (42.136, -0.408), "Jaca": (42.570, -0.549), "Sallent": (42.771, -0.333),
    "Laruns": (42.988, -0.427), "Oloron": (43.194, -0.607), "Pau": (43.296, -0.370),
    "Balaitous": (42.839, -0.290), "Midi d'Ossau": (42.843, -0.438),
}

G = 9.80665

# ----------------------------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------------------------

def _download(url: str, dst: Path):
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  found {dst.name}")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dst.name} ...")
    req = urllib.request.Request(url, headers={"X-Requested-With": "XMLHttpRequest"})
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)
    return dst


def load_links(path=LINKS_FILE):
    """Read the private data links (and optional data_root) from data_links.yml."""
    import yaml
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found: copy data_links.example.yml to data_links.yml and fill in "
                                "the links you received for the workshop (the file is not version-controlled).")
    return yaml.safe_load(path.read_text()) or {}


def download_inputs(raw_dir, links):
    """Download (once) the ERA5 zip, the Izas xlsx and the Copernicus GLO-30 DEM tiles."""
    raw = Path(raw_dir)
    z = _download(links["era5_zip_url"], raw / "ERA5_Pyr_2019_2020.zip")
    daily = raw / "era5_daily"
    if len(glob.glob(str(daily / "d*_*.nc"))) < 1462:
        print("  unzipping ERA5 ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(daily)
    _download(links["izas_xlsx_url"], raw / "Izas_Meteo_2019-2021.xlsx")
    tiles = [_download(COP30_URL.format(t=t), raw / "dem_tiles" / f"Copernicus_DSM_COG_10_{t}_DEM.tif")
             for t in DEM_TILES]
    return dict(era5_daily=daily, izas_xlsx=raw / "Izas_Meteo_2019-2021.xlsx", dem_tiles=tiles)

# ----------------------------------------------------------------------------------------------
# ERA5 -> TopoPyScale input format
# ----------------------------------------------------------------------------------------------

def _drop_extra(ds):
    return ds.drop_vars(["number", "expver"], errors="ignore")


def prepare_era5(daily_dir, climate_dir, years=(2019, 2020), bbox=ERA5_BBOX):
    """
    Convert the daily dSURF_/dPLEV_ files into yearly SURF_YYYY.nc / PLEV_YYYY.nc as read by TopoPyScale:
    PLEV: valid_time -> time, pressure_level -> level (ascending), extra coords dropped; both cropped to bbox.
    Original files are left untouched.
    """
    climate_dir = Path(climate_dir)
    climate_dir.mkdir(parents=True, exist_ok=True)
    s, n, w, e = bbox
    crop = dict(latitude=slice(n, s), longitude=slice(w, e))
    enc = lambda ds: {v: {"zlib": True, "complevel": 1, "dtype": "float32"} for v in ds.data_vars}
    for y in years:
        fs, fp = climate_dir / f"SURF_{y}.nc", climate_dir / f"PLEV_{y}.nc"
        if fs.exists() and fp.exists():
            print(f"  found {fs.name}, {fp.name}")
            continue
        print(f"  writing {fs.name}, {fp.name} ...")
        surf = xr.open_mfdataset(sorted(glob.glob(f"{daily_dir}/dSURF_{y}*.nc")), combine="nested",
                                 concat_dim="time", preprocess=_drop_extra, coords="minimal",
                                 compat="override", data_vars="all").sel(crop).load()
        plev = xr.open_mfdataset(sorted(glob.glob(f"{daily_dir}/dPLEV_{y}*.nc")), combine="nested",
                                 concat_dim="valid_time", preprocess=_drop_extra, coords="minimal",
                                 compat="override", data_vars="all")
        plev = plev.rename({"valid_time": "time", "pressure_level": "level"}).sortby("level").sel(crop).load()
        surf.to_netcdf(fs, encoding=enc(surf))
        plev.to_netcdf(fp, encoding=enc(plev))
    return check_era5(climate_dir, years)


def build_zarr(climate_dir, store="ERA5.zarr"):
    """
    Convert the yearly SURF_/PLEV_ files into one zarr store with TopoPyScale's own converter
    (fetch_era5.convert_netcdf_stack_to_zarr, as Topoclass.era5_obj.to_zarr()). The zarr path of the downscaling is
    much faster than reading the NetCDF files.
    """
    from TopoPyScale import fetch_era5 as fe
    climate_dir = Path(climate_dir)
    if not (climate_dir / store).exists():
        n_time = xr.open_mfdataset(sorted(glob.glob(f"{climate_dir}/SURF_*.nc"))).sizes["time"]
        n_lev = xr.open_dataset(sorted(climate_dir.glob("PLEV_*.nc"))[0]).sizes["level"]
        fe.convert_netcdf_stack_to_zarr(path_to_netcdfs=str(climate_dir), zarrout=str(climate_dir / store),
                                        chunks={"latitude": 3, "longitude": 3, "time": n_time, "level": n_lev})
    return climate_dir / store


def _to_tps_format(ds, ref, plev=False):
    """CDS netcdf -> TopoPyScale names, cropped and snapped to the reference (6-hourly) grid."""
    ds = ds.drop_vars(["number", "expver"], errors="ignore")
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if plev:
        ds = ds.rename({"pressure_level": "level"}).sortby("level")
    sub = ds.sel(latitude=ref.latitude, longitude=ref.longitude, method="nearest", tolerance=0.01)
    return sub.assign_coords(latitude=ref.latitude.values, longitude=ref.longitude.values)


def prepare_era5_1h(monthly_dir, out_dir, ref_dir, start="2019-11-01", end="2020-04-30"):
    """
    Hourly ERA5 (monthly SURF / 7-day PLEV files from download_era5_1h.py) -> yearly SURF_YYYY.nc / PLEV_YYYY.nc on
    exactly the same grid as the 6-hourly files in ref_dir, with checks: complete hourly time axis, and the values at
    00/06/12/18 UTC equal to the 6-hourly files (same ERA5 fields).
    """
    monthly_dir, out_dir = Path(monthly_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_s = xr.open_mfdataset(sorted(glob.glob(f"{ref_dir}/SURF_*.nc")), combine="by_coords")
    ref_p = xr.open_mfdataset(sorted(glob.glob(f"{ref_dir}/PLEV_*.nc")), combine="by_coords")
    hours = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta("23h"), freq="1h")
    surf = xr.concat([_to_tps_format(xr.open_dataset(f), ref_s)
                      for f in sorted(monthly_dir.glob("mSURF_*.nc"))], dim="time").sortby("time").load()
    plev = xr.concat([_to_tps_format(xr.open_dataset(f), ref_p, plev=True)
                      for f in sorted(monthly_dir.glob("mPLEV_*.nc"))], dim="time").sortby("time").load()
    for name, ds in (("SURF", surf), ("PLEV", plev)):
        t = pd.DatetimeIndex(ds.time.values)
        assert t.equals(hours), f"{name}: hourly time axis incomplete ({len(t)} of {len(hours)} steps)"
    common = pd.DatetimeIndex(ref_s.time.values).intersection(hours)
    for v, tol in (("t2m", 0.01), ("ssrd", 50.0), ("tp", 1e-6), ("sp", 5.0)):
        d = float(abs(surf[v].sel(time=common) - ref_s[v].sel(time=common)).max())
        assert d < tol, f"{v}: hourly and 6-hourly ERA5 differ by {d}"
    d = float(abs(plev.t.sel(time=common) - ref_p.t.sel(time=common)).max())
    assert d < 0.01, f"PLEV t: hourly and 6-hourly ERA5 differ by {d}"
    enc = lambda ds: {v: {"zlib": True, "complevel": 1, "dtype": "float32"} for v in ds.data_vars}
    for year in np.unique(hours.year):
        s_, p_ = surf.sel(time=str(year)), plev.sel(time=str(year))
        s_.to_netcdf(out_dir / f"SURF_{year}.nc", encoding=enc(s_))
        p_.to_netcdf(out_dir / f"PLEV_{year}.nc", encoding=enc(p_))
    print(f"  hourly ERA5 ready: {len(hours)} steps, identical to the 6-hourly files at {len(common)} common times")
    return out_dir


def install_climate(src_dir, project_dir, store="ERA5.zarr"):
    """Copy the ERA5 zarr store into <project>/inputs/climate (where TopoPyScale looks for climate.era5.zarr_store)."""
    import shutil
    dst = Path(project_dir) / "inputs" / "climate"
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / store).exists():
        shutil.copytree(Path(src_dir) / store, dst / store)
    return dst


def check_era5(climate_dir, years=(2019, 2020)):
    """Sanity checks required by TopoPyScale (and by the 6-hourly / 1-hour-accumulation assumption)."""
    surf = xr.open_mfdataset(sorted(glob.glob(f"{climate_dir}/SURF_*.nc")), combine="by_coords")
    plev = xr.open_mfdataset(sorted(glob.glob(f"{climate_dir}/PLEV_*.nc")), combine="by_coords")
    expected = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 18:00", freq="6h")
    assert np.array_equal(surf.latitude, plev.latitude) and np.array_equal(surf.longitude, plev.longitude), \
        "SURF and PLEV grids differ"
    assert pd.DatetimeIndex(surf.time.values).equals(expected), "SURF time axis is not complete 6-hourly"
    assert pd.DatetimeIndex(plev.time.values).equals(expected), "PLEV time axis is not complete 6-hourly"
    assert np.all(np.diff(plev.level.values) > 0), "PLEV levels must be ascending"
    ssrd_max = float(surf.ssrd.max())
    # a 1-hour mean cannot exceed the solar constant (1361 W/m2); 6-hour accumulations would be ~4-6x larger
    assert ssrd_max < 1361 * 3600, f"ssrd max {ssrd_max:.3g} J/m2 > 1361*3600: not a 1-hour accumulation?"
    info = dict(n_times=len(expected), lat=(float(surf.latitude.min()), float(surf.latitude.max())),
                lon=(float(surf.longitude.min()), float(surf.longitude.max())),
                levels=plev.level.values.tolist(), ssrd_max_Wm2=ssrd_max / 3600,
                tp_max_mm_per_h=float(surf.tp.max()) * 1000)
    print("  ERA5 checks passed:", info)
    return info

# ----------------------------------------------------------------------------------------------
# DEM
# ----------------------------------------------------------------------------------------------
to_utm = Transformer.from_crs(4326, EPSG, always_xy=True)
to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)


def transect_bounds(res=30.0, buffer_km=10.0, **kw):
    """UTM rectangle of the transect (xmin, ymin, xmax, ymax) and of the buffered DEM, snapped to res."""
    t = {**TRANSECT, **kw}
    xc, _ = to_utm.transform(t["lon_c"], 0.5 * (t["lat_s"] + t["lat_n"]))
    _, ys = to_utm.transform(t["lon_c"], t["lat_s"])
    _, yn = to_utm.transform(t["lon_c"], t["lat_n"])
    half = t["width_km"] * 500.0
    snap = lambda v: np.round(v / res) * res
    tb = tuple(snap(v) for v in (xc - half, ys, xc + half, yn))
    b = buffer_km * 1000.0
    db = (tb[0] - b, tb[1] - b, tb[2] + b, tb[3] + b)
    # both ERA5 columns must lie inside the transect along its whole length
    for lon in ERA5_COLS:
        for lat in (t["lat_s"], t["lat_n"]):
            x, _ = to_utm.transform(lon, lat)
            assert tb[0] + 2000 < x < tb[2] - 2000, f"ERA5 column {lon} not inside transect at {lat}N"
    return tb, db


def build_dem(tiles, out_tif, bounds, res=30.0, mask_tif=None, mask_bounds=None):
    """Mosaic GLO-30 tiles and reproject to EPSG:32630 on an exact (bounds, res) grid; optional 0/1 mask."""
    from rioxarray.merge import merge_arrays
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bounds
    nx, ny = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    if not out_tif.exists():
        lon, lat = to_ll.transform([xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax])
        das = [xr.open_dataarray(t, engine="rasterio").squeeze("band", drop=True) for t in tiles]
        mos = merge_arrays(das).rio.clip_box(min(lon) - 0.02, min(lat) - 0.02, max(lon) + 0.02, max(lat) + 0.02)
        dem = mos.rio.reproject(f"EPSG:{EPSG}", transform=from_origin(xmin, ymax, res, res),
                                shape=(ny, nx), resampling=Resampling.bilinear).astype("float32")
        assert int(dem.isnull().sum()) == 0 and float(dem.min()) > -100, "DEM has nodata"
        dem.attrs = {"units": "m", "long_name": "Copernicus GLO-30 elevation"}
        dem.rio.write_nodata(None, inplace=True)
        dem.rio.to_raster(out_tif, compress="deflate")
    dem = xr.open_dataarray(out_tif, engine="rasterio").squeeze("band", drop=True)
    if mask_tif is not None and not Path(mask_tif).exists():
        mx0, my0, mx1, my1 = mask_bounds
        m = ((dem.x > mx0) & (dem.x < mx1) & (dem.y > my0) & (dem.y < my1)).astype("uint8")
        m = m.transpose("y", "x")                       # (x & y) broadcasting yields (x, y)
        m = m.rio.write_crs(f"EPSG:{EPSG}").rio.write_transform(dem.rio.transform())
        m.rio.to_raster(mask_tif)
    print(f"  {out_tif.name}: {dem.shape[0]} x {dem.shape[1]} px ({dem.size / 1e6:.1f} Mpx) at {res:.0f} m, "
          f"elevation {float(dem.min()):.0f}-{float(dem.max()):.0f} m")
    return dem


def cut_window(dem_tif, out_tif, lat, lon, half_km=12.0):
    """Cut a square window (same grid) from an existing DEM GeoTIFF."""
    out_tif = Path(out_tif)
    if not out_tif.exists():
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        dem = xr.open_dataarray(dem_tif, engine="rasterio").squeeze("band", drop=True)
        x, y = to_utm.transform(lon, lat)
        h = half_km * 1000
        win = dem.sel(x=slice(x - h, x + h), y=slice(y + h, y - h))
        win.rio.to_raster(out_tif, compress="deflate")
    return xr.open_dataarray(out_tif, engine="rasterio").squeeze("band", drop=True)

# ----------------------------------------------------------------------------------------------
# TopoPyScale configs / fixes / mapping
# ----------------------------------------------------------------------------------------------

def render_config(template, out_file, **values):
    """Fill a $placeholder YAML template (string.Template) and write it into the project directory."""
    txt = string.Template(Path(template).read_text()).substitute(**{k: str(v) for k, v in values.items()})
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(txt)
    return txt


def compass_aspect(aspect_rad):
    """TopoPyScale aspect (0 = South, +pi/2 = East, radians) -> compass aspect in degrees (0 = N, 90 = E)."""
    return np.mod(180.0 - np.rad2deg(aspect_rad), 360.0)


def numpy_compass_aspect(elev, res):
    """Independent compass aspect (deg, downslope direction, 0 = N) from a north-up elevation array."""
    dz_drow, dz_dcol = np.gradient(np.asarray(elev, dtype=float), res)
    dz_dx, dz_dy = dz_dcol, -dz_drow          # rows go north -> south
    return np.mod(np.rad2deg(np.arctan2(-dz_dx, -dz_dy)), 360.0)


def check_aspect_convention(ds_param, res, min_slope_deg=15):
    """Assert that TopoPyScale's aspect matches an independent estimate once converted with compass_aspect()."""
    a_tps = compass_aspect(ds_param.aspect.values)
    a_np = numpy_compass_aspect(ds_param.elevation.values, res)
    steep = np.rad2deg(ds_param.slope.values) > min_slope_deg
    agreement = float(np.nanmean(np.cos(np.deg2rad(a_tps - a_np))[steep]))
    print(f"  aspect convention check: mean cos(difference) = {agreement:.3f} on {steep.sum()} steep pixels")
    assert agreement > 0.8, "Aspect convention mismatch!"
    return agreement


def _as_series(values):
    if isinstance(values, xr.DataArray):
        values = values.to_series()
    return pd.Series(values)


def _cluster_index(ds_param, stride=1):
    """Integer cluster index per pixel (-1 outside the clustering mask), optionally subsampled for display."""
    ind = np.asarray(ds_param.point_ind.values)[::stride, ::stride]
    ind = np.where(np.isfinite(ind), ind, -1).astype(np.int64)
    return np.where(ind < 0, -1, ind)


def to_grid(values, ds_param, stride=1):
    """
    Map per-cluster values (pd.Series / 1-D DataArray indexed by point_name) onto the DEM grid, in memory only
    (used for plotting a single 2-D field; nothing is written to disk). stride > 1 subsamples for display.
    """
    v = _as_series(values)
    ind = _cluster_index(ds_param, stride)
    lut = np.full(max(int(ind.max()), int(np.max(v.index.astype(int)))) + 2, np.nan)
    lut[v.index.astype(int).to_numpy()] = v.to_numpy(dtype=float)
    grid = np.where(ind >= 0, lut[ind], np.nan)
    return xr.DataArray(grid, coords={"y": ds_param.y.values[::stride], "x": ds_param.x.values[::stride]},
                        dims=("y", "x"))


def lat_weights(ds_param, bin_km=5.0, by=None):
    """
    Pixel-count weights (row-normalised) of each cluster in each latitude bin, computed from the cluster labels.
    A latitudinal profile of any per-cluster variable v (indexed by point_name) is then  W @ v.
    by: optional 2-D boolean DataArray (e.g. an elevation band or slope mask) restricting the pixels counted.
    Nothing gridded is stored: the result is a small (n_bins x n_clusters) table.
    """
    ind = _cluster_index(ds_param)
    lat_row = to_ll.transform(np.full(ds_param.y.size, float(ds_param.x.mean())), ds_param.y.values)[1]
    step = bin_km / 111.2
    row_bin = np.round(lat_row / step).astype(np.int64)
    ok = ind >= 0
    if by is not None:
        ok &= np.asarray(by, dtype=bool)
    b = np.broadcast_to(row_bin[:, None], ind.shape)[ok]
    c = ind[ok]
    nb, nc = b.max() - b.min() + 1, c.max() + 1
    counts = np.bincount((b - b.min()) * nc + c, minlength=nb * nc).reshape(nb, nc)
    keep_b, keep_c = counts.sum(1) > 0, counts.sum(0) > 0
    W = pd.DataFrame(counts[keep_b][:, keep_c], index=np.round((np.arange(nb)[keep_b] + b.min()) * step, 4),
                     columns=np.arange(nc)[keep_c])
    W.index.name, W.columns.name = "lat", "point_ind"
    return W


def profile(W, values):
    """Latitude profile (pixel-weighted mean) from lat_weights() and per-cluster values indexed by point_name."""
    v = _as_series(values)
    v.index = v.index.astype(int)
    v = v.reindex(W.columns)
    good = v.notna().to_numpy()
    Wg = W.loc[:, good]
    return (Wg @ v[good]) / Wg.sum(axis=1).replace(0, np.nan)


def hillshade(elev, res, azdeg=315, altdeg=40):
    from matplotlib.colors import LightSource
    return LightSource(azdeg=azdeg, altdeg=altdeg).hillshade(np.asarray(elev, float), dx=res, dy=res, vert_exag=1)


def centroid_table(mp):
    """df_centroids + convenient derived columns (compass aspect, northness, slope in deg)."""
    df = mp.toposub.df_centroids.copy()
    df["aspect_deg"] = compass_aspect(df["aspect"])
    df["northness"] = np.cos(np.deg2rad(df["aspect_deg"])) * np.sin(df["slope"])
    df["slope_deg"] = np.rad2deg(df["slope"])
    df = df.set_index("point_name")
    if "mask" in df:
        df = df.drop(columns="mask")
    return df


def _pixels_near(ds_param, x0, y0, radius):
    sub = ds_param[["elevation", "slope", "aspect", "svf"]].sel(
        x=slice(x0 - radius, x0 + radius), y=slice(y0 + radius, y0 - radius))
    df = sub.to_dataframe().reset_index()
    df["dist"] = np.hypot(df.x - x0, df.y - y0)
    df["asp"] = compass_aspect(df.aspect)
    df["slope_deg"] = np.rad2deg(df.slope)
    return df[df.dist <= radius]


DIRECTIONS = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}


def pick_aspect_rose(ds_param, x0, y0, radius=3000.0, d_elev=100.0, slope_deg=(20, 35), max_dev_deg=15):
    """One pixel per compass direction (8) near (x0, y0), at the elevation of (x0, y0) +/- d_elev."""
    e0 = float(ds_param.elevation.sel(x=x0, y=y0, method="nearest"))
    df = _pixels_near(ds_param, x0, y0, radius)
    cand = df[(np.abs(df.elevation - e0) < d_elev) & df.slope_deg.between(*slope_deg)]
    out = []
    for name, target in DIRECTIONS.items():
        dev = np.abs(((cand.asp - target + 180) % 360) - 180)
        c = cand[dev < max_dev_deg]
        if c.empty:
            print(f"  WARNING: no {name}-facing pixel found")
            continue
        r = c.assign(score=c.dist / radius + dev[dev < max_dev_deg] / max_dev_deg).sort_values("score").iloc[0]
        out.append(dict(name=f"rose_{name}", x=r.x, y=r.y, group="aspect_rose", target=target))
    return pd.DataFrame(out)


def pick_elevation_series(ds_param, x0, y0, levels=range(1200, 2801, 200), radius=6000.0, d_elev=40.0,
                          max_slope_deg=15.0):
    """One gentle-slope pixel per elevation level near (x0, y0): isolates the elevation effect."""
    df = _pixels_near(ds_param, x0, y0, radius)
    out = []
    for z in levels:
        c = df[np.abs(df.elevation - z) < d_elev]
        if c.empty:
            print(f"  WARNING: no pixel at {z} m within {radius / 1000:.0f} km")
            continue
        gentle = c[c.slope_deg < max_slope_deg]
        c = gentle if not gentle.empty else c
        r = c.assign(score=c.slope_deg / max_slope_deg + c.dist / radius).sort_values("score").iloc[0]
        out.append(dict(name=f"elev_{z}", x=r.x, y=r.y, group="elevation_series", target=z))
    return pd.DataFrame(out)


def izas_point_table(ds_param, station=IZAS):
    """Izas AWS + elevation series + aspect rose, as the points CSV for TopoPyScale (x, y in EPSG:32630)."""
    x0, y0 = to_utm.transform(station["lon"], station["lat"])
    rows = [dict(name=station["name"], x=x0, y=y0, group="station", target=np.nan)]
    return pd.concat([pd.DataFrame(rows), pick_elevation_series(ds_param, x0, y0),
                      pick_aspect_rose(ds_param, x0, y0)], ignore_index=True)

# ----------------------------------------------------------------------------------------------
# Raw ERA5 baseline
# ----------------------------------------------------------------------------------------------

def rh_from_dewpoint(t_k, td_k):
    es = lambda T: 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    return 100.0 * es(td_k) / es(t_k)


def open_era5_surf(climate_dir):
    return xr.open_mfdataset(sorted(glob.glob(f"{climate_dir}/SURF_*.nc")), combine="by_coords")


def era5_columns(climate_dir, lat_range=(42.0, 43.5), lons=ERA5_COLS):
    """ERA5 surface fields of the grid cells along the two transect columns (dims latitude, longitude, time)."""
    surf = open_era5_surf(climate_dir)
    lat_sel = surf.latitude[(surf.latitude >= lat_range[0]) & (surf.latitude <= lat_range[1])]
    c = surf.sel(longitude=list(lons), method="nearest").sel(latitude=lat_sel).load()
    c["oro"] = c.z.isel(time=0) / G
    return c


def era5_display_map(da_ll, x, y):
    """
    Show an ERA5 field (dims latitude, longitude) at native resolution on a UTM display grid (x, y 1-D):
    each display pixel takes the value of its nearest ERA5 cell (the 'blocky' view ERA5 really has).
    """
    X, Y = np.meshgrid(x, y)
    lon, lat = to_ll.transform(X.ravel(), Y.ravel())
    ilat = np.abs(da_ll.latitude.values[None, :] - lat[:, None]).argmin(1)
    ilon = np.abs(da_ll.longitude.values[None, :] - lon[:, None]).argmin(1)
    vals = np.asarray(da_ll.transpose("latitude", "longitude").values)[ilat, ilon].reshape(X.shape)
    return xr.DataArray(vals, coords={"y": y, "x": x}, dims=("y", "x"))


def era5_nearest(climate_dir, lat, lon):
    """Raw ERA5 at the nearest grid cell (no interpolation). tp is scaled to mm per 6 h like TopoPyScale."""
    surf = xr.open_mfdataset(sorted(glob.glob(f"{climate_dir}/SURF_*.nc")), combine="by_coords")
    c = surf.sel(latitude=lat, longitude=lon, method="nearest").load()
    df = pd.DataFrame({
        "T": c.t2m - 273.15, "RH": rh_from_dewpoint(c.t2m, c.d2m), "P": c.sp / 100,
        "SW": c.ssrd / 3600, "LW": c.strd / 3600, "tp": c.tp * 1000 * 6,
    }, index=pd.DatetimeIndex(c.time.values, name="time"))
    df.attrs = dict(lat=float(c.latitude), lon=float(c.longitude), oro=float(c.z.isel(time=0)) / G)
    return df

# ----------------------------------------------------------------------------------------------
# Izas station
# ----------------------------------------------------------------------------------------------
_IZAS_COLS = {"TEMP_Avg": "T", "RH_Avg": "RH", "BARO_Avg": "P", "PYRA_Avg": "SW", "ALBEDO_Avg": "SWup",
              "WS_Avg": "WS", "WD_Avg": "WD", "Pluvio": "Pluvio", "HS_Avg": "HS"}


def load_izas(xlsx, cache=None):
    """Read the 10-min Izas AWS file (UTC, end-of-interval labels) with light QC."""
    cache = Path(cache) if cache else Path(xlsx).with_suffix(".pkl")
    if cache.exists():
        return pd.read_pickle(cache)
    raw = pd.read_excel(xlsx, sheet_name="Datos")
    t = pd.to_datetime(pd.to_datetime(raw["#Fecha"]).dt.strftime("%Y-%m-%d") + " " + raw["Hora"].astype(str))
    df = pd.DataFrame(index=pd.DatetimeIndex(t, name="time"))
    for col in raw.columns:
        for key, new in _IZAS_COLS.items():
            if str(col).startswith(key + "(") or str(col) == key:
                df[new] = raw[col].to_numpy(dtype=float)
    df["SW"] = df["SW"].clip(lower=0)
    df["SWup"] = df["SWup"].clip(lower=0)
    df.to_pickle(cache)
    return df


def izas_precip(df10, max_step_mm=5.0):
    """
    10-min precipitation from the Geonor bucket level ('Pluvio', cumulative, emptying = drops).
    Negative steps (emptying, noise) are discarded, and so are implausible positive jumps
    (> max_step_mm in 10 min, e.g. antifreeze added after emptying).
    """
    d = df10["Pluvio"].diff()
    d = d.where((d > 0) & (d < max_step_mm), 0.0)
    return d.where(df10["Pluvio"].notna() & df10["Pluvio"].shift().notna())


def izas_snow_depth(df10):
    """Daily median snow depth (m) from the SR50A, negative noise removed."""
    hs = df10["HS"].where(df10["HS"] > -5).clip(lower=0) / 100
    return hs.resample("1D").median()


def izas_at(df10, times):
    """
    Station values matched to model timestamps:
      instantaneous variables (T, RH, P, WS): mean of the 10-min values around t (t-10, t, t+10 min labels)
      SW: mean over (t-1h, t], matching the ERA5 1-hour accumulation ending at t
      precip: 6-h total over (t-6h, t]
    """
    inst = df10[["T", "RH", "P", "WS"]].rolling(3, center=True).mean()
    sw = df10["SW"].rolling(6).mean()
    pr = izas_precip(df10).rolling(36, min_periods=30).sum()
    out = pd.concat([inst, sw, pr.rename("precip")], axis=1).reindex(pd.DatetimeIndex(times))
    out.index.name = "time"
    return out

# ----------------------------------------------------------------------------------------------
# Downscaled point -> DataFrame, metrics
# ----------------------------------------------------------------------------------------------

def downscaled_df(ds_pt, solar_pt=None):
    """
    One downscaled point (dims: time) -> DataFrame in station units.
    TopoPyScale's SW is on the inclined surface of the DEM pixel. With the point's solar geometry (solar_pt), also
    return 'SW_horiz': the shortwave a levelled pyranometer at that point would receive (same diffuse/sky view and
    shading, direct beam on a horizontal plane: SW_direct * mu0 / cos_illumination).
    """
    from TopoPyScale import meteo_util as mu
    ds_pt = ds_pt.load()
    df = pd.DataFrame({
        "T": ds_pt.t - 273.15, "RH": 100 * mu.q_2_rh(ds_pt.t.values, ds_pt.p.values, ds_pt.q.values),
        "P": ds_pt.p / 100, "SW": ds_pt.SW, "SW_direct": ds_pt.SW_direct, "SW_diffuse": ds_pt.SW_diffuse,
        "LW": ds_pt.LW, "WS": ds_pt.ws, "tp": ds_pt.tp, "tp_nolapse": ds_pt.tp / ds_pt.precip_lapse_rate,
    }, index=pd.DatetimeIndex(ds_pt.time.values, name="time"))
    if solar_pt is not None:
        mu0 = solar_pt.mu0.sel(time=ds_pt.time).values
        cosi = ds_pt.cos_illumination.values
        ratio = np.where(cosi > 0.01, mu0 / np.where(cosi > 0.01, cosi, 1.0), 0.0)
        df["SW_horiz"] = ds_pt.SW_diffuse.values + ds_pt.SW_direct.values * ratio
    return df


def metrics(obs, mod):
    o, m = pd.Series(obs), pd.Series(mod)
    ok = o.notna() & m.notna()
    o, m = o[ok], m[ok]
    return pd.Series({"bias": (m - o).mean(), "RMSE": np.sqrt(((m - o) ** 2).mean()),
                      "r": np.corrcoef(o, m)[0, 1] if len(o) > 2 else np.nan, "n": int(ok.sum())})

# ----------------------------------------------------------------------------------------------
# FSM
# ----------------------------------------------------------------------------------------------

def run_fsm(mp, fsm_exec, ds=None, tag="FSM", nave=4, nconfig=31, n_core=8):
    """
    Export forcing with the (timestep-aware) TopoPyScale FSM exporter and run FSM for all points.
    ds: optional modified downscaled dataset (e.g. precipitation without lapse rate); default mp.downscaled_pts.
    Returns snow depth (m) and SWE (kg m-2) as DataArrays (point_name, time) with daily values.
    """
    from TopoPyScale import sim_fsm, topo_export as te

    ds = mp.downscaled_pts if ds is None else ds
    proj = Path(mp.config.project.directory)
    tstep = pd.Timedelta(mp.config.climate[mp.config.project.climate].timestep).total_seconds()
    fmt = f"{tag}_pt_*.txt"
    with contextlib.chdir(proj):        # FSM (Fortran) needs short relative paths
        for f in glob.glob(f"outputs/{fmt}") + glob.glob(f"fsm_sims/sim_{tag}_pt_*.txt"):
            os.remove(f)
        te.to_fsm(ds, Path("outputs") / fmt, tstep_seconds=tstep)
        sim_fsm.fsm_sim_parallel(fsm_input=f"outputs/{fmt}", fsm_nconfig=nconfig, fsm_nave=nave,
                                 fsm_exec=str(Path(fsm_exec).resolve()), n_core=n_core, fsm_dt=tstep)
        names = ds.point_name.values
        snd, swe = {}, {}
        for f in sorted(glob.glob(f"fsm_sims/sim_{tag}_pt_*.txt")):
            i = int(Path(f).stem.split("_")[-1])
            df = sim_fsm.read_pt_fsm(f)
            snd[names[i]], swe[names[i]] = df["snd"], df["swe"]
    to_da = lambda d, n: xr.DataArray(pd.DataFrame(d).T.sort_index(), dims=("point_name", "time"), name=n)
    return to_da(snd, "snd"), to_da(swe, "swe")
