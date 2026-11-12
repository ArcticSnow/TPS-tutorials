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
    # same ERA5 fields; only GRIB packing differs between requests (daily vs monthly) -> physical tolerances
    diffs = {}
    for v, tol in (("t2m", 0.05), ("ssrd", 3600.0), ("strd", 3600.0), ("tp", 1e-5), ("sp", 10.0)):
        diffs[v] = d = float(abs(surf[v].sel(time=common) - ref_s[v].sel(time=common)).max())
        assert d < tol, f"{v}: hourly and 6-hourly ERA5 differ by {d}"
    for v, tol in (("t", 0.05), ("q", 1e-5), ("u", 0.05), ("z", 5.0)):
        diffs[f"plev {v}"] = d = float(abs(plev[v].sel(time=common) - ref_p[v].sel(time=common)).max())
        assert d < tol, f"PLEV {v}: hourly and 6-hourly ERA5 differ by {d}"
    print("  max |hourly - 6-hourly| at common times:", {k: f"{v:.3g}" for k, v in diffs.items()})
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
    print(f"  {out_tif.name}: {dem.shape[0]} x {dem.shape[1]} px ({dem.size / 1e6:.1f} million pixels) at {res:.0f} m, "
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
    """Raw ERA5 at the nearest grid cell (no interpolation). tp is scaled to mm per time step like TopoPyScale."""
    surf = xr.open_mfdataset(sorted(glob.glob(f"{climate_dir}/SURF_*.nc")), combine="by_coords")
    c = surf.sel(latitude=lat, longitude=lon, method="nearest").load()
    tstep_h = (c.time[1] - c.time[0]).values / np.timedelta64(1, "h")   # ERA5 time step [h]
    df = pd.DataFrame({
        "T": c.t2m - 273.15, "RH": rh_from_dewpoint(c.t2m, c.d2m), "P": c.sp / 100,
        "SW": c.ssrd / 3600, "LW": c.strd / 3600, "tp": c.tp * 1000 * tstep_h,
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

@contextlib.contextmanager
def _quiet():
    """
    Silence the per-file messages of TopoPyScale and FSM: Python prints of this process and everything written to
    file descriptor 1 (worker processes started from here, the FSM executable). Errors (stderr) still show.
    """
    import io
    import sys
    sys.stdout.flush()
    devnull, saved = os.open(os.devnull, os.O_WRONLY), os.dup(1)
    os.dup2(devnull, 1)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def run_fsm(mp, fsm_exec, ds=None, tag="FSM", nave=4, nconfig=31, n_core=8, quiet=True):
    """
    Export forcing with the (timestep-aware) TopoPyScale FSM exporter and run FSM for all points.
    ds: optional modified downscaled dataset (e.g. precipitation without lapse rate); default mp.downscaled_pts.
    Returns snow depth (m) and SWE (kg m-2) as DataArrays (point_name, time) with daily values.
    quiet: hide the per-file messages (thousands of lines for thousands of clusters).
    """
    from TopoPyScale import sim_fsm, topo_export as te

    ds = mp.downscaled_pts if ds is None else ds
    proj = Path(mp.config.project.directory)
    tstep = pd.Timedelta(mp.config.climate[mp.config.project.climate].timestep).total_seconds()
    fmt = f"{tag}_pt_*.txt"
    with contextlib.chdir(proj), (_quiet() if quiet else contextlib.nullcontext()):   # FSM needs short relative paths
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
    print(f"FSM ({tag}): {len(snd)} points simulated, time step {tstep:.0f} s, daily outputs")
    return to_da(snd, "snd"), to_da(swe, "swe")

# ----------------------------------------------------------------------------------------------
# Shared by the two notebooks (1_downscaling, 2_analysis)
# ----------------------------------------------------------------------------------------------

def load_times(times_file):
    import json
    times_file = Path(times_file)
    return json.loads(times_file.read_text()) if times_file.exists() else {}


@contextlib.contextmanager
def timed(label, times_file, cached=False):
    """
    Time a notebook step and keep its compute time in times_file (JSON, shared by both notebooks). When the step is
    only loaded from existing outputs (cached), report the time it took when it was actually computed.
    """
    import json
    import time
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    times = load_times(times_file)
    if cached and label in times:
        print(f"[{label}] loaded from disk in {dt:.0f} s (computed in {times[label] / 60:.1f} min)")
    elif cached:                     # outputs made elsewhere (e.g. copied from another project): no compute time
        print(f"[{label}] loaded from disk in {dt:.0f} s")
    else:
        times[label] = dt
        Path(times_file).parent.mkdir(parents=True, exist_ok=True)
        Path(times_file).write_text(json.dumps(times, indent=1))
        print(f"[{label}] computed in {dt / 60:.1f} min")


def disk_usage(p):
    """Size of a file or folder in bytes."""
    p = Path(p)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def km(v):
    return np.asarray(v) / 1000.0


def install_dem(files, project_dir):
    """Copy the DEM (and mask) GeoTIFFs into <project>/inputs/dem, where TopoPyScale looks for them."""
    import shutil
    for f in files:
        dst = Path(project_dir) / "inputs" / "dem" / Path(f).name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)


CLUSTER_VARS = ["cluster_labels", "point_name", "point_ind", "mask"]   # written into ds_param.nc by the clustering


def _dims_x_first(ds):
    """
    Same dataset with dimension order (x, y) and variables still (y, x), as produced by TopoPyScale's open_dem().
    TopoPyScale's clustering writes the cluster labels back with np.reshape(..., order='F'), which is only right for
    this order: a ds_param.nc opened from file has order (y, x) and would get scrambled label maps.
    """
    out = xr.Dataset(coords={"x": ds.x, "y": ds.y}, attrs=ds.attrs)
    for v in ds.variables:
        if v not in ("x", "y"):
            out[v] = ds[v]
    return out


def dem_param(mp):
    """
    Step 1 without re-processing: if ds_param.nc already exists, open it instead of calling mp.compute_dem_param(),
    which re-reads and rewrites the whole file even when nothing has to be computed (~8 min for the transect).
    If the clustering still has to run (it rewrites ds_param.nc), the file is loaded in memory, without old labels
    and in the dimension order the clustering expects.
    """
    out = Path(mp.config.outputs.path)
    f = out / mp.config.outputs.file.ds_param
    if not f.exists():
        mp.compute_dem_param()
    elif (out / mp.config.outputs.file.df_centroids).exists():
        mp.toposub.ds_param = xr.open_dataset(f)
    else:
        mp.toposub.ds_param = _dims_x_first(xr.load_dataset(f).drop_vars(CLUSTER_VARS, errors="ignore"))
    return mp.toposub.ds_param


def seed_project(src_project, dst_project):
    """
    Re-use steps 1, 2 and 4 of a project with the same DEM and sampling but another ERA5 time step (they do not depend
    on the time step): copy ds_param.nc (terrain + cluster labels) and df_centroids.pck, link da_horizon.nc. Files
    that already exist in dst_project, or are missing in src_project, are left alone (they are then computed).
    """
    import shutil
    src, dst = Path(src_project) / "outputs", Path(dst_project) / "outputs"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("ds_param.nc", "df_centroids.pck"):
        if (src / f).exists() and not (dst / f).exists():
            shutil.copyfile(src / f, dst / f".{f}.part")
            os.replace(dst / f".{f}.part", dst / f)
    if (src / "da_horizon.nc").exists() and not (dst / "da_horizon.nc").exists():
        (dst / "da_horizon.nc").symlink_to((src / "da_horizon.nc").resolve())


def clearest_winter_noon(climate_dir, months=(12, 1)):
    """The 12 UTC time step of December/January with the most ERA5 shortwave along the transect (a clear day)."""
    cols = era5_columns(climate_dir)
    t = cols.time.to_index()
    sel = np.where((t.hour == 12) & t.month.isin(months))[0]
    return t[sel][int(np.argmax(cols.ssrd.isel(time=sel).mean(["latitude", "longitude"]).values))]


def seed_terrain(src_outputs, dst_outputs, ds_param="ds_param.nc", da_horizon="da_horizon.nc"):
    """
    Re-use the terrain analysis of a project on the same DEM in another project (slope, aspect, sky view and horizons
    do not depend on the clustering):
    - ds_param: copied without the cluster variables, with the original encoding. TopoPyScale writes the cluster labels
      into this file, so two projects cannot share it;
    - da_horizon: symbolic link (TopoPyScale only reads it).
    """
    src, dst = Path(src_outputs), Path(dst_outputs)
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / ds_param).exists():
        import warnings
        with xr.open_dataset(src / ds_param) as ds, warnings.catch_warnings():
            warnings.simplefilter("ignore", xr.SerializationWarning)     # no _FillValue: there are no NaNs
            ds = ds.drop_vars(CLUSTER_VARS, errors="ignore")
            ds.to_netcdf(dst / f".{ds_param}.part")
        os.replace(dst / f".{ds_param}.part", dst / ds_param)
    if not (dst / da_horizon).exists():
        (dst / da_horizon).symlink_to((src / da_horizon).resolve())


def era5_cells_in(oro, bounds):
    """ERA5 cell centres (lat, lon) of the field oro (dims latitude, longitude) whose centre lies within UTM bounds."""
    out = []
    for la in oro.latitude.values:
        for lo in oro.longitude.values:
            x, y = to_utm.transform(lo, la)
            if bounds[0] < x < bounds[2] and bounds[1] < y < bounds[3]:
                out.append((la, lo))
    return out


def draw_era5_cells(ax, cells, half=0.125, **kw):
    """Outline ERA5 cells (lat, lon centres) on a UTM map in km; + marks the cell centres."""
    from matplotlib.patches import Polygon
    for la, lo in cells:
        corners = [(lo - half, la - half), (lo + half, la - half), (lo + half, la + half), (lo - half, la + half)]
        xy = np.array([to_utm.transform(a, b) for a, b in corners]) / 1000
        ax.add_patch(Polygon(xy, fill=False, **kw))
        x, y = to_utm.transform(lo, la)
        ax.plot(x / 1000, y / 1000, "r+", ms=7)


def map_clusters(ax, values, cmap, label, vmin=None, vmax=None, stride=6, params=None, dem=None, extent=None,
                 res=30.0, colorbar=True):
    """
    Map per-cluster values (indexed by point_name) onto the pixels of each cluster (params = ds_param with the
    cluster labels) over a hillshade of dem, with the Izas AWS as a gold star. stride > 1 subsamples for display.
    """
    g = to_grid(values, params, stride=stride)
    e = extent or [km(g.x[0]), km(g.x[-1]), km(g.y[-1]), km(g.y[0])]
    d = dem.sel(x=g.x, y=g.y, method="nearest")
    ax.imshow(hillshade(d.values, res * stride), cmap="gray", extent=e)
    im = ax.imshow(g, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, extent=e, interpolation="nearest")
    if colorbar:
        ax.figure.colorbar(im, ax=ax, shrink=0.6, label=label)   # ax.figure: also works inside subfigures
    x0, y0 = to_utm.transform(IZAS["lon"], IZAS["lat"])
    ax.plot(km(x0), km(y0), "*", color="gold", mec="k", ms=10)
    ax.set_xticks([])
    return im


def izas_point_info(mp_i, station=IZAS):
    """
    The Izas points (station, elevation series, aspect rose) with their DEM parameters, indexed by point_name.
    Group and target are recovered from the point names written by izas_point_table().
    """
    info = mp_i.toposub.df_centroids.set_index("point_name")
    info["aspect_deg"] = compass_aspect(info.aspect)
    info["slope_deg"] = np.rad2deg(info.slope)

    def group_target(n):
        if n == station["name"]:
            return "station", np.nan
        kind, tgt = n.split("_", 1)
        return ("elevation_series", float(tgt)) if kind == "elev" else ("aspect_rose", float(DIRECTIONS[tgt]))

    info["group"], info["target"] = zip(*[group_target(n) for n in info.index])
    return info


def read_downscaled(downscaled_dir, pattern="down_pt_*.nc"):
    """
    All downscaled points of a project, in memory (dims point_name, time). Reading the files one by one is much faster
    than TopoPyScale's lazy open_mfdataset() + .load() when there are thousands of them (3000 files: ~1 min vs ~20).
    """
    files = sorted(glob.glob(f"{downscaled_dir}/{pattern}"))
    return xr.concat([xr.load_dataset(f) for f in files], dim="point_name", coords="different", compat="equals")


def load_results(mp):
    """
    Read the outputs of a finished project, like Topoclass.load_project() but with read_downscaled():
    df_centroids, ds_param (with the cluster labels), ds_solar and the downscaled points (returned).
    """
    out, f = Path(mp.config.outputs.path), mp.config.outputs.file
    mp.toposub.df_centroids = pd.read_pickle(out / f.df_centroids)
    mp.toposub.ds_param = xr.open_dataset(out / f.ds_param)
    mp.ds_solar = xr.open_dataset(out / f.ds_solar)
    mp.downscaled_pts = read_downscaled(out / "downscaled")
    return mp.downscaled_pts


def open_era5_plev(climate_dir):
    return xr.open_mfdataset(sorted(glob.glob(f"{climate_dir}/PLEV_*.nc")), combine="by_coords")


def toposcale_weights(lat, lon, x, y, era_lat, era_lon):
    """
    The 3 x 3 ERA5 cells TopoPyScale uses for a point (centred on the nearest cell) and their horizontal weights
    (inverse squared distance in the DEM projection, normalised), as in topo_scale_zarr.create_subset_indices()
    and downscale_atmo(). Returns a (latitude, longitude) DataArray of weights.
    """
    ilat, ilon = np.abs(era_lat - lat).argmin(), np.abs(era_lon - lon).argmin()
    la, lo = era_lat[ilat - 1:ilat + 2], era_lon[ilon - 1:ilon + 2]
    LO, LA = np.meshgrid(lo, la)
    X, Y = to_utm.transform(LO, LA)
    w = 1.0 / ((x - X) ** 2 + (y - Y) ** 2)
    return xr.DataArray(w / w.sum(), coords={"latitude": la, "longitude": lo}, dims=("latitude", "longitude"))


def plot_workflow(times=None, n_clusters=300, timestep="6h", tag=""):
    """
    Flowchart of the TopoPyScale steps used in the notebooks: TopoPyScale call, what it computes, output file.
    times: optional {label: seconds} (compute_times.json) to print the compute time of each step.
    timestep: ERA5 time step ("6h" or "1h"); tag: suffix of the labels of the steps that depend on it (3 and 5).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    times = times or {}
    def tmin(*keys):                                             # longest compute time among these labels
        ts = [times[k] for k in keys if k in times]
        return f"\n{max(ts) / 60:.1f} min" if ts else ""

    fig = plt.figure(figsize=(13, 8.6))
    ax = fig.add_axes([0.005, 0.0, 0.99, 0.94])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    W, XS, YT, YB, HT, HB = 0.225, (0.12, 0.375, 0.625, 0.88), 0.815, 0.345, 0.33, 0.43
    boxes = {   # key: (x centre, y centre, height, colour, title, body)
        "dem": (XS[0], YT, HT, "#e6e6e6", "DEM, 30 m",
                "Copernicus GLO-30\nUTM 30N\ntransect + 10 km\nbuffer"),
        "param": (XS[1], YT, HT, "#d9ead3", "1. Terrain",
                  "compute_dem_param()\nslope, aspect,\nsky-view factor\nper pixel\n→ ds_param.nc"
                  + tmin("1. Terrain: transect")),
        "sub": (XS[2], YT, HT, "#d9ead3", "2. TopoSUB",
                f"extract_topo_param()\n4.6 million pixels\n→ {n_clusters} clusters\n(or a list of points)\n→ df_centroids.pck"
                + tmin(f"2. TopoSUB: {n_clusters} clusters")),
        "solar": (XS[3], YT, HT, "#fff2cc", "3. Solar geometry",
                  "compute_solar_geometry()\nsun position per\ncluster and time step\n→ ds_solar.nc"
                  + tmin(f"3. Solar geometry: {n_clusters} clusters{tag}")),
        "era5": (XS[0], YB, HB, "#cfe2f3", "ERA5, " + {"6h": "6-hourly", "1h": "hourly"}.get(timestep, timestep),
                 "surface and\npressure levels\n→ ERA5.zarr"),
        "hor": (XS[1], YB, HB, "#d9ead3", "4. Horizons",
                "compute_horizon()\n36 azimuths per pixel\n→ da_horizon.nc" + tmin(*[k for k in times if k.startswith("4. Horizons")])),
        "scale": (XS[2], YB, HB, "#fce5cd", "5. TopoScale",
                  "downscale_climate()\nper cluster, from the\n3 × 3 ERA5 cells:\nT, q, wind interpolated\n"
                  "to the cluster elevation;\nSW: slope, shadows,\nsky view\n→ down_pt_*.nc"
                  + tmin(f"5. TopoScale: {n_clusters} clusters{tag}")),
        "out": (XS[3], YB, HB, "#ead1dc", "Results",
                "one time series\nper cluster\n→ maps: painted back\non the cluster pixels\n→ FSM snow model\n"
                "(notebook 2)"),
    }
    for x, y, h, c, title, body in boxes.values():
        ax.add_patch(FancyBboxPatch((x - W / 2, y - h / 2), W, h, boxstyle="round,pad=0.005,rounding_size=0.015",
                                    fc=c, ec="0.3", lw=1.2))
        ax.text(x, y + h / 2 - 0.02, title, ha="center", va="top", fontsize=16, weight="bold")
        ax.text(x, y + h / 2 - 0.075, body, ha="center", va="top", fontsize=13, linespacing=1.25)

    def arrow(p0, p1, ls="-", **kw):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.6, ls=ls,
                                                          mutation_scale=18, **kw))

    def label(x, y, text, **kw):
        ax.text(x, y, text, fontsize=12, style="italic", color="0.3", **kw)

    e = W / 2
    for a, b in zip(XS[:-1], XS[1:]):                            # DEM -> 1 -> 2 -> 3
        arrow((a + e, YT), (b - e, YT))
    arrow((XS[1] + e, YB), (XS[2] - e, YB))                      # horizons -> TopoScale
    arrow((XS[2] + e, YB), (XS[3] - e, YB))                      # TopoScale -> results
    arrow((XS[0] + 0.03, YT - HT / 2), (XS[1] - e, YB + 0.1))   # DEM -> horizons
    arrow((XS[2] - 0.03, YT - HT / 2), (XS[1] + e, YB + 0.12), ls="--")   # clusters -> horizons (sampling)
    label(XS[1] + 0.04, 0.6, "sampled at the\ncluster centroids", ha="left", va="center")
    arrow((XS[2] + 0.03, YT - HT / 2), (XS[2] + 0.03, YB + HB / 2))   # clusters -> TopoScale
    arrow((XS[3] - 0.03, YT - HT / 2), (XS[2] + e, YB + 0.14))        # solar -> TopoScale
    y0 = 0.05                                                      # ERA5 -> TopoScale, below the boxes
    ax.plot([XS[0], XS[0], XS[2] - 0.05], [YB - HB / 2, y0, y0], color="0.25", lw=1.6)
    arrow((XS[2] - 0.05, y0), (XS[2] - 0.05, YB - HB / 2))
    label(XS[1], y0 + 0.022, "ERA5 cells around each cluster", ha="center", va="center")
    fig.suptitle("TopoPyScale workflow (Topoclass methods; outputs in <project>/outputs)", fontsize=17)
    return fig
