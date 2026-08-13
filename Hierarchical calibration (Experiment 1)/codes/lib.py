"""item1 shared library.

Three things live here and nowhere else:

  1. The cut-mode extension (Part C).  `analysis.analyse` is not edited,
     instead
     `install_cut_modes()` replaces the module-level function that `analyse`
     calls internally, with a dispatcher that reproduces the original code path
     bitwise when cut_mode == "mad_raw".  Every cut therefore still runs INSIDE
     `analyse`, identically for real data, nulls and injections.

  2. Configuration, cadence and injection construction.

  3. The batch runner: heartbeat, thermal pauses, per-row flush, resume.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ITEM1 = os.path.dirname(HERE)
HCAL = os.path.dirname(ITEM1)
PARENT = os.path.dirname(HCAL)
# APPEND, never insert at 0: hcal/code contains modules whose names collide with
# item1's own (make_figs.py, make_config.py).  Inserting at the front silently
# shadowed item1's modules with the parent run's, which the dry run caught.
sys.path.append(os.path.join(HCAL, "code"))

import analysis  # noqa: E402
import gen       # noqa: E402

RESULTS = os.path.join(ITEM1, "results")
FIGS = os.path.join(ITEM1, "figs")
SPEC_VERSION = "hcal-item1-v1"
SEED_BASE = 20260811

PART_INDEX = dict(C=3, D=4, E=5, F=6, G=7, H=8)


def seed_for(part_index, config_index, i):
    return SEED_BASE + 1_000_000 * part_index + 1000 * config_index + i


_ORIGINAL_CUTS = analysis.apply_quality_cuts


@dataclass(frozen=True)
class SpecExt(analysis.Spec):
    """`analysis.Spec` plus the cut mode.  Frozen, like its parent."""
    cut_mode: str = "mad_raw"


def _robust_sd(x):
    """1.4826 * MAD, the consistent-for-Gaussian robust standard deviation."""
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def apply_quality_cuts_ext(t, y, sig, b, catflags, spec):
    """Dispatcher installed in place of `analysis.apply_quality_cuts`.

    Cuts 1 (catflags) and 2 (magerr) are unchanged in every mode, both are
    exact no-ops on synthetic data.  Only cut 3 differs:

      mad_raw     |resid from a running median| > 5 * MAD          (as-built,
                  unscaled MAD, so a 3.37 sigma clip for Gaussian residuals)
      mad_scaled  |resid from a running median| > 5 * 1.4826 * MAD (a true 5 sigma)
      sigma_clip  |y - median(y_band)| > 5 * 1.4826 * MAD(y_band),
                  no running median at all, which is what separates
                  "the scaling was wrong" from "the running median was wrong"
    """
    mode = getattr(spec, "cut_mode", "mad_raw")
    if mode == "mad_raw":
        return _ORIGINAL_CUTS(t, y, sig, b, catflags, spec)

    n = t.size
    keep = np.ones(n, dtype=bool)
    counts = dict(n_removed_catflags=0, n_removed_magerr=0, n_removed_mad=0)

    if spec.cut_catflags:
        bad = np.asarray(catflags) != 0
        counts["n_removed_catflags"] = int(np.count_nonzero(bad & keep))
        keep &= ~bad

    bad = sig > spec.cut_magerr_max
    counts["n_removed_magerr"] = int(np.count_nonzero(bad & keep))
    keep &= ~bad

    idx = np.flatnonzero(keep)
    if idx.size >= 3:
        tk, yk, bk = t[idx], y[idx], b[idx]
        if mode == "mad_scaled":
            rm = analysis._running_median_per_band(
                tk, yk, bk, spec.cut_running_median_window_d)
            resid = yk - rm
        elif mode == "sigma_clip":
            resid = np.empty_like(yk)
            for lab in np.unique(bk):
                m = bk == lab
                resid[m] = yk[m] - np.median(yk[m])
        else:
            raise ValueError(f"unknown cut_mode {mode!r}")
        drop_local = np.zeros(idx.size, dtype=bool)
        for lab in np.unique(bk):
            m = bk == lab
            r = resid[m]
            s = _robust_sd(r)
            if s > 0.0:
                drop_local[m] = np.abs(r) > spec.cut_mad_n * s
        counts["n_removed_mad"] = int(np.count_nonzero(drop_local))
        keep[idx[drop_local]] = False

    counts["n_removed_total"] = int(n - np.count_nonzero(keep))
    return keep, counts


_INSTALLED = False


def install_cut_modes():
    """Idempotent.  Must be in force before any analyse() call in item1."""
    global _INSTALLED
    if not _INSTALLED:
        analysis.apply_quality_cuts = apply_quality_cuts_ext
        _INSTALLED = True


def cut_impl_name():
    """Which cut function analyse() will actually call.  Recorded on every row
    so that a silently unpatched worker can never go unnoticed again."""
    return analysis.apply_quality_cuts.__name__


def assert_cut_mode_active(spec):
    """Fail loudly if a non-default cut mode was requested but the dispatcher
    is not installed.  Without this check a worker process silently falls
    back to the default cut mode and reports results as if it had not."""
    if getattr(spec, "cut_mode", "mad_raw") != "mad_raw" and not _INSTALLED:
        raise RuntimeError(
            f"cut_mode={spec.cut_mode!r} requested but the cut-mode dispatcher "
            f"is not installed in this process, analyse() would silently run "
            f"mad_raw")


# Install on IMPORT, not merely when a script's __main__ block runs.  joblib
# pickles a worker defined in a __main__ module BY VALUE, so that module's
# top-level statements never execute in the child process, only imported
# modules like this one do.  Installing here is what makes the child honour the
# cut mode at all.
install_cut_modes()


def load_hcal_config():
    with open(os.path.join(HCAL, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def load_item1_config():
    with open(os.path.join(ITEM1, "config.json"), encoding="utf-8") as f:
        return json.load(f)


FIDUCIAL_CADENCE = dict(
    bands=["g", "r"], baseline_d=2000.0, t_start=0.0,
    season_obs_d=240.0, season_gap_d=125.0,
    cadence_g_d=3.0, cadence_r_d=3.0, r_offset_d=1.0,
    jitter_halfwidth_d=0.5, drop_fraction=0.15,
    sigma_err_min_mag=0.015, sigma_err_max_mag=0.025,
)
FIDUCIAL_DUTY = 240.0 / 365.0
SEASON_PERIOD_D = 365.0


def make_cadence(baseline_d=2000.0, dt_d=3.0, n_bands=2, duty_cycle=FIDUCIAL_DUTY):
    """Fiducial season model with baseline, sampling interval, bands and duty
    cycle as free parameters.  Everything else is the fiducial cadence."""
    cad = dict(FIDUCIAL_CADENCE)
    cad["baseline_d"] = float(baseline_d)
    cad["cadence_g_d"] = float(dt_d)
    cad["cadence_r_d"] = float(dt_d)
    cad["season_obs_d"] = float(duty_cycle) * SEASON_PERIOD_D
    cad["season_gap_d"] = SEASON_PERIOD_D - cad["season_obs_d"]
    cad["duty_cycle"] = float(duty_cycle)
    cad["bands"] = ["g", "r"] if n_bands == 2 else ["g"]
    cad["n_bands"] = int(n_bands)
    return cad


def make_grid(baseline_d, P_n=160, P_min_d=60.0, P_max_frac=2.5, spacing="log"):
    return dict(n=int(P_n), min_d=float(P_min_d),
                max_d=float(baseline_d) / float(P_max_frac), spacing=spacing)


ETA_GRID = dict(min=-0.60, max=0.60, step=0.05, n=25, zero_on_grid=True)


def make_entry(config_index, label, baseline_d=2000.0, dt_d=3.0, n_bands=2,
               duty_cycle=FIDUCIAL_DUTY, tau_d=320.0, sigma_ratio=3.0,
               grid=None, axis="", level=""):
    cad = make_cadence(baseline_d, dt_d, n_bands, duty_cycle)
    sigma_phot = 0.020
    target_var = (sigma_ratio * sigma_phot) ** 2
    return dict(
        config_index=int(config_index), label=label, axis=axis, level=level,
        baseline_d=float(baseline_d), n_bands=int(n_bands),
        duty_cycle=float(duty_cycle), sampling_dt_d=float(dt_d),
        tau_over_T=float(tau_d) / float(baseline_d), tau_d=float(tau_d),
        sigma_ratio=float(sigma_ratio),
        target_in_window_variance_mag2=float(target_var),
        cadence=cad,
        grid_P=grid if grid is not None else make_grid(baseline_d),
        grid_eta=dict(ETA_GRID),
    )


def build_spec(entry, cut_mode="mad_raw", window_d=50.0, freq_uniform=False):
    """Build a SpecExt from an entry.  `freq_uniform` is Part D's alternative
    node spacing: uniform in frequency rather than in log P, same node count."""
    cfg = load_hcal_config()
    base = analysis.Spec.from_config(cfg, entry)
    d = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    d["cut_running_median_window_d"] = float(window_d)
    d["label"] = entry.get("label", "")
    return SpecExt(cut_mode=cut_mode, **d)


_ORIGINAL_BUILD_GRID = analysis.build_grid
_FREQ_UNIFORM = {"on": False}


def _build_grid_freq_uniform(spec):
    f = np.linspace(1.0 / spec.P_max_d, 1.0 / spec.P_min_d, spec.P_n)
    P = 1.0 / f[::-1]
    step = (spec.eta_max - spec.eta_min) / (spec.eta_n - 1)
    eta = np.round(np.arange(spec.eta_min, spec.eta_max + 0.5 * step, step), 10)
    assert eta.size == spec.eta_n
    assert np.isclose(eta, 0.0).sum() == 1
    return P, eta


def set_grid_spacing(freq_uniform):
    """Switch the period-node spacing globally.  Recorded in config.json."""
    _FREQ_UNIFORM["on"] = bool(freq_uniform)
    analysis.build_grid = (_build_grid_freq_uniform if freq_uniform
                           else _ORIGINAL_BUILD_GRID)


class GenCache:
    """One Generator per (entry signature).  Building a Generator is O(n^2) in
    epochs (the in-window variance factor), so it must not be repeated."""

    def __init__(self, cadence_seed=20260807):
        self.cadence_seed = int(cadence_seed)
        self._d = {}

    @staticmethod
    def key(entry):
        c = entry["cadence"]
        return (round(c["baseline_d"], 6), round(c["cadence_g_d"], 8),
                tuple(c["bands"]), round(c["season_obs_d"], 6),
                round(entry["tau_d"], 6),
                round(entry["target_in_window_variance_mag2"], 12))

    def get(self, entry):
        k = self.key(entry)
        g = self._d.get(k)
        if g is None:
            g = gen.Generator(entry, self.cadence_seed)
            self._d[k] = g
        return g


def inject_chirp(g, times, bands, P_ref_d, eta, amp1, amp2, phases):
    """Add the two-harmonic chirp.  `phases` is {band: (ph1, ph2)}, which is how
    the free-phase two-band structure is expressed: same amplitudes, independent
    phase per band.  A single-phase injection passes the same phase for both."""
    y = np.zeros(times.size, dtype=float)
    for lab, (ph1, ph2) in phases.items():
        m = np.asarray(bands) == lab
        if not m.any():
            continue
        y[m] = gen.chirp_signal(times[m], P_ref_d, eta, amp1, amp2, ph1, ph2,
                                g.t_ref, g.T_span)
    return y


def sigma_eff(g, path):
    """SNR convention: quadrature sum of the median photometric error and the
    background's realised in-window RMS.  Declared in config.json."""
    return float(np.hypot(np.median(g.sig), np.std(path, ddof=1)))


class Heartbeat:
    def __init__(self, part, every=10):
        self.part = part
        self.every = int(every)
        self.t0 = time.time()
        self.n = 0
        self.json = os.path.join(ITEM1, "HEARTBEAT.json")
        self.log = os.path.join(ITEM1, "HEARTBEAT.log")

    def beat(self, config_index, i, rows_written, force=False):
        self.n += 1
        if not force and (self.n % self.every) != 0:
            return
        el = time.time() - self.t0
        d = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                 spec_version=SPEC_VERSION, part=self.part,
                 config_index=config_index, i=i, rows_written=int(rows_written),
                 elapsed_s=round(el, 2),
                 mean_s_per_fit=round(el / max(self.n, 1), 4),
                 fits_this_session=self.n)
        tmp = self.json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, self.json)
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")


def existing_keys(path, cols=("part", "config_index", "i")):
    """Resume support: the (part, config_index, i) triples already on disk."""
    if not os.path.exists(path):
        return set()
    import pandas as pd
    try:
        df = pd.read_csv(path, usecols=list(cols))
    except Exception:
        return set()
    return set(zip(*[df[c].tolist() for c in cols]))


class RowWriter:
    """One row per fit, flushed immediately.  A kill leaves whole rows."""

    def __init__(self, path, fields):
        self.path = path
        self.fields = list(fields)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        if not new:
            # Appending with a different field list writes values in THIS
            # object's order under the file's existing header, silently
            # shifting every column.  That happened once, to Part D's G7 rows.
            with open(path, newline="", encoding="utf-8") as fh:
                existing = next(csv.reader(fh), None)
            if existing is not None and existing != self.fields:
                raise ValueError(
                    f"{os.path.basename(path)} already exists with a different "
                    f"header.\n  on disk: {existing}\n  requested: {self.fields}\n"
                    f"Appending would misalign every column.  Write to a new "
                    f"file, or migrate the existing one first.")
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.fields,
                                extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.f.flush()
        self.n = 0

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()
        os.fsync(self.f.fileno())
        self.n += 1

    def close(self):
        self.f.close()


def run_batch(tasks, worker, out_path, fields, part, n_jobs=2,
              batch_size=None, pause_between_batches_s=5.0,
              pause_every=300, pause_long_s=90.0, heartbeat_every=10,
              resume=True):
    """Run `worker(task) -> dict (or list of dicts)` over `tasks`.

    Thermal discipline, after four shutdowns: n_jobs 2 by default and never
    above 4, BLAS pinned to one thread, a 5 s pause between joblib batches, a
    90 s pause every 300 fits, one row flushed per fit.
    """
    from joblib import Parallel, delayed

    n_jobs = max(1, min(int(n_jobs), 4))
    if batch_size is None:
        batch_size = 2 * n_jobs

    done = existing_keys(out_path) if resume else set()
    todo = [t for t in tasks
            if (part, int(t["config_index"]), int(t["i"])) not in done]
    print(f"[{part}] {len(tasks)} tasks, {len(done)} already on disk, "
          f"{len(todo)} to run, n_jobs={n_jobs}")
    print(f"RESUME COMMAND:\n"
          f"    python "
          f"{os.path.basename(sys.argv[0])}\n")
    if not todo:
        return 0

    wr = RowWriter(out_path, fields)
    hb = Heartbeat(part, every=heartbeat_every)
    n_since_long = 0
    t0 = time.time()
    try:
        with Parallel(n_jobs=n_jobs, backend="loky", batch_size=1,
                      max_nbytes=None) as pool:
            for k in range(0, len(todo), batch_size):
                chunk = todo[k:k + batch_size]
                out = pool(delayed(worker)(t) for t in chunk)
                for t, res in zip(chunk, out):
                    for row in (res if isinstance(res, list) else [res]):
                        if row is not None:
                            wr.write(row)
                    hb.beat(int(t["config_index"]), int(t["i"]), wr.n)
                n_since_long += len(chunk)
                el = time.time() - t0
                rate = wr.n / el * 60 if el > 0 else 0
                print(f"  [{part}] {wr.n}/{len(todo)} rows  "
                      f"{el/60:.1f} min  {rate:.1f} rows/min", flush=True)
                if n_since_long >= pause_every:
                    print(f"  [{part}] thermal pause {pause_long_s:.0f} s",
                          flush=True)
                    time.sleep(pause_long_s)
                    n_since_long = 0
                elif k + batch_size < len(todo):
                    time.sleep(pause_between_batches_s)
    finally:
        hb.beat(-1, -1, wr.n, force=True)
        wr.close()
    print(f"[{part}] done: {wr.n} rows in {(time.time()-t0)/60:.1f} min")
    return wr.n


def boot_pct(x, q, n_boot=2000, seed=12345):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    bs = np.percentile(x[idx], q, axis=1)
    return (float(np.percentile(x, q)), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))
