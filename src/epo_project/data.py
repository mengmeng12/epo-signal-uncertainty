"""Step 1 (industry returns) and Step 2 (momentum and risk inputs).

Step 1, prepare_returns():
    data/raw/<source files>, data/raw/source_manifest.json
    data/processed/returns.csv       one row per (date, industry): total, rf, excess
    data/processed/eligibility.csv   one row per (date, industry): history available at t

Step 2, build_inputs():
    data/interim/momentum.csv        trailing 12-month return and momentum score z
    data/interim/risk_models.npz     Sigma_hat_t and Q_t, padded to all 49 industries
    data/interim/risk_index.csv      formation date, window, industry order per matrix

Timing: everything indexed by a month-end t uses returns of months <= t only.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as cfg


# ====================================================================== Step 1

def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_sources(force: bool = False) -> dict:
    """Download the French zip files into data/raw (kept unchanged) and record
    their provenance. Existing files are reused unless force=True."""
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.RAW_DIR / "source_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    manifest = {}
    for key, (zip_name, csv_name, _) in cfg.SOURCES.items():
        zip_path, csv_path = cfg.RAW_DIR / zip_name, cfg.RAW_DIR / csv_name
        url = cfg.FRENCH_BASE_URL + zip_name
        if force or not zip_path.exists():
            with urllib.request.urlopen(url) as resp:
                zip_path.write_bytes(resp.read())
            downloaded = datetime.now(timezone.utc).isoformat(timespec="seconds")
        else:
            downloaded = old.get(key, {}).get("downloaded_utc", "unknown")
        payload = zip_path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            member = next(n for n in zf.namelist() if n.lower() == csv_name.lower())
            csv_bytes = zf.read(member)
        csv_path.write_bytes(csv_bytes)

        vintage = re.search(r"created using the (\d{6}) CRSP database", csv_bytes.decode("latin-1"))
        manifest[key] = {
            "url": url,
            "downloaded_utc": downloaded,
            "zip_file": zip_name,
            "zip_sha256": _sha256(payload),
            "csv_file": csv_name,
            "csv_sha256": _sha256(csv_bytes),
            "crsp_vintage": vintage.group(1) if vintage else None,
        }
    return manifest


def read_french_table(csv_name: str, marker: str) -> pd.DataFrame:
    """Read one monthly table from a French CSV (several tables are stacked in
    one file). Returns decimals indexed by month-end; missing codes -> NaN."""
    lines = (cfg.RAW_DIR / csv_name).read_text(encoding="latin-1").splitlines()
    start = next(i for i, line in enumerate(lines) if marker in line)
    header = start
    while not lines[header].startswith(","):
        header += 1
    columns = [c.strip() for c in lines[header].split(",")[1:]]

    dates, rows = [], []
    for line in lines[header + 1:]:
        parts = line.split(",")
        token = parts[0].strip()
        if not (token.isdigit() and len(token) == 6):
            break
        dates.append(token)
        rows.append([float(v) for v in parts[1:]])

    index = pd.to_datetime(dates, format="%Y%m") + pd.offsets.MonthEnd(0)
    table = pd.DataFrame(rows, columns=columns, index=index)
    table = table.mask(table <= cfg.MISSING_CODE_THRESHOLD)   # missing stays missing
    table.index.name = "date"
    return table / 100.0


def _consecutive_months(available: pd.DataFrame) -> pd.DataFrame:
    """Number of consecutive non-missing months ending at each date (0 if missing)."""
    out = np.zeros(available.shape, dtype=int)
    values = available.to_numpy()
    for j in range(values.shape[1]):
        run = 0
        for i in range(values.shape[0]):
            run = run + 1 if values[i, j] else 0
            out[i, j] = run
    return pd.DataFrame(out, index=available.index, columns=available.columns)


def prepare_returns(force_download: bool = False) -> dict:
    """Step 1: raw files, returns.csv, eligibility.csv and source_manifest.json."""
    manifest = download_sources(force=force_download)

    _, ind_csv, ind_marker = cfg.SOURCES["industry49_monthly"]
    _, ff3_csv, ff3_marker = cfg.SOURCES["ff3_monthly"]
    total = read_french_table(ind_csv, ind_marker)
    rf = read_french_table(ff3_csv, ff3_marker)["RF"].reindex(total.index)
    if rf.isna().any():
        raise ValueError("risk-free rate missing for some industry months")
    excess = total.sub(rf, axis=0)

    long = (total.stack(future_stack=True).rename("total_return").to_frame()
            .join(excess.stack(future_stack=True).rename("excess_return")))
    long.index.names = ["date", "industry"]
    long = long.reset_index()
    long["rf"] = long["date"].map(rf)
    long = long[["date", "industry", "total_return", "rf", "excess_return"]]

    run = _consecutive_months(total.notna())
    elig = run.stack().rename("consecutive_months").reset_index()
    elig.columns = ["date", "industry", "consecutive_months"]
    elig["has_return"] = elig["consecutive_months"] > 0
    elig["momentum_ready"] = elig["consecutive_months"] >= cfg.MOMENTUM_MONTHS
    elig["risk_ready"] = elig["consecutive_months"] >= cfg.RISK_WINDOW
    elig["benchmark_eligible"] = elig["momentum_ready"] & elig["risk_ready"]

    # completion criteria
    assert not long.duplicated(["date", "industry"]).any(), "duplicate date-industry rows"
    assert long["total_return"].abs().max() < 5, "returns do not look like decimals"
    assert not elig.duplicated(["date", "industry"]).any()

    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fmt = {"date_format": "%Y-%m-%d", "index": False}
    long.to_csv(cfg.PROCESSED_DIR / "returns.csv", float_format="%.6f", **fmt)
    elig.to_csv(cfg.PROCESSED_DIR / "eligibility.csv", **fmt)

    manifest["_summary"] = {
        "first_month": str(total.index.min().date()),
        "last_common_month": str(total.index.max().date()),
        "industries": list(total.columns),
        "missing_industry_months": int(total.isna().sum().sum()),
        "rf_source": "F-F_Research_Data_Factors (monthly RF, one-month T-bill)",
        "eligibility_rule": (f"momentum_ready: >= {cfg.MOMENTUM_MONTHS} consecutive months ending at t; "
                             f"risk_ready: >= {cfg.RISK_WINDOW}; benchmark_eligible: both"),
    }
    (cfg.RAW_DIR / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest["_summary"]


def load_returns(kind: str = "excess_return") -> pd.DataFrame:
    """Wide date x industry table of total_return, rf or excess_return."""
    long = pd.read_csv(cfg.PROCESSED_DIR / "returns.csv", parse_dates=["date"])
    if kind == "rf":
        return long.drop_duplicates("date").set_index("date")["rf"]
    wide = long.pivot(index="date", columns="industry", values=kind)
    return wide[long["industry"].drop_duplicates()]          # keep French column order


def load_eligibility(column: str = "benchmark_eligible") -> pd.DataFrame:
    long = pd.read_csv(cfg.PROCESSED_DIR / "eligibility.csv", parse_dates=["date"])
    wide = long.pivot(index="date", columns="industry", values=column)
    return wide[long["industry"].drop_duplicates()]


# ====================================================================== Step 2

def momentum_score(trailing: pd.Series) -> pd.Series:
    """Demean across industries, then scale the positive and negative legs to
    sum to +1 and -1 (the paper's cross-sectional momentum signal)."""
    d = trailing - trailing.mean()
    pos, neg = d[d > 0].sum(), -d[d < 0].sum()
    return d.where(d <= 0, d / pos).where(d >= 0, d / neg)


def build_inputs() -> dict:
    """Step 2: momentum scores and the 60-month risk models for every month-end t."""
    total = load_returns("total_return")
    excess = load_returns("excess_return")
    eligible = load_eligibility("benchmark_eligible")
    industries = list(total.columns)
    n = len(industries)

    trailing = (1 + total).rolling(cfg.MOMENTUM_MONTHS, min_periods=cfg.MOMENTUM_MONTHS).apply(np.prod, raw=True) - 1

    mom_rows, index_rows, sigmas, qs = [], [], [], []
    for i, t in enumerate(total.index):
        members = [c for c in industries if eligible.at[t, c]]
        if len(members) < cfg.MIN_INDUSTRIES:
            continue
        z = momentum_score(trailing.loc[t, members])

        window = excess.iloc[i - cfg.RISK_WINDOW + 1:i + 1][members]
        assert window.notna().all().all() and len(window) == cfg.RISK_WINDOW
        sigma_hat = window.cov(ddof=1).to_numpy()
        v = np.diag(np.diag(sigma_hat))
        q = (1 - cfg.PRELIM_SHRINK) * sigma_hat + cfg.PRELIM_SHRINK * v

        # completion criteria
        assert abs(z[z > 0].sum() - 1) < 1e-10 and abs(z[z < 0].sum() + 1) < 1e-10
        np.linalg.cholesky(q)                                   # raises if not positive definite
        assert np.allclose(np.diag(q), np.diag(sigma_hat))

        pos = [industries.index(c) for c in members]
        pad_s, pad_q = np.full((n, n), np.nan), np.full((n, n), np.nan)
        pad_s[np.ix_(pos, pos)], pad_q[np.ix_(pos, pos)] = sigma_hat, q
        sigmas.append(pad_s)
        qs.append(pad_q)

        for c in members:
            mom_rows.append((t, c, trailing.at[t, c], z[c]))
        index_rows.append({"formation_date": t, "window_start": window.index[0],
                           "last_input_date": window.index[-1], "n_industries": len(members),
                           "min_eigenvalue_q": float(np.linalg.eigvalsh(q).min()),
                           "industries": "|".join(members)})

    cfg.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    fmt = {"date_format": "%Y-%m-%d", "index": False}
    mom = pd.DataFrame(mom_rows, columns=["date", "industry", "trailing_12m_return", "z"])
    mom.to_csv(cfg.INTERIM_DIR / "momentum.csv", float_format="%.10g", **fmt)
    risk_index = pd.DataFrame(index_rows)
    risk_index.to_csv(cfg.INTERIM_DIR / "risk_index.csv", **fmt)
    np.savez_compressed(cfg.INTERIM_DIR / "risk_models.npz",
                        dates=risk_index["formation_date"].dt.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
                        industries=np.array(industries), sigma_hat=np.stack(sigmas), q=np.stack(qs))
    return {"formation_dates": len(risk_index),
            "first": str(risk_index["formation_date"].min().date()),
            "last": str(risk_index["formation_date"].max().date()),
            "industries_min_max": (int(risk_index["n_industries"].min()), int(risk_index["n_industries"].max())),
            "smallest_eigenvalue_q": float(risk_index["min_eigenvalue_q"].min())}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Momentum scores (wide, NaN outside the universe), the risk index, and the
    risk models as {formation_date: (industry list, Sigma_hat, Q)}."""
    mom = pd.read_csv(cfg.INTERIM_DIR / "momentum.csv", parse_dates=["date"])
    z = mom.pivot(index="date", columns="industry", values="z")
    risk_index = pd.read_csv(cfg.INTERIM_DIR / "risk_index.csv",
                             parse_dates=["formation_date", "window_start", "last_input_date"])
    with np.load(cfg.INTERIM_DIR / "risk_models.npz", allow_pickle=False) as arrays:
        industries = list(arrays["industries"])
        sigma_hat, q = arrays["sigma_hat"], arrays["q"]          # decompress once
    models = {}
    for k, row in risk_index.iterrows():
        members = row["industries"].split("|")
        pos = [industries.index(c) for c in members]
        models[row["formation_date"]] = (members, sigma_hat[k][np.ix_(pos, pos)], q[k][np.ix_(pos, pos)])
    return z, risk_index, models


if __name__ == "__main__":
    print("Step 1:", prepare_returns())
    print("Step 2:", build_inputs())
