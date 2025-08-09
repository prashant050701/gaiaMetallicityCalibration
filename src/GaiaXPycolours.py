import os, gc, re, numpy as np, pandas as pd
from gaiaxpy import generate, PhotometricSystem

XP_COEFFS_PATH = "/kaggle/input/gaiaxpcoefficients/XP_coeffs_all_deduped.csv"
GAIA_PATH      = "/kaggle/input/gaiadr3matchedlamostdr10/calibrationLAMOST-result.csv"
OUT_CSV        = "/kaggle/working/xp_colours_only.csv"
SKIP_LOG       = "/kaggle/working/skipped_sources.csv"

CHUNK_SIZE     = 200_000
USE_OBSERVED_G = True

wanted = ["Gaia_DR3_Vega", "JPLUS", "SDSS", "PanSTARRS1", "Stromgren"]
available_map = {ps.name: ps for ps in PhotometricSystem}
systems = [available_map[n] for n in wanted if n in available_map]
print("Will synthesise for systems:"); [print("   ", s.name) for s in systems]

gaia_g = (
    pd.read_csv(GAIA_PATH, usecols=["source_id", "phot_g_mean_mag"])
      .drop_duplicates("source_id")
      .set_index("source_id")["phot_g_mean_mag"])

num_pat = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

def parse_coeff(s):
    if not isinstance(s, str): return None
    vals = num_pat.findall(s)
    if not vals: return None
    return np.array([float(v) for v in vals], dtype=float)

INT_STUBS  = ["bp_n_parameters","rp_n_parameters","bp_n_transits","rp_n_transits","bp_n_relevant_bases","rp_n_relevant_bases"]
FLOAT_STUBS= ["bp_standard_deviation","rp_standard_deviation","bp_chi_squared","rp_chi_squared"]
OBJ_STUBS  = ["bp_coefficient_errors","rp_coefficient_errors","bp_coefficient_correlations","rp_coefficient_correlations"]

def ensure_required_columns(df):
    df["bp_n_parameters"] = df["bp_coefficients"].map(lambda a: 0 if a is None else int(a.shape[0]))
    df["rp_n_parameters"] = df["rp_coefficients"].map(lambda a: 0 if a is None else int(a.shape[0]))
    for c in ["bp_n_transits","rp_n_transits","bp_n_relevant_bases","rp_n_relevant_bases"]:
        if c not in df.columns: df[c] = 0
    for c in ["bp_standard_deviation","rp_standard_deviation","bp_chi_squared","rp_chi_squared"]:
        if c not in df.columns: df[c] = 0.0
    for c in ["bp_coefficient_errors","rp_coefficient_errors","bp_coefficient_correlations","rp_coefficient_correlations"]:
        if c not in df.columns: df[c] = None
    return df

def patch_gaiaxpy():
    from gaiaxpy.generator import synthetic_photometry_generator as _spg
    from gaiaxpy.spectrum.absolute_sampled_spectrum import AbsoluteSampledSpectrum
    from gaiaxpy.spectrum.photometric_absolute_sampled_spectrum import PhotometricAbsoluteSampledSpectrum
    import gaiaxpy.core.generic_functions as gf
    import gaiaxpy.generator.generator as gen

    _spg.get_covariance_matrix = lambda row, band: np.zeros((1,1), dtype=float)
    AbsoluteSampledSpectrum.get_available_bands = staticmethod(lambda xp: [b for b in xp.keys() if xp[b] is not None])

    def _zero_error(self, spectra_covariance, design_matrix, stdev):
        n = design_matrix.shape[1] if hasattr(design_matrix,"shape") and len(design_matrix.shape)==2 else len(design_matrix)
        return np.zeros(n, dtype=float)
    PhotometricAbsoluteSampledSpectrum._sample_error = _zero_error

    gf.cast_output  = lambda df: df
    gen.cast_output = lambda df: df

patch_gaiaxpy()

def _norm(c):
    return (c.replace(" ", "").replace("-", "").replace("/", "").replace("__", "_"))

def build_colours(df_phot, anchor_g_series):
    df = df_phot.copy()
    df.columns = [_norm(c) for c in df.columns]
    g_cols = [c for c in df.columns if c.lower().endswith("_mag_g")]
    if not g_cols: raise RuntimeError("No synthetic G magnitude column found.")
    prefer = [c for c in g_cols if "gaiadr3vega" in c.lower()]
    g_syn_col = prefer[0] if prefer else g_cols[0]
    g_syn = df[g_syn_col].values
    obs_g = anchor_g_series.reindex(df["source_id"]).values
    g_anchor = np.where(np.isfinite(obs_g), obs_g, g_syn) if USE_OBSERVED_G else g_syn
    mag_cols = [c for c in df.columns if "_mag_" in c and c != g_syn_col]
    out = {"source_id": df["source_id"].values}
    for c in mag_cols:
        out["col_G_minus_" + c.replace("_mag_", "_")] = (g_anchor - df[c].values).astype(np.float32)
    return pd.DataFrame(out)

def has_any_coeff_str(x):
    return isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]")

if os.path.exists(OUT_CSV): os.remove(OUT_CSV)
if os.path.exists(SKIP_LOG): os.remove(SKIP_LOG)

first = True
total_rows = 0
total_skipped = 0
skip_header_written = False

reader = pd.read_csv(
    XP_COEFFS_PATH,
    usecols=["source_id","bp_coefficients","rp_coefficients"],
    chunksize=CHUNK_SIZE)

print("Starting chunks...")
for i, chunk in enumerate(reader, 1):
    chunk["source_id"] = pd.to_numeric(chunk["source_id"], errors="coerce").astype("Int64")
    chunk["bp_coefficients"] = chunk["bp_coefficients"].astype(str)
    chunk["rp_coefficients"] = chunk["rp_coefficients"].astype(str)

    bad = (~chunk["bp_coefficients"].map(has_any_coeff_str)) & (~chunk["rp_coefficients"].map(has_any_coeff_str))
    if bad.any():
        ids = chunk.loc[bad, "source_id"].dropna()
        mode = "a" if skip_header_written else "w"
        ids.to_frame("source_id").to_csv(SKIP_LOG, index=False, mode=mode, header=not skip_header_written)
        skip_header_written = True
        total_skipped += int(bad.sum())
        print(f"[chunk {i}] skipped {int(bad.sum())} (cum {total_skipped}) with neither BP nor RP")
    chunk = chunk.loc[~bad].copy()
    if chunk.empty: continue

    bad_id = chunk["source_id"].isna()
    if bad_id.any():
        ids = chunk.loc[bad_id, "source_id"]
        mode = "a" if skip_header_written else "w"
        pd.DataFrame({"source_id": ids.dropna().astype("Int64")}).to_csv(SKIP_LOG, index=False, mode=mode, header=not skip_header_written)
        skip_header_written = True
        total_skipped += int(bad_id.sum())
        print(f"[chunk {i}] skipped {int(bad_id.sum())} (cum {total_skipped}) invalid source_id")
        chunk = chunk.loc[~bad_id]

    chunk["bp_coefficients"] = chunk["bp_coefficients"].map(parse_coeff)
    chunk["rp_coefficients"] = chunk["rp_coefficients"].map(parse_coeff)

    both_none = chunk["bp_coefficients"].isna() & chunk["rp_coefficients"].isna()
    if both_none.any():
        ids = chunk.loc[both_none, "source_id"]
        mode = "a" if skip_header_written else "w"
        ids.to_frame("source_id").to_csv(SKIP_LOG, index=False, mode=mode, header=not skip_header_written)
        skip_header_written = True
        total_skipped += int(both_none.sum())
        print(f"[chunk {i}] skipped {int(both_none.sum())} (cum {total_skipped}) unparsable coeff arrays")
        chunk = chunk.loc[~both_none]

    if chunk.empty: continue

    chunk = ensure_required_columns(chunk)

    chunk["source_id"] = chunk["source_id"].astype("int64")

    try:
        phot = generate(
            chunk,
            photometric_system=systems,
            error_correction=False,
            save_file=False
        )
    except Exception as e:
        print(f"[chunk {i}] generate() failed, isolating bad rows: {e}")
        ok_parts = []
        for sub_idx, sub in enumerate(np.array_split(chunk, 8)):
            try:
                sub_phot = generate(sub, photometric_system=systems, error_correction=False, save_file=False)
                ok_parts.append(sub_phot)
            except Exception as e2:
                bad_ids_local = []
                for rid, row in sub.iterrows():
                    try:
                        _ = generate(sub.loc[[rid]], photometric_system=systems, error_correction=False, save_file=False)
                    except Exception:
                        bad_ids_local.append(int(row["source_id"]))
                if bad_ids_local:
                    mode = "a" if skip_header_written else "w"
                    pd.DataFrame({"source_id": bad_ids_local}).to_csv(SKIP_LOG, index=False, mode=mode, header=not skip_header_written)
                    skip_header_written = True
                    total_skipped += len(bad_ids_local)
                    preview = ", ".join(map(str, bad_ids_local[:5]))
                    extra = "" if len(bad_ids_local) <= 5 else " ..."
                    print(f"[chunk {i}] skipped {len(bad_ids_local)} rows in subchunk {sub_idx}; first 5: {preview}{extra}")

        if not ok_parts:
            continue
        phot = pd.concat(ok_parts, ignore_index=True)

    colours = build_colours(phot, gaia_g)
    colours.to_csv(OUT_CSV, index=False, mode="w" if first else "a", header=first)
    first = False
    total_rows += len(colours)

    del chunk, phot, colours
    gc.collect()

print(f"\nSaved {total_rows:,} rows of colours to: {OUT_CSV}")
print(f"Total skipped sources: {total_skipped:,}")
if total_skipped: print(f"Logged to: {SKIP_LOG}")
