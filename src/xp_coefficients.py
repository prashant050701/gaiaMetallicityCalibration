import os, time, pandas as pd, pyvo
tap = pyvo.dal.TAPService("https://gaia.aip.de/tap")
ids = pd.read_csv("gaia_ids.csv", dtype={"source_id": str})["source_id"].astype(int).tolist()
batch = 1000
out = "XP_coeffs_all.csv"
if os.path.exists(out):
    os.remove(out)
for start in range(0, len(ids), batch):
    chunk = ids[start:start + batch]
    idlist = ",".join(map(str, chunk))
    adql = f"""
    SELECT source_id, bp_coefficients, rp_coefficients
    FROM gaiadr3.xp_continuous_mean_spectrum
    WHERE source_id IN ({idlist})
    """
    for _ in range(3):
        try:
            df = tap.run_sync(adql, maxrec=batch).to_table().to_pandas()
            hdr = not os.path.exists(out)
            df.to_csv(out, mode="a", header=hdr, index=False)
            break
        except Exception:
            time.sleep(5)
