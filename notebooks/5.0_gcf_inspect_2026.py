import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pyxdf
    from pathlib import Path

    BIDS_ROOT = Path("data/bids")
    FILES = sorted(p for p in BIDS_ROOT.rglob("*.xdf") if p.is_file())
    return BIDS_ROOT, FILES, mo, pyxdf


@app.cell
def _(BIDS_ROOT, FILES, mo):
    file_picker = mo.ui.dropdown(
        options={str(p.relative_to(BIDS_ROOT)): p for p in FILES},
        value=str(FILES[0].relative_to(BIDS_ROOT)) if FILES else None,
        label="XDF file",
        full_width=True,
    )
    file_picker
    return (file_picker,)


@app.cell
def _(file_picker, mo, pyxdf):
    mo.stop(file_picker.value is None, mo.md("Select a file above."))

    streams, header = pyxdf.load_xdf(file_picker.value)
    return header, streams


@app.cell
def _(mo, streams):
    # One row per stream, so the tree below is easier to navigate
    summary = [
        {
            "name": s["info"]["name"][0],
            "type": s["info"]["type"][0],
            "channels": int(s["info"]["channel_count"][0]),
            "nominal_srate": float(s["info"]["nominal_srate"][0]),
            "effective_srate": s["info"]["effective_srate"],
            "samples": len(s["time_stamps"]),
        }
        for s in streams
    ]
    mo.vstack([mo.md("## Streams"), mo.ui.table(summary, selection=None)])
    return


@app.cell
def _(header, mo, streams):
    mo.vstack(
        [
            mo.md("## File header"),
            mo.tree(header),
            mo.md("## Stream metadata"),
            mo.accordion(
                {
                    f"{i}: {s['info']['name'][0]}": mo.tree(s["info"])
                    for i, s in enumerate(streams)
                }
            ),
        ]
    )
    return


@app.cell
def _(mo, streams):
    def channel_rows(stream):
        # pyxdf wraps every XML value in a list; desc may be None or lack channels
        desc = (stream["info"]["desc"] or [None])[0] or {}
        channels = (desc.get("channels") or [{}])[0].get("channel", [])
        return [
            {"index": i, **{k: v[0] if v else None for k, v in ch.items()}}
            for i, ch in enumerate(channels)
        ]

    mo.vstack(
        [
            mo.md("## Channels"),
            mo.ui.tabs(
                {
                    f"{i}: {s['info']['name'][0]}": (
                        mo.ui.table(rows, selection=None, page_size=64)
                        if (rows := channel_rows(s))
                        else mo.md("_No channel metadata in this stream._")
                    )
                    for i, s in enumerate(streams)
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Signal quality

    Checks on every regularly sampled numeric stream. Thresholds:

    - **gaps**: consecutive timestamps further apart than 2 sample periods
    - **flat**: std ≈ 0, or more than 50 % of samples equal to the previous one
    - **clipped**: more than 1 % of samples sit exactly on the channel's min/max
    - **line noise**: power in 50 ± 1 Hz over power in 1–100 Hz above 0.5
    - **outlier**: std more than 5× above or below the median std of the stream
    """)
    return


@app.cell
def _(mo, streams):
    import numpy as np
    from scipy.signal import welch

    LINE_FREQ = 50.0

    def is_signal(stream):
        info = stream["info"]
        numeric = info["channel_format"][0] not in ("string",)
        return numeric and float(info["nominal_srate"][0]) > 0

    def stream_quality(stream):
        info = stream["info"]
        srate = float(info["nominal_srate"][0])
        ts = stream["time_stamps"]
        dt = np.diff(ts)
        gaps = dt[dt > 2 / srate]
        duration = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        return {
            "name": info["name"][0],
            "duration_s": round(duration, 2),
            "samples": len(ts),
            "expected_samples": int(round(duration * srate)) + 1,
            "nominal_srate": srate,
            "effective_srate": round(info["effective_srate"], 3),
            "n_gaps": len(gaps),
            "longest_gap_s": round(float(gaps.max()), 4) if len(gaps) else 0.0,
            "non_monotonic_ts": int((dt <= 0).sum()),
        }

    def channel_quality(stream):
        info = stream["info"]
        srate = float(info["nominal_srate"][0])
        x = np.asarray(stream["time_series"], dtype=float)
        desc = (info["desc"] or [None])[0] or {}
        chans = (desc.get("channels") or [{}])[0].get("channel", [])
        labels = [c["label"][0] if c.get("label") else None for c in chans]
        labels += [f"ch{i}" for i in range(len(labels), x.shape[1])]

        stds = np.nanstd(x, axis=0)
        median_std = np.median(stds[stds > 0]) if (stds > 0).any() else 0.0
        nperseg = int(min(len(x), 4 * srate))
        freqs, psd = welch(np.nan_to_num(x), fs=srate, nperseg=nperseg, axis=0)
        band = (freqs >= 1) & (freqs <= min(100, srate / 2))
        line = np.abs(freqs - LINE_FREQ) <= 1

        rows = []
        for i in range(x.shape[1]):
            c = x[:, i]
            valid = c[~np.isnan(c)]
            flat_pct = 100 * np.mean(np.diff(valid) == 0) if len(valid) > 1 else 100
            lo, hi = (valid.min(), valid.max()) if len(valid) else (0, 0)
            clip_pct = 100 * np.mean((valid == lo) | (valid == hi)) if hi > lo else 0
            band_power = psd[band, i].sum()
            line_ratio = psd[line, i].sum() / band_power if band_power > 0 else 0

            flags = []
            if stds[i] < 1e-12 or flat_pct > 50:
                flags.append("flat")
            if clip_pct > 1:
                flags.append("clipped")
            if line.any() and line_ratio > 0.5:
                flags.append("line noise")
            if median_std > 0 and not 0.2 < stds[i] / median_std < 5:
                flags.append("outlier")
            if len(valid) < len(c):
                flags.append("NaN")

            rows.append(
                {
                    "index": i,
                    "label": labels[i],
                    "status": ", ".join(flags) or "ok",
                    "mean": float(np.mean(valid)) if len(valid) else None,
                    "std": float(stds[i]),
                    "rms": float(np.sqrt(np.mean(valid**2))) if len(valid) else None,
                    "peak_to_peak": float(hi - lo),
                    "nan_pct": round(100 * (1 - len(valid) / len(c)), 3),
                    "flat_pct": round(float(flat_pct), 3),
                    "clip_pct": round(float(clip_pct), 3),
                    "line_ratio": round(float(line_ratio), 3),
                }
            )
        return rows

    signal_streams = [s for s in streams if is_signal(s) and len(s["time_stamps"])]
    mo.stop(not signal_streams, mo.md("_No regularly sampled numeric streams._"))

    channel_tables = {}
    for _i, _s in enumerate(signal_streams):
        _rows = channel_quality(_s)
        _bad = sum(r["status"] != "ok" for r in _rows)
        channel_tables[f"{_s['info']['name'][0]} ({_bad}/{len(_rows)} flagged)"] = (
            mo.ui.table(_rows, selection=None, page_size=64)
        )

    mo.vstack(
        [
            mo.md("### Timing"),
            mo.ui.table([stream_quality(s) for s in signal_streams], selection=None),
            mo.md("### Channels"),
            mo.ui.tabs(channel_tables),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
