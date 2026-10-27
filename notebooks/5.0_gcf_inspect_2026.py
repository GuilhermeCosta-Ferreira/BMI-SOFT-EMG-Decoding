import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pyxdf
    from pathlib import Path
    from scipy.signal import welch

    BIDS_ROOT = Path("data/bids")
    QC_CSV = Path("data/quality/channel_qc.csv")
    FILES = sorted(p for p in BIDS_ROOT.rglob("*.xdf") if p.is_file())
    return BIDS_ROOT, FILES, Path, QC_CSV, mo, np, pd, pyxdf, welch


@app.cell
def _(FILES, mo):
    mo.md(rf"""
    # Inspecting the 2026 XDF recordings

    This notebook checks the raw `.xdf` recordings in `data/bids` before they
    go through the pruning pipeline. It is split into four parts:

    1. **Session selection**: pick one of the {len(FILES)} recordings.
    2. **Metadata**: the file header, every stream's `info` block and its
       channel list, as written by LabRecorder.
    3. **Signal quality**: timing and per-channel checks on one signal stream,
       to catch acquisition problems (dead electrodes, saturation, dropped
       samples, mains interference).
    4. **Batch export**: run the channel checks on one or all sessions and
       save them to a CSV, so the results can be reloaded without recomputing.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Session selection

    Each XDF file holds one recording session and can contain several LSL
    streams (e.g. the amplifier and a marker stream). Loading reads every
    sample, so switching sessions takes a few seconds.
    """)
    return


@app.cell
def _(BIDS_ROOT, FILES, mo):
    file_picker = mo.ui.dropdown(
        options={str(p.relative_to(BIDS_ROOT)): p for p in FILES},
        value=str(FILES[0].relative_to(BIDS_ROOT)) if FILES else None,
        label="Session",
        full_width=True,
    )
    file_picker
    return (file_picker,)


@app.cell
def _(file_picker, mo, pyxdf):
    mo.stop(file_picker.value is None, mo.md("Select a session above."))

    streams, header = pyxdf.load_xdf(file_picker.value)
    return header, streams


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Metadata

    ### 2.1 Streams overview

    One row per stream. The *nominal* rate is what the device declared; the
    *effective* rate is what pyxdf measured from the timestamps,

    $$
    f_\text{eff} = \frac{N - 1}{t_{N} - t_{1}},
    $$

    so a large mismatch between the two already hints at dropped samples or
    clock problems. Streams with $f_\text{nom} = 0$ are irregular (markers).
    """)
    return


@app.cell
def _(mo, streams):
    summary = [
        {
            "name": s["info"]["name"][0],
            "type": s["info"]["type"][0],
            "format": s["info"]["channel_format"][0],
            "channels": int(s["info"]["channel_count"][0]),
            "nominal_srate": float(s["info"]["nominal_srate"][0]),
            "effective_srate": round(s["info"]["effective_srate"], 3),
            "samples": len(s["time_stamps"]),
        }
        for s in streams
    ]
    mo.ui.table(summary, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.2 File header and stream info

    The raw XML metadata, as parsed by pyxdf. Every leaf value is wrapped in a
    one-element list because XML elements may repeat.
    """)
    return


@app.cell
def _(header, mo, streams):
    mo.vstack(
        [
            mo.accordion({"File header": mo.tree(header)}),
            mo.accordion(
                {
                    f"Stream {i}: {s['info']['name'][0]}": mo.tree(s["info"])
                    for i, s in enumerate(streams)
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.3 Channels

    The `desc/channels` block of each stream, flattened into one row per
    channel. Streams that do not describe their channels show a note instead.
    """)
    return


@app.function
def channel_info(stream):
    """Return the stream's `desc/channels` entries as flat dicts."""
    # pyxdf wraps every XML value in a list; desc may be None or lack channels
    desc = (stream["info"]["desc"] or [None])[0] or {}
    channels = (desc.get("channels") or [{}])[0].get("channel", [])
    return [{k: v[0] if v else None for k, v in ch.items()} for ch in channels]


@app.cell
def _(mo, streams):
    mo.ui.tabs(
        {
            f"{i}: {s['info']['name'][0]}": (
                mo.ui.table(
                    [{"index": j, **c} for j, c in enumerate(rows)],
                    selection=None,
                    page_size=64,
                )
                if (rows := channel_info(s))
                else mo.md("_No channel metadata in this stream._")
            )
            for i, s in enumerate(streams)
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Signal quality

    Only regularly sampled numeric streams can be checked. Pick one below; the
    checks run on that stream only.

    ### 3.1 Metrics

    For a channel $x_1, \dots, x_N$ (NaNs removed):

    | Metric | Definition | Flag when |
    |---|---|---|
    | mean, std | $\bar x$, $\sigma = \sqrt{\tfrac1N \sum_n (x_n - \bar x)^2}$ | $\sigma < 10^{-12}$ → **flat** |
    | RMS | $\sqrt{\tfrac1N \sum_n x_n^2}$ | |
    | peak-to-peak | $\max_n x_n - \min_n x_n$ | |
    | flat % | $\tfrac{100}{N-1} \, \#\{n : x_{n+1} = x_n\}$ | $> 50\,\%$ → **flat** |
    | clip % | $\tfrac{100}{N} \, \#\{n : x_n \in \{\min x, \max x\}\}$ | $> 1\,\%$ → **clipped** |
    | line ratio | $\rho_{50} = \dfrac{\sum_{\lvert f - 50 \rvert \le 1} P(f)}{\sum_{1 \le f \le 100} P(f)}$ | $> 0.5$ → **line noise** |
    | relative std | $\sigma / \operatorname{median}_c \sigma_c$ across the stream's channels | $\notin (0.2, 5)$ → **outlier** |
    | NaN % | share of missing samples | $> 0$ → **NaN** |

    $P(f)$ is the Welch power spectral density with 4 s segments. A channel
    that is flat or clipped most likely lost contact or saturated the
    amplifier; an outlier is either a bad electrode or a channel of a
    different kind (e.g. EMG next to EEG), so check its `type` first.

    The timing check flags a **gap** whenever two consecutive timestamps are
    further apart than two sample periods, $t_{n+1} - t_n > 2 / f_\text{nom}$.
    """)
    return


@app.function
def is_signal(stream):
    """True for numeric streams with a fixed sampling rate."""
    info = stream["info"]
    return (
        info["channel_format"][0] != "string"
        and float(info["nominal_srate"][0]) > 0
        and len(stream["time_stamps"]) > 1
    )


@app.cell
def _(np):
    def stream_timing(stream):
        """Duration, sample counts and timestamp gaps of a signal stream."""
        info = stream["info"]
        srate = float(info["nominal_srate"][0])
        ts = stream["time_stamps"]
        dt = np.diff(ts)
        gaps = dt[dt > 2 / srate]
        duration = float(ts[-1] - ts[0])
        return {
            "duration_s": round(duration, 2),
            "samples": len(ts),
            "expected_samples": int(round(duration * srate)) + 1,
            "nominal_srate": srate,
            "effective_srate": round(info["effective_srate"], 3),
            "n_gaps": len(gaps),
            "longest_gap_s": round(float(gaps.max()), 4) if len(gaps) else 0.0,
            "non_monotonic_ts": int((dt <= 0).sum()),
        }

    return (stream_timing,)


@app.cell
def _(np, welch):
    LINE_FREQ = 50.0

    def channel_quality(stream):
        """One row of quality metrics and flags per channel of a signal stream."""
        srate = float(stream["info"]["nominal_srate"][0])
        x = np.asarray(stream["time_series"], dtype=float)
        meta = channel_info(stream)
        labels = [c.get("label") for c in meta]
        labels += [f"ch{i}" for i in range(len(labels), x.shape[1])]
        types = [c.get("type") for c in meta] + [None] * x.shape[1]

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
            has_data = len(valid) > 0
            flat_pct = 100 * np.mean(np.diff(valid) == 0) if len(valid) > 1 else 100
            lo, hi = (valid.min(), valid.max()) if has_data else (0.0, 0.0)
            clip_pct = 100 * np.mean((valid == lo) | (valid == hi)) if hi > lo else 0
            band_power = psd[band, i].sum()
            line_ratio = psd[line, i].sum() / band_power if band_power > 0 else 0
            rel_std = stds[i] / median_std if median_std > 0 else np.nan

            flags = []
            if not stds[i] >= 1e-12 or flat_pct > 50:
                flags.append("flat")
            if clip_pct > 1:
                flags.append("clipped")
            if line.any() and line_ratio > 0.5:
                flags.append("line noise")
            if not 0.2 < rel_std < 5:
                flags.append("outlier")
            if len(valid) < len(c):
                flags.append("NaN")

            rows.append(
                {
                    "index": i,
                    "label": labels[i],
                    "type": types[i],
                    "status": ", ".join(flags) or "ok",
                    "mean": float(np.mean(valid)) if has_data else None,
                    "std": float(stds[i]),
                    "rms": float(np.sqrt(np.mean(valid**2))) if has_data else None,
                    "peak_to_peak": float(hi - lo),
                    "rel_std": round(float(rel_std), 3),
                    "nan_pct": round(100 * (1 - len(valid) / len(c)), 3),
                    "flat_pct": round(float(flat_pct), 3),
                    "clip_pct": round(float(clip_pct), 3),
                    "line_ratio": round(float(line_ratio), 3),
                }
            )
        return rows

    return (channel_quality,)


@app.cell
def _(mo, streams):
    signal_streams = {
        f"{i}: {s['info']['name'][0]} ({s['info']['type'][0]})": s
        for i, s in enumerate(streams)
        if is_signal(s)
    }
    mo.stop(not signal_streams, mo.md("_No regularly sampled numeric streams._"))

    stream_picker = mo.ui.dropdown(
        options=signal_streams,
        value=next(iter(signal_streams)),
        label="Stream to check",
        full_width=True,
    )
    stream_picker
    return (stream_picker,)


@app.cell
def _(mo, stream_picker, stream_timing):
    mo.vstack(
        [
            mo.md("### 3.2 Timing"),
            mo.ui.table([stream_timing(stream_picker.value)], selection=None),
        ]
    )
    return


@app.cell
def _(channel_quality, mo, stream_picker):
    _rows = channel_quality(stream_picker.value)
    _bad = sum(r["status"] != "ok" for r in _rows)
    mo.vstack(
        [
            mo.md(f"### 3.3 Channels\n\n**{_bad} of {len(_rows)}** channels flagged."),
            mo.ui.table(_rows, selection=None, page_size=64),
        ]
    )
    return


@app.cell
def _(QC_CSV, mo):
    mo.md(rf"""
    ## 4. Batch export

    Runs the channel checks of section 3 and writes them to
    `{QC_CSV}`, one row per (session, stream, channel). In each session the
    checked streams are those with the same `type` as the stream picked above,
    so marker streams are skipped.

    - **This session** recomputes only the selected file and replaces its rows
      in the CSV, keeping the others.
    - **All sessions** loads every file, which takes a while, and rewrites the
      whole CSV.

    Next time, skip the computation: section 4.2 loads the CSV if it exists.
    """)
    return


@app.cell
def _(mo):
    export_scope = mo.ui.radio(
        options=["This session", "All sessions"],
        value="This session",
        label="Scope",
        inline=True,
    )
    export_button = mo.ui.run_button(label="Compute and save to CSV")
    mo.hstack([export_scope, export_button], justify="start", gap=2)
    return export_button, export_scope


@app.cell
def _(
    BIDS_ROOT,
    FILES,
    QC_CSV,
    channel_quality,
    export_button,
    export_scope,
    file_picker,
    mo,
    pd,
    pyxdf,
    stream_picker,
    streams,
):
    mo.stop(not export_button.value)

    def session_rows(path, file_streams):
        rows = []
        for s in file_streams:
            if not is_signal(s) or s["info"]["type"][0] != target_type:
                continue
            for r in channel_quality(s):
                rows.append(
                    {
                        "session": str(path.relative_to(BIDS_ROOT)),
                        "stream": s["info"]["name"][0],
                        **r,
                    }
                )
        return rows

    target_type = stream_picker.value["info"]["type"][0]

    if export_scope.value == "This session":
        new = pd.DataFrame(session_rows(file_picker.value, streams))
        session = str(file_picker.value.relative_to(BIDS_ROOT))
        old = pd.read_csv(QC_CSV) if QC_CSV.exists() else pd.DataFrame()
        if not old.empty:
            old = old[old["session"] != session]
        qc = pd.concat([old, new], ignore_index=True)
    else:
        _rows = []
        for _path in mo.status.progress_bar(FILES, title="Checking sessions"):
            _rows += session_rows(_path, pyxdf.load_xdf(_path)[0])
        qc = pd.DataFrame(_rows)

    QC_CSV.parent.mkdir(parents=True, exist_ok=True)
    qc.sort_values(["session", "stream", "index"]).to_csv(QC_CSV, index=False)
    mo.callout(
        mo.md(f"Saved {len(qc)} rows to `{QC_CSV}`. Section 4.2 reloads it."),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.2 Saved results

    The table below is read from the CSV. The second table summarises it by
    session: $n_\text{flagged} / n_\text{channels}$ and how often each flag
    appears.
    """)
    return


@app.cell
def _(QC_CSV, export_button, mo, pd):
    # Re-read after every export, the button value changes on each click
    export_button
    mo.stop(
        not QC_CSV.exists(),
        mo.md(f"_No CSV yet at `{QC_CSV}`; run the export above._"),
    )

    saved = pd.read_csv(QC_CSV)
    flags = saved["status"].str.get_dummies(sep=", ").drop(columns="ok", errors="ignore")
    per_session = (
        pd.concat([saved[["session", "stream"]], flags], axis=1)
        .assign(channels=1, flagged=(saved["status"] != "ok").astype(int))
        .groupby(["session", "stream"], as_index=False)
        .sum()
    )
    mo.vstack(
        [
            mo.ui.table(per_session, selection=None),
            mo.ui.table(saved, selection=None, page_size=64),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
