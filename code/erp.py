import gc
from functools import partial

import h5py
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from scipy.stats import zscore


def main(
    bids_root: str,
    stimuli_root: str,
    out_dir: str,
    subject: str,
    band: str,
    tmin: float,
    tmax: float,
    chunk_size: int,
    overwrite: bool,
):

    df = pd.read_csv(f"{stimuli_root}/podcast_transcript.csv")
    df.dropna(subset=["start"], inplace=True)
    df.sort_values("start", inplace=True)

    fif_path = BIDSPath(
        root=bids_root,
        subject=subject,
        datatype="ieeg",
        description=band,
        extension=".fif",
    )
    for fif_path in fif_path.match():

        out_path = fif_path.copy().update(
            root=out_dir,
            datatype="erp",
            suffix="result",
            extension=".h5",
            check=False,
        )
        if out_path.fpath.exists() and not overwrite:
            print(f"Skipping {out_path.basename}, already done")
            continue

        print(f"Subject {fif_path.subject}")
        raw = mne.io.read_raw_fif(fif_path, verbose=False)
        sfreq = int(raw.info["sfreq"])
        # mne stores meters, nilearn plots in mm
        coords = np.vstack([ch["loc"][:3] for ch in raw.info["chs"]]) * 1000

        events = np.zeros((len(df), 3), dtype=int)
        events[:, 0] = (df.start * sfreq).astype(int)

        # Same chunking as encoding.py, zscoring and averaging are both per electrode
        evoked = []
        for i in range(0, len(raw.ch_names), chunk_size):
            chunk = (
                raw.copy()
                .pick(raw.ch_names[i : i + chunk_size])
                .load_data(verbose=False)
            )
            chunk = chunk.apply_function(partial(zscore, axis=1), channel_wise=False)
            epochs = mne.Epochs(
                chunk,
                events,
                tmin=tmin,
                tmax=tmax,
                proj=False,
                baseline=None,
                event_id=None,
                preload=True,
                event_repeated="merge",
                verbose=False,
            )
            evoked.append(epochs.average().data)
            lags = epochs.times
            del chunk, epochs
        evoked = np.concatenate(evoked)  # electrodes x lags

        out_path.mkdir()
        with h5py.File(out_path, "w") as f:
            f.create_dataset(name="evoked", data=evoked)
            f.create_dataset(name="lags", data=lags)
            f.create_dataset(name="coords", data=coords)
            f.create_dataset(name="ch_names", data=np.array(raw.ch_names, dtype="S"))

        del raw, evoked
        gc.collect()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--bids-root", type=str, default="data/derivatives/ecogprep")
    parser.add_argument("--stimuli-root", type=str, default="data/stimuli")
    parser.add_argument("-o", "--out-dir", type=str, default="results")
    parser.add_argument("-s", "--subject", type=str, default=None)
    parser.add_argument("-b", "--band", type=str, default="highgamma")
    parser.add_argument("--tmin", type=float, default=-2)
    parser.add_argument("--tmax", type=float, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    _args = parser.parse_args()
    main(**vars(_args))
