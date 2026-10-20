import gc

import h5py
import mne
import numpy as np
import pandas as pd
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from mne_bids import BIDSPath
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_features(stimuli_root: str, modelname: str, layer: int = None):

    with h5py.File(f"{stimuli_root}/{modelname}/features.hdf5", "r") as f:
        datasets = list(f.keys())
        if layer is not None:
            embeddings = f[f"layer-{layer}"][...]
        elif len(datasets) == 1:
            embeddings = f[datasets[0]][...]
        else:
            raise ValueError(f"{modelname} has several layers, pass --layer")

    df = pd.read_csv(
        f"{stimuli_root}/{modelname}/transcript.tsv", sep="\t", index_col=0
    )

    # Reduce tokens to words by averaging
    aligned_embeddings = []
    for _, group in df.groupby("word_idx"):
        indices = group.index.to_numpy()
        aligned_embeddings.append(embeddings[indices].mean(0))
    aligned_embeddings = np.stack(aligned_embeddings)

    df = df.groupby("word_idx").agg(dict(word="first", start="first", end="last"))

    # Remove words with no onsets
    good_mask = df["start"].notna().to_numpy()
    aligned_embeddings = aligned_embeddings[good_mask]
    df = df[good_mask]

    return df, aligned_embeddings


def main(
    bids_root: str,
    stimuli_root: str,
    out_dir: str,
    subject: str,
    band: str,
    modelname: str,
    layer: int,
    n_folds: int,
    tmin: float,
    tmax: float,
    chunk_size: int,
    overwrite: bool,
):

    # falls back to numpy without a GPU
    backend = set_backend("torch_cuda", on_error="warn")
    print(f"Using {backend.__name__}")

    df, embeddings = get_features(stimuli_root, modelname, layer)

    desc = modelname if layer is None else f"{modelname}-l{layer}"
    desc = desc.replace("-", ".").replace("_", ".")

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
            datatype="encoding",
            description=desc,
            suffix="result",
            extension=".h5",
            check=False,
        )
        if out_path.fpath.exists() and not overwrite:
            print(f"Skipping {out_path.basename}, already done")
            continue

        print(f"Subject {fif_path.subject}, {modelname}")
        raw = mne.io.read_raw_fif(fif_path, verbose=False)
        raw = raw.pick("ecog", exclude="bads")
        sfreq = int(raw.info["sfreq"])
        # mne stores meters, nilearn plots in mm
        coords = np.vstack([ch["loc"][:3] for ch in raw.info["chs"]]) * 1000

        events = np.zeros((len(df), 3), dtype=int)
        events[:, 0] = (df.start * sfreq).astype(int)

        # Epoch a few electrodes at a time, the 512 Hz epochs for every electrode
        # don't fit in Colab memory. Every step here is per electrode so the result
        # is the same as epoching them all at once
        print("Epoching data")
        epochs_data = []
        for i in range(0, len(raw.ch_names), chunk_size):
            chunk = (
                raw.copy()
                .pick(raw.ch_names[i : i + chunk_size])
                .load_data(verbose=False)
            )
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
            epochs = epochs.resample(
                sfreq=32,
                npad="auto",
                method="fft",
                window="hamming",
                n_jobs=-1,
                verbose=False,
            )
            epochs_data.append(epochs.get_data(copy=True))
            del chunk
        epochs_data = np.concatenate(epochs_data, axis=1)  # words x electrodes x lags
        epochs_shape = epochs_data.shape[1:]
        lags = epochs.times

        X = embeddings[epochs.selection].astype(np.float32)
        Y = epochs_data.reshape(len(epochs_data), -1).astype(np.float32)

        model = make_pipeline(
            StandardScaler(),
            RidgeCV(
                alphas=np.logspace(1, 10, 10),
                fit_intercept=True,
                cv=KFold(n_splits=5, shuffle=False),
            ),
        )

        print("Fitting models")
        scores = []
        kfold = KFold(n_folds, shuffle=False)
        for train_index, test_index in kfold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            # Standardize Y
            scaler = StandardScaler()
            Y_train = scaler.fit_transform(Y_train)
            Y_test = scaler.transform(Y_test)

            model.fit(X_train, Y_train)
            Y_preds = model.predict(X_test)
            corr = correlation_score(Y_test, Y_preds)
            scores.append(backend.to_numpy(corr).reshape(epochs_shape))
        scores = np.stack(scores)  # folds x electrodes x lags

        out_path.mkdir()
        with h5py.File(out_path, "w") as f:
            f.create_dataset(name="scores", data=scores)
            f.create_dataset(name="lags", data=lags)
            f.create_dataset(name="coords", data=coords)
            f.create_dataset(name="ch_names", data=np.array(raw.ch_names, dtype="S"))

        del raw, epochs, epochs_data, X, Y
        gc.collect()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--bids-root", type=str, default="data/derivatives/ecogprep")
    parser.add_argument("--stimuli-root", type=str, default="data/stimuli")
    parser.add_argument("-o", "--out-dir", type=str, default="results")
    parser.add_argument("-s", "--subject", type=str, default=None)
    parser.add_argument("-b", "--band", type=str, default="highgamma")
    parser.add_argument("-m", "--modelname", type=str, default="gpt2-xl")
    parser.add_argument("-l", "--layer", type=int, default=None)
    parser.add_argument("-k", "--n-folds", type=int, default=2)
    parser.add_argument("--tmin", type=float, default=-2)
    parser.add_argument("--tmax", type=float, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    _args = parser.parse_args()
    main(**vars(_args))
