import gc
import re

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
from wordfreq import zipf_frequency

from encoding import get_features


def get_word_info(stimuli_root: str, modelname: str, df: pd.DataFrame) -> pd.DataFrame:

    tokens = pd.read_csv(
        f"{stimuli_root}/{modelname}/transcript.tsv", sep="\t", index_col=0
    )

    # rank of the first token, the one at word onset
    df["rank"] = tokens.groupby("word_idx")["rank"].first().loc[df.index]
    df["predicted"] = df["rank"] == 0

    words = [re.sub(r"[^\w']", "", word).lower() for word in df.word]
    df["zipf"] = [zipf_frequency(word, "en") for word in words]
    df["duration"] = df.end - df.start

    # half-Zipf frequency bins and duration quartiles, used for matching
    df["freq_bin"] = (df.zipf * 2).round() / 2
    df["dur_bin"] = pd.qcut(df.duration, 4, labels=False)

    return df


def matched_sample(df: pd.DataFrame, rng: np.random.Generator):

    # same number of predicted and unpredicted words in every bin
    pred_idx, unpred_idx = [], []
    for _, group in df.groupby(["freq_bin", "dur_bin"]):
        pred = group.index[group.predicted].to_numpy()
        unpred = group.index[~group.predicted].to_numpy()
        n = min(len(pred), len(unpred))
        pred_idx.append(rng.choice(pred, n, replace=False))
        unpred_idx.append(rng.choice(unpred, n, replace=False))

    return np.concatenate(pred_idx), np.concatenate(unpred_idx)


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
    n_iters: int,
    seed: int,
):

    # falls back to numpy without a GPU
    backend = set_backend("torch_cuda", on_error="warn")
    print(f"Using {backend.__name__}")

    df, embeddings = get_features(stimuli_root, modelname, layer)
    df = get_word_info(stimuli_root, modelname, df)
    assert df.predicted.sum() == 1485
    assert (~df.predicted).sum() == 3651

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
            datatype="prediction",
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

        # keep the word table in the same order as the epochs
        df_sel = df.iloc[epochs.selection].reset_index(drop=True)
        print(
            f"{df_sel.predicted.sum()} predicted and {(~df_sel.predicted).sum()} unpredicted words after epoching"
        )

        model = make_pipeline(
            StandardScaler(),
            RidgeCV(
                alphas=np.logspace(1, 10, 10),
                fit_intercept=True,
                cv=KFold(n_splits=5, shuffle=False),
            ),
        )

        print("Fitting models")
        rng = np.random.default_rng(seed)
        scores = []
        scores_pred = np.zeros((n_folds, n_iters, *epochs_shape), dtype=np.float32)
        scores_unpred = np.zeros_like(scores_pred)
        # folds x iterations x groups x (zipf, duration)
        balance = np.zeros((n_folds, n_iters, 2, 2), dtype=np.float32)
        n_matched = np.zeros(n_folds, dtype=int)

        kfold = KFold(n_folds, shuffle=False)
        for k, (train_index, test_index) in enumerate(kfold.split(X)):
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

            df_test = df_sel.iloc[test_index].reset_index(drop=True)
            for i in range(n_iters):
                pred_idx, unpred_idx = matched_sample(df_test, rng)
                for g, idx in enumerate([pred_idx, unpred_idx]):
                    corr = correlation_score(Y_test[idx], Y_preds[idx])
                    score = backend.to_numpy(corr).reshape(epochs_shape)
                    if g == 0:
                        scores_pred[k, i] = score
                    else:
                        scores_unpred[k, i] = score
                    balance[k, i, g] = df_test.loc[idx, ["zipf", "duration"]].mean()
            n_matched[k] = len(pred_idx)
            print(f"Fold {k}: {n_matched[k]} words per group")

        scores = np.stack(scores)  # folds x electrodes x lags

        out_path.mkdir()
        with h5py.File(out_path, "w") as f:
            f.create_dataset(name="scores", data=scores)
            # iterations x electrodes x lags
            f.create_dataset(name="scores_pred", data=scores_pred.mean(0))
            f.create_dataset(name="scores_unpred", data=scores_unpred.mean(0))
            # iterations x groups x (zipf, duration)
            f.create_dataset(name="balance", data=balance.mean(0))
            f.create_dataset(name="n_matched", data=n_matched)
            f.create_dataset(name="predicted", data=df_sel.predicted.to_numpy())
            f.create_dataset(name="zipf", data=df_sel.zipf.to_numpy())
            f.create_dataset(name="duration", data=df_sel.duration.to_numpy())
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
    parser.add_argument("-s", "--subject", type=str, default="03")
    parser.add_argument("-b", "--band", type=str, default="highgamma")
    parser.add_argument("-m", "--modelname", type=str, default="gpt2-xl")
    parser.add_argument("-l", "--layer", type=int, default=None)
    parser.add_argument("-k", "--n-folds", type=int, default=2)
    parser.add_argument("--tmin", type=float, default=-2)
    parser.add_argument("--tmax", type=float, default=2)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    _args = parser.parse_args()
    main(**vars(_args))
