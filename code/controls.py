import gc

import h5py
import mne
import numpy as np
from himalaya.backend import get_backend, set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from mne_bids import BIDSPath
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from encoding import get_features
from prediction import get_word_info, matched_sample


def window_r(Y_true, Y_pred, window_cols, n_electrodes):
    # mean correlation over the window lags, one value per electrode
    corr = correlation_score(Y_true[:, window_cols], Y_pred[:, window_cols])
    return get_backend().to_numpy(corr).reshape(n_electrodes, -1).mean(-1)


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
    n_shifts: int,
    n_draws: int,
    min_shift: int,
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
            datatype="controls",
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

        # previous word's embedding, the same as shift-embn in hassonlab/247-encoding
        has_prev = epochs.selection > 0
        X_prev = np.zeros_like(X)
        X_prev[has_prev] = embeddings[epochs.selection[has_prev] - 1].astype(np.float32)

        # current then previous, the order concat-embn uses
        X_joint = np.hstack([X, X_prev])

        window = (lags >= -1) & (lags < 0)
        n_electrodes = epochs_shape[0]
        window_cols = (
            np.arange(n_electrodes)[:, None] * len(lags)
            + np.flatnonzero(window)[None, :]
        ).ravel()

        rng = np.random.default_rng(seed)  # same draws as prediction.py
        rng_null = np.random.default_rng(seed + 1)

        scores, scores_prev, scores_joint = [], [], []
        obs_diff = np.zeros((n_folds, n_electrodes))
        null_diff = np.zeros((n_folds, n_shifts, n_electrodes))
        # current, previous, joint
        null_r = np.zeros((3, n_folds, n_shifts, n_electrodes))
        offsets = np.zeros((n_folds, n_shifts), dtype=int)

        print("Fitting models")
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

            # Observed difference, drawn exactly as in prediction.py
            df_test = df_sel.iloc[test_index].reset_index(drop=True)
            for i in range(n_iters):
                pred_idx, unpred_idx = matched_sample(df_test, rng)
                obs_diff[k] += window_r(
                    Y_test[pred_idx], Y_preds[pred_idx], window_cols, n_electrodes
                ) - window_r(
                    Y_test[unpred_idx], Y_preds[unpred_idx], window_cols, n_electrodes
                )
            obs_diff[k] /= n_iters

            # Label-shift null, the predicted labels slide against the words
            offsets[k] = rng_null.integers(
                min_shift, len(test_index) - min_shift, n_shifts
            )
            predicted = df_test.predicted.to_numpy()
            for s, offset in enumerate(offsets[k]):
                df_shift = df_test.assign(predicted=np.roll(predicted, offset))
                for d in range(n_draws):
                    pred_idx, unpred_idx = matched_sample(df_shift, rng_null)
                    null_diff[k, s] += window_r(
                        Y_test[pred_idx], Y_preds[pred_idx], window_cols, n_electrodes
                    ) - window_r(
                        Y_test[unpred_idx],
                        Y_preds[unpred_idx],
                        window_cols,
                        n_electrodes,
                    )
                null_diff[k, s] /= n_draws

            # Prediction-shift null, the predictions slide against the brain data
            Y_preds = backend.to_numpy(Y_preds)
            for s, offset in enumerate(offsets[k]):
                null_r[0, k, s] = window_r(
                    Y_test, np.roll(Y_preds, offset, axis=0), window_cols, n_electrodes
                )

            # Previous-word and joint models skip the first word, which has no previous word
            train_prev = train_index[has_prev[train_index]]
            test_prev = test_index[has_prev[test_index]]
            scaler = StandardScaler()
            Y_train = scaler.fit_transform(Y[train_prev])
            Y_test = scaler.transform(Y[test_prev])
            for m, (X_m, out) in enumerate(
                [(X_prev, scores_prev), (X_joint, scores_joint)], start=1
            ):
                model.fit(X_m[train_prev], Y_train)
                Y_preds = backend.to_numpy(model.predict(X_m[test_prev]))
                corr = correlation_score(Y_test, Y_preds)
                out.append(backend.to_numpy(corr).reshape(epochs_shape))
                for s, offset in enumerate(offsets[k]):
                    null_r[m, k, s] = window_r(
                        Y_test,
                        np.roll(Y_preds, offset, axis=0),
                        window_cols,
                        n_electrodes,
                    )
            print(f"Fold {k} done")

        out_path.mkdir()
        with h5py.File(out_path, "w") as f:
            # folds x electrodes x lags
            f.create_dataset(name="scores", data=np.stack(scores))
            f.create_dataset(name="scores_prev", data=np.stack(scores_prev))
            f.create_dataset(name="scores_joint", data=np.stack(scores_joint))
            f.create_dataset(name="obs_diff", data=obs_diff.mean(0))  # electrodes
            # shifts x electrodes
            f.create_dataset(name="null_diff", data=null_diff.mean(0))
            # models x shifts x electrodes
            f.create_dataset(name="null_r", data=null_r.mean(1))
            f.create_dataset(name="offsets", data=offsets)
            f.create_dataset(name="window", data=window)
            f.create_dataset(name="lags", data=lags)
            f.create_dataset(name="coords", data=coords)
            f.create_dataset(name="ch_names", data=np.array(raw.ch_names, dtype="S"))
            f.attrs["n_draws"] = n_draws
            f.attrs["min_shift"] = min_shift

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
    parser.add_argument("--n-shifts", type=int, default=1000)
    parser.add_argument("--n-draws", type=int, default=20)
    parser.add_argument("--min-shift", type=int, default=100)
    _args = parser.parse_args()
    main(**vars(_args))
