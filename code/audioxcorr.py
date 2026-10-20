import h5py
import mne
import numpy as np
from mne_bids import BIDSPath
from scipy import signal
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags


def xcorr(x: np.ndarray, y: np.ndarray, maxlags: int = None):
    """Normalized cross correlation of two 1D signals, using scipy"""

    x = x - x.mean()
    y = y - y.mean()

    corr = correlate(x, y, mode="full", method="fft")
    lags = correlation_lags(x.size, y.size, mode="full")
    corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    if maxlags is not None:
        middle = (lags == 0).nonzero()[0].item()
        lags = np.arange(-maxlags, maxlags + 1)
        corr = corr[middle - maxlags : middle + maxlags + 1]

    return corr, lags


def preprocess_audio(x, fs, to_fs, lowcut=200, highcut=5000):

    # See https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    nyq = 0.5 * fs
    b, a = signal.butter(5, [lowcut / nyq, highcut / nyq], btype="band")

    assert x.ndim == 1

    # Bandpass the high quality audio
    y = signal.lfilter(b, a, x)

    # Downsample to the clinical system's rate, N = n * (to_fs / fs)
    y = signal.resample(y, num=round(x.size / fs * to_fs))

    # Take the audio envelope
    envelope = np.abs(signal.hilbert(y - y.mean()))

    return envelope


def main(
    bids_root: str,
    stimuli_root: str,
    out_dir: str,
    subject: str,
    band: str,
    lag: int,
    overwrite: bool,
):

    audio_sf, audio_wave = wavfile.read(f"{stimuli_root}/podcast.wav")
    if audio_wave.ndim > 1:
        audio_wave = audio_wave[:, 0]  # take the first channel

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
            datatype="audioxcorr",
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

        audio_env = preprocess_audio(audio_wave, audio_sf, sfreq)
        data = raw.get_data()
        maxlag = int(lag * sfreq)

        all_corrs = []
        for i in range(len(data)):
            corrs, lags = xcorr(data[i], audio_env, maxlags=maxlag)
            all_corrs.append(corrs)
        all_corrs = np.stack(all_corrs)  # electrodes x lags

        out_path.mkdir()
        with h5py.File(out_path, "w") as f:
            f.create_dataset(name="corrs", data=all_corrs)
            f.create_dataset(name="lags", data=lags / sfreq)
            f.create_dataset(name="coords", data=coords)
            f.create_dataset(name="ch_names", data=np.array(raw.ch_names, dtype="S"))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--bids-root", type=str, default="data/derivatives/ecogprep")
    parser.add_argument("--stimuli-root", type=str, default="data/stimuli")
    parser.add_argument("-o", "--out-dir", type=str, default="results")
    parser.add_argument("-s", "--subject", type=str, default=None)
    parser.add_argument("-b", "--band", type=str, default="highgamma")
    parser.add_argument("-l", "--lag", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    _args = parser.parse_args()
    main(**vars(_args))
