import os
import urllib.request

S3_URL = "https://s3.amazonaws.com/openneuro.org/ds005574"


def get_paths(subjects: list[str], features: list[str]) -> list[str]:
    paths = ["stimuli/podcast.wav", "stimuli/podcast_transcript.csv"]
    for feature in features:
        paths.append(f"stimuli/{feature}/features.hdf5")
        paths.append(f"stimuli/{feature}/transcript.tsv")
    for sub in subjects:
        paths.append(
            f"derivatives/ecogprep/sub-{sub}/ieeg/sub-{sub}_task-podcast_desc-highgamma_ieeg.fif"
        )
    return paths


def main(data_root: str, subjects: list[str], features: list[str]):

    for path in get_paths(subjects, features):
        out_path = os.path.join(data_root, path)
        if os.path.exists(out_path):
            continue  # already have it, same as wget -nc

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(f"Downloading {path}")

        # temp name so an interrupted download isn't mistaken for a finished one
        urllib.request.urlretrieve(f"{S3_URL}/{path}", out_path + ".part")
        os.rename(out_path + ".part", out_path)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument(
        "--subjects", nargs="+", default=[f"{i:02d}" for i in range(1, 10)]
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["spectral", "phonetic", "syntactic", "en_core_web_lg", "gpt2-xl"],
    )
    _args = parser.parse_args()
    main(**vars(_args))
