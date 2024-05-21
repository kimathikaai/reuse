import csv
import logging
import os
import shutil
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List

from tqdm import tqdm


def setup_logger():
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d]-2s %(message)s"
    )
    logger.setLevel(level=logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger()

EXTENSIONS = [".png", ".jpg", ".bmp", ".tif", ".tiff"]


def get_file_stem(path_list: List[str]):
    """
    Remove the exensions from file name and return the
    extension free file name. This function is required
    because some file names include '.'
    which naively splitting based on '.' on file names
    without extensions will produce non existent filenames
    """
    _path_list = []
    for path in path_list:
        if Path(path).suffix in EXTENSIONS:
            _path_list.append(Path(path).stem)
        else:
            _path_list.append(Path(path).name)

    return _path_list


def read_paths_csv(csv_path: str) -> List[str]:
    """
    Given a csv of paths, read them as a comma separated row
    """
    path_list = []

    with open(csv_path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            path_list.extend(row)

    logger.info(f"Read {len(path_list)} filenames from {csv_path}")

    return path_list


def copy_files_from_csv(source_dir: str, csv_file: str, commit=None):
    """
    Move files from a directory `source_dir` into a different
    directory (i.e. `source_dir + time`) if the are in the `csv_file`.
    Note that this method is agnostic of file extensions
    """
    # Validate the directory existence
    assert os.path.isdir(source_dir)

    # Create a target directory
    now = datetime.now().strftime("%d%m%Y%H%M%S")
    target_dir = str(Path(source_dir)) + "-" + now

    # Read the csv of paths
    included_paths = read_paths_csv(csv_file)
    included_paths_stems = get_file_stem(included_paths)
    file_paths = glob(os.path.join(source_dir, "*"))
    sample_paths = [x for x in file_paths if Path(x).stem in included_paths_stems]
    _sample_stems = [Path(x).stem for x in sample_paths]

    # Visualize any differences
    logger.info(
        (
            "Path differences: \n"
            f"{set(_sample_stems).symmetric_difference(set(included_paths_stems))}"
        )
    )

    # Assert that the csv and found files are the same
    assert_string = f"{len(sample_paths) = }, {len(included_paths) = }"
    assert len(included_paths) == len(sample_paths), assert_string

    # Create the target directory
    if commit:
        os.mkdir(target_dir)
        for path in tqdm(sample_paths):
            shutil.copy(path, target_dir)

    logger.info(
        f"Copied {len(sample_paths)} files from {source_dir} to {target_dir} using {csv_file}"
    )


if __name__ == "__main__":
    copy_files_from_csv(*sys.argv[1:])
