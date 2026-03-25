"""Download BGMN executable for Linux, Mac, and Windows."""
import os
import platform
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_bgmn():
    """Download BGMN executable for Linux, Mac, and Windows."""
    # get os
    os_name = platform.system()  # Darwin, Linux, Windows

    if os_name not in ["Darwin", "Linux", "Windows"]:
        raise Exception("Unsupported OS: " + os_name + ".")

    # get url
    URL = f"https://cedergrouphub.github.io/dara/_static/bgmnwin_{os_name}.zip"

    # download
    r = requests.get(URL, stream=True, timeout=30)
    if r.status_code != 200:
        raise Exception(
            f"Cannot download from {URL}. Please check your internet connection or contact the developer."
        )

    bgmn_folder = Path(__file__).parent

    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="iB", unit_scale=True)
    with (bgmn_folder / "bgmnwin.zip").open("wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    # unzip
    with zipfile.ZipFile((bgmn_folder / "bgmnwin.zip").as_posix(), "r") as zip_ref:
        zip_ref.extractall(path=bgmn_folder.as_posix())

    # delete zip
    os.remove((bgmn_folder / "bgmnwin.zip").as_posix())

    # give permission
    if os_name == "Linux" or os_name == "Darwin":
        os.system(f"chmod +x {bgmn_folder}/BGMNwin/bgmn")
        os.system(f"chmod +x {bgmn_folder}/BGMNwin/teil")
        os.system(f"chmod +x {bgmn_folder}/BGMNwin/eflech")
        os.system(f"chmod +x {bgmn_folder}/BGMNwin/output")


if __name__ == "__main__":
    download_bgmn()
