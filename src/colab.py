"""Running in Colab: find or fetch the release, and fail early with something actionable.

Every notebook here runs in its own virtual machine. `/content` is wiped between them, while
Drive persists. That asymmetry caused a real failure: notebook 00 extracted the release to
`/content/raw`, notebook 01 assumed it was still there, and a fresh VM gave

    RELEASE = sorted({...})[0]
    IndexError: list index out of range

ten lines into the notebook, naming neither the cause nor the cure. The rules that follow from
it, and that this module enforces:

1. **A notebook may not assume another notebook's VM.** Anything on `/content` is this
   session's alone. Only Drive and the Hub survive.
2. **Every prerequisite is checked in one preflight cell**, before any slow work, and the
   error says which notebook produces the missing thing.
3. **There is one implementation of each step.** Two copies of the download logic would drift.
"""

from __future__ import annotations

import glob
import os
import zipfile

REPO_ID = "realslimman/REFUGE-MultiRater"
ZIP_NAME = "REFUGE-Multirater.zip"


def find_release(raw_dir: str) -> str | None:
    """The folder holding Training-400 / Validation-400 / Test-400, or None if absent."""
    matches = glob.glob(os.path.join(raw_dir, "**", "Training-400"), recursive=True)
    roots = sorted({os.path.dirname(p) for p in matches})
    return roots[0] if roots else None


def ensure_release(raw_dir: str, cache_dir: str = "/content/hf") -> str:
    """Return the release root, downloading and extracting it if this VM does not have it.

    The Hub repo holds ONE 1.5 GB zip, not an unpacked tree, so `snapshot_download` plus a
    glob finds a zip and no folders. Roughly 3-4 minutes on a cold VM, instant once extracted.
    """
    root = find_release(raw_dir)
    if root:
        print(f"release already extracted in this session: {root}")
        return root

    from huggingface_hub import hf_hub_download

    os.makedirs(raw_dir, exist_ok=True)
    print(f"{raw_dir} is empty - this is a fresh Colab VM. Fetching the release (~1.5 GB)...")
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_NAME, repo_type="dataset",
                               local_dir=cache_dir)
    print(f"downloaded {os.path.getsize(zip_path) / 1e9:.2f} GB; extracting...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)

    root = find_release(raw_dir)
    if root is None:
        raise RuntimeError(
            f"Extracted {ZIP_NAME} into {raw_dir} but found no Training-400 folder. "
            "The release layout has changed; check the archive before going further.")
    print(f"release ready: {root}")
    return root


def require(paths: dict[str, str]) -> None:
    """Check every prerequisite at once and name the notebook that produces each missing one.

    `paths` maps a file path to the thing that creates it. Raising once, with the whole list,
    beats failing on the first import and hiding the other three.
    """
    missing = [(path, source) for path, source in paths.items() if not os.path.exists(path)]
    if not missing:
        return
    lines = "\n".join(f"    {path}\n        produced by: {source}" for path, source in missing)
    raise FileNotFoundError(
        f"\n\n{len(missing)} prerequisite(s) are missing:\n\n{lines}\n\n"
        "Run the notebook named above first. Drive persists between sessions, so this only "
        "needs doing once - unlike /content, which is wiped with every VM.\n")


def require_gpu(minimum_gb: float = 14.0) -> tuple[str, float]:
    """Stop now, with the menu path, rather than at the first .cuda() call twenty minutes in."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n\nNo GPU in this session. Runtime -> Change runtime type -> GPU -> L4, then "
            "Run all again.\nNotebook 00 is the only one that runs on CPU.\n")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram < minimum_gb:
        raise RuntimeError(
            f"\n\n{name} has only {vram:.0f} GB. This needs about {minimum_gb:.0f} GB at the "
            "configured batch size.\nRuntime -> Change runtime type -> GPU -> L4.\n")
    return name, vram


def batch_size_for(vram_gb: float) -> int:
    """U-Net + EfficientNet-B4 at 512 px with AMP needs roughly 1.5 GB per sample."""
    return 24 if vram_gb > 60 else 16 if vram_gb > 35 else 8 if vram_gb > 20 else 4


def archive_crops(work_dir: str, out_path: str = "/content/crops.tar") -> tuple[str, int, int]:
    """Pack the per-eye .npz crops into ONE tar, written to local disk.

    The archive copy exists for reproducibility, not for training - notebook 02 reads the
    individual .npz files from Drive. So it goes to the Hub as a single file rather than
    1,200 small ones. That matters for more than tidiness:

    * 1,200 separate LFS uploads take far longer and fail *partially*, leaving a repo that
      looks populated but is not. A single file either lands or does not.
    * `upload_large_folder` has no `path_in_repo` parameter, so on any version that provides
      it the files land at the repo ROOT instead of under `crops/`, quietly invalidating
      every path in `manifest.csv`. One file with an explicit path cannot drift.

    Returns (tar path, number of crops packed, bytes).
    """
    import tarfile

    crops_dir = os.path.join(work_dir, "crops")
    names = sorted(f for f in os.listdir(crops_dir) if f.endswith(".npz"))
    if not names:
        raise FileNotFoundError(f"no .npz crops in {crops_dir} - run notebook 01's export first")
    with tarfile.open(out_path, "w") as tar:
        for i, name in enumerate(names, 1):
            tar.add(os.path.join(crops_dir, name), arcname=name)
            if i % 300 == 0:
                print(f"   packed {i}/{len(names)}")
    size = os.path.getsize(out_path)
    print(f"packed {len(names)} crops into {out_path} ({size / 1e6:.0f} MB)")
    return out_path, len(names), size


def verify_upload(api, repo_id: str, expected: dict[str, int]) -> None:
    """Confirm the Hub actually holds what we just sent, with the sizes we sent.

    An upload that half-succeeds is worse than one that fails: the repo looks populated.
    `expected` maps a path in the repo to its byte size.
    """
    info = api.dataset_info(repo_id, files_metadata=True)
    actual = {s.rfilename: s.size for s in info.siblings}
    problems = []
    for path, size in expected.items():
        if path not in actual:
            problems.append(f"    {path} is missing from the repo")
        elif actual[path] is not None and size and actual[path] != size:
            problems.append(f"    {path} is {actual[path]} bytes on the Hub, {size} locally")
    if problems:
        raise RuntimeError("upload did not complete:\n" + "\n".join(problems))
    print(f"verified on the Hub: {', '.join(sorted(expected))}")
