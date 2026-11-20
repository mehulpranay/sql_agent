"""
make_demo_assets.py  --  prepare the small set of files the container ships with.

Run it ONCE, on your own machine, where the full spider_schema_lookup.pkl (made by
preprocessing.py) and Spider's `database` folder already exist:

    python make_demo_assets.py

It creates a folder `demo_assets/` that you commit to the repo:

    demo_assets/spider_schema_lookup.pkl         only the demo databases (see demo_dbs.py)
    demo_assets/databases/<db_id>/<db_id>.sqlite the demo databases themselves (read-only data)
    demo_assets/constraints.txt                  exact versions of numpy, networkx and
                                                 sentence-transformers used to make the pickle
    demo_assets/ATTRIBUTION.md                   where the data comes from

Why this way, instead of rebuilding everything while the image is built
    * The schema lookup is built from the SAME databases it describes, so they must travel
      together. Shipping both in the image makes that impossible to get wrong.
    * Rebuilding needs your whole Spider download and a dataset download at build time. The
      host would have to do that, and you could not test it locally.
    * The pickle holds numpy arrays and a networkx graph. Pickles load safely only with
      compatible library versions, so constraints.txt makes the image install the versions
      that made it.
"""

import os
import sys
import pickle
import shutil
import argparse
import subprocess
from pathlib import Path
from importlib import metadata

from demo_dbs import DEMO_DATABASES

REQUIRED_KEYS = ('schema', 'col_dicts', 'embeddings', 'graph')
EMBEDDING_DIM = 384
PINNED_PACKAGES = ('numpy', 'networkx', 'sentence-transformers')

# Libraries the pickle must NOT need. The service image does not install them (they are large
# and only preprocessing.py uses them), so a pickle that refers to them would fail at startup.
NOT_IN_SERVICE_IMAGE = ('pandas', 'sklearn', 'datasets', 'pyarrow')

ATTRIBUTION = """# Data attribution

The SQLite databases in this folder come from the Spider text-to-SQL dataset
(Yu et al., Yale University LILY Lab, https://yale-lily.github.io/spider).
Spider is distributed under the CC BY-SA 4.0 license; check the license file that came with
your download. The databases are included unmodified.
"""


def _check_entry(db_id, entry):
    missing = [k for k in REQUIRED_KEYS if k not in entry]
    if missing:
        raise SystemExit(f'{db_id}: the schema lookup entry is missing {missing}.')
    n_cols, dim = entry['embeddings'].shape
    if dim != EMBEDDING_DIM or n_cols != len(entry['col_dicts']):
        raise SystemExit(
            f'{db_id}: embeddings have shape {entry["embeddings"].shape} but there are '
            f'{len(entry["col_dicts"])} columns (expected one {EMBEDDING_DIM}-number row per column).'
        )


def _check_loads_without_heavy_libraries(pickle_path):
    """Load the pickle in a fresh interpreter in which pandas, sklearn etc. cannot be imported."""
    code = (
        'import sys, pickle\n'
        f'for name in {NOT_IN_SERVICE_IMAGE!r}: sys.modules[name] = None\n'
        f'pickle.load(open({str(pickle_path)!r}, "rb"))\n'
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            'The trimmed pickle needs a library the service image does not install:\n'
            + result.stderr.strip().splitlines()[-1]
        )


def build_assets(src_pkl, src_db_dir, out_dir):
    src_pkl, src_db_dir, out = Path(src_pkl), Path(src_db_dir), Path(out_dir)

    if not src_pkl.is_file():
        raise SystemExit(f'Schema lookup not found: {src_pkl}. Run preprocessing.py first.')
    if not src_db_dir.is_dir():
        raise SystemExit(f'Spider database folder not found: {src_db_dir}. Set SPIDER_DB_PATH.')

    with open(src_pkl, 'rb') as f:
        full_lookup = pickle.load(f)

    trimmed, sqlite_files = {}, {}
    for db_id in DEMO_DATABASES:
        if db_id not in full_lookup:
            raise SystemExit(f'{db_id!r} is not in {src_pkl}.')
        sqlite_file = src_db_dir / db_id / f'{db_id}.sqlite'
        if not sqlite_file.is_file():
            raise SystemExit(f'Database file not found: {sqlite_file}')
        _check_entry(db_id, full_lookup[db_id])
        trimmed[db_id], sqlite_files[db_id] = full_lookup[db_id], sqlite_file

    # start clean, so a database removed from demo_dbs.py does not linger in the folder
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    pickle_path = out / 'spider_schema_lookup.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(trimmed, f)

    for db_id, sqlite_file in sqlite_files.items():
        target = out / 'databases' / db_id
        target.mkdir(parents=True)
        shutil.copy2(sqlite_file, target / f'{db_id}.sqlite')

    lines = []
    for package in PINNED_PACKAGES:
        try:
            lines.append(f'{package}=={metadata.version(package)}')
        except metadata.PackageNotFoundError:
            print(f'WARNING: {package} is not installed here, so its version is not pinned.')
    (out / 'constraints.txt').write_text('\n'.join(lines) + '\n')
    (out / 'ATTRIBUTION.md').write_text(ATTRIBUTION)

    _check_loads_without_heavy_libraries(pickle_path)

    total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    print(f'Wrote {out}/  ({total / 1e6:.2f} MB in total)')
    for db_id in trimmed:
        print(f'  {db_id}: {len(trimmed[db_id]["col_dicts"])} columns')
    print('Pinned versions:', ', '.join(lines))
    return out


def main():
    parser = argparse.ArgumentParser(description='Collect the files the container ships with.')
    parser.add_argument('--out', default='demo_assets')
    parser.add_argument('--schema-pkl', default=os.environ.get('SCHEMA_PKL_PATH', 'spider_schema_lookup.pkl'))
    parser.add_argument('--spider-db', default=os.environ.get('SPIDER_DB_PATH'))
    args = parser.parse_args()

    if not args.spider_db:
        raise SystemExit('Set SPIDER_DB_PATH (or pass --spider-db) to Spider\'s `database` folder.')
    build_assets(args.schema_pkl, args.spider_db, args.out)


if __name__ == '__main__':
    main()
