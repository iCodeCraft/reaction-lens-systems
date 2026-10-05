"""Download only the pinned inference and evaluation artifacts, verifying all hashes."""
import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from reaction_lens.artifacts import verify_files, load_bundle


def ensure(repo, revision, folder, hashes, anonymous):
    if not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Use an immutable 40-character commit hash')
    if not hashes:
        raise ValueError('Artifact hash manifest must not be empty')
    try:
        verify_files(folder, hashes)
        return
    except ValueError:
        pass
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    try:
        snapshot_download(repo, revision=revision, local_dir=folder,
                          allow_patterns=list(hashes), token=False if anonymous else None)
    except (HfHubHTTPError, LocalEntryNotFoundError) as error:
        raise SystemExit(f'Cannot obtain pinned artifacts for {repo}@{revision}. '
                         'Check repository access and network connectivity; see README. '
                         f'{type(error).__name__}') from None
    verify_files(folder, hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', action='store_true')
    parser.add_argument('--anonymous', action='store_true', help='Disable saved HF credentials')
    args = parser.parse_args()
    pin = json.loads((ROOT / 'artifact-manifest.json').read_text())
    bundle = ROOT / 'artifacts/structured-seed23'
    ensure(pin['repo_id'], pin['revision'], bundle, pin['files_sha256'], args.anonymous)
    manifest, _, _ = load_bundle(bundle)
    if args.encoder:
        for name in ('base', 'adapter'):
            spec = manifest['encoder'][name]
            ensure(spec['repo'], spec['revision'], ROOT / 'artifacts/encoder' / name,
                   spec['files_sha256'], args.anonymous)
    print('Artifact hashes verified. Existing verified files require no network access.')


if __name__ == '__main__':
    main()
