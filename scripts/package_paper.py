"""Package the separate paper, its numerical evidence and license deterministically."""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
names=['paper/reaction-lens-systems.tex','paper/reaction-lens-systems.bbl',
       'paper/preamble.tex','paper/numbers.tex','paper/artifact-links.tex',
       'paper/references.bib','paper/LICENSE','NUMBERS.csv',
       'results/table1_scoring.csv','results/table2_end_to_end.csv',
       'results/table3_contracts.csv','results/fresh-evaluation.json',
       'results/system-benchmark.json','PROTOCOL.md']
dest=ROOT/'.local/paper/reaction-lens-systems-source.zip'
dest.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(dest,'w',compression=ZIP_DEFLATED) as z:
    for name in sorted(names):
        entry=ZipInfo(name,date_time=(2026,10,5,0,0,0))
        entry.compress_type=ZIP_DEFLATED
        entry.external_attr=0o100644 << 16
        z.writestr(entry,(ROOT/name).read_bytes())
print(dest)
