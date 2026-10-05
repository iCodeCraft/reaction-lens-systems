"""Bind paper evidence links to a verified immutable source commit."""
import argparse
import re
import subprocess
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',required=True,help='https://github.com/owner/repository')
p.add_argument('--revision',required=True,help='Full immutable git commit SHA')
p.add_argument('--source-tree', type=Path, default=ROOT, help='Local Git checkout containing the evidence commit')
a=p.parse_args()
if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',a.repo):
    p.error('Expected a GitHub repository URL without a trailing slash')
if not re.fullmatch(r'[a-f0-9]{40}',a.revision):
    p.error('Expected a full lowercase git commit SHA')
# A syntactically plausible SHA is not evidence of a real source revision.
try:
    subprocess.run(['git','-C',str(a.source_tree),'cat-file','-e',a.revision+'^{commit}'],
                   check=True, capture_output=True)
    required=set(json.loads((ROOT/'results/evidence-manifest.json').read_text()))
    required.update(json.loads((ROOT/'measured-source-hashes.json').read_text()))
    required.update(['NUMBERS.csv','artifact-manifest.json','src/reaction_lens/evaluation.py',
                     'results/table1_scoring.csv','results/table2_end_to_end.csv',
                     'results/table3_contracts.csv','results/evidence-manifest.json'])
    for name in sorted(required):
        committed=subprocess.check_output(['git','-C',str(a.source_tree),'show',a.revision+':'+name],
                                          stderr=subprocess.PIPE)
        if committed!=(ROOT/name).read_bytes():
            p.error('Evidence differs from requested commit: '+name)
except subprocess.CalledProcessError:
    p.error('Commit or required evidence file is absent from the local source checkout')
url=a.repo+'/blob/'+a.revision
(ROOT/'paper/artifact-links.tex').write_text(
    '% Public source revision; availability must be verified separately.\n'
    '\\newcommand{\\artifactroot}{'+url+'}\n'
    '\\newcommand{\\repositorynotice}{Code and data: \\url{'+a.repo+'}.}\n'
    '\\newcommand{\\evidence}[2]{\\href{\\artifactroot/#1}{#2}}\n')
print('Bound numeric evidence links to '+url)
print('Run make paper to rebuild the PDF and local source archive. Remote access is not checked.')
