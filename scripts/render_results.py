"""Rebuild tables, paper macros and numerical evidence from recorded observations."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def csv_text(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def produce():
    raw = json.loads((ROOT/'results/system-benchmark.json').read_text())
    ev = json.loads((ROOT/'results/fresh-evaluation.json').read_text())
    numbers, macros = [], []

    def claim(name, value, unit, source, pointer, places=None):
        display = str(value) if places is None else f'{value:.{places}f}'
        numbers.append(dict(claim=name, value=value, displayed_value=display,
                            unit=unit, source=source, pointer=pointer,
                            generator='scripts/render_results.py'))
        macros.append('\\newcommand{\\' + name + '}{\\evidence{NUMBERS.csv}{' + display + '}}')
        return display

    src = 'results/system-benchmark.json'
    claim('SampleCount', raw['protocol']['samples'], 'publications', src, '/protocol/samples')
    claim('RepeatCount', raw['protocol']['repeats'], 'repeats', src, '/protocol/repeats')
    claim('MemoryBatch', raw['memory_runs'][0]['batch'], 'publications', src, '/memory_runs/0/batch')
    claim('MemorySpans', raw['memory_runs'][0]['max_sentences'], 'spans', src, '/memory_runs/0/max_sentences')
    c = raw['contract_summary']
    claim('ContractChecks', c['checks'], 'checks', src, '/contract_summary/checks')
    claim('ScoreComparisons', c['score_comparisons'], 'comparisons', src, '/contract_summary/score_comparisons')
    claim('MaxError', c['max_absolute_logit_error'], 'logit', src, '/contract_summary/max_absolute_logit_error', 1)
    claim('DecisionFlips', c['selection_flips'], 'decisions', src, '/contract_summary/selection_flips')
    rows = []
    for i, run in enumerate(raw['cached_feature_runs']):
        mem = raw['memory_runs'][i]
        if mem['block_size'] != run['block_size']:
            raise ValueError('Memory and latency block identities differ')
        tag = ['Small', 'Default', 'Full'][i]
        root = f'/cached_feature_runs/{i}'
        claim(tag+'Block', run['block_size'], 'options', src, root+'/block_size')
        med = claim(tag+'Median', run['latency_seconds']['median']*1000, 'ms', src, root+'/timings/*/seconds -> median * 1000', 3)
        p95 = claim(tag+'Pninetyfive', run['latency_seconds']['p95']*1000, 'ms', src, root+'/timings/*/seconds -> quantile(.95) * 1000', 3)
        rss = claim(tag+'Memory', mem['process_peak_rss_bytes']/2**20, 'MiB', src, f'/memory_runs/{i}/process_peak_rss_bytes / 1048576', 2)
        rows.append(dict(block_size=run['block_size'], median_ms=med, p95_ms=p95,
                         process_peak_rss_MiB=rss, requests=len(run['timings'])))
    end = raw['end_to_end']
    claim('EndRequests', len(end['rows']), 'requests', src, '/end_to_end/rows -> length')
    claim('EndExamples', len({x['pmid'] for x in end['rows']}), 'publications', src, '/end_to_end/rows/*/pmid -> unique count')
    claim('EndMedian', end['predict_seconds']['median']*1000, 'ms', src, '/end_to_end/rows/*/predict_seconds -> median * 1000', 2)
    claim('EndPninetyfive', end['predict_seconds']['p95']*1000, 'ms', src, '/end_to_end/rows/*/predict_seconds -> quantile(.95) * 1000', 2)
    claim('SetupSeconds', end['setup_seconds'], 's', src, '/end_to_end/setup_seconds', 2)
    claim('EndMemory', end['process_peak_rss_bytes']/2**20, 'MiB', src, '/end_to_end/process_peak_rss_bytes / 1048576', 2)
    fraction = sum(x['encoding_seconds'] for x in end['rows'])/sum(x['predict_seconds'] for x in end['rows'])
    claim('EncodingShare', fraction*100, 'percent', src, '/end_to_end/rows -> sum(encoding_seconds) / sum(predict_seconds) * 100', 2)
    es = 'results/fresh-evaluation.json'
    for name, key in [('PublicationCount','publications'), ('CatalogCount','catalog_reactions'), ('KnownLinks','known_links')]:
        claim(name, ev[key], 'count', es, '/'+key)
    for name, key in [('MeasuredMAP','MAP'),('MeasuredHit','Hit@1'),('MeasuredRecall','Recall@100')]:
        claim(name, ev['measured'][key], 'fraction', es, '/measured/'+key, 4)
    claim('MetricError', max(ev['absolute_differences'].values()), 'absolute difference', es, '/absolute_differences -> max', 1)
    result = {
        'NUMBERS.csv': csv_text(numbers),
        'results/table1_scoring.csv': csv_text(rows),
        'results/table2_end_to_end.csv': csv_text(end['rows']),
        'results/table3_contracts.csv': csv_text([dict(block_size=r['block_size'], **x) for r in raw['cached_feature_runs'] for x in r['contracts']]),
        'paper/numbers.tex': '\n'.join(macros)+'\n',
    }
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ['results/system-benchmark.json','results/fresh-evaluation.json','PROTOCOL.md',
               'scripts/benchmark_system.py','scripts/check_release.py','scripts/render_results.py']}
    result['results/evidence-manifest.json'] = json.dumps(hashes,indent=2)+'\n'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    result=produce()
    for name, text in result.items():
        path=ROOT/name
        if args.check:
            if not path.exists() or path.read_bytes()!=text.encode():
                raise SystemExit('Generated evidence is stale: '+name)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(text.encode())
    print(json.dumps({'status':'passed','generated_files':len(result),'mode':'check' if args.check else 'render'}))


if __name__=='__main__':
    main()
