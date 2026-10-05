"""Validate raw observations and preserved implementation using only the stdlib."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def quantile(values, q):
    values=sorted(values)
    at=(len(values)-1)*q
    lo=math.floor(at); hi=math.ceil(at)
    return values[lo]+(values[hi]-values[lo])*(at-lo)


def summary(rows, key, stored):
    values=[r[key] for r in rows]
    require(all(math.isfinite(x) and x>=0 for x in values),'Invalid timing')
    expected={'count':len(values),'median':statistics.median(values),
              'p95':quantile(values,.95),'min':min(values),'max':max(values)}
    for k,v in expected.items():
        require(math.isclose(v,stored[k],rel_tol=1e-12,abs_tol=1e-12), 'Summary mismatch: '+k)


def main():
    hashes=json.loads((ROOT/'measured-source-hashes.json').read_text())
    for name, expected in hashes.items():
        local=name
        require(hashlib.sha256((ROOT/local).read_bytes()).hexdigest()==expected,
                'Measured implementation changed: '+local)
    raw=json.loads((ROOT/'results/system-benchmark.json').read_text())
    for name, expected in raw['source_sha256'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'Measured source changed: '+name)
    ev=json.loads((ROOT/'results/fresh-evaluation.json').read_text())
    pin=json.loads((ROOT/'artifact-manifest.json').read_text())
    require(raw['bundle_sha256']==pin['files_sha256']['manifest.json'], 'Bundle identity mismatch')
    require(raw['evaluation_sha256']==pin['files_sha256']['evaluation/manifest.json'], 'Evaluation identity mismatch')
    catalog=ev['catalog_reactions']
    blocks=[128,1024,catalog]
    require([r['block_size'] for r in raw['cached_feature_runs']]==blocks, 'Block coverage mismatch')
    require([r['block_size'] for r in raw['memory_runs']]==blocks, 'Memory block coverage mismatch')
    require(raw['protocol']['absolute_logit_tolerance']==1e-5, 'Unexpected contract tolerance')
    require(raw['protocol']['warmup_queries']==5 and raw['protocol']['seed']==20261005, 'Protocol mismatch')
    checks=[]
    samples=raw['protocol']['samples']; repeats=raw['protocol']['repeats']
    for run in raw['cached_feature_runs']:
        require(len(run['sample_indices'])==samples and len(set(run['sample_indices']))==samples,'Sample mismatch')
        require(len(run['timings'])==samples*repeats,'Missing latency observations')
        require(len(run['contracts'])==samples*4,'Missing contract probes')
        require({x['scenario'] for x in run['contracts']}==
                {'permutation','subset','append_duplicates','unblocked_reference'},'Missing scenario')
        require({(r['repeat'],r['row']) for r in run['timings']}==
                {(rep,i) for rep in range(repeats) for i in run['sample_indices']},'Timing coverage mismatch')
        summary(run['timings'],'seconds',run['latency_seconds'])
        require(run['catalog_size']==catalog, 'Catalog mismatch')
        require(run['sample_indices']==raw['cached_feature_runs'][0]['sample_indices'], 'Unpaired samples')
        require(all(type(i) is int and 0<=i<ev['publications'] for i in run['sample_indices']), 'Invalid sample index')
        require({(c['row'],c['scenario']) for c in run['contracts']}==
                {(i,k) for i in run['sample_indices'] for k in
                 ('permutation','subset','append_duplicates','unblocked_reference')}, 'Contract coverage mismatch')
        for c in run['contracts']:
            require(c['comparisons']==(catalog//2 if c['scenario']=='subset' else catalog), 'Wrong comparison denominator')
            require(math.isfinite(c['max_absolute_logit_error']) and c['max_absolute_logit_error']>=0,
                    'Invalid contract error')
            require(type(c['selection_flips']) is int and 0<=c['selection_flips']<=c['comparisons'], 'Invalid flip count')
        checks.extend(run['contracts'])
    expected={'checks':len(checks),'score_comparisons':sum(x['comparisons'] for x in checks),
              'max_absolute_logit_error':max(x['max_absolute_logit_error'] for x in checks),
              'selection_flips':sum(x['selection_flips'] for x in checks)}
    require(expected==raw['contract_summary'],'Contract aggregate mismatch')
    require(expected['max_absolute_logit_error']<=raw['protocol']['absolute_logit_tolerance']
            and expected['selection_flips']==0 and raw['status']=='passed','Contract failed')
    for row in raw['memory_runs']:
        require(row['output_shape']==[row['batch'],catalog] and row['batch']==min(16,samples), 'Memory shape mismatch')
        require(row['all_finite'] and row['process_peak_rss_bytes']>=row['pre_scoring_peak_rss_bytes'],
                'Invalid memory observation')
    end=raw['end_to_end']
    summary(end['rows'],'predict_seconds',end['predict_seconds'])
    require(hashlib.sha256((ROOT/'examples/latency-examples.json').read_bytes()).hexdigest()==
            end['examples_sha256'],'Latency input changed')
    examples=json.loads((ROOT/'examples/latency-examples.json').read_text())['examples']
    require(len(end['rows'])==len(examples)*repeats and
            {(r['repeat'],r['pmid']) for r in end['rows']}==
            {(rep,x['pmid']) for rep in range(repeats) for x in examples}, 'End-to-end coverage mismatch')
    require(end['warmup_queries']==1, 'End-to-end warmup mismatch')
    for row in end['rows']:
        require(row['options_scored']==catalog, 'Incomplete end-to-end catalog')
        require(all(math.isfinite(row[k]) and row[k]>=0 for k in
                    ('encoding_seconds','scoring_seconds','request_seconds','predict_seconds','predict_plus_json_seconds')),
                'Invalid component timing')
        require(row['encoding_seconds']<=row['predict_seconds']<=row['predict_plus_json_seconds'],
                'Invalid component timing')
    ev=json.loads((ROOT/'results/fresh-evaluation.json').read_text())
    require(ev['absolute_tolerance']==0.00005, 'Unexpected metric tolerance')
    require(set(ev['measured'])==set(ev['expected'])==set(ev['absolute_differences'])==
            {'MAP','Hit@1','Recall@100'}, 'Missing metric')
    require(all(type(ev[k]) is int and ev[k]>0 for k in ('publications','catalog_reactions','known_links')), 'Invalid metric denominator')
    for key,v in ev['measured'].items():
        require(math.isfinite(v) and 0<=v<=1 and math.isfinite(ev['expected'][key]) and
                0<=ev['expected'][key]<=1, 'Invalid metric')
        require(abs(v-ev['expected'][key])==ev['absolute_differences'][key], 'Metric difference mismatch')
        require(ev['absolute_differences'][key]<=ev['absolute_tolerance'],'Metric tolerance failed')
    require(ev['status']=='passed','Model evaluation failed')
    print(json.dumps({'status':'passed','measured_files_unchanged':len(hashes),
                      'contract_checks':len(checks),'fresh_evaluation':'passed',
                      'scope':'Local evidence consistency; not external review or public access validation'}))


if __name__=='__main__':
    main()
