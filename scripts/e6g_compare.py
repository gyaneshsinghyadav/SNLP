#!/usr/bin/env python3
"""Analyze E6-G E3 vs E4 on *matched* DBLP candidate tasks.

Requires identical per-seed candidate fingerprints across variants. For
statistical statements, all paired comparisons are exploratory and five-seed
sign-flip p-values are discrete. Retention metrics use training-era validation.
"""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

DEFAULT_SEEDS=(7,13,42,101,202)
VARIANTS=('e3_control','e4_retention')

def compare(root,output,seeds=DEFAULT_SEEDS,variants=VARIANTS,require_five=True,reference_root=None):
    rows=[];memories=[];retention=[];decisions=[];checkpoints=[]
    for seed in seeds:
        reference=None
        for variant in variants:
            folder=root/variant/f'e6g_dblp_seed{seed}'
            # Use verified archived E3 logs only when no newly executed control exists.
            if (variant=='e3_control' and not (folder/'model_summary.csv').is_file()
                and reference_root is not None):
                folder=reference_root/'e3_control'/f'e6g_dblp_seed{seed}'
            files={key:folder/f'{key}.csv' for key in
                   ('model_summary','candidate_fingerprints','task_audit',
                    'retention','replay_memory','controller_decisions','checkpoint_selection')}
            if not files['model_summary'].exists():
                if require_five: raise FileNotFoundError(files['model_summary'])
                continue
            for name,path in files.items():
                if name=='checkpoint_selection' and variant=='e3_control' and not path.exists():
                    continue  # Archived E3 did not implement E4 checkpoint logs.
                if not path.exists():raise FileNotFoundError(path)
            keys=pd.read_csv(files['candidate_fingerprints'],dtype={'time':str})
            if reference is None:reference=keys
            else:pd.testing.assert_frame_equal(reference,keys,check_exact=True)
            audits=pd.read_csv(files['task_audit'])
            if 'label_overlap_count' in audits and audits.label_overlap_count.ne(0).any():
                raise RuntimeError(f'Task label leakage audit failed: {folder}')
            summary=pd.read_csv(files['model_summary'])
            if set(summary.seed)!={seed}:raise RuntimeError(f'Seed mismatch: {folder}')
            summary.insert(0,'variant',variant)
            rows.append(summary)
            for name,dest in [('retention',retention),('replay_memory',memories),
                              ('controller_decisions',decisions),('checkpoint_selection',checkpoints)]:
                if not files[name].exists():continue
                frame=pd.read_csv(files[name])
                if not frame.empty:
                    frame.insert(0,'variant',variant)
                    dest.append(frame)
    if not rows: raise RuntimeError('No completed results')
    df=pd.concat(rows,ignore_index=True)
    if df.duplicated(['variant','seed','backbone','policy']).any():
        raise RuntimeError('Repeated policy/seed configuration')
    if require_five:
        if set(seeds)!=set(DEFAULT_SEEDS):raise RuntimeError('Require prespecified five seeds')
        for (variant,policy),group in df.groupby(['variant','policy']):
            if set(group.seed)!=set(DEFAULT_SEEDS):raise RuntimeError(f'Missing seeds {variant} {policy}')
    if retention:
        hist=pd.concat(retention,ignore_index=True)
        historical=[]
        for (variant,seed,policy),group in hist.groupby(['variant','seed','policy']):
            oldest=min(group.historical_year.astype(int))
            final=str(sorted(group.after_year.astype(str))[-1])
            sub=group[(group.historical_year.astype(int)==oldest)&
                      (group.after_year.astype(str)==final)]
            if len(sub)==1:
                historical.append({'variant':variant,'seed':seed,'policy':policy,
                    'oldest_year':oldest,'oldest_final_ap':float(sub.iloc[0].historical_ap),
                    'oldest_final_drop':float(sub.iloc[0].ap_drop)})
        df=df.merge(pd.DataFrame(historical),on=['variant','seed','policy'],how='left')
    if decisions:
        acts=pd.concat(decisions,ignore_index=True)
        plasticity=acts.groupby(['variant','seed','policy'],as_index=False).agg(
            mean_current_validation_gain=('current_val_gain','mean'),
            total_current_validation_gain=('current_val_gain','sum'))
        df=df.merge(plasticity,on=['variant','seed','policy'],how='left')
    if memories:
        mem=pd.concat(memories,ignore_index=True)
        summaries=[]
        for (variant,seed,policy),group in mem.groupby(['variant','seed','policy']):
            last=group.iloc[-1]
            summaries.append({'variant':variant,'seed':seed,'policy':policy,
                              'final_memory_positive_share':float(last.positive_share),
                              'final_memory_periods':int(last.periods),
                              'final_memory_count':int(last.stored_pairs)})
        df=df.merge(pd.DataFrame(summaries),on=['variant','seed','policy'],how='left')
    output.mkdir(parents=True,exist_ok=True)
    df.to_csv(output/'matched_seed_scores.csv',index=False)
    if retention:hist.to_csv(output/'retention_all.csv',index=False)
    if memories:mem.to_csv(output/'memory_all.csv',index=False)
    if decisions:pd.concat(decisions,ignore_index=True).to_csv(output/'controller_actions_all.csv',index=False)
    if checkpoints:pd.concat(checkpoints,ignore_index=True).to_csv(output/'checkpoint_selection_all.csv',index=False)
    metrics=('test_mean_ap','test_mean_roc_auc','mean_forgetting','peak_forgetting',
             'oldest_final_ap','oldest_final_drop','final_memory_positive_share','replayed_pairs',
             'rollback_years','rejected_epochs','mean_current_validation_gain')
    aggregate=df.groupby(['variant','backbone','policy'],as_index=False).agg(
        **{f'{k}_mean':(k,'mean') for k in metrics if k in df.columns},
        **{f'{k}_sd':(k,lambda x: x.std(ddof=1)) for k in metrics if k in df.columns},
        n_seeds=('seed','nunique'))
    aggregate.to_csv(output/'five_seed_summary.csv',index=False)
    comps=[]
    for (backbone,policy),grp in df.groupby(['backbone','policy']):
        base=grp[grp.variant=='e3_control'].set_index('seed')
        if base.empty:continue
        for variant in variants:
            if variant=='e3_control':continue
            test=grp[grp.variant==variant].set_index('seed')
            common=sorted(set(base.index)&set(test.index))
            if len(common)<2:continue
            for metric in ('test_mean_ap','mean_forgetting','peak_forgetting','oldest_final_drop'):
                if metric not in test:continue
                diff=(test.loc[common,metric]-base.loc[common,metric]).to_numpy(float)
                n=len(diff);mean=float(diff.mean());ci=float(t.ppf(.975,n-1)*diff.std(ddof=1)/np.sqrt(n))
                null=np.array([abs(float(np.mean(diff*np.asarray(signs))))
                               for signs in product((-1,1),repeat=n)])
                comps.append({'variant':variant,'baseline':'e3_control','backbone':backbone,
                    'policy':policy,'metric':metric,'n':n,'difference':mean,
                    'ci95_low':mean-ci,'ci95_high':mean+ci,
                    'two_sided_exact_sign_flip_p':float(np.mean(null>=abs(mean)-1e-12)),
                    'improved_seeds':int((diff>0).sum() if metric=='test_mean_ap' else (diff<0).sum())})
    pd.DataFrame(comps).to_csv(output/'paired_E3_comparisons.csv',index=False)
    try:
        import matplotlib.pyplot as plt
        for metric,label,filename in [('test_mean_ap','Test AP','ap_by_variant.png'),
                                      ('mean_forgetting','Mean historical forgetting','forgetting_by_variant.png'),
                                      ('oldest_final_drop','Oldest-task final AP drop','oldest_task_drop_by_variant.png')]:
            if metric not in df:continue
            selected=df[df.policy=='dqn'].groupby('variant')[metric].agg(['mean','std']).reindex(variants).dropna()
            if selected.empty:continue
            fig,ax=plt.subplots(figsize=(7,4))
            ax.bar(selected.index.str.upper(),selected['mean'],yerr=selected['std'],capsize=4)
            ax.set_ylabel(f'{label} (fraction 0–1)')
            ax.set_title('E6-G DQN: E3 vs E4 matched DBLP tasks')
            ax.grid(axis='y',alpha=.25)
            fig.tight_layout();fig.savefig(output/filename,dpi=160);plt.close(fig)
    except ImportError:pass
    print(aggregate.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    print('Wrote',output)
    return aggregate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,default=Path('results/e6g_checkpoint'))
    p.add_argument('--output',type=Path,default=Path('results/e6g_checkpoint_analysis'))
    p.add_argument('--seeds',default='7,13,42,101,202')
    p.add_argument('--variants',default='e3_control,e4_retention')
    p.add_argument('--require-five',action='store_true')
    p.add_argument('--reference-root',type=Path,default=Path('docs/e6g_e3_reference'))
    a=p.parse_args()
    compare(a.results_dir,a.output,tuple(int(x) for x in a.seeds.split(',')),
            tuple(x.strip() for x in a.variants.split(',')),a.require_five,a.reference_root)
if __name__=='__main__':main()
