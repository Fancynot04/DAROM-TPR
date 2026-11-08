"""Read-only reward-component analysis for B0--B3 artifacts.

Inputs may be a shadow-audit directory containing ``audit.json`` or JSONL
records with per-step ``reward_terms``.  Missing step terms are warnings, never
converted to zero-valued samples.
"""
from __future__ import annotations

import argparse, hashlib, json, math, warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVENTS = ('success', 'collision', 'missed_exit', 'teleport', 'timeout')
DENSE = ('efficiency', 'route')
COLORS = {'events': '#4c78a8', 'efficiency': '#f58518', 'route': '#54a24b'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_terms(terms):
    names = {item['name'] for item in terms}
    return 'e76_d3_dense_v1' if names.intersection(DENSE) else 'events_only_v1'


def audit_records(path, gamma):
    source = path / 'audit.json' if path.is_dir() else path
    payload = json.loads(source.read_text())
    if not isinstance(payload, list):
        raise ValueError(f'{source} is not a shadow-audit list')
    records, warnings_out = [], []
    warnings_out.append(f'{source}: no resolved reward configuration hash; profile is inferred from terms only')
    for episode, item in enumerate(payload):
        for row in item.get('rows', []):
            if 'terms' not in row:
                warnings_out.append(f'{item.get("name", episode)}: missing per-step terms')
                continue
            records.append({'source': str(source), 'source_sha256': sha256(source),
                            'group': path.name if path.is_dir() else source.stem,
                            'episode_id': item.get('name', str(episode)), 'step': row['step'] - 1,
                            'gamma': gamma, 'reward': float(row['reward']),
                            'outcome': row.get('terminal') or None, 'profile': profile_terms(row['terms']),
                            'terms': row['terms']})
    return records, warnings_out


def jsonl_records(path, gamma):
    records, warnings_out = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try: row = json.loads(line)
        except json.JSONDecodeError as error:
            warnings_out.append(f'{path}:{number}: invalid JSON: {error}')
            continue
        if 'reward_terms' not in row:
            warnings_out.append(f'{path}:{number}: no per-step reward_terms; skipped')
            continue
        if 'reward' not in row:
            warnings_out.append(f'{path}:{number}: no environment reward; skipped')
            continue
        records.append({'source': str(path), 'source_sha256': sha256(path), 'group': path.parent.name,
                        'episode_id': str(row.get('episode_id', row.get('episode', 'unknown'))),
                        'step': int(row.get('episode_step', row.get('step', number - 1))), 'gamma': gamma,
                        'reward': float(row['reward']), 'outcome': row.get('terminal_event'),
                        'terms': row['reward_terms']})
    return records, warnings_out


def load_inputs(paths, gamma):
    all_records, messages = [], []
    for raw in paths:
        path = Path(raw)
        if path.is_dir() and (path / 'audit.json').exists():
            records, warning_messages = audit_records(path, gamma)
        elif path.is_file() and path.name == 'audit.json':
            records, warning_messages = audit_records(path, gamma)
        elif path.is_file() and path.suffix == '.jsonl':
            records, warning_messages = jsonl_records(path, gamma)
        else:
            messages.append(f'{path}: unsupported input; expected audit.json directory/file or JSONL')
            continue
        all_records.extend(records); messages.extend(warning_messages)
    if not all_records:
        raise ValueError('No usable per-step reward records found')
    return all_records, messages


def analyse(records, tolerance=1e-8):
    episodes, samples, issues = [], [], []
    by_episode = {}
    for record in records:
        by_episode.setdefault((record['group'], record['episode_id']), []).append(record)
    for (group, episode_id), rows in by_episode.items():
        rows.sort(key=lambda row: row['step'])
        gamma = rows[0]['gamma']
        paid_by_name, discounted_by_name = {}, {}
        total, discounted_total = 0.0, 0.0
        for index, row in enumerate(rows):
            if row['step'] != index:
                issues.append(f'{group}/{episode_id}: non-contiguous or nonzero-based step index')
            reward = row['reward']; total += reward; discounted_total += (gamma ** index) * reward
            paid = 0.0
            for term in row['terms']:
                required = {'name', 'raw', 'weighted', 'enabled', 'active', 'clipped'}
                if not required.issubset(term):
                    issues.append(f'{group}/{episode_id}/step{index}: incomplete term {term.get("name")}')
                    continue
                name = term['name']; raw = float(term['raw']); clipped = float(term.get('clipped_value', raw) if term.get('clipped_value') is not None else raw)
                weighted = float(term['weighted'])
                effective_weight = float(term.get('effective_weight', term.get('weight', 0.0)))
                base_weight = float(term.get('base_weight', term.get('weight', 0.0)))
                decay_factor = term.get('decay_factor')
                if decay_factor is not None:
                    decay_factor = float(decay_factor)
                finite = (raw, clipped, weighted, effective_weight, base_weight)
                if (not all(math.isfinite(value) for value in finite)
                        or (decay_factor is not None and not math.isfinite(decay_factor))):
                    issues.append(f'{group}/{episode_id}/step{index}: nonfinite {name}')
                    continue
                paid += weighted; paid_by_name[name] = paid_by_name.get(name, 0.0) + weighted
                discounted_by_name[name] = discounted_by_name.get(name, 0.0) + (gamma ** index) * weighted
                samples.append({'group': group, 'episode_id': episode_id, 'step': index, 'name': name,
                                'raw': raw, 'clipped_value': clipped, 'weighted': weighted,
                                'effective_weight': effective_weight, 'base_weight': base_weight,
                                'decay_factor': decay_factor,
                                'enabled': bool(term['enabled']), 'active': bool(term['active']),
                                'clipped': bool(term['clipped']), 'gamma': gamma})
            if not math.isclose(paid, reward, abs_tol=tolerance, rel_tol=0):
                issues.append(f'{group}/{episode_id}/step{index}: term sum {paid} != reward {reward}')
        if not math.isclose(sum(paid_by_name.values()), total, abs_tol=tolerance, rel_tol=0):
            issues.append(f'{group}/{episode_id}: undiscounted decomposition mismatch')
        if not math.isclose(sum(discounted_by_name.values()), discounted_total, abs_tol=tolerance, rel_tol=0):
            issues.append(f'{group}/{episode_id}: discounted decomposition mismatch')
        absolute = sum(abs(value) for value in discounted_by_name.values())
        episode = {'group': group, 'episode_id': episode_id, 'gamma': gamma, 'steps': len(rows),
                   'total_reward': total, 'discounted_return': discounted_total,
                   'outcome': next((r['outcome'] for r in reversed(rows) if r['outcome']), None)}
        for name, value in paid_by_name.items():
            episode[f'{name}_undiscounted'] = value
            episode[f'{name}_discounted'] = discounted_by_name[name]
            episode[f'{name}_discounted_abs_share'] = (abs(discounted_by_name[name]) / absolute if absolute else np.nan)
        episodes.append(episode)
    return pd.DataFrame(episodes), pd.DataFrame(samples), issues


def component_stats(samples, episodes):
    rows = []
    for (group, name), frame in samples.groupby(['group', 'name']):
        expected = int(episodes.loc[episodes.group == group, 'steps'].sum())
        result = {'group': group, 'name': name, 'samples': len(frame), 'term_absent_steps': expected - len(frame),
                  'active_samples': int(frame.active.sum()),
                  'enabled_samples': int(frame.enabled.sum()), 'disabled_samples': int((~frame.enabled).sum()),
                  'clipped_samples': int(frame.clipped.sum()), 'clipped_ratio': float(frame.clipped.mean())}
        for prefix, series in (('raw', frame.raw), ('clipped', frame.clipped_value), ('weighted', frame.weighted),
                               ('effective_weight', frame.effective_weight), ('base_weight', frame.base_weight)):
            absolute = series.abs(); nonzero = absolute[absolute > 0]
            result.update({f'{prefix}_mean': series.mean(), f'{prefix}_min': series.min(), f'{prefix}_max': series.max(),
                           f'{prefix}_p95': series.quantile(.95), f'{prefix}_p99': series.quantile(.99),
                           f'{prefix}_abs_p95': absolute.quantile(.95), f'{prefix}_abs_p99': absolute.quantile(.99),
                           f'{prefix}_nonzero_abs_p95': nonzero.quantile(.95) if len(nonzero) else np.nan,
                           f'{prefix}_nonzero_abs_p99': nonzero.quantile(.99) if len(nonzero) else np.nan,
                           f'{prefix}_max_abs': absolute.max(), f'{prefix}_nonzero_ratio': float((absolute > 0).mean())})
        rows.append(result)
    return pd.DataFrame(rows)


def plots(episodes, samples, output):
    groups = episodes.group.unique(); names = [name for name in ('success','collision','missed_exit','teleport','timeout','efficiency','route') if f'{name}_discounted' in episodes]
    fig, ax = plt.subplots(figsize=(9, 5)); means = episodes.groupby('group')[[f'{n}_discounted' for n in names]].mean()
    x = np.arange(len(means)); bottom_pos=np.zeros(len(means)); bottom_neg=np.zeros(len(means))
    for name in names:
        values=means[f'{name}_discounted'].to_numpy(); color=COLORS['events' if name in EVENTS else name]
        ax.bar(x, np.maximum(values,0), bottom=bottom_pos, label=name, color=color); bottom_pos += np.maximum(values,0)
        ax.bar(x, np.minimum(values,0), bottom=bottom_neg, color=color); bottom_neg += np.minimum(values,0)
    ax.set_xticks(x, means.index); ax.set_ylabel('mean discounted paid contribution'); ax.legend(ncol=2); ax.axhline(0,color='black',lw=.7); fig.tight_layout(); fig.savefig(output/'reward_contributions.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); x=np.arange(len(episodes)); bottom_pos=np.zeros(len(episodes)); bottom_neg=np.zeros(len(episodes))
    for name in names:
        values=episodes[f'{name}_undiscounted'].fillna(0).to_numpy(); color=COLORS['events' if name in EVENTS else name]
        ax.bar(x,np.maximum(values,0),bottom=bottom_pos,label=name,color=color); bottom_pos+=np.maximum(values,0)
        ax.bar(x,np.minimum(values,0),bottom=bottom_neg,color=color); bottom_neg+=np.minimum(values,0)
    ax.plot(x,episodes.total_reward,'ko',label='total return'); ax.set_ylabel('undiscounted paid return'); ax.legend(ncol=3); ax.axhline(0,color='black',lw=.7); fig.tight_layout(); fig.savefig(output/'return_decomposition.png',dpi=180); plt.close(fig)
    dense=samples[samples.name.isin(DENSE)]; fig, axes=plt.subplots(2,2,figsize=(10,7))
    for axis, name in zip(axes.flat, DENSE):
        frame=dense[dense.name==name]
        if frame.empty: axis.text(.5,.5,'N/A: no samples',ha='center'); continue
        axis.hist(frame.raw,bins=50,alpha=.55,label='raw'); axis.hist(frame.clipped_value,bins=50,alpha=.55,label='clipped')
        axis.axvline(-1,color='red',ls='--'); axis.axvline(0,color='black',ls='--'); axis.set_title(name); axis.legend()
    axes[1,0].set_xlabel('value'); axes[1,1].axis('off'); fig.tight_layout(); fig.savefig(output/'reward_clipping_distribution.png',dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('inputs',nargs='+'); parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--gamma',type=float,default=.999)
    args=parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(f'Output exists: {args.output_dir}')
    args.output_dir.mkdir(parents=True)
    records, warnings_out=load_inputs(args.inputs,args.gamma); episodes,samples,issues=analyse(records)
    components=component_stats(samples, episodes); episodes.to_csv(args.output_dir/'episodes.csv',index=False); components.to_csv(args.output_dir/'components.csv',index=False); plots(episodes,samples,args.output_dir)
    summary={'inputs':[str(Path(x)) for x in args.inputs],'gamma':args.gamma,'step_samples':len(samples),'episodes':len(episodes),'warnings':warnings_out,'consistency_issues':issues,'components':json.loads(components.to_json(orient='records'))}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'README.md').write_text('Read-only reward decomposition. raw is pre-clip; clipped_value is post-clip; weighted is paid. Missing per-step logs are warnings, not zeros.\n')
    print(json.dumps({'episodes':len(episodes),'samples':len(samples),'warnings':len(warnings_out),'issues':len(issues)}))

if __name__=='__main__': main()
