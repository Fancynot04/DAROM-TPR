"""Plot the archived D metrics and digitized paper curves without network access."""
import argparse
import csv
import os
import subprocess
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]
PROJECT = ARCHIVE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', type=int, default=50000, help='Trailing loss-mean window in environment steps')
    args = parser.parse_args()
    if args.window <= 0:
        parser.error('--window must be positive')
    output = PROJECT/'tmp/ai/research-archive/d-paper-comparison.png'
    subprocess.run(['git','check-ignore','--quiet',str(output)],cwd=PROJECT,check=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Configure only this plotting process; do not change the shell or training environment.
    os.environ['MPLCONFIGDIR'] = str(output.parent)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (ARCHIVE/'reproduction/d_metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    for ax,(name,key) in zip(axes,[('return','train/reward_mean_100'),('critic','train/critic_loss'),('actor','train/actor_loss')]):
        selected=[r for r in rows if r[key]]
        x=np.array([int(r['env_steps']) for r in selected])
        y=np.array([float(r[key]) for r in selected])
        ax.plot(x/1e6,y,alpha=.18,color='#D55E00',linewidth=.6,label='D logged values')
        if name!='return':
            left=np.searchsorted(x,x-args.window,side='right')
            sums=np.concatenate(([0.],np.cumsum(y)))
            y=(sums[1:]-sums[left])/(np.arange(len(y))+1-left)
        ax.plot(x/1e6,y,color='#D55E00',label='D reward mean 100 episodes' if name=='return' else f'D trailing {args.window:,}-step mean')
        with (ARCHIVE/f'reproduction/paper_comparison/paper-hard-gru-{name}-digitized.csv').open() as f:
            paper=list(csv.DictReader(f))
        ax.plot([float(r['env_steps_estimated'])/1e6 for r in paper],
                [float(r['figure_value_estimated']) for r in paper],color='#0072B2',label='Paper displayed curve (approx.)')
        if name=='critic':
            ax.set_yscale('symlog',linthresh=1)
        ax.set_ylabel(name+(' (symlog)' if name=='critic' else ''))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Environment steps (millions)')
    fig.suptitle('D seed 42 vs paper Hard/GRU — archived snapshot, not a final 7M result')
    fig.text(.5,.01,'Paper smoothing/seed aggregation unknown; loss similarity alone does not establish successful reproduction.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96))
    fig.savefig(output,dpi=150)
    plt.close(fig)
    print(output)


if __name__=='__main__':
    main()
