"""Rescore saved probability forecasts at a declared interval confidence level."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from base.metrics import Metrics, gaussian_intervals

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--alpha', type=float, default=.05)
    args = p.parse_args()
    rows = json.loads((args.directory/'comparison.json').read_text())
    for row in rows:
        with np.load(args.directory/f"predictions_{row['budget']}.npz") as a:
            mu, cov, y = [torch.from_numpy(a[k]) for k in ['mean','covariance','target']]
        metrics = Metrics('MGAU',['MAE','RMSE','MAPE','KL','CRPS','MPIW','COV','PAPER_HALF_WIDTH'])
        metrics.interval_alpha = args.alpha
        metrics.compute_one_batch(mu,y,float('nan'),'test',scale=cov)
        row.update({k:v[0] for k,v in zip(metrics.metric_lst,metrics.test_res)})
        row['MAPE_ratio'] = row['MAPE']/100
        row['interval_alpha'] = args.alpha
        lo,hi = gaussian_intervals(mu,cov,args.alpha)
        np.savez_compressed(args.directory/f"intervals_{row['budget']}_alpha_{args.alpha}.npz",
                            lower=lo.numpy(),upper=hi.numpy(),interval_alpha=args.alpha)
    dest = args.directory/f'comparison_alpha_{args.alpha}.json'
    dest.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__ == '__main__':
    main()
