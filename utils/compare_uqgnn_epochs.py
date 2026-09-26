"""Controlled epoch-budget comparison; select by validation NLL, test at end."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from base.metrics import Metrics, gaussian_intervals, mnormal_loss
from src.flow.uqgnn.uqgnn_model import UQGNN
from utils.args import set_seed
from utils.dataloader import get_dataset_info
from utils.generate import reconstruct_scaler
from utils.graph_algo import normalize_adj_mx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgets', nargs='+', type=int, default=[20, 50, 100, 200, 400])
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--interval-alpha', type=float, default=.05)
    args = parser.parse_args()
    budgets = sorted(set(args.budgets))
    set_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    out = root / 'result/epoch_comparison' / time.strftime('%Y-%m-%d_%H-%M-%S')
    out.mkdir(parents=True, exist_ok=False)
    path, adj_path, nodes = get_dataset_info('chicago_15min', '2022')
    folder = Path(path) / '2022'
    data = torch.tensor(np.load(folder / 'his.npz')['data'], dtype=torch.float32, device='cuda')
    idx = {s: torch.tensor(np.load(folder / f'idx_{s}.npy'), device='cuda') for s in ['train', 'val', 'test']}
    scaler = reconstruct_scaler(json.loads((folder / 'meta.json').read_text()))
    span = (scaler.data_max_ - scaler.data_min_).cuda()
    offsets = torch.arange(-11, 1, device='cuda')
    gso = normalize_adj_mx(np.load(adj_path)-np.eye(nodes), 'uqgnn')[0]
    model = UQGNN(A=gso, node_num=nodes, hidden_dim_s=64, hidden_dim_t=64,
                  emb_dim=32, kernel_size=3, temporal_layers=2, num_timesteps_output=1,
                  device=torch.device('cuda'), input_dim=3, output_dim=3,
                  seq_len=12, min_vec=1e-6, horizon=1)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0005)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=.95)
    config = dict(vars(args), dataset='chicago_15min', years='2022',
                  selection='minimum validation Gaussian NLL within budget',
                  early_stopping=False, learning_rate=.001, weight_decay=.0005,
                  scheduler='StepLR(200,0.95)', loss_scale='normalized; affine-equivalent to correctly scaled original NLL',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    def batches(ids):
        for b in ids.split(args.bs):
            yield data[b[:, None]+offsets], data[b+1].unsqueeze(1)
    @torch.no_grad()
    def validate():
        model.eval()
        total = 0.
        for x, y in batches(idx['val']):
            mu, cov = model(x)
            total += mnormal_loss(mu, y, float('nan'), cov).item()*len(x)
        return total/len(idx['val'])
    best, best_epoch, history, selected = float('inf'), 0, [], {}
    start = time.time()
    for epoch in range(1, max(budgets)+1):
        model.train()
        total = 0.
        order = idx['train'][torch.randperm(len(idx['train']), device='cuda')]
        for x, y in batches(order):
            opt.zero_grad()
            mu, cov = model(x)
            loss = mnormal_loss(mu, y, float('nan'), cov)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm):
                raise RuntimeError(f'Nonfinite gradient at epoch {epoch}')
            opt.step()
            total += loss.item()*len(x)
        val = validate()
        sched.step()
        history.append(dict(epoch=epoch, train_nll=total/len(order), val_nll=val))
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch in budgets:
            torch.save(best_state, out / f'best_within_{epoch}.pt')
            selected[epoch] = dict(selected_epoch=best_epoch, validation_nll=best)
        if epoch % 10 == 0 or epoch == 1:
            (out / 'history.json').write_text(json.dumps(history), encoding='utf-8')
            torch.save(dict(epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict()), out/'last.pt')
            print(f'epoch={epoch} train={total/len(order):.4f} val={val:.4f} best_epoch={best_epoch} elapsed={time.time()-start:.0f}s', flush=True)
    # Only now inspect the held-out test set, for all predeclared budgets.
    rows = []
    for budget in budgets:
        model.load_state_dict(torch.load(out/f'best_within_{budget}.pt', weights_only=True))
        model.eval()
        mus, covs, ys = [], [], []
        with torch.no_grad():
            for x,y in batches(idx['test']):
                mu,cov = model(x)
                mus.append(scaler.inverse_transform(mu).cpu())
                ys.append(scaler.inverse_transform(y).cpu())
                covs.append((cov*span[:,None]*span[None,:]).cpu())
        mu,cov,y = torch.cat(mus),torch.cat(covs),torch.cat(ys)
        tracker = Metrics('MGAU', ['MAE','RMSE','MAPE','KL','CRPS','MPIW','COV'])
        tracker.interval_alpha = args.interval_alpha
        tracker.compute_one_batch(mu,y,float('nan'),'test',scale=cov)
        row = dict(budget=budget, **selected[budget], **{k:v[0] for k,v in zip(tracker.metric_lst,tracker.test_res)})
        rows.append(row)
        lo,hi = gaussian_intervals(mu,cov,args.interval_alpha)
        np.savez_compressed(out/f'predictions_{budget}.npz',mean=mu.numpy(),covariance=cov.numpy(),target=y.numpy(),lower=lo.numpy(),upper=hi.numpy())
        print(json.dumps(row), flush=True)
    (out/'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print('RESULT_DIRECTORY', out, flush=True)


if __name__ == '__main__':
    main()
