"""Validate prepared files and one real-data GPU optimization step."""
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.dataloader import get_dataset_info, load_dataset
from utils.graph_algo import normalize_adj_mx
from src.flow.uqgnn.uqgnn_model import UQGNN
from base.metrics import mnormal_loss


def main():
    torch.manual_seed(2025)
    path, adj_path, nodes = get_dataset_info('chicago_15min', '2022')
    folder = Path(path) / '2022'
    raw = np.load(ROOT / 'datasets/raw/chicago_paper_2022/chicago_taxi_bike_crime_2022_09_11_TND.npy')
    assert raw.shape == (8736, 77, 3) and np.isfinite(raw).all() and (raw >= 0).all()
    adj = np.load(adj_path)
    assert adj.shape == (77, 77) and np.isfinite(adj).all() and np.allclose(adj, adj.T)
    indices = [np.load(folder / f'idx_{s}.npy') for s in ['train', 'val', 'test']]
    assert indices[0][-1] < indices[1][0] and indices[1][-1] < indices[2][0]
    info = json.loads((folder / 'info.json').read_text())
    assert info['config']['train_only_scaler']
    args = SimpleNamespace(years='2022', seq_len=12, horizon=1, bs=8)
    loaders, scaler = load_dataset(path, args, logging.getLogger('verify'))
    stop = info['config']['scaler_fit_stop_exclusive']
    np.testing.assert_allclose(scaler.data_max_.numpy(), raw[:stop].max(axis=(0, 1)))
    normalized = np.load(folder / 'his.npz')['data']
    np.testing.assert_allclose(scaler.inverse_transform(normalized), raw, atol=1e-4)
    assert torch.cuda.is_available()
    gso = normalize_adj_mx(adj - np.eye(nodes), 'uqgnn')[0]
    model = UQGNN(A=gso, node_num=nodes, hidden_dim_s=64, hidden_dim_t=64,
                  emb_dim=32, kernel_size=3, temporal_layers=2,
                  num_timesteps_output=1, device=torch.device('cuda:0'),
                  input_dim=3, output_dim=3, seq_len=12, min_vec=1e-6, horizon=1)
    x, y = next(loaders['train_loader'].get_iterator())
    x, y = x.cuda(), y.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mu, covariance = model(x)
    assert mu.shape == y.shape == (8, 1, 77, 3)
    assert covariance.shape == (8, 1, 77, 3, 3)
    loss = mnormal_loss(mu, y, float('nan'), covariance)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    report = dict(status='passed', shape=list(raw.shape), totals=raw.sum(axis=(0, 1)).tolist(),
                  splits={s: len(i) for s, i in zip(['train', 'val', 'test'], indices)},
                  gpu=torch.cuda.get_device_name(), batch_shape=list(x.shape),
                  loss=float(loss.detach()), checks=['shape', 'finite_nonnegative', 'adjacency',
                  'chronological_splits', 'train_only_scaler', 'inverse_transform',
                  'GPU_forward', 'Gaussian_NLL', 'finite_gradients', 'optimizer_step'])
    (folder / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
