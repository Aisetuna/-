"""Analytical/regression tests for the Gaussian evaluation path."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import properscoring as ps
from base.metrics import masked_crps, gaussian_intervals, Metrics
from base.engine import BaseEngine
from utils.generate import MinMaxScaler

mean = torch.tensor([[0., 2., -1.]], dtype=torch.float64)
y = torch.tensor([[0., -3., 1.]], dtype=torch.float64)
sd = torch.tensor([[1., 2., .5]], dtype=torch.float64)
cov = torch.diag_embed(sd.square())
np.testing.assert_allclose(masked_crps(mean, y, float('nan'), cov).item(),
                           ps.crps_gaussian(y.numpy(), mean.numpy(), sd.numpy()).mean())
lo, hi = gaussian_intervals(mean, cov)
np.testing.assert_allclose((hi-lo).numpy(), 2*1.6448536269514722*sd.numpy())
engine = BaseEngine.__new__(BaseEngine)
engine._loss_fn = 'MGAU'
engine._scaler = MinMaxScaler([0, 0, 0], [2, 3, 4])
np.testing.assert_allclose(engine._inverse_covariance(cov).numpy(),
                           (cov * torch.tensor([2., 3., 4.])[:, None] * torch.tensor([2., 3., 4.])[None, :]).numpy())
metrics = Metrics('MGAU', ['MAE', 'MAPE', 'RMSE', 'CRPS', 'MPIW', 'COV'])
metrics.compute_one_batch(mean, y, float('nan'), 'test', scale=cov)
assert all(np.isfinite(v).all() for v in metrics.test_res)
# Interval-based engines must retain their existing MPIW path.
interval = Metrics('MAE', ['MAE', 'MPIW', 'CRPS'])
interval.compute_one_batch(mean, y, float('nan'), 'test', lower=lo, upper=hi)
print('Gaussian CRPS reference, interval quantiles, covariance scaling, dispatch: PASS')

# NLL in count units differs only by the fixed affine log-Jacobian.
from base.metrics import mnormal_loss
span = torch.tensor([2.,3.,4.], dtype=torch.float64)
shift = torch.tensor([1.,2.,3.], dtype=torch.float64)
mu = mean.clone().requires_grad_()
raw_loss = mnormal_loss(mu*span+shift, y*span+shift, float('nan'), engine._inverse_covariance(cov))
norm_loss = mnormal_loss(mu, y, float('nan'), cov)
torch.testing.assert_close(raw_loss-norm_loss, span.log().sum())
torch.testing.assert_close(torch.autograd.grad(raw_loss, mu)[0], torch.autograd.grad(norm_loss, mu)[0])
point = masked_crps(mean,y,float('nan'))
torch.testing.assert_close(point, (mean-y).abs().mean())
try:
    gaussian_intervals(mean,cov,1.)
    raise AssertionError('invalid alpha accepted')
except ValueError:
    pass
print('Affine NLL/gradient equivalence and point-forecast regression: PASS')
