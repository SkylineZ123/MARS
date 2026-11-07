"""Opt-in real release checks: MARS_RELEASE_ASSETS=/path/to/bundle pytest this file."""
import os
from pathlib import Path
import pytest
import torch
import yaml

ASSETS = os.environ.get('MARS_RELEASE_ASSETS')
pytestmark = pytest.mark.skipif(not ASSETS, reason='set MARS_RELEASE_ASSETS to test real pretrained assets')


def test_released_loaders_cpu():
    from mars.controller import MARSController
    from mars.rltoken.adapter import load_adapter
    from mars.rltoken.learner import load_specialist
    from mars.tubes.assets_cli import safe_path, verify
    root = Path(ASSETS)
    verify(root)
    cfg = yaml.safe_load((root / 'simulation.yaml').read_text())
    adapter = load_adapter(safe_path(root, cfg['adapter']))
    assert adapter.z_dim == 960
    for paths in cfg['seeds'].values():
        ctrl = MARSController.from_checkpoint(safe_path(root, paths['router']))
        assert ctrl.classes == ['vla', 'pickup', 'insert']
        for name in ctrl.classes[1:]:
            expert = load_specialist(safe_path(root, paths[name]))
            assert expert.C == 5
            z, q, ref = (torch.zeros(1, size) for size in (expert.z_dim, expert.q_dim, expert.C*expert.d_a))
            out = expert.act(z[0].numpy(), q[0].numpy(), ref.reshape(expert.C, expert.d_a).numpy())
            assert out.shape == (5, 7) and torch.isfinite(torch.from_numpy(out)).all()
