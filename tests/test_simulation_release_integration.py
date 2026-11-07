"""Release integration checks, independent of simulator/model downloads."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from mars.backbones.base import PrefixFeatures, _AffineNormalizer
from mars.rltoken.controller import RoutedRLTPolicy
from mars.tubes.assets_cli import download, safe_path, verify
from mars.tubes.noise import episode_key, vla_noise_key
from mars.tubes.runtime import RoundedExpert, SimulationRuntime, evaluate


def test_manifest_integrity_and_traversal(tmp_path):
    (tmp_path / 'asset').write_bytes(b'abc')
    manifest = {'format': 'mars-assets-v1', 'files': {'asset': hashlib.sha256(b'abc').hexdigest()}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert verify(tmp_path) == manifest
    (tmp_path / 'asset').write_bytes(b'bad')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify(tmp_path)
    for name in ('../escape', '/absolute', 'x/../../escape', 'a\\b'):
        with pytest.raises(ValueError):
            safe_path(tmp_path, name)
    for revision in ('', 'main', 'v1', 'a'*39):
        with pytest.raises(ValueError):
            download('unused/repo', revision, tmp_path)


def test_download_pins_every_file(monkeypatch, tmp_path):
    import huggingface_hub
    calls = []
    def fetch(repo, filename, revision, local_dir):
        calls.append((repo, filename, revision))
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        if filename == 'manifest.json':
            p.write_text(json.dumps({'format': 'mars-assets-v1', 'files': {'weights/x': hashlib.sha256(b'x').hexdigest()}}))
        else:
            p.write_bytes(b'x')
        return str(p)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fetch)
    download('org/model', 'a'*40, tmp_path)
    assert [c[1] for c in calls] == ['manifest.json', 'weights/x']
    assert all(c[2] == 'a'*40 for c in calls)


@pytest.mark.parametrize("scene,seed,expected_key,expected_noise", [
    (0, 0, 2650463210756714728,
     [247482029033203862, 478144806140697542, 2729974743796427598, 2783494682575226108]),
    (30000, 1, 1298345114169258611,
     [3667351185356237647, 3449781856587538614, 3778292217740363529, 3012358229952340903]),
    (30099, 2, 3134127565942771486,
     [1340339052470538783, 2404662838197574814, 3023761331606118981, 1661037544717256054]),
])
def test_noise_matches_migration_reference(scene, seed, expected_key, expected_noise):
    # Fixed outputs captured from the original runner; no external checkout required.
    key = episode_key('tubes_v2', 0, scene, seed)
    assert key == expected_key
    assert [vla_noise_key(key, t) for t in (0, 5, 17, 1199)] == expected_noise


def test_normalization_matches_reference():
    norm = _AffineNormalizer(np.array([1, 2]), np.array([2, 3]))
    x = np.array([3, 5], np.float32)
    np.testing.assert_array_equal(norm.normalize(x), (x-norm.mean)/(norm.std+1e-8))


class Adapter:
    def eval(self): return self
    def encode_features(self, features): return torch.tensor([[0.1234567]])


class Controller:
    classes = ['vla', 'pickup', 'insert']
    router = SimpleNamespace(needs_modalities=False)
    def reset(self): self.active = 0; self.t = 0
    @property
    def active_name(self): return self.classes[self.active]
    def query(self, z, q):
        old = self.active
        self.active = 1 if self.t >= 3 else 0
        self.t += 1
        return self.active, old != self.active, np.eye(3)[self.active]


class Expert:
    C = 5
    def act(self, z, q, ref, **kwargs):
        self.z = z
        return ref + 100


def test_controller_cadence_handoff_and_fp16_expert():
    controller, expert = Controller(), Expert()
    references = []
    def reference(features, q):
        references.append(len(actions))
        return np.arange(10, dtype=np.float32)[:, None]
    policy = RoutedRLTPolicy(controller, Adapter(), reference,
                             {'pickup': RoundedExpert(expert), 'insert': RoundedExpert(Expert())})
    features = PrefixFeatures(torch.zeros(1, 1, 1), {}, {}, torch.zeros(1, 1))
    actions = []
    for _ in range(14):
        actions.append(float(policy.step(features, np.zeros(1)).action[0]))
    assert references == [0, 3, 8, 13]
    assert actions[:4] == [0, 1, 2, 100]
    np.testing.assert_array_equal(expert.z, np.array([0.1234567]).astype(np.float16).astype(np.float32))
    policy.reset()
    assert not policy.queue and controller.active == 0


@pytest.mark.parametrize("mode,cadence", [("base", 10), ("base_c5", 5)])
def test_runtime_keys_reset_and_action_space(tmp_path, mode, cadence):
    runtime = SimulationRuntime.__new__(SimulationRuntime)
    runtime.mode = mode
    runtime.model_seed = 0
    runtime.root = tmp_path
    runtime.config_path = tmp_path / 'simulation.yaml'
    runtime.config_path.write_text('test')
    (tmp_path / 'manifest.json').write_text('{}')
    runtime.cfg = {'max_steps': 12}
    queries = []
    class Backbone:
        c_vla = 10
        def prefill(self, batch): return None
        def preprocess(self, obs, task): return obs
        def normalize_state(self, q): return q
        def state_from_raw(self, obs): return np.zeros(8)
        def make_noise(self, key): queries.append(key); return key
        def reference_chunk(self, features, noise): return torch.ones(1, 50, 7)*2
        def unnormalize_action(self, action): return action
    class Env:
        task_description = 'test'
        spec = SimpleNamespace(parts=3)
        completed = [0, 1, 1]
        stage_index = 4
        def seated(self, i): return False
        def reset(self, init_id, seed):
            self.reset_args = init_id, seed
            self.n = 0
            return {}
        def step(self, action):
            np.testing.assert_array_equal(action, np.ones(7))
            self.n += 1
            return {}, self.n == 12, {}
    runtime.backbone = Backbone()
    env = Env()
    row = runtime.run(env, 30000, 2)
    assert row['success'] and row['steps'] == 12
    assert env.reset_args == (30000, 2)
    key = episode_key('tubes_v2', 0, 30000, 2)
    assert queries == [vla_noise_key(key, t) for t in range(0, 12, cadence)]
    assert row["completed_tubes"] == 2 and row["scr"] == 2/3
    assert "switches" not in row
    assert row["stages_completed"] == 4 and row["termination_reason"] == "success"
    result = evaluate(runtime, env, [30000], [2])
    assert result['n'] == 1 and result['success_rate'] == 1


def test_metrics_have_explicit_denominators():
    from mars.tubes.runtime import summarize
    rows = [dict(success=True, steps=100, completed_tubes=3, total_tubes=3),
            dict(success=False, steps=1200, completed_tubes=1, total_tubes=3)]
    summary = summarize(rows)
    assert summary['success_rate'] == .5
    assert summary['scr'] == 4/6
    assert summary['steps_all'] == {'mean': 650, 'n': 2}
    assert summary['steps_success_only'] == {'mean': 100, 'n': 1}
    assert summarize(rows[1:])['steps_success_only'] == {'mean': None, 'n': 0}


@pytest.mark.parametrize("seeds,explicit", [([0], False), ([0, 1, 2], False), ([0, 1, 2], True)])
def test_cli_default_evaluation_grid(monkeypatch, tmp_path, capsys, seeds, explicit):
    import sys
    from mars.tubes import runtime, cli
    runs = []
    class Runtime:
        def __init__(self, assets, seed, device, mode, config): self.seed, self.mode = seed, mode
    def fake_evaluate(rt, env, scenes, noises):
        runs.append((rt.mode, rt.seed, list(scenes), list(noises)))
        return {'episodes': [{'mode': rt.mode, 'model_seed': rt.seed, 'success': True, 'steps': 1,
                              'completed_tubes': 3, 'total_tubes': 3} for _ in range(100)]}
    monkeypatch.setattr(runtime, 'SimulationRuntime', Runtime)
    monkeypatch.setattr(runtime, 'evaluate', fake_evaluate)
    class Env:
        def __init__(self, **kwargs): pass
        def close(self): pass
    monkeypatch.setitem(sys.modules, 'mars.tubes.environment', SimpleNamespace(TubesEnv=Env))
    cfg = yaml.safe_load(Path('configs/tubes_simulation.yaml').read_text())
    cfg['seeds'] = {seed: cfg['seeds'][0] for seed in seeds}
    config_path = tmp_path / ('custom.yaml' if explicit else 'simulation.yaml')
    config_path.write_text(yaml.safe_dump(cfg))
    arguments = ['--config', str(config_path), '--model-seeds', *map(str, seeds)] if explicit else []
    cli.main(['evaluate', '--assets', str(tmp_path), *arguments])
    result = json.loads(capsys.readouterr().out)
    assert len(runs) == 2 + len(seeds)
    assert all(r[2] == list(range(30050, 30100)) and r[3] == [0, 1] for r in runs)
    assert result['model_seeds'] == seeds
    assert set(result['by_mode']) == {'base', 'base_c5', 'mars'}

    assert result['by_mode']['base']['n'] == 100
    assert result['by_mode']['base_c5']['n'] == 100
    assert result['by_mode']['mars']['n'] == 100 * len(seeds)
    assert result['by_mode']['mars']['seed_success_rate_sample_std'] == (0 if len(seeds) > 1 else None)
    assert result['by_mode']['base']['seed_success_rate_sample_std'] is None


def test_seed_standard_deviation_is_sample_std():
    from mars.tubes.runtime import summarize_modes
    rows = [dict(mode='mars', model_seed=seed, success=bool(seed), steps=1,
                 completed_tubes=3*seed, total_tubes=3) for seed in (0, 1)]
    result = summarize_modes(rows)['mars']
    assert result['seed_count'] == 2
    assert result['seed_success_rate_mean'] == .5
    assert result['seed_success_rate_sample_std'] == pytest.approx(np.sqrt(.5))


def test_invalid_cli_grid_fails_before_simulator_import(tmp_path):
    from mars.tubes.cli import main
    for arguments in (['--episodes', '3'], ['--scene', '30000'],
                      ['--model-seeds', '0', '0'], ['--noise-seeds', '0', '0']):
        with pytest.raises(SystemExit):
            main(['evaluate', '--assets', str(tmp_path), *arguments])


def test_single_run_includes_reproducibility(monkeypatch, tmp_path, capsys):
    import sys
    from mars.tubes import runtime, cli
    class Runtime:
        cfg = {'max_steps': 1200}
        def __init__(self, *args): pass
        def run(self, env, scene, noise, limit): return {'scene': scene, 'noise_seed': noise, 'steps': limit}
    class Env:
        def __init__(self, **kwargs): pass
        def close(self): pass
    monkeypatch.setattr(runtime, 'SimulationRuntime', Runtime)
    monkeypatch.setattr(runtime, 'reproducibility', lambda rt, scenes, noises: {'scenes': scenes, 'noise_seeds': noises})
    monkeypatch.setitem(sys.modules, 'mars.tubes.environment', SimpleNamespace(TubesEnv=Env))
    (tmp_path / 'simulation.yaml').write_text(Path('configs/tubes_simulation.yaml').read_text())
    cli.main(['run', '--assets', str(tmp_path), '--max-steps', '12'])
    result = json.loads(capsys.readouterr().out)
    assert result['reproducibility'] == {'scenes': [30050], 'noise_seeds': [0], 'max_steps': 12}


@pytest.mark.parametrize('arguments', [
    ['run', '--seed', '1'], ['evaluate', '--model-seeds', '0', '1'],
    ['evaluate', '--seed', '2'],
])
def test_missing_seed_fails_before_heavy_imports(monkeypatch, tmp_path, capsys, arguments):
    import builtins
    from mars.tubes.cli import main
    (tmp_path / 'simulation.yaml').write_text(Path('configs/tubes_simulation.yaml').read_text())
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name in ('runtime', 'environment', 'mars.backbones.smolvla', 'robosuite'):
            pytest.fail(f'heavy import before seed validation: {name}')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    with pytest.raises(SystemExit) as exc:
        main([arguments[0], '--assets', str(tmp_path), *arguments[1:]])
    assert exc.value.code == 2
    assert 'unavailable model seeds' in capsys.readouterr().err


def test_runtime_missing_seed_fails_before_backbone_or_manifest(monkeypatch, tmp_path):
    from mars.tubes import runtime
    (tmp_path / 'simulation.yaml').write_text(Path('configs/tubes_simulation.yaml').read_text())
    def forbidden(*args, **kwargs):
        pytest.fail('loaded assets before seed validation')
    monkeypatch.setattr(runtime, 'make_backbone', forbidden)
    monkeypatch.setattr(runtime, 'verify', forbidden)
    with pytest.raises(ValueError, match='unavailable model seed 1'):
        runtime.SimulationRuntime(tmp_path, seed=1)


@pytest.mark.parametrize('mismatch', [None, 'adapter', 'router_q', 'expert_z', 'expert_q', 'expert_action'])
def test_runtime_rejects_incompatible_dimensions(monkeypatch, tmp_path, mismatch):
    from mars.tubes import runtime
    (tmp_path / 'simulation.yaml').write_text(Path('configs/tubes_simulation.yaml').read_text())
    monkeypatch.setattr(runtime, 'verify', lambda root: None)
    monkeypatch.setattr(runtime, 'make_backbone', lambda cfg: SimpleNamespace(c_vla=10, d_q=8, d_a=7))
    router = SimpleNamespace(z_dim=960, q_dim=9 if mismatch == 'router_q' else 8)
    controller = SimpleNamespace(classes=['vla', 'pickup', 'insert'], router=router)
    monkeypatch.setattr(runtime.MARSController, 'from_checkpoint', lambda *a, **kw: controller)
    monkeypatch.setattr(runtime, 'load_adapter', lambda *a: SimpleNamespace(z_dim=961 if mismatch == 'adapter' else 960))
    expert = SimpleNamespace(C=5, z_dim=961 if mismatch == 'expert_z' else 960,
                             q_dim=9 if mismatch == 'expert_q' else 8,
                             d_a=6 if mismatch == 'expert_action' else 7)
    loaded = []
    def load(path, **kwargs):
        loaded.append(path.relative_to(tmp_path).as_posix())
        return expert
    monkeypatch.setattr(runtime, 'load_specialist', load)
    if mismatch is None:
        monkeypatch.setattr(runtime, 'RoutedRLTPolicy', lambda *a, **kw: SimpleNamespace())
        rt = runtime.SimulationRuntime(tmp_path, device='cpu')
        assert rt.model_seed == 0
    else:
        with pytest.raises(ValueError, match='mismatch'):
            runtime.SimulationRuntime(tmp_path, device='cpu')
    assert loaded == ['actors/pickup.pt', 'actors/insert.pt']
