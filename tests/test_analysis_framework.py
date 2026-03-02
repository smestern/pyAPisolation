"""
Tests for the new analysis framework.

Covers:
- AnalysisBase subclassing and parameter introspection
- AnalysisResult creation, concatenation, DataFrame export
- Registry register / get / list
- run() with per_sweep and per_file modes
- Integration with demo ABF files (if available)
"""

import os
import numpy as np
import pandas as pd
import pytest

# ======================================================================
# Fixtures
# ======================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEMO_ABF_1 = os.path.join(DATA_DIR, "demo_data_1.abf")
DEMO_ABF_2 = os.path.join(DATA_DIR, "demo_data_2.abf")

HAS_DEMO_DATA = os.path.exists(DEMO_ABF_1)


def _make_fake_sweep(n_points=10000, dt=1e-4, spike=False):
    """Generate a simple synthetic sweep (1-D arrays)."""
    x = np.arange(n_points) * dt
    c = np.zeros(n_points)
    # Subthreshold baseline near -70 mV
    y = np.full(n_points, -70.0)
    if spike:
        # Insert a crude spike in the middle
        mid = n_points // 2
        y[mid - 5 : mid + 5] = 30.0
    return x, y, c


def _make_fake_data_2d(n_sweeps=3, n_points=10000, dt=1e-4):
    """Generate multi-sweep 2-D arrays."""
    xs, ys, cs = [], [], []
    for _ in range(n_sweeps):
        x, y, c = _make_fake_sweep(n_points, dt)
        xs.append(x)
        ys.append(y)
        cs.append(c)
    return np.array(xs), np.array(ys), np.array(cs)


# ======================================================================
# 1) AnalysisResult
# ======================================================================

class TestAnalysisResult:
    def test_basic_creation(self):
        from pyAPisolation.analysis.result import AnalysisResult
        r = AnalysisResult(name="test", file_path="foo.abf")
        assert r.success is True
        assert r.name == "test"

    def test_add_error_marks_failure(self):
        from pyAPisolation.analysis.result import AnalysisResult
        r = AnalysisResult(name="test")
        r.add_error("something broke")
        assert r.success is False
        assert "something broke" in r.errors

    def test_to_dataframe_single(self):
        from pyAPisolation.analysis.result import AnalysisResult
        r = AnalysisResult(name="test", data={"peak": 42, "width": 1.5})
        df = r.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["peak"].iloc[0] == 42

    def test_to_dataframe_sweep_results(self):
        from pyAPisolation.analysis.result import AnalysisResult
        r = AnalysisResult(name="test", file_path="f.abf", data={"cell_id": "A"},
                           sweep_results=[{"v": -70}, {"v": -65}, {"v": -60}])
        df = r.to_dataframe()
        assert len(df) == 3
        assert list(df["sweep"]) == [0, 1, 2]
        # file-level data is broadcast
        assert all(df["cell_id"] == "A")

    def test_concatenate(self):
        from pyAPisolation.analysis.result import AnalysisResult
        r1 = AnalysisResult(name="t", file_path="a.abf", data={"x": 1})
        r2 = AnalysisResult(name="t", file_path="b.abf", data={"x": 2})
        combined = AnalysisResult.concatenate([r1, r2])
        df = combined.to_dataframe()
        assert len(df) == 2
        assert set(df["file"]) == {"a.abf", "b.abf"}

    def test_concatenate_empty(self):
        from pyAPisolation.analysis.result import AnalysisResult
        combined = AnalysisResult.concatenate([])
        assert combined.success is False


# ======================================================================
# 2) AnalysisBase — subclassing and parameter introspection
# ======================================================================

class TestAnalysisBase:
    def _make_simple_module(self):
        from pyAPisolation.analysis.base import AnalysisBase

        class SimpleModule(AnalysisBase):
            name = "simple"
            sweep_mode = "per_sweep"
            threshold: float = -20.0
            window_ms: int = 5

            def analyze(self, x, y, c, **kwargs):
                return {"max_v": float(np.max(y)), "above": float(np.max(y)) > self.threshold}

        return SimpleModule

    def test_param_discovery(self):
        Cls = self._make_simple_module()
        m = Cls()
        params = m.get_parameters()
        assert "threshold" in params
        assert "window_ms" in params
        assert params["threshold"]["default"] == -20.0
        assert params["threshold"]["type"] is float

    def test_param_override_in_constructor(self):
        Cls = self._make_simple_module()
        m = Cls(threshold=-40.0)
        assert m.threshold == -40.0

    def test_set_parameters(self):
        Cls = self._make_simple_module()
        m = Cls()
        m.set_parameters(threshold=-50.0, window_ms=10)
        assert m.threshold == -50.0
        assert m.window_ms == 10

    def test_collect_param_dict(self):
        Cls = self._make_simple_module()
        m = Cls(threshold=-30.0)
        d = m._collect_param_dict()
        assert d == {"threshold": -30.0, "window_ms": 5}

    def test_auto_name(self):
        from pyAPisolation.analysis.base import AnalysisBase

        class MyFancyAnalysis(AnalysisBase):
            def analyze(self, x, y, c, **kwargs):
                return {}

        m = MyFancyAnalysis()
        assert m.name == "myfancyanalysis"

    def test_run_per_sweep_with_arrays(self):
        Cls = self._make_simple_module()
        m = Cls()
        x, y, c = _make_fake_data_2d(n_sweeps=3)
        result = m.run(x=x, y=y, c=c)
        assert result.success
        assert len(result.sweep_results) == 3
        # Each sweep should have max_v of -70
        for sr in result.sweep_results:
            assert sr["max_v"] == pytest.approx(-70.0)

    def test_run_per_file_mode(self):
        from pyAPisolation.analysis.base import AnalysisBase

        class FileMode(AnalysisBase):
            name = "filemode"
            sweep_mode = "per_file"

            def analyze(self, x, y, c, **kwargs):
                # x, y, c are 2-D here
                return {"n_sweeps": x.shape[0], "mean_v": float(np.mean(y))}

        m = FileMode()
        x, y, c = _make_fake_data_2d(n_sweeps=4)
        result = m.run(x=x, y=y, c=c)
        assert result.success
        assert result.data["n_sweeps"] == 4
        assert result.data["mean_v"] == pytest.approx(-70.0)

    def test_run_handles_exceptions(self):
        from pyAPisolation.analysis.base import AnalysisBase

        class Crasher(AnalysisBase):
            name = "crasher"
            sweep_mode = "per_sweep"

            def analyze(self, x, y, c, **kwargs):
                raise RuntimeError("intentional crash")

        m = Crasher()
        x, y, c = _make_fake_sweep()
        # Should not raise — error is captured in the result
        result = m.run(x=np.array([x]), y=np.array([y]), c=np.array([c]))
        # The sweep-level failure is a warning, not a top-level error
        assert len(result.sweep_results) == 1
        assert "_error" in result.sweep_results[0]

    def test_selected_sweeps(self):
        Cls = self._make_simple_module()
        m = Cls()
        x, y, c = _make_fake_data_2d(n_sweeps=5)
        result = m.run(x=x, y=y, c=c, selected_sweeps=[0, 2, 4])
        assert len(result.sweep_results) == 3
        assert result.sweep_results[0]["sweep_number"] == 0
        assert result.sweep_results[1]["sweep_number"] == 2
        assert result.sweep_results[2]["sweep_number"] == 4


# ======================================================================
# 3) Registry
# ======================================================================

class TestRegistry:
    def test_register_and_get(self):
        from pyAPisolation.analysis.registry import register, get, clear
        from pyAPisolation.analysis.base import AnalysisBase

        class Dummy(AnalysisBase):
            name = "test_dummy"
            def analyze(self, x, y, c, **kwargs):
                return {}

        # Save state and register
        register(Dummy)
        m = get("test_dummy")
        assert m is not None
        assert m.name == "test_dummy"

    def test_list_modules(self):
        from pyAPisolation.analysis.registry import list_modules
        names = list_modules()
        assert isinstance(names, list)
        # Built-ins should be registered by default
        assert "spike" in names
        assert "subthreshold" in names
        assert "peak_detector" in names

    def test_register_with_overrides(self):
        from pyAPisolation.analysis.registry import register, get
        from pyAPisolation.analysis.base import AnalysisBase

        class ParamTest(AnalysisBase):
            name = "param_test_reg"
            cutoff: float = 5.0
            def analyze(self, x, y, c, **kwargs):
                return {}

        register(ParamTest, cutoff=99.0)
        m = get("param_test_reg")
        assert m.cutoff == 99.0


# ======================================================================
# 4) Integration with demo ABF data
# ======================================================================

@pytest.mark.skipif(not HAS_DEMO_DATA, reason="Demo ABF files not found")
class TestWithDemoData:
    def test_spike_analysis_runs(self):
        from pyAPisolation.analysis import get
        module = get("spike")
        result = module.run(file=DEMO_ABF_1)
        assert result.success
        assert len(result.sweep_results) > 0
        df = result.to_dataframe()
        assert "spike_count" in df.columns

    def test_subthreshold_analysis_runs(self):
        from pyAPisolation.analysis import get
        module = get("subthreshold")
        result = module.run(file=DEMO_ABF_1)
        assert result.success
        df = result.to_dataframe()
        assert "sag_ratio" in df.columns

    def test_peak_detector_runs(self):
        from pyAPisolation.analysis import get
        module = get("peak_detector")
        result = module.run(file=DEMO_ABF_1)
        assert result.success
        df = result.to_dataframe()
        assert "peak_voltage" in df.columns

    def test_run_batch(self):
        from pyAPisolation.analysis import get, run_batch
        module = get("peak_detector")
        result = run_batch(module, DATA_DIR)
        df = result.to_dataframe()
        # Should have processed at least the 2 demo ABF files
        assert len(df) > 0


# ======================================================================
# 5) Legacy import compatibility
# ======================================================================

class TestLegacyImports:
    def test_feature_extractor_still_importable(self):
        from pyAPisolation.featureExtractor import analyze, batch_feature_extract, analyze_sweep
        assert callable(analyze)
        assert callable(batch_feature_extract)
        assert callable(analyze_sweep)

    def test_dataset_still_importable(self):
        from pyAPisolation.dataset import cellData
        assert callable(cellData)
