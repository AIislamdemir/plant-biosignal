"""
tests/test_statistical_features.py
=====================================
features/statistical_features.py için birim testleri.

En kritik testler:
  - test_constant_signal_returns_zero_not_nan: NaN sızıntısını önleyen
    kenar-durum korumasının gerçekten çalıştığını kanıtlar
  - test_build_feature_matrix_alignment: X/y/groups'un sırasının
    kaymadığını (off-by-one riskinin olmadığını) kanıtlar
"""

from __future__ import annotations

import numpy as np
import pytest

from config import StimulusLabel
from data.labeling_protocol import LabeledWindow
from features.statistical_features import (
    STATISTICAL_FEATURE_NAMES,
    build_feature_matrix,
    extract_statistical_features,
    extract_statistical_features_dict,
)


# ---------------------------------------------------------------------------
# extract_statistical_features - temel doğruluk testleri
# ---------------------------------------------------------------------------
class TestExtractStatisticalFeatures:
    def test_output_shape_and_order(self):
        """Çıktı, STATISTICAL_FEATURE_NAMES ile aynı sayıda (6) elemana
        sahip olmalı - sıra sözleşmesi burada test ediliyor."""
        window = np.random.default_rng(0).normal(size=1024)
        result = extract_statistical_features(window)
        assert result.shape == (len(STATISTICAL_FEATURE_NAMES),) == (6,)

    def test_known_values_on_simple_signal(self):
        """Elle hesaplanabilir basit bir sinyalde mean/min/max/variance'ın
        doğru çıktığını kontrol ediyoruz (skew/kurtosis'i scipy'ye güveniyoruz,
        onları ayrı test ediyoruz)."""
        window = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        feats = extract_statistical_features_dict(window)
        assert feats["mean"] == pytest.approx(3.0)
        assert feats["min"] == pytest.approx(1.0)
        assert feats["max"] == pytest.approx(5.0)
        assert feats["variance"] == pytest.approx(2.0)  # popülasyon varyansı, ddof=0

    def test_constant_signal_returns_zero_not_nan(self):
        """Sabit (sıfır varyanslı) bir sinyalde skewness/kurtosis NaN değil,
        açıkça 0.0 dönmeli - aksi halde eğitim verisine NaN sızar."""
        window = np.full(1024, fill_value=7.5)
        feats = extract_statistical_features_dict(window)
        assert feats["variance"] == pytest.approx(0.0)
        assert feats["skewness"] == 0.0
        assert feats["kurtosis"] == 0.0
        assert not np.isnan(feats["skewness"])
        assert not np.isnan(feats["kurtosis"])

    def test_sharp_spike_has_high_kurtosis(self):
        """Sezgisel doğrulama: tek keskin bir spike içeren sinyal, düz
        gürültülü bir sinyale göre BELİRGİN olarak daha yüksek kurtosis
        üretmeli (basıklığın 'ani olayı yakalama' işini yaptığının kanıtı)."""
        rng = np.random.default_rng(1)
        quiet = rng.normal(0, 0.02, 1024)
        spiky = quiet.copy()
        spiky[500] += 5.0

        quiet_feats = extract_statistical_features_dict(quiet)
        spiky_feats = extract_statistical_features_dict(spiky)
        assert spiky_feats["kurtosis"] > quiet_feats["kurtosis"] * 10

    def test_rejects_non_1d_input(self):
        with pytest.raises(ValueError):
            extract_statistical_features(np.zeros((10, 10)))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            extract_statistical_features(np.array([]))


# ---------------------------------------------------------------------------
# build_feature_matrix - LabeledWindow entegrasyon testleri
# ---------------------------------------------------------------------------
class TestBuildFeatureMatrix:
    def _make_labeled_window(self, label: StimulusLabel, plant_id: str, seed: int) -> LabeledWindow:
        samples = np.random.default_rng(seed).normal(size=1024)
        return LabeledWindow(
            samples=samples, label=label, plant_id=plant_id, trial_id=f"trial_{seed}", window_index=0
        )

    def test_output_shapes_are_consistent(self):
        windows = [
            self._make_labeled_window(StimulusLabel.BASELINE, "plant_0", 0),
            self._make_labeled_window(StimulusLabel.MECHANICAL_TOUCH, "plant_1", 1),
            self._make_labeled_window(StimulusLabel.BASELINE, "plant_1", 2),
        ]
        X, y, groups = build_feature_matrix(windows)
        assert X.shape == (3, 6)
        assert y.shape == (3,)
        assert groups.shape == (3,)

    def test_build_feature_matrix_alignment(self):
        """X, y, groups arasında sıra kayması OLMAMALI: i. satırın etiketi
        y[i], plant_id'si groups[i] olan pencereyle birebir eşleşmeli."""
        windows = [
            self._make_labeled_window(StimulusLabel.BASELINE, "plant_A", 10),
            self._make_labeled_window(StimulusLabel.MECHANICAL_TOUCH, "plant_B", 20),
            self._make_labeled_window(StimulusLabel.LIGHT_TRANSITION, "plant_C", 30),
        ]
        X, y, groups = build_feature_matrix(windows)

        assert list(y) == ["baseline", "mechanical_touch", "light_transition"]
        assert list(groups) == ["plant_A", "plant_B", "plant_C"]

        # Her satırın, doğrudan aynı pencereden extract edilen özelliklerle
        # birebir aynı olduğunu doğrula (yeniden hesaplayıp karşılaştır)
        for i, w in enumerate(windows):
            expected = extract_statistical_features(w.samples)
            np.testing.assert_array_almost_equal(X[i], expected)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            build_feature_matrix([])