"""
tests/test_labeling_protocol.py
=================================
data/labeling_protocol.py için birim testleri.

En kritik testler:
  - test_guard_band_windows_are_excluded: "ground truth netliği" kısıtının
    fiilen uygulandığını kanıtlar (geçiş anına yakın pencereler ele alınmaz)
  - test_build_labeled_dataset_preserves_plant_id: bitki-bazlı CV'nin
    ihtiyaç duyduğu bilginin kaybolmadığını kanıtlar
"""

from __future__ import annotations

import numpy as np
import pytest

from config import StimulusLabel, WindowConfig
from data.labeling_protocol import (
    ProtocolMinimums,
    Trial,
    build_labeled_dataset,
    label_window,
    validate_dataset_protocol,
)


def make_trial(**overrides) -> Trial:
    """Testlerde tekrar tekrar yazmamak için makul varsayılanlarla bir
    Trial üreten yardımcı fabrika fonksiyonu."""
    defaults = dict(
        trial_id="t1",
        plant_id="plant_0",
        label=StimulusLabel.MECHANICAL_TOUCH,
        sampling_rate_hz=100.0,
        stimulus_onset_time_s=120.0,
        baseline_duration_s=120.0,
        post_stimulus_duration_s=120.0,
        guard_band_s=2.0,
    )
    defaults.update(overrides)
    return Trial(**defaults)


# ---------------------------------------------------------------------------
# Trial doğrulama testleri
# ---------------------------------------------------------------------------
class TestTrialValidation:
    def test_baseline_label_rejected(self):
        """Bir trial'ın kendi uyaran sınıfı BASELINE olamaz (baseline
        otomatik olarak uyaran-öncesi pencerelere atanır)."""
        with pytest.raises(ValueError):
            make_trial(label=StimulusLabel.BASELINE)

    def test_below_minimum_baseline_duration_rejected(self):
        with pytest.raises(ValueError):
            make_trial(baseline_duration_s=10.0, stimulus_onset_time_s=10.0)

    def test_below_minimum_post_stimulus_duration_rejected(self):
        with pytest.raises(ValueError):
            make_trial(post_stimulus_duration_s=10.0)

    def test_onset_before_baseline_end_rejected(self):
        """Uyaran, baseline süresi dolmadan gerçekleşemez (kayıt onset'ten
        önce en az baseline_duration_s kadar sürmüş olmalı)."""
        with pytest.raises(ValueError):
            make_trial(stimulus_onset_time_s=50.0, baseline_duration_s=120.0)


# ---------------------------------------------------------------------------
# label_window - çekirdek zaman-tabanlı etiketleme testleri
# ---------------------------------------------------------------------------
class TestLabelWindow:
    def test_window_before_onset_is_baseline(self):
        """Onset'ten (120s) tamamen önce biten bir pencere baseline olmalı."""
        trial = make_trial(stimulus_onset_time_s=120.0, guard_band_s=2.0)
        wc = WindowConfig(window_size=1000, hop_size=1000)  # fs=100Hz -> 10s pencere
        # window_index=5 -> [50s, 60s) -> guard_start=118s'den önce bitiyor
        assert label_window(trial, window_index=5, window_config=wc) == StimulusLabel.BASELINE

    def test_window_after_guard_is_stimulus_label(self):
        """Guard band'den tamamen sonra başlayan bir pencere trial'ın
        uyaran sınıfıyla etiketlenmeli."""
        trial = make_trial(stimulus_onset_time_s=120.0, guard_band_s=2.0)
        wc = WindowConfig(window_size=1000, hop_size=1000)
        # window_index=13 -> [130s, 140s) -> guard_end=122s'den sonra başlıyor
        assert label_window(trial, window_index=13, window_config=wc) == trial.label

    def test_guard_band_window_is_excluded(self):
        """Onset anını (120s) içine alan pencere guard band'e denk gelir
        ve None dönmeli (belirsiz, veri setine dahil edilmemeli)."""
        trial = make_trial(stimulus_onset_time_s=120.0, guard_band_s=2.0)
        wc = WindowConfig(window_size=1000, hop_size=1000)
        # window_index=12 -> [120s, 130s) -> guard_start=118, guard_end=122
        # pencere guard_end'i (122s) kesiyor -> ne tam öncesi ne tam sonrası
        assert label_window(trial, window_index=12, window_config=wc) is None

    def test_window_outside_trial_bounds_is_none(self):
        """Trial'ın tanımlı kayıt aralığının (baseline_start..post_end)
        tamamen dışında kalan pencere None dönmeli."""
        trial = make_trial(
            stimulus_onset_time_s=120.0, baseline_duration_s=120.0, post_stimulus_duration_s=120.0
        )
        wc = WindowConfig(window_size=1000, hop_size=1000)
        # window_index=30 -> [300s, 310s) -> post_end=240s'i çoktan geçmiş
        assert label_window(trial, window_index=30, window_config=wc) is None


# ---------------------------------------------------------------------------
# build_labeled_dataset - uçtan uca entegrasyon testleri
# ---------------------------------------------------------------------------
class TestBuildLabeledDataset:
    def test_build_labeled_dataset_preserves_plant_id(self):
        """Üretilen her LabeledWindow, hangi bitkiden geldiğini doğru
        taşımalı - bitki-bazlı cross-validation buna bağımlı."""
        fs = 100.0
        onset = 120.0
        duration_s = onset + 120.0
        n_samples = int(duration_s * fs)
        rng = np.random.default_rng(0)

        trial = make_trial(
            trial_id="t_plantA", plant_id="plant_A", sampling_rate_hz=fs, stimulus_onset_time_s=onset
        )
        signal = rng.normal(size=n_samples)

        dataset = build_labeled_dataset([trial], {"t_plantA": signal})

        assert len(dataset) > 0
        assert all(lw.plant_id == "plant_A" for lw in dataset)
        assert all(lw.trial_id == "t_plantA" for lw in dataset)

    def test_labels_present_are_only_baseline_or_trial_label(self):
        """Üretilen etiketler sadece BASELINE veya trial'ın kendi uyaran
        sınıfı olmalı - guard band'e denk gelenler zaten elenmiş olmalı."""
        fs = 100.0
        onset = 120.0
        n_samples = int((onset + 120.0) * fs)
        rng = np.random.default_rng(1)

        trial = make_trial(sampling_rate_hz=fs, stimulus_onset_time_s=onset)
        dataset = build_labeled_dataset([trial], {trial.trial_id: rng.normal(size=n_samples)})

        found_labels = {lw.label for lw in dataset}
        assert found_labels <= {StimulusLabel.BASELINE, trial.label}

    def test_missing_raw_signal_raises_key_error(self):
        trial = make_trial()
        with pytest.raises(KeyError):
            build_labeled_dataset([trial], {})


# ---------------------------------------------------------------------------
# validate_dataset_protocol testleri
# ---------------------------------------------------------------------------
class TestValidateDatasetProtocol:
    def test_insufficient_repeats_produces_warning(self):
        trials = [
            make_trial(trial_id=f"t{i}", plant_id=f"plant_{i % 3}")
            for i in range(5)  # asgari 20'nin altında
        ]
        warnings = validate_dataset_protocol(trials)
        assert any("tekrar" in w for w in warnings)

    def test_insufficient_plants_produces_warning(self):
        trials = [
            make_trial(trial_id=f"t{i}", plant_id="only_one_plant")
            for i in range(ProtocolMinimums.MIN_REPEATS_PER_CLASS)
        ]
        warnings = validate_dataset_protocol(trials)
        assert any("bitki" in w for w in warnings)

    def test_sufficient_data_produces_no_warnings(self):
        trials = [
            make_trial(trial_id=f"t{i}", plant_id=f"plant_{i % 4}")
            for i in range(ProtocolMinimums.MIN_REPEATS_PER_CLASS)
        ]
        assert validate_dataset_protocol(trials) == []