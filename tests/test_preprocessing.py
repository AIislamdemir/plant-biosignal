"""
tests/test_preprocessing.py
============================
data/preprocessing.py için birim testleri.

En kritik test `test_offline_online_window_count_matches` - bu proje
için sadece bir "nice to have" değil, mimarinin temel iddiasının
doğrulamasıdır: offline ve online mod aynı sayıda ve aynı boyutta
pencere üretmeli, çünkü ikisi de aynı `push()` çekirdeğini kullanıyor.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import FilterConfig, WindowConfig
from data.preprocessing import (
    HighPassFilter,
    PreprocessingPipeline,
    SlidingWindower,
)


# ---------------------------------------------------------------------------
# SlidingWindower testleri
# ---------------------------------------------------------------------------
class TestSlidingWindower:
    def test_no_window_before_buffer_full(self):
        """Tampon dolmadan hiçbir pencere üretilmemeli."""
        windower = SlidingWindower(WindowConfig(window_size=10, hop_size=10))
        for i in range(9):
            assert windower.push(float(i)) is None

    def test_first_window_shape(self):
        """İlk pencere tam window_size uzunluğunda olmalı."""
        windower = SlidingWindower(WindowConfig(window_size=10, hop_size=10))
        window = None
        for i in range(10):
            window = windower.push(float(i))
        assert window is not None
        assert window.shape == (10,)
        np.testing.assert_array_equal(window, np.arange(10, dtype=np.float64))

    def test_non_overlapping_window_count(self):
        """hop_size == window_size iken örtüşme olmamalı; N örnekten
        N // window_size tam pencere çıkmalı."""
        window_size = 100
        windower = SlidingWindower(WindowConfig(window_size=window_size, hop_size=window_size))
        n_samples = 350
        count = sum(1 for i in range(n_samples) if windower.push(float(i)) is not None)
        assert count == n_samples // window_size  # 3

    def test_overlapping_window_count(self):
        """hop_size < window_size iken örtüşme sonucu daha fazla pencere
        üretilmeli (örn. %50 overlap -> yaklaşık 2 katı pencere)."""
        window_size, hop_size = 100, 50
        windower = SlidingWindower(WindowConfig(window_size=window_size, hop_size=hop_size))
        n_samples = 350
        count = sum(1 for i in range(n_samples) if windower.push(float(i)) is not None)
        # ilk pencere 100. örnekte, sonrakiler her 50 örnekte bir
        expected = (n_samples - window_size) // hop_size + 1
        assert count == expected

    def test_returned_window_is_independent_copy(self):
        """Döndürülen pencere, tamponun canlı referansı DEĞİL, bağımsız
        bir kopya olmalı (aksi halde bir sonraki push çağrısı, çağıranın
        elindeki pencereyi sessizce bozar)."""
        windower = SlidingWindower(WindowConfig(window_size=5, hop_size=5))
        window = None
        for i in range(5):
            window = windower.push(float(i))
        snapshot = window.copy()
        for i in range(5, 10):
            windower.push(float(i))
        np.testing.assert_array_equal(window, snapshot)

    def test_invalid_hop_size_raises(self):
        with pytest.raises(ValueError):
            SlidingWindower(WindowConfig(window_size=10, hop_size=0))
        with pytest.raises(ValueError):
            SlidingWindower(WindowConfig(window_size=10, hop_size=11))

    def test_reset_clears_state(self):
        windower = SlidingWindower(WindowConfig(window_size=5, hop_size=5))
        for i in range(5):
            windower.push(float(i))
        windower.reset()
        for i in range(4):
            assert windower.push(float(i)) is None


# ---------------------------------------------------------------------------
# HighPassFilter testleri
# ---------------------------------------------------------------------------
class TestHighPassFilter:
    def test_dc_component_is_attenuated_offline(self):
        """Sabit (DC, 0 Hz) bir sinyal, yüksek geçiren filtreden sonra
        neredeyse sıfıra yaklaşmalı (tanım gereği 0 Hz her zaman kesim
        frekansının altındadır)."""
        fs = 100.0
        dc_signal = np.full(2000, fill_value=3.0)
        hpf = HighPassFilter(FilterConfig(cutoff_hz=0.5, order=4), sampling_rate_hz=fs)
        filtered = hpf.process_offline(dc_signal)
        # Filtrenin oturması için başlangıç kısmını atlayıp son kısma bakıyoruz
        assert np.abs(filtered[-500:]).max() < 0.05

    def test_invalid_cutoff_raises(self):
        """Kesim frekansı Nyquist limitini aşarsa (fs/2) hata vermeli."""
        with pytest.raises(ValueError):
            HighPassFilter(FilterConfig(cutoff_hz=60.0), sampling_rate_hz=100.0)  # 60 > 50 (Nyquist)

    def test_online_push_matches_offline_length(self):
        """Online modda örnek-örnek filtrelenen sinyal, offline filtrelenen
        sinyalle aynı uzunlukta olmalı (örnek kaybı/tekrar olmamalı)."""
        fs = 100.0
        rng = np.random.default_rng(0)
        signal = rng.normal(size=500)

        hpf_offline = HighPassFilter(sampling_rate_hz=fs)
        offline_result = hpf_offline.process_offline(signal)

        hpf_online = HighPassFilter(sampling_rate_hz=fs)
        online_result = np.array([hpf_online.push(float(s)) for s in signal])

        assert len(online_result) == len(offline_result) == len(signal)


# ---------------------------------------------------------------------------
# PreprocessingPipeline - offline/online PARİTE testi (en kritik test)
# ---------------------------------------------------------------------------
class TestPreprocessingPipelineParity:
    def test_offline_online_window_count_matches(self):
        """PROJENİN TEMEL MİMARİ İDDİASI: aynı sinyal offline ve online
        modda işlendiğinde, AYNI SAYIDA pencere üretilmeli.

        Bu test kırılırsa, offline eğitim ile online inference'ın farklı
        veri gördüğü anlamına gelir (train/serve skew) - bu ciddi bir
        regresyon sinyalidir.
        """
        fs = 100.0
        rng = np.random.default_rng(1)
        n_samples = 5000
        signal = rng.normal(size=n_samples)

        offline_pipeline = PreprocessingPipeline(sampling_rate_hz=fs)
        offline_windows = offline_pipeline.process_offline(signal)

        online_pipeline = PreprocessingPipeline(sampling_rate_hz=fs)
        online_windows = [
            w for s in signal if (w := online_pipeline.process_sample(float(s))) is not None
        ]

        assert len(offline_windows) == len(online_windows)
        for ow, onw in zip(offline_windows, online_windows):
            assert ow.samples.shape == onw.samples.shape == (1024,)

    def test_reset_allows_fresh_session(self):
        """reset() sonrası pipeline, sanki hiç veri görmemiş gibi
        davranmalı (yeni bir bitki/oturuma geçişi simüle eder)."""
        fs = 100.0
        pipeline = PreprocessingPipeline(sampling_rate_hz=fs, window_config=WindowConfig(window_size=10, hop_size=10))
        for i in range(10):
            pipeline.process_sample(float(i))
        pipeline.reset()
        for i in range(9):
            assert pipeline.process_sample(float(i)) is None