"""
features/mfcc_features.py
============================
MFCC (mel-frekans kepstral katsayıları) tabanlı frekans özellik çıkarımı.
Prompt'ta belirtildiği gibi, MFCC "sinyaldeki frekans modülasyonunu güçlü
şekilde yakaladığı gösterilmiştir" - bu modül, statistical_features.py'nin
ZAMAN domenindeki özelliklerine (mean, variance, skewness...) ek olarak,
FREKANS domenindeki bilgiyi yakalıyor.

Neden `librosa` gibi hazır bir kütüphane KULLANMADIK?
--------------------------------------------------------
MFCC'nin adımlarını (FFT -> mel filtrebank -> log -> DCT) elle yazmak,
hem gereksiz ağır bir bağımlılık (librosa'nın numba/llvmlite gibi
büyük alt bağımlılıkları var) eklememizi önlüyor, hem de her adımın
NE yaptığını şeffaf tutuyor - bu proje için "kod çalışıyor ama içi
kara kutu" kabul edilebilir değil.

ŞEFFAFLİK NOTU (bkz. config.py MFCCConfig docstring'i):
------------------------------------------------------------
"Mel" skalası insan işitme algısından ödünç alınmış bir yöntemdir.
Burada "bitki sesi duyuyor" gibi bir biyolojik iddia YOK - mel-warped
filtrebank, literatürde ampirik olarak frekans modülasyonunu yakalamada
işe yaradığı gösterildiği için bir sinyal işleme TEKNİĞİ olarak
kullanılıyor.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.fft import dct

from config import CONFIG, MFCCConfig
from data.labeling_protocol import LabeledWindow


# ---------------------------------------------------------------------------
# 1) Mel <-> Hz dönüşümleri
# ---------------------------------------------------------------------------
def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Hz cinsinden frekansı mel skalasına çevirir.

    Formül (Slaney/HTK yaklaşımlarından biri, en yaygın kullanılan
    versiyon): mel = 2595 * log10(1 + hz/700). Bu formülün özelliği:
    düşük frekanslarda neredeyse doğrusal, yüksek frekanslarda
    logaritmik sıkıştırma yapması - yani filtrebank düşük frekanslarda
    daha "sık" (yüksek çözünürlük), yüksek frekanslarda daha "seyrek"
    (düşük çözünürlük) filtreler yerleştirir.
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """hz_to_mel()'in tersi - mel skalasından Hz'e geri döner."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# ---------------------------------------------------------------------------
# 2) Mel-warped üçgen filtrebank inşası
# ---------------------------------------------------------------------------
def _build_mel_filterbank(
    n_filters: int, n_fft: int, sampling_rate_hz: float, low_freq_hz: float, high_freq_hz: float
) -> np.ndarray:
    """(n_filters, n_fft//2 + 1) şeklinde bir üçgen filtrebank matrisi
    üretir. Her satır, FFT'nin güç spektrumuna uygulandığında TEK bir
    mel-bandının enerjisini toplayan bir üçgen ağırlık fonksiyonudur.

    Adımlar:
      1) [low_freq_hz, high_freq_hz] aralığını mel skalasına çevir
      2) Bu mel aralığını n_filters+2 EŞİT ARALIKLI noktaya böl (mel
         skalasında eşit aralık - bu, filtrelerin Hz cinsinden GİDEREK
         GENİŞLEDİĞİ anlamına gelir, çünkü mel dönüşümü logaritmiktir)
      3) Bu noktaları geri Hz'e çevirip en yakın FFT bin indeksine yuvarla
      4) Ardışık üç nokta (sol, tepe, sağ) arasında üçgen bir ağırlık
         fonksiyonu oluştur
    """
    mel_low = hz_to_mel(np.array(low_freq_hz))
    mel_high = hz_to_mel(np.array(high_freq_hz))
    mel_points = np.linspace(mel_low, mel_high, n_filters + 2)
    hz_points = mel_to_hz(mel_points)

    # Hz noktalarını FFT bin indekslerine çevir
    bin_indices = np.floor((n_fft + 1) * hz_points / sampling_rate_hz).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft // 2)

    n_freq_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_filters, n_freq_bins))

    for filter_idx in range(n_filters):
        left, center, right = bin_indices[filter_idx : filter_idx + 3]

        # Dejenere durum: komşu mel noktaları aynı FFT bin'ine yuvarlanmışsa
        # (çok düşük n_fft veya çok yüksek n_filters ile olabilir), o
        # filtreyi sıfır bırakıyoruz (hiçbir enerji toplamaz) - bölme
        # hatasına (ZeroDivisionError) düşmek yerine güvenli bir varsayılan.
        if left == center or center == right:
            continue

        # Sol yamaç: left -> center arasında 0'dan 1'e doğrusal artış
        filterbank[filter_idx, left:center] = (
            np.arange(left, center) - left
        ) / (center - left)
        # Sağ yamaç: center -> right arasında 1'den 0'a doğrusal azalış
        filterbank[filter_idx, center:right] = (
            right - np.arange(center, right)
        ) / (right - center)

    return filterbank


# ---------------------------------------------------------------------------
# 3) Tek bir pencereden MFCC çıkarımı
# ---------------------------------------------------------------------------
def get_mfcc_feature_names(mfcc_config: Optional[MFCCConfig] = None) -> tuple[str, ...]:
    """MFCC özellik isimlerini (mfcc_1, mfcc_2, ...) döndürür.

    statistical_features.py'deki STATISTICAL_FEATURE_NAMES ile aynı
    gerekçe: eğitim ve inference arasında sütun sırasının KAYMAMASI için
    isimler tek bir yerden üretiliyor.
    """
    config = mfcc_config or CONFIG.mfcc
    return tuple(f"mfcc_{i + 1}" for i in range(config.n_mfcc))


def extract_mfcc_features(
    window: np.ndarray, sampling_rate_hz: float, mfcc_config: Optional[MFCCConfig] = None
) -> np.ndarray:
    """Bir pencereden (1D numpy array) MFCC katsayılarını çıkarır.

    ÖNEMLİ FARK: statistical_features.py'deki fonksiyonların aksine, bu
    fonksiyon `sampling_rate_hz` parametresi GEREKTİRİR. Sebep: mel
    filtrebank'ın hangi Hz aralığını kapsayacağı (ve dolayısıyla hangi
    FFT bin'lerinin hangi filtreye denk geldiği), örnekleme hızına
    doğrudan bağlıdır - aynı pencere 10 Hz'de ve 100 Hz'de tamamen
    farklı bir frekans içeriğini temsil eder.

    Adımlar (klasik MFCC pipeline'ı, konuşma/ses işlemeden ödünç alınmış):
      1) Pencereye bir Hamming penceresi uygula (kenar etkilerini/spektral
         sızıntıyı azaltmak için - FFT'nin sinyalin SONSUZ tekrar ettiğini
         varsaydığını, pencere kenarlarındaki ani kesintinin sahte
         frekans bileşenleri (spektral sızıntı) yaratabileceğini unutma)
      2) FFT ile güç spektrumunu hesapla
      3) Mel-warped üçgen filtrebank ile spektrumu n_mel_filters banda
         indirge (her bandın enerjisini topla)
      4) Enerjilerin logaritmasını al (insan işitmesi/genel sinyal
         dinamiği logaritmik algılanır/dağılır - büyük enerji
         farklarını sıkıştırır)
      5) DCT (ayrık kosinüs dönüşümü) ile bandlar arası KORELASYONU
         azalt ve enerjiyi ilk birkaç katsayıda yoğunlaştır, ilk
         n_mfcc katsayıyı tut
    """
    config = mfcc_config or CONFIG.mfcc
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 1:
        raise ValueError(f"1 boyutlu pencere bekleniyor, gelen şekil: {window.shape}")

    n = window.size
    high_freq_hz = config.high_freq_hz or (sampling_rate_hz / 2.0)  # varsayılan: Nyquist

    # Adım 1-2: pencereleme + güç spektrumu
    windowed_signal = window * np.hamming(n)
    power_spectrum = (np.abs(np.fft.rfft(windowed_signal, n=n)) ** 2) / n

    # Adım 3: mel filtrebank uygulaması
    filterbank = _build_mel_filterbank(
        config.n_mel_filters, n, sampling_rate_hz, config.low_freq_hz, high_freq_hz
    )
    filter_energies = filterbank @ power_spectrum

    # Adım 4: log - sıfır enerjili bantlarda log(0) = -inf hatasından
    # kaçınmak için küçük bir epsilon ile tabana çekiyoruz.
    filter_energies = np.maximum(filter_energies, np.finfo(np.float64).eps)
    log_energies = np.log(filter_energies)

    # Adım 5: DCT-II, ortonormal norm (enerji-korur, literatürde standart)
    mfcc = dct(log_energies, type=2, norm="ortho")[: config.n_mfcc]

    return mfcc


def extract_mfcc_features_dict(
    window: np.ndarray, sampling_rate_hz: float, mfcc_config: Optional[MFCCConfig] = None
) -> dict[str, float]:
    """extract_mfcc_features()'ın isim->değer sözlüğü döndüren, insan
    okunur versiyonu (debug/EDA için)."""
    values = extract_mfcc_features(window, sampling_rate_hz, mfcc_config)
    names = get_mfcc_feature_names(mfcc_config)
    return dict(zip(names, values))


def build_mfcc_feature_matrix(
    labeled_windows: Sequence[LabeledWindow],
    sampling_rate_hz: float,
    mfcc_config: Optional[MFCCConfig] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """statistical_features.py'deki build_feature_matrix() ile AYNI
    (X, y, groups) sözleşmesini uygular - farkı sadece özellik çıkarım
    fonksiyonunun MFCC olması. Bu tutarlılık BİLİNÇLİ: models/
    katmanı, hangi özellik setiyle beslendiğini bilmeden aynı
    evaluate_models_grouped_cv() fonksiyonunu kullanabiliyor.
    """
    if len(labeled_windows) == 0:
        raise ValueError("Boş LabeledWindow listesinden özellik matrisi üretilemez.")

    features: list[np.ndarray] = []
    labels: list[str] = []
    plant_ids: list[str] = []

    for lw in labeled_windows:
        features.append(extract_mfcc_features(lw.samples, sampling_rate_hz, mfcc_config))
        labels.append(lw.label.value)
        plant_ids.append(lw.plant_id)

    X = np.stack(features, axis=0)
    y = np.array(labels)
    groups = np.array(plant_ids)
    return X, y, groups


if __name__ == "__main__":
    # Self-check: farklı frekans içeriğine sahip iki sentetik pencerede
    # MFCC katsayılarının GERÇEKTEN farklı çıktığını gösteriyoruz -
    # bu, "frekans modülasyonunu yakalama" iddiasının kanıtı.
    fs = CONFIG.acquisition.event_sampling_rate_hz  # 100 Hz
    n = CONFIG.window.window_size
    t = np.arange(n) / fs

    low_freq_signal = np.sin(2 * np.pi * 2.0 * t)     # 2 Hz - yavaş salınım
    high_freq_signal = np.sin(2 * np.pi * 20.0 * t)   # 20 Hz - hızlı salınım
    rng = np.random.default_rng(0)
    noisy_signal = rng.normal(0, 1.0, n)               # geniş bantlı gürültü

    for name, sig in [
        ("2 Hz sinüs (yavaş)", low_freq_signal),
        ("20 Hz sinüs (hızlı)", high_freq_signal),
        ("beyaz gürültü", noisy_signal),
    ]:
        mfcc = extract_mfcc_features(sig, sampling_rate_hz=fs)
        print(f"\n[{name}]")
        print(f"  İlk 5 katsayı: {np.round(mfcc[:5], 3)}")

    print(f"\nÖzellik isimleri: {get_mfcc_feature_names()}")