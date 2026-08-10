"""
config.py
=========
Proje genelinde kullanılan tüm sinyal işleme, veri toplama ve etiketleme
parametrelerinin **tek gerçek kaynağı** (single source of truth).

Neden merkezi bir config dosyası?
----------------------------------
Bu projenin özgünlük ekseni "aynı pencereleme/filtreleme mantığının hem
offline (eğitim) hem online (gerçek zamanlı inference) modunda birebir
aynı şekilde çalışması" üzerine kurulu. Eğer pencere boyutu veya filtre
kesim frekansı iki farklı yerde (örn. training script'inde ve
inference.py'de) ayrı ayrı hardcode edilirse, zamanla ikisi birbirinden
sessizce sapar (bu duruma literatürde "train/serve skew" denir) ve model
canlı ortamda eğitildiği dağılımdan farklı bir girdi görmeye başlar.
Bu yüzden her modül (preprocessing, features, models, inference) sinyal
parametrelerini DOĞRUDAN buradan okuyacak, kendi içinde tekrar tanımlamayacak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# 1) Etiket seti
# ---------------------------------------------------------------------------
class StimulusLabel(str, Enum):
    """Kontrollü, tekrarlanabilir uyaran sınıfları.

    str alt sınıfından türetildi ki hem tip güvenliği (enum) hem de
    doğrudan string gibi kullanılabilirlik (örn. dosya adlandırma,
    pandas DataFrame kolonu) elde edelim.

    Bilinçli olarak antropomorfik OLMAYAN, fizyolojik/deneysel
    terminoloji kullanıldı (proje kısıtı: "mutlu/üzgün/korkuyor" gibi
    ifadeler yasak).
    """

    BASELINE = "baseline"
    MECHANICAL_TOUCH = "mechanical_touch"          # en güçlü, en hızlı sinyal
    LIGHT_TRANSITION = "light_transition"           # aydınlık/karanlık geçişi
    DROUGHT_STRESS = "drought_stress"               # kademeli, gün(ler) süren
    CHEMICAL_SALT_STRESS = "chemical_salt_stress"
    TEMPERATURE_SHOCK = "temperature_shock"

    @classmethod
    def fast_event_labels(cls) -> tuple["StimulusLabel", ...]:
        """Saniyeler-dakikalar mertebesinde gelişen, yüksek örnekleme hızı
        gerektiren sınıflar (bkz. AcquisitionConfig.event_sampling_rate_hz)."""
        return (cls.BASELINE, cls.MECHANICAL_TOUCH, cls.LIGHT_TRANSITION)

    @classmethod
    def slow_process_labels(cls) -> tuple["StimulusLabel", ...]:
        """Saatler-günler mertebesinde kademeli gelişen, düşük örnekleme
        hızıyla uzun süreli izleme gerektiren sınıflar."""
        return (cls.DROUGHT_STRESS, cls.CHEMICAL_SALT_STRESS, cls.TEMPERATURE_SHOCK)


# ---------------------------------------------------------------------------
# 2) Veri toplama (acquisition) parametreleri
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AcquisitionConfig:
    """Donanımdan sinyal okuma ile ilgili parametreler.

    İki farklı örnekleme rejimi tanımlıyoruz çünkü tek bir sabit hız
    literatürdeki iki farklı fizyolojik zaman ölçeğini karşılayamıyor:

    - baseline_sampling_rate_hz: PhytoNode tarzı sürekli/uzun-vadeli izleme
      (kuraklık, kimyasal stres gibi yavaş süreçler) için düşük güç
      tüketimli, düşük hızlı kayıt.
    - event_sampling_rate_hz: Mekanik dokunma/yaralanma gibi hızlı aksiyon
      potansiyellerini kaçırmamak için gereken yüksek hız.

    NOT: Bu iki değer arasında sistemin hangi modda çalışacağını
    inference.py çalışma zamanında (hangi sınıf ailesi izleniyorsa)
    seçecek; bu config sadece izin verilen değerleri tanımlıyor.
    """

    baseline_sampling_rate_hz: float = 10.0     # PhytoNode referansı
    event_sampling_rate_hz: float = 100.0       # dokunma/yaralanma için min. hız
    adc_resolution_bits: int = 16
    input_range_mv: float = 5.0                 # tipik plant SpikerBox çıkış aralığı


# ---------------------------------------------------------------------------
# 3) Filtre parametreleri
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterConfig:
    """Yüksek geçiren (high-pass) IIR filtre parametreleri.

    Amaç: Ham sinyaldeki yavaş baseline kaymasını (elektrot polarizasyonu,
    sıcaklık kaynaklı yavaş drift vb.) elemek ve uyaran-kaynaklı hızlı,
    stokastik bileşeni öne çıkarmak (Sai, Sood, Saini 2022 metodolojisi).

    ÖNEMLİ - şeffaflık notu: Kaynak makaleler kesim frekansının TAM
    sayısal değerini raporlamıyor; bu yüzden cutoff_hz ve order burada
    mühendislik pratiğine dayalı, TÜNE EDİLMESİ GEREKEN başlangıç
    değerleri olarak seçildi (bitki biyoelektrik sinyalinde tipik ilgi
    alanı > 0.1-0.5 Hz). Bunları makalelerden birebir alınmış kesin
    sayılar gibi sunmuyoruz; deney verisiyle çapraz doğrulanmalı.
    """

    cutoff_hz: float = 0.5
    order: int = 4               # Butterworth filtre derecesi
    filter_type: str = "highpass"


# ---------------------------------------------------------------------------
# 4) Pencereleme (windowing) parametreleri
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowConfig:
    """Sabit boyutlu pencere parametreleri.

    window_size: Literatürdeki 1024 örneklik sabit pencere (Sai, Sood,
    Saini 2022). Örnekleme hızına göre bu pencerenin süresi değişir:
        - event_sampling_rate_hz=100 Hz  -> 1024/100  = 10.24 saniyelik pencere
        - baseline_sampling_rate_hz=10Hz -> 1024/10   = 102.4 saniyelik pencere
    Bu kasıtlı: hızlı olaylar kısa pencerede, yavaş süreçler doğal olarak
    daha uzun bir zaman diliminde değerlendiriliyor.

    hop_size: Ardışık pencereler arasındaki kayma miktarı (örnek sayısı).
    hop_size < window_size ise pencereler örtüşür (overlap) — hem offline
    eğitim verisini zenginleştirmek hem de online modda daha sık tahmin
    üretmek için kullanışlı. hop_size == window_size ise örtüşme yoktur.
    """

    window_size: int = 1024
    hop_size: int = 512          # %50 overlap, varsayılan


# ---------------------------------------------------------------------------
# 5) Sınıflandırma / gerçek zamanlı performans hedefleri
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RealtimeConfig:
    """inference.py için performans bütçesi.

    max_inference_latency_ratio: Bir pencerenin gerçek dünya süresine göre,
    o pencere için özellik çıkarımı + sınıflandırmanın alabileceği azami
    süre oranı. 1.0'dan küçük olmalı ki sistem sinyali "yetişerek" işlesin
    (aksi halde giriş kuyruğu birikir ve gecikme kümülatif artar).
    """

    max_inference_latency_ratio: float = 0.5   # pencere süresinin en fazla yarısı


# ---------------------------------------------------------------------------
# 6) Dosya yolları
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "datasets" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "datasets" / "processed")
    model_artifacts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "artifacts")


# ---------------------------------------------------------------------------
# 7) Hepsini bir araya getiren üst-config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectConfig:
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Modül seviyesinde tek bir örnek (singleton benzeri kullanım):
# diğer modüller `from config import CONFIG` ile bunu import edecek.
CONFIG = ProjectConfig()


if __name__ == "__main__":
    # Hızlı doğrulama: config'in beklenen gibi kurulduğunu göster.
    import json

    print("Aktif proje konfigürasyonu:")
    print(f"  Baseline örnekleme hızı : {CONFIG.acquisition.baseline_sampling_rate_hz} Hz")
    print(f"  Event örnekleme hızı    : {CONFIG.acquisition.event_sampling_rate_hz} Hz")
    print(f"  Filtre kesim frekansı   : {CONFIG.filter.cutoff_hz} Hz (order={CONFIG.filter.order})")
    print(f"  Pencere boyutu          : {CONFIG.window.window_size} örnek")
    print(f"  Event modunda pencere süresi: {CONFIG.window.window_size / CONFIG.acquisition.event_sampling_rate_hz:.2f} s")
    print(f"  Baseline modunda pencere süresi: {CONFIG.window.window_size / CONFIG.acquisition.baseline_sampling_rate_hz:.2f} s")