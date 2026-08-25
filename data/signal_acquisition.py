"""
data/signal_acquisition.py
=============================
Ham sinyal örneklerinin NEREDEN geldiğini soyutlayan modül.

Prompt'un donanım önerisi: sıfırdan analog amplifikatör tasarlamak yerine
Backyard Brains'in Plant SpikerBox'ı (elektrot + amplifikatör + Arduino
arayüzü hazır, açık kaynak) kullanılmalı. Bu cihaz, okunan ADC değerlerini
Arduino üzerinden SERİ PORT (USB) üzerinden ASCII sayı olarak gönderir.

Bu modülün tasarım ilkesi: inference.py'nin (ve dolayısıyla tüm pipeline'ın)
örneklerin GERÇEK donanımdan mı yoksa bir SİMÜLASYONDAN mı geldiğini
BİLMEMESİ gerekir. Her iki kaynak da aynı `SignalSource` arayüzünü
uyguluyor - `RealtimeClassifier.process_sample()` hangi kaynaktan
beslendiğinden habersiz, sadece "bir sonraki float değeri" alıyor.

Bu, tam olarak preprocessing.py'deki offline/online paylaşımlı çekirdek
mantığının bir üst katmandaki (donanım soyutlaması) yansıması: aynı
felsefe, projenin her katmanında tekrarlanıyor.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from config import CONFIG, AcquisitionConfig

try:
    import serial as _pyserial

    _PYSERIAL_AVAILABLE = True
except ImportError:  # pragma: no cover - donanım olmayan ortamlarda beklenir
    _PYSERIAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1) ADC kalibrasyonu - config.py'deki AcquisitionConfig alanlarının
#    ilk kez GERÇEK bir işlevi burada kullanılıyor
# ---------------------------------------------------------------------------
def adc_counts_to_millivolts(
    raw_count: int, acquisition_config: Optional[AcquisitionConfig] = None
) -> float:
    """Ham ADC sayacını (örn. 0-65535 aralığında bir tam sayı) milivolt
    cinsinden gerçek bir gerilim değerine çevirir.

    Formül: (ham_deger / max_deger) * giris_araligi_mV

    Neden bu dönüşüm gerekli? ADC, donanımın fiziksel giriş aralığını
    (örn. 0-5V) sabit bit derinliğinde (örn. 16 bit -> 0-65535) sayısal
    değerlere böler. Ham sayaçları DOĞRUDAN filtre/model'e vermek yerine
    fiziksel birime (mV) çevirmek, farklı ADC donanımları (farklı bit
    derinliği/aralık) arasında GEÇİŞ YAPILDIĞINDA modelin yeniden
    eğitilmesi gerekmeden çalışabilmesini sağlar - kalibrasyon farkı
    burada, tek bir yerde absorbe ediliyor.
    """
    config = acquisition_config or CONFIG.acquisition
    max_count = (2**config.adc_resolution_bits) - 1
    return (raw_count / max_count) * config.input_range_mv


# ---------------------------------------------------------------------------
# 2) Soyut arayüz - hem gerçek donanım hem simülasyon bunu uygular
# ---------------------------------------------------------------------------
class SignalSource(ABC):
    """Tek bir örnek üretebilen herhangi bir kaynağın uyması gereken
    sözleşme. `inference.py`'deki RealtimeClassifier, SADECE bu arayüzü
    bilir - somut implementasyonu (gerçek donanım mı, simülasyon mu)
    hiç bilmez.

    Context manager (`with ... as source:`) olarak kullanılabilir olması
    ZORUNLU tutuldu: seri port gibi kaynaklar açık kalırsa (örn. bir
    hata sonucu erken çıkılırsa) donanım kilitli kalabilir - `with`
    bloğu, hata olsa bile `close()`'un çağrılmasını garanti eder.
    """

    @property
    @abstractmethod
    def sampling_rate_hz(self) -> float:
        """Bu kaynağın ürettiği örneklerin nominal örnekleme hızı."""

    @abstractmethod
    def read_sample(self) -> float:
        """Bir sonraki örneği döndürür (bloklayıcı olabilir - donanımdan
        yeni veri gelene kadar bekleyebilir)."""

    def close(self) -> None:
        """Kaynağı serbest bırakır (seri port kapatma vb.). Varsayılan
        olarak hiçbir şey yapmaz - alt sınıflar gerektiğinde override eder."""

    def __enter__(self) -> "SignalSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# 3) Gerçek donanım: Plant SpikerBox (Arduino, seri port)
# ---------------------------------------------------------------------------
class SerialADCSource(SignalSource):
    """Backyard Brains Plant SpikerBox'tan (veya benzeri, Arduino tabanlı,
    seri port üzerinden ASCII tam sayı gönderen herhangi bir cihazdan)
    canlı örnek okur.

    Beklenen protokol: Arduino firmware'i her satırda TEK bir ham ADC
    değerini (tam sayı, newline ile ayrılmış) seri porta yazıyor -
    Plant SpikerBox'ın standart/varsayılan çıktı formatı budur.

    NOT: Bu sınıf gerçek donanım BAĞLANTISI gerektirir; donanım
    bağlanmadan test edilemez. Donanım olmadan pipeline'ı test etmek
    için `SimulatedSignalSource` kullanılmalı (bkz. aşağısı).
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        sampling_rate_hz: Optional[float] = None,
        acquisition_config: Optional[AcquisitionConfig] = None,
        timeout_s: float = 2.0,
    ) -> None:
        if not _PYSERIAL_AVAILABLE:
            raise ImportError(
                "pyserial kurulu değil. Gerçek donanımdan okumak için "
                "'pip install pyserial' çalıştırın."
            )
        self.acquisition_config = acquisition_config or CONFIG.acquisition
        self._sampling_rate_hz = sampling_rate_hz or self.acquisition_config.event_sampling_rate_hz
        self._serial = _pyserial.Serial(port=port, baudrate=baud_rate, timeout=timeout_s)

    @property
    def sampling_rate_hz(self) -> float:
        return self._sampling_rate_hz

    def read_sample(self) -> float:
        """Seri porttan bir satır okur, ham ADC sayacını parse edip
        milivolt cinsine çevirir.

        Bozuk/okunamayan bir satır gelirse (örn. USB kablosu gürültüsü,
        firmware başlangıç mesajı) ValueError fırlatmak yerine 0.0
        DÖNMÜYORUZ bilerek - sessizce yanlış bir değer enjekte etmek,
        gerçek bir okuma hatasını maskeler. Çağıran taraf (örn. bir
        retry döngüsü) bu hatayı ele almalı.
        """
        line = self._serial.readline().decode("ascii", errors="strict").strip()
        if not line:
            raise TimeoutError(
                f"Seri porttan {self._serial.timeout}s içinde veri gelmedi - "
                f"bağlantıyı/donanımı kontrol edin."
            )
        raw_count = int(line)
        return adc_counts_to_millivolts(raw_count, self.acquisition_config)

    def close(self) -> None:
        if self._serial.is_open:
            self._serial.close()


# ---------------------------------------------------------------------------
# 4) Simülasyon: donanım olmadan geliştirme/test/demo için
# ---------------------------------------------------------------------------
class SimulatedSignalSource(SignalSource):
    """Elde zaten kaydedilmiş (veya sentetik) bir sinyal dizisini, SANKİ
    canlı bir donanımdan geliyormuş gibi, GERÇEK DUVAR SAATİ zamanlamasıyla
    yeniden oynatan kaynak.

    inference.py'deki `simulate_realtime_stream()`'den farkı: o fonksiyon
    zamanlamayı HİÇ simüle etmiyordu (hızlı test için bilerek); bu sınıf
    ise `time.sleep()` ile GERÇEK örnekleme aralığını bekliyor - donanım
    henüz elimize geçmeden, sistemin GERÇEK ZAMANLI kısıtlar altında
    (örn. gecikme bütçesi) nasıl davrandığını dürüstçe test etmek için.

    realtime_factor: Gerçek bekleme süresini hızlandırmak için (örn. 10.0
    -> her şey 10 kat hızlı oynatılır). Demo/geliştirme sırasında 30
    saniyelik bir kaydı gerçekten 30 saniye beklemek pratik değil;
    realtime_factor=1.0 dürüst bir gerçek-zamanlı test, >1.0 hızlandırılmış
    bir önizleme sağlar.
    """

    def __init__(
        self, signal: np.ndarray, sampling_rate_hz: float, realtime_factor: float = 1.0
    ) -> None:
        if realtime_factor <= 0:
            raise ValueError("realtime_factor pozitif olmalı.")
        self._signal = np.asarray(signal, dtype=np.float64)
        self._sampling_rate_hz = sampling_rate_hz
        self._realtime_factor = realtime_factor
        self._index = 0
        self._sample_interval_s = (1.0 / sampling_rate_hz) / realtime_factor

    @property
    def sampling_rate_hz(self) -> float:
        return self._sampling_rate_hz

    def read_sample(self) -> float:
        if self._index >= len(self._signal):
            raise StopIteration("Simüle edilen sinyalin sonuna ulaşıldı.")
        time.sleep(self._sample_interval_s)
        sample = float(self._signal[self._index])
        self._index += 1
        return sample

    @property
    def remaining_samples(self) -> int:
        return max(0, len(self._signal) - self._index)


# ---------------------------------------------------------------------------
# 5) Kaynağı doğrudan bir RealtimeClassifier'a bağlayan çalışma döngüsü
# ---------------------------------------------------------------------------
def run_realtime_loop(
    source: SignalSource,
    classifier,  # inference.RealtimeClassifier (döngüsel import'tan kaçınmak için type hint'siz)
    on_result: Callable,
    max_samples: Optional[int] = None,
) -> int:
    """Bir SignalSource'tan örnek okuyup RealtimeClassifier'a besleyen,
    her yeni sonuç üretildiğinde `on_result` callback'ini çağıran ana
    döngü. Hem gerçek donanımla (SerialADCSource) hem simülasyonla
    (SimulatedSignalSource) İDENTİK şekilde çalışır - kaynağı değiştirmek
    tek satırlık bir değişiklik.

    Dönüş değeri: işlenen toplam örnek sayısı (loglama/debug için).
    """
    n_processed = 0
    while max_samples is None or n_processed < max_samples:
        try:
            sample = source.read_sample()
        except StopIteration:
            break

        result = classifier.process_sample(sample)
        if result is not None:
            on_result(result)

        n_processed += 1

    return n_processed


if __name__ == "__main__":
    # Demo: donanım olmadan, SimulatedSignalSource + gerçek RealtimeClassifier
    # ile uçtan uca bir "canlı" döngü çalıştırıyoruz. realtime_factor=200 ile
    # 20 saniyelik bir kaydı ~0.1 saniyede oynatıyoruz (demo hızlı olsun diye).
    from features.statistical_features import extract_statistical_features
    from inference import RealtimeClassifier
    from models.classical_models import train_final_model

    fs = CONFIG.acquisition.event_sampling_rate_hz
    rng = np.random.default_rng(0)

    # ADC kalibrasyon demosu
    print("--- ADC kalibrasyon örneği ---")
    for raw in [0, 32768, 65535]:
        mv = adc_counts_to_millivolts(raw)
        print(f"  ham ADC={raw:6d} -> {mv:.3f} mV")

    # Hızlı bir model eğit (inference.py demosuyla aynı mantık)
    def make_labeled_feature(is_touch: bool) -> np.ndarray:
        window = rng.normal(0, 0.02, CONFIG.window.window_size)
        if is_touch:
            window[CONFIG.window.window_size // 2] += 3.0
        return extract_statistical_features(window)

    X_train = np.stack(
        [make_labeled_feature(False) for _ in range(40)]
        + [make_labeled_feature(True) for _ in range(40)]
    )
    y_train = np.array(["baseline"] * 40 + ["mechanical_touch"] * 40)
    trained_model = train_final_model("random_forest", X_train, y_train)
    classifier = RealtimeClassifier(trained_model, sampling_rate_hz=fs)

    # Simüle edilmiş "canlı" sinyal - 20 saniye, ortasında dokunma
    duration_s = 20
    n_samples = int(duration_s * fs)
    live_signal = rng.normal(0, 0.02, n_samples)
    live_signal[n_samples // 2] += 3.0

    print(f"\n--- Simüle edilmiş canlı akış (gerçek {duration_s}s, {200}x hızlandırılmış) ---")
    results = []
    with SimulatedSignalSource(live_signal, sampling_rate_hz=fs, realtime_factor=200.0) as source:
        n_processed = run_realtime_loop(
            source, classifier, on_result=lambda r: results.append(r), max_samples=n_samples
        )

    print(f"{n_processed} örnek işlendi, {len(results)} pencere sınıflandırıldı:")
    for r in results:
        print(f"  Pencere {r.window_index} -> '{r.label}' ({r.inference_latency_s * 1000:.2f} ms)")