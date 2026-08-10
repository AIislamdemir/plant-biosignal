"""
data/preprocessing.py
======================
Ham bitki biyoelektrik sinyaline uygulanan iki temel ön işleme adımı:

    1) HighPassFilter  -> yavaş baseline kaymasını eler
    2) SlidingWindower  -> sinyali sabit boyutlu pencerelere böler

Bu modülün TASARIM İLKESİ (projenin özgünlük ekseniyle doğrudan bağlantılı):
------------------------------------------------------------------------
Literatürdeki çalışmaların hepsi offline çalışıyor: tüm sinyal önce
diskte/bellekte toplanıyor, SONRA bir bütün olarak filtrelenip
pencereleniyor. Biz ise `inference.py`'de sinyali örnek-örnek CANLI
okuyacağız. Eğer offline filtreleme/pencereleme mantığını online mod
için AYRI bir kod olarak yazarsak, iki mantık zamanla birbirinden
sapabilir ve modelin gördüğü eğitim verisi ile canlı ortamda gördüğü
veri arasında ince farklar oluşur (train/serve skew) - bu, gerçek
zamanlı ML sistemlerindeki en sinsi hata kaynaklarından biridir.

Bunu önlemek için burada TEK bir "çekirdek" işlem tanımlıyoruz
(`push` metodu, örnek-örnek çalışır) ve offline mod (`process_offline`)
bu çekirdeği sadece bir döngü içinde çağırarak kullanıyor. Yani offline
kod, online kodun bir "toplu" (batch) sarmalayıcısından ibaret - ayrı
bir implementasyon değil.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from config import CONFIG, FilterConfig, WindowConfig


# ---------------------------------------------------------------------------
# 1) Yüksek geçiren IIR filtre
# ---------------------------------------------------------------------------
class HighPassFilter:
    """Butterworth yüksek geçiren filtre; hem offline (zero-phase) hem
    online (causal, stateful) kullanım için TEK bir filtre tasarımını
    paylaşır.

    Neden iki farklı uygulama metodu var (process_offline vs push)?
    -----------------------------------------------------------------
    Bu, sinyal işlemede sık yapılan bir hatanın kaynağı olduğu için
    özellikle açıklıyorum:

    - `scipy.signal.sosfiltfilt` (offline'da kullandığımız) sinyali
      HEM ileri HEM geri yönde filtreler. Bu "zero-phase" bir filtre
      üretir (faz kayması yoktur, sinyal şekli bozulmaz) — ama bunu
      yapabilmek için sinyalin TAMAMINA, yani GELECEK örneklere de
      ihtiyaç duyar. Bu yüzden SADECE offline/batch modda kullanılabilir.

    - `scipy.signal.sosfilt` (online'da kullandığımız) sinyali sadece
      ileri yönde, örnek geldikçe filtreler ("causal" filtre). Gelecek
      örneklere ihtiyaç duymaz, bu yüzden gerçek zamanlı kullanılabilir.
      Bedeli: küçük bir faz gecikmesi (phase delay) oluşur — sınıflandırma
      için bu genelde sorun değildir çünkü model zaten filtrelenmiş
      sinyal üzerinde eğitilecek, ama okuyucunun bu farkı bilmesi önemli.

    Online modda filtrenin "hafızası" (iç durumu, `zi`) her yeni örnekte
    güncellenip saklanmalı; aksi halde her örnekte filtre sıfırdan
    başlar ve pencere sınırlarında sahte geçiş (transient) sıçramaları
    oluşur. Bu yüzden `self._zi` iki çağrı arasında kalıcı tutulur.
    """

    def __init__(self, filter_config: Optional[FilterConfig] = None, sampling_rate_hz: float = 100.0) -> None:
        self.config = filter_config or CONFIG.filter
        self.sampling_rate_hz = sampling_rate_hz

        nyquist_hz = sampling_rate_hz / 2.0
        normalized_cutoff = self.config.cutoff_hz / nyquist_hz
        if not (0.0 < normalized_cutoff < 1.0):
            raise ValueError(
                f"Kesim frekansı ({self.config.cutoff_hz} Hz) örnekleme hızına "
                f"({sampling_rate_hz} Hz) göre geçersiz. Nyquist limiti: {nyquist_hz} Hz."
            )

        # SOS (second-order sections) formu tercih edildi: yüksek dereceli
        # filtrelerde klasik (b, a) katsayı formuna göre sayısal olarak
        # çok daha kararlıdır (özellikle order >= 4 için).
        self._sos = butter(
            N=self.config.order,
            Wn=normalized_cutoff,
            btype=self.config.filter_type,
            output="sos",
        )

        # Online (causal) filtrenin durumu (state). None -> henüz hiç
        # örnek işlenmedi, ilk çağrıda sıfırdan başlatılacak.
        self._zi: Optional[np.ndarray] = None

    def process_offline(self, signal: np.ndarray) -> np.ndarray:
        """Tüm sinyali TEK SEFERDE, zero-phase (ileri-geri) filtreler.

        SADECE offline/batch işleme için kullanılmalı (eğitim verisi
        hazırlarken). Gerçek zamanlı akışta kullanılamaz çünkü gelecek
        örneklere ihtiyaç duyar.
        """
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim != 1:
            raise ValueError(f"1 boyutlu sinyal bekleniyor, gelen şekil: {signal.shape}")
        return sosfiltfilt(self._sos, signal)

    def push(self, sample: float) -> float:
        """Tek bir örneği causal (nedensel) olarak filtreler ve filtrelenmiş
        değeri döndürür. Gerçek zamanlı akış için kullanılır.

        İç durum (`self._zi`) çağrılar arasında korunur; bu sayede sanki
        sürekli akan bir sinyalmiş gibi doğru şekilde filtrelenir.
        """
        if self._zi is None:
            # scipy'nin sosfilt_zi'si "kararlı duruma" (steady-state) en
            # hızlı ulaşacak başlangıç durumunu hesaplar; bu, filtrenin
            # ilk örneklerde sahte bir sıçrama (transient) üretmesini azaltır.
            from scipy.signal import sosfilt_zi

            zi_unit = sosfilt_zi(self._sos)
            # İlk örnek değeriyle ölçekleyerek başlatmak, sinyal sıfırdan
            # uzak bir DC seviyesinden başlıyorsa transienti daha da azaltır.
            self._zi = zi_unit * sample

        filtered, self._zi = sosfilt(self._sos, [sample], zi=self._zi)
        return float(filtered[0])

    def reset(self) -> None:
        """Filtre durumunu sıfırlar. Yeni bir kayıt oturumu (örn. yeni bir
        bitkiye/deneye geçerken) başlarken çağrılmalı; aksi halde önceki
        oturumdan kalan filtre hafızası yeni sinyale sızar."""
        self._zi = None


# ---------------------------------------------------------------------------
# 2) Kayan pencere (sliding window)
# ---------------------------------------------------------------------------
class SlidingWindower:
    """Sabit boyutlu, opsiyonel örtüşmeli (overlap) pencereleme.

    Çekirdek mantık `push()` metodunda: örnek örnek beslenir, dahili bir
    `deque` (sabit maksimum uzunluklu tampon) doldurulur, tampon dolduğunda
    ve "hop" sayacı tetiklendiğinde pencerenin bir KOPYASI döndürülür.

    `process_offline()` bu metodu bir dizinin üzerinde döngüyle çağırmaktan
    başka bir şey yapmaz -> offline ve online mod GERÇEKTEN aynı kodu
    çalıştırır, sadece girdi kaynağı farklıdır (hazır dizi vs. tek tek gelen
    canlı örnek).

    Neden kopya (`.copy()`) döndürüyoruz?
    ---------------------------------------
    `deque`'in içindeki numpy array'i doğrudan döndürürsek, bir sonraki
    `push()` çağrısı tamponun içeriğini değiştirir ve çağıran taraf elinde
    tuttuğu "pencere"nin farkında olmadan bozulduğunu görür. Bu, gerçek
    zamanlı sistemlerde yakalanması çok zor bir hataya (silent data
    corruption) yol açar; bu yüzden burada performanstan feragat edip
    açıkça bir kopya veriyoruz.
    """

    def __init__(self, window_config: Optional[WindowConfig] = None) -> None:
        self.config = window_config or CONFIG.window
        if self.config.hop_size <= 0 or self.config.hop_size > self.config.window_size:
            raise ValueError(
                f"hop_size (0, window_size] aralığında olmalı. "
                f"Gelen: hop_size={self.config.hop_size}, window_size={self.config.window_size}"
            )

        self._buffer: deque[float] = deque(maxlen=self.config.window_size)
        # Bir sonraki pencerenin üretilmesine kaç örnek kaldığını sayar.
        # Başlangıçta window_size kadar örnek gerekiyor (ilk pencere dolmalı).
        self._samples_until_next_window = self.config.window_size

    def push(self, sample: float) -> Optional[np.ndarray]:
        """Tek bir (filtrelenmiş) örneği tampona ekler.

        Tampon dolu VE hop sayacı sıfırlandıysa, `window_size` uzunluğunda
        bir numpy array (pencere) döndürür. Aksi halde None döner (henüz
        yeni bir pencere üretecek kadar örnek birikmedi).
        """
        self._buffer.append(sample)
        self._samples_until_next_window -= 1

        buffer_is_full = len(self._buffer) == self.config.window_size
        hop_reached = self._samples_until_next_window <= 0

        if buffer_is_full and hop_reached:
            window = np.array(self._buffer, dtype=np.float64)  # kopya üretir
            self._samples_until_next_window = self.config.hop_size
            return window

        return None

    def process_offline(self, signal: np.ndarray) -> list[np.ndarray]:
        """Bir dizinin tamamını `push()` üzerinden geçirip üretilen tüm
        pencerelerin listesini döndürür.

        Bilinçli olarak: bu metot KENDİ pencereleme mantığını yazmaz;
        sadece `push()`'u çağırır. Böylece offline ile online arasında
        davranış farkı olması imkansız hale gelir (aynı fonksiyon,
        farklı çağıran).
        """
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim != 1:
            raise ValueError(f"1 boyutlu sinyal bekleniyor, gelen şekil: {signal.shape}")

        windows: list[np.ndarray] = []
        for sample in signal:
            window = self.push(float(sample))
            if window is not None:
                windows.append(window)
        return windows

    def reset(self) -> None:
        """Tamponu ve hop sayacını sıfırlar. Yeni bir kayıt/oturuma
        geçerken çağrılmalı (bkz. HighPassFilter.reset ile aynı gerekçe)."""
        self._buffer.clear()
        self._samples_until_next_window = self.config.window_size


# ---------------------------------------------------------------------------
# 3) İkisini birleştiren uçtan uca pipeline
# ---------------------------------------------------------------------------
@dataclass
class PreprocessedWindow:
    """Bir pencere ve onunla ilgili meta veriyi bir arada taşıyan sonuç tipi.

    post_stimulus_only bayrağı, prompttaki "post-stimulus kısmın
    kullanılması" gerekliliğini işaretlemek için var; etiketleme
    katmanı (labeling_protocol.py, sonraki adımda yazılacak) bu bilgiyi
    kullanarak uyarandan ÖNCEKİ pencereleri baseline, SONRAKİ pencereleri
    ilgili uyaran sınıfı olarak işaretleyecek.
    """

    samples: np.ndarray
    window_index: int
    is_post_stimulus: Optional[bool] = None


class PreprocessingPipeline:
    """HighPassFilter + SlidingWindower'ı zincirleyen kullanıcı-dostu arayüz.

    Bu sınıf, `features/*.py` ve `inference.py`'nin doğrudan kullanacağı
    ana giriş noktası olacak. Offline eğitim scripti `process_offline()`,
    canlı `inference.py` ise `process_sample()` çağıracak — ikisi de
    aynı filtre + pencereleme mantığını (yukarıda açıklanan sebeple)
    paylaşıyor.
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        filter_config: Optional[FilterConfig] = None,
        window_config: Optional[WindowConfig] = None,
    ) -> None:
        self.sampling_rate_hz = sampling_rate_hz
        self.filter = HighPassFilter(filter_config, sampling_rate_hz=sampling_rate_hz)
        self.windower = SlidingWindower(window_config)
        self._window_counter = 0

    def process_offline(self, raw_signal: np.ndarray) -> list[PreprocessedWindow]:
        """Ham (filtrelenmemiş) sinyali baştan sona işler: önce zero-phase
        filtre uygulanır, sonra pencerelere bölünür.

        NOT: Filtreleme burada TÜM sinyale bir kerede (process_offline)
        uygulanıyor, çünkü offline modda zero-phase filtrenin avantajından
        (faz kayması yok) faydalanmak istiyoruz. Online modda ise filtre
        örnek-örnek (causal) uygulanacak - bkz. process_sample.
        """
        filtered_signal = self.filter.process_offline(raw_signal)
        raw_windows = self.windower.process_offline(filtered_signal)
        return [
            PreprocessedWindow(samples=w, window_index=i)
            for i, w in enumerate(raw_windows)
        ]

    def process_sample(self, sample: float) -> Optional[PreprocessedWindow]:
        """Canlı akıştan gelen TEK bir örneği işler.

        Sırasıyla: causal filtre -> sliding window tamponuna ekleme.
        Yeni bir pencere tamamlandıysa onu döndürür, aksi halde None.
        Bu metot `inference.py` tarafından her yeni ADC örneği geldiğinde
        çağrılacak.
        """
        filtered_sample = self.filter.push(sample)
        window = self.windower.push(filtered_sample)
        if window is None:
            return None

        result = PreprocessedWindow(samples=window, window_index=self._window_counter)
        self._window_counter += 1
        return result

    def reset(self) -> None:
        """Yeni bir kayıt oturumu / yeni bir bitki başlarken filtre ve
        pencere durumunu sıfırlar."""
        self.filter.reset()
        self.windower.reset()
        self._window_counter = 0


if __name__ == "__main__":
    # Küçük bir kendi-kendini-doğrulama (self-check) demosu:
    # Aynı sentetik sinyali hem offline hem online modda işleyip,
    # üretilen pencere sayısının ve kaba istatistiklerin tutarlı
    # olduğunu gösteriyoruz. Gerçek birim testleri tests/ altında olacak.
    rng = np.random.default_rng(seed=42)
    fs = CONFIG.acquisition.event_sampling_rate_hz
    duration_s = 30
    n_samples = int(fs * duration_s)

    # Yavaş bir baseline drift + yüksek frekanslı gürültü içeren sentetik sinyal
    t = np.arange(n_samples) / fs
    synthetic_signal = 0.5 * np.sin(2 * np.pi * 0.01 * t) + rng.normal(0, 0.05, n_samples)

    pipeline_offline = PreprocessingPipeline(sampling_rate_hz=fs)
    offline_windows = pipeline_offline.process_offline(synthetic_signal)
    print(f"[Offline] {n_samples} örnekten {len(offline_windows)} pencere üretildi.")
    print(f"[Offline] İlk pencere şekli: {offline_windows[0].samples.shape}")

    pipeline_online = PreprocessingPipeline(sampling_rate_hz=fs)
    online_window_count = 0
    for sample in synthetic_signal:
        w = pipeline_online.process_sample(float(sample))
        if w is not None:
            online_window_count += 1
    print(f"[Online]  {n_samples} örnek tek tek beslendi, {online_window_count} pencere üretildi.")