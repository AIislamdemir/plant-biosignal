"""
data/labeling_protocol.py
===========================
Kontrollü deney protokolünü (Trial) tanımlayan ve preprocessing.py'nin
ürettiği pencerelere OTOMATİK, ZAMAN-TABANLI etiket atayan modül.

Bu modül prompt'taki iki kısıtı doğrudan koda döker:
---------------------------------------------------------
1) "Ground truth netliği": Her etiket, deney sırasında BİLİNÇLİ olarak
   uygulanan bir uyarana (stimulus_onset_time_s) dayanmalı. Sinyalden
   "yorumlanan" ikincil etiketler (ör. sunum dili) ayrı tutulmalı ve
   bilimsel iddia olarak sunulmamalı.
   -> Burada etiketleme SADECE deneyci tarafından kaydedilen uyaran
   zamanına göre yapılıyor; sinyalin şekline bakarak "bu bir stres
   tepkisine benziyor" gibi bir çıkarım YAPILMIYOR. Ayrıca uyaran
   geçiş anının etrafında bilinçli bir "guard band" (belirsizlik
   bandı) bırakılıyor - o bölgeye denk gelen pencereler HİÇBİR
   sınıfa atanmıyor (None), veri setine dahil edilmiyor. Böyle bir
   guard band olmazsa, tam geçiş anına denk gelen bir pencere hem
   baseline hem uyaran karakteristiği taşır ve modele gürültülü/
   çelişkili bir örnek olarak sızar.

2) "Bitki-bazlı cross-validation": Aynı bitkiden gelen örnekler hem
   train hem test setinde olmamalı (data leakage riski).
   -> Trial ve LabeledWindow, plant_id'yi ZORUNLU bir alan olarak
   taşıyor ki ileride models/ katmanında sklearn'ün GroupKFold'u
   (group=plant_id) ile bölünebilsin. Bu dosyada CV yapmıyoruz, ama
   CV'nin ihtiyaç duyacağı bilgiyi burada kaybetmeden taşıyoruz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from config import CONFIG, FilterConfig, StimulusLabel, WindowConfig
from data.preprocessing import PreprocessingPipeline


# ---------------------------------------------------------------------------
# 1) Protokol sabitleri (Veri Toplama Protokolü bölümünden)
# ---------------------------------------------------------------------------
class ProtocolMinimums:
    """Prompt'un "Veri Toplama Protokolü" bölümünde verilen somut asgari
    değerler. Sabit tutuluyor (config.py'ye değil buraya konuldu çünkü
    bunlar sinyal işleme parametresi değil, DENEY TASARIMI kısıtları)."""

    MIN_BASELINE_DURATION_S: float = 120.0        # tekrar başına en az 2 dk baseline
    MIN_POST_STIMULUS_DURATION_S: float = 120.0    # tekrar başına en az 2 dk uyaran-sonrası
    MIN_REPEATS_PER_CLASS: int = 20                # sınıf başına en az 20 tekrar
    MIN_DISTINCT_PLANTS: int = 3                   # bitki-bazlı CV için zorunlu alt sınır


# ---------------------------------------------------------------------------
# 2) Tek bir deney tekrarını (trial) temsil eden veri yapısı
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Trial:
    """Tek bir kontrollü deney tekrarını (örn. "bitki #3, dokunma uyaranı,
    3. tekrar") temsil eder.

    Zaman alanlarının hepsi TRIAL'IN KENDİ KAYIT BAŞLANGICINA GÖRE
    (relative), saniye cinsinden. Yani stimulus_onset_time_s=120.0 demek
    "bu trial'ın kaydı başladıktan 120 saniye sonra uyaran uygulandı"
    demektir - global bir saate göre değil.

    guard_band_s: Uyaran anının HEMEN öncesi ve sonrasındaki belirsizlik
    penceresi. Örn. guard_band_s=2.0 ise, onset_time'dan 2 saniye önce
    başlayıp 2 saniye sonra biten aralığa denk gelen pencereler
    etiketlenmeden atlanır (bkz. modül docstring'i, madde 1).
    """

    trial_id: str
    plant_id: str
    label: StimulusLabel                    # bu trial'da uygulanan uyaran sınıfı
    sampling_rate_hz: float
    stimulus_onset_time_s: float
    baseline_duration_s: float = ProtocolMinimums.MIN_BASELINE_DURATION_S
    post_stimulus_duration_s: float = ProtocolMinimums.MIN_POST_STIMULUS_DURATION_S
    guard_band_s: float = 1.0
    # Bilimsel iddia olarak KULLANILMAYAN, sadece insan-okunur bağlam
    # notu (ör. "sunumda kullanılacak dil"). Eğitim/etiketleme
    # mantığının hiçbir yerinde bu alana bakılmaz - kasıtlı olarak.
    interpretive_note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.label == StimulusLabel.BASELINE:
            raise ValueError(
                "Trial.label BASELINE olamaz; baseline etiketi otomatik olarak "
                "uyaran-öncesi pencerelere atanır, bir trial'ın 'kendi' uyaran "
                "sınıfı olmalı (örn. MECHANICAL_TOUCH)."
            )
        if self.baseline_duration_s < ProtocolMinimums.MIN_BASELINE_DURATION_S:
            raise ValueError(
                f"baseline_duration_s ({self.baseline_duration_s}s) protokol asgari "
                f"değerinin ({ProtocolMinimums.MIN_BASELINE_DURATION_S}s) altında."
            )
        if self.post_stimulus_duration_s < ProtocolMinimums.MIN_POST_STIMULUS_DURATION_S:
            raise ValueError(
                f"post_stimulus_duration_s ({self.post_stimulus_duration_s}s) protokol "
                f"asgari değerinin ({ProtocolMinimums.MIN_POST_STIMULUS_DURATION_S}s) altında."
            )
        if self.guard_band_s < 0:
            raise ValueError("guard_band_s negatif olamaz.")
        if self.stimulus_onset_time_s < self.baseline_duration_s:
            raise ValueError(
                f"stimulus_onset_time_s ({self.stimulus_onset_time_s}s), baseline_duration_s "
                f"({self.baseline_duration_s}s)'den küçük olamaz - kayıt, uyarandan önce "
                f"en az baseline_duration_s kadar sürmüş olmalı."
            )


# ---------------------------------------------------------------------------
# 3) Etiketlenmiş pencere sonuç tipi
# ---------------------------------------------------------------------------
@dataclass
class LabeledWindow:
    """build_labeled_dataset()'in ürettiği, eğitime hazır tek bir örnek.

    plant_id burada BİLİNÇLİ olarak taşınıyor (preprocessing.py'deki
    PreprocessedWindow'da yok) - features/ ve models/ katmanları bu
    alanı sklearn GroupKFold(groups=plant_id) için kullanacak.
    """

    samples: np.ndarray
    label: StimulusLabel
    plant_id: str
    trial_id: str
    window_index: int


# ---------------------------------------------------------------------------
# 4) Çekirdek etiketleme mantığı: pencere zamanı -> etiket
# ---------------------------------------------------------------------------
def _window_time_span(
    window_index: int, window_config: WindowConfig, sampling_rate_hz: float
) -> tuple[float, float]:
    """Bir pencerenin trial-lokal zaman ekseninde [başlangıç, bitiş)
    aralığını (saniye) hesaplar.

    SlidingWindower.push() mantığıyla BİREBİR tutarlı olmalı: window_index
    (0-tabanlı) pencerenin bitiş örneği = window_size + window_index * hop_size,
    başlangıç örneği = bitiş - window_size = window_index * hop_size.
    (preprocessing.py'deki SlidingWindower.push() implementasyonuna bakınız.)
    """
    start_sample = window_index * window_config.hop_size
    end_sample = start_sample + window_config.window_size
    return start_sample / sampling_rate_hz, end_sample / sampling_rate_hz


def label_window(
    trial: Trial, window_index: int, window_config: Optional[WindowConfig] = None
) -> Optional[StimulusLabel]:
    """Bir pencerenin trial içindeki zaman konumuna göre etiketini belirler.

    Dönüş değeri None ise, pencere ya:
      (a) trial'ın tanımlı kayıt aralığının (baseline_start .. post_end) dışında, ya da
      (b) guard band içinde (uyaran geçişine çok yakın, belirsiz)
    demektir - HER İKİ DURUMDA DA bu pencere eğitim setine DAHİL EDİLMEMELİ.

    Bu fonksiyon SADECE zamana bakar, sinyalin kendisine bakmaz - bu,
    "ground truth netliği" kısıtının doğrudan uygulanmasıdır.
    """
    window_config = window_config or CONFIG.window
    start_t, end_t = _window_time_span(window_index, window_config, trial.sampling_rate_hz)

    baseline_start = trial.stimulus_onset_time_s - trial.baseline_duration_s
    guard_start = trial.stimulus_onset_time_s - trial.guard_band_s
    guard_end = trial.stimulus_onset_time_s + trial.guard_band_s
    post_end = trial.stimulus_onset_time_s + trial.post_stimulus_duration_s

    # (a) Trial'ın tanımlı kayıt aralığının tamamen dışında kalan pencere
    if start_t < baseline_start or end_t > post_end:
        return None

    # Pencere TAMAMEN guard band'den önce bitiyor -> baseline
    if end_t <= guard_start:
        return StimulusLabel.BASELINE

    # Pencere TAMAMEN guard band'den sonra başlıyor -> trial'ın uyaran sınıfı
    if start_t >= guard_end:
        return trial.label

    # (b) Geriye kalan tek durum: pencere guard band'i bir şekilde
    # kesişiyor (uyaran geçişine çok yakın) -> belirsiz, ele
    return None


# ---------------------------------------------------------------------------
# 5) Tüm trial'ları işleyip etiketlenmiş veri setini üreten üst fonksiyon
# ---------------------------------------------------------------------------
def build_labeled_dataset(
    trials: Sequence[Trial],
    raw_signals: Mapping[str, np.ndarray],
    window_config: Optional[WindowConfig] = None,
    filter_config: Optional[FilterConfig] = None,
) -> list[LabeledWindow]:
    """Trial listesini ve her trial'a ait ham sinyali alır; her trial için
    preprocessing.py'deki OFFLINE pipeline'ı çalıştırır (zero-phase filtre
    + pencereleme), sonra her pencereyi label_window() ile etiketler.

    Guard band'e veya trial kapsamı dışına denk gelen pencereler (None
    dönenler) SESSİZCE atlanır - bu bilinçli bir tasarım kararı: belirsiz
    örnekleri "en yakın sınıfa yuvarlamak" yerine veri setinden çıkarmak,
    prompttaki "ground truth netliği" kısıtına daha sadık.

    Neden offline pipeline (process_offline) kullanılıyor, online değil?
    ------------------------------------------------------------------
    Bu fonksiyon, ELİMİZDE ZATEN TAMAMLANMIŞ kayıtlar (trials) olduğu
    eğitim-veri-hazırlama aşaması için. Gerçek zamanlı akış inference.py'de
    ayrı olarak PreprocessingPipeline.process_sample() ile ele alınacak;
    orada etiketleme değil, TAHMİN üretimi yapılacak.
    """
    window_config = window_config or CONFIG.window
    labeled_windows: list[LabeledWindow] = []

    for trial in trials:
        if trial.trial_id not in raw_signals:
            raise KeyError(f"'{trial.trial_id}' için ham sinyal raw_signals içinde bulunamadı.")

        raw_signal = raw_signals[trial.trial_id]
        pipeline = PreprocessingPipeline(
            sampling_rate_hz=trial.sampling_rate_hz,
            filter_config=filter_config,
            window_config=window_config,
        )
        preprocessed_windows = pipeline.process_offline(raw_signal)

        for pw in preprocessed_windows:
            label = label_window(trial, pw.window_index, window_config)
            if label is None:
                continue  # kapsam dışı veya guard band -> veri setine girmiyor

            # preprocessing.py'de tanımlı ama boş bırakılmış alanı burada dolduruyoruz
            pw.is_post_stimulus = label != StimulusLabel.BASELINE

            labeled_windows.append(
                LabeledWindow(
                    samples=pw.samples,
                    label=label,
                    plant_id=trial.plant_id,
                    trial_id=trial.trial_id,
                    window_index=pw.window_index,
                )
            )

    return labeled_windows


# ---------------------------------------------------------------------------
# 6) Veri seti protokol uyum kontrolü
# ---------------------------------------------------------------------------
def validate_dataset_protocol(trials: Sequence[Trial]) -> list[str]:
    """Trial listesinin prompt'taki asgari veri toplama gereksinimlerini
    karşılayıp karşılamadığını kontrol eder ve İHLALLERİ metin olarak
    döndürür (raise etmez - eğitim/keşif aşamasında eksik veriyle de
    çalışabilmek isteyebilirsin, ama eksikliğin FARKINDA olmalısın).

    Kontrol edilenler:
      - Sınıf başına tekrar sayısı >= MIN_REPEATS_PER_CLASS
      - Toplam farklı bitki sayısı >= MIN_DISTINCT_PLANTS
        (bitki-bazlı cross-validation'ın anlamlı olması için şart)
    """
    warnings: list[str] = []

    repeats_per_label: dict[StimulusLabel, int] = {}
    for trial in trials:
        repeats_per_label[trial.label] = repeats_per_label.get(trial.label, 0) + 1

    for label, count in repeats_per_label.items():
        if count < ProtocolMinimums.MIN_REPEATS_PER_CLASS:
            warnings.append(
                f"'{label.value}' sınıfı için {count} tekrar var, "
                f"protokol asgari değeri {ProtocolMinimums.MIN_REPEATS_PER_CLASS}."
            )

    distinct_plants = {trial.plant_id for trial in trials}
    if len(distinct_plants) < ProtocolMinimums.MIN_DISTINCT_PLANTS:
        warnings.append(
            f"Toplam {len(distinct_plants)} farklı bitki var, "
            f"protokol asgari değeri {ProtocolMinimums.MIN_DISTINCT_PLANTS}. "
            f"Bitki-bazlı cross-validation güvenilir olmayabilir."
        )

    return warnings


if __name__ == "__main__":
    # Küçük bir uçtan-uca demo: sentetik trial'lar oluşturup etiketleme +
    # protokol kontrolünün beklendiği gibi çalıştığını gösteriyoruz.
    rng = np.random.default_rng(7)
    fs = CONFIG.acquisition.event_sampling_rate_hz  # 100 Hz

    demo_trials = []
    demo_signals: dict[str, np.ndarray] = {}
    for plant_idx in range(2):  # bilinçli olarak protokol asgarisinin (3) altında
        for rep in range(2):    # bilinçli olarak protokol asgarisinin (20) altında
            trial_id = f"plant{plant_idx}_touch_rep{rep}"
            onset = 120.0
            total_duration_s = onset + 120.0
            n_samples = int(total_duration_s * fs)
            signal = rng.normal(0, 0.05, n_samples)
            # Uyaran sonrası kısma yapay bir genlik artışı ekle (demo amaçlı)
            onset_sample = int(onset * fs)
            signal[onset_sample:] += rng.normal(0, 0.3, n_samples - onset_sample)

            demo_signals[trial_id] = signal
            demo_trials.append(
                Trial(
                    trial_id=trial_id,
                    plant_id=f"plant_{plant_idx}",
                    label=StimulusLabel.MECHANICAL_TOUCH,
                    sampling_rate_hz=fs,
                    stimulus_onset_time_s=onset,
                    guard_band_s=1.0,
                )
            )

    dataset = build_labeled_dataset(demo_trials, demo_signals)
    label_counts: dict[str, int] = {}
    for lw in dataset:
        label_counts[lw.label.value] = label_counts.get(lw.label.value, 0) + 1

    print(f"Toplam etiketlenmiş pencere: {len(dataset)}")
    print(f"Sınıf dağılımı: {label_counts}")
    print(f"Örnek plant_id'ler: {sorted({lw.plant_id for lw in dataset})}")

    print("\nProtokol uyum kontrolü:")
    for warning in validate_dataset_protocol(demo_trials):
        print(f"  [UYARI] {warning}")