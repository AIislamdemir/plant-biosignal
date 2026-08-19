"""
inference.py
===============
PROJENİN ANA TEKNİK İDDİASI: canlı akan bitki biyoelektrik sinyalini
örnek-örnek okuyup, her tamamlanan pencerede ANINDA bir sınıflandırma
sonucu üreten gerçek zamanlı motor.

Bu dosya, şimdiye kadar yazdığımız HER modülü birbirine bağlıyor:

    ham örnek (ADC'den)
        -> data/preprocessing.py   : PreprocessingPipeline.process_sample()
        -> features/statistical_features.py : extract_statistical_features()
        -> models/classical_models.py       : TrainedModel.predict_labels()
        -> InferenceResult (anlık tahmin + gecikme ölçümü)

Literatürdeki OFFLINE yaklaşımlardan fark:
--------------------------------------------
Najdenovska (2021), Reissig (2021), Buss (2023/2025) dahil tüm çalışmalar
önce TÜM veriyi toplayıp SONRA laboratuvarda ayrı bir analiz aşamasında
sınıflandırıyor. Burada ise pencereleme VE filtreleme, preprocessing.py'de
tanımlı AYNI push() çekirdeğiyle örnek geldikçe çalışıyor - training'de
kullanılan mantığın BİREBİR AYNISI, ayrı bir "online implementasyon"
yazılmadı. Bu, projenin offline/online tutarlılık garantisinin (bkz.
data/preprocessing.py docstring'i) fiilen meyvesini verdiği yer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from config import CONFIG, FilterConfig, RealtimeConfig, WindowConfig
from data.preprocessing import PreprocessingPipeline
from features.statistical_features import extract_statistical_features
from models.classical_models import TrainedModel


# ---------------------------------------------------------------------------
# 1) Tek bir pencere için üretilen sonuç
# ---------------------------------------------------------------------------
@dataclass
class InferenceResult:
    """Bir pencere tamamlandığında üretilen anlık tahmin ve onunla ilgili
    performans bilgisi.

    inference_latency_s: SADECE özellik çıkarımı + model tahmini için
    geçen süre (sinyal toplama/filtreleme süresi dahil değil - o zaten
    örnek geldikçe, pencere tamamlanmadan ÖNCE dağıtık şekilde yapılıyor).
    Bu, "pencere tamamlandıktan sonra sonucu üretmek ne kadar sürdü"
    sorusuna cevap veriyor - gerçek zamanlı sistemin darboğazı burası.

    within_latency_budget: RealtimeConfig.max_inference_latency_ratio'ya
    göre belirlenen bütçenin içinde kalınıp kalınmadığı. False ise,
    sistem "yetişemiyor" demektir - girdi kuyruğu birikmeye başlar.
    """

    label: str
    window_index: int
    inference_latency_s: float
    within_latency_budget: bool


# ---------------------------------------------------------------------------
# 2) Gerçek zamanlı sınıflandırma motoru
# ---------------------------------------------------------------------------
class RealtimeClassifier:
    """Canlı akan örnekleri alıp, her tamamlanan pencerede bir
    InferenceResult üreten durumlu (stateful) sınıf.

    Kullanım (inference.py'nin canlı bir donanım döngüsünde kullanımı):

        classifier = RealtimeClassifier(trained_model, sampling_rate_hz=100.0)
        while True:
            sample = read_from_adc()          # donanımdan tek örnek oku
            result = classifier.process_sample(sample)
            if result is not None:
                display_prediction(result)     # utils/visualization.py (sonraki adım)
    """

    def __init__(
        self,
        trained_model: TrainedModel,
        sampling_rate_hz: float,
        filter_config: Optional[FilterConfig] = None,
        window_config: Optional[WindowConfig] = None,
        realtime_config: Optional[RealtimeConfig] = None,
    ) -> None:
        self.trained_model = trained_model
        self.sampling_rate_hz = sampling_rate_hz
        self.window_config = window_config or CONFIG.window
        self.realtime_config = realtime_config or CONFIG.realtime

        self.pipeline = PreprocessingPipeline(
            sampling_rate_hz=sampling_rate_hz,
            filter_config=filter_config,
            window_config=self.window_config,
        )

        # Bir pencerenin GERÇEK DÜNYA süresi (saniye) - performans
        # bütçesinin dayandığı referans değer.
        self.window_duration_s = self.window_config.window_size / sampling_rate_hz
        self.max_allowed_latency_s = (
            self.window_duration_s * self.realtime_config.max_inference_latency_ratio
        )

    def process_sample(self, sample: float) -> Optional[InferenceResult]:
        """Tek bir ham örneği işler. Pencere henüz tamamlanmadıysa None
        döner (bu NORMAL - sistem sessizce örnek biriktirmeye devam
        ediyor demektir). Pencere tamamlandıysa, özellik çıkarımı +
        model tahmini yapıp bir InferenceResult döner.
        """
        window = self.pipeline.process_sample(sample)
        if window is None:
            return None

        # SADECE bu kısmı (özellik çıkarımı + tahmin) zamanlıyoruz -
        # filtreleme/pencereleme maliyeti zaten örnek geldikçe dağıtık
        # şekilde ödendi, pencere "tamamlandığı an" ekstra bir maliyeti yok.
        start = time.perf_counter()
        feature_vector = extract_statistical_features(window.samples).reshape(1, -1)
        predicted_label = self.trained_model.predict_labels(feature_vector)[0]
        elapsed_s = time.perf_counter() - start

        within_budget = elapsed_s <= self.max_allowed_latency_s
        if not within_budget:
            # Sistemi DURDURMUYORUZ (gerçek zamanlı akışta durmak en kötü
            # seçenek olurdu) - ama bunu görünür kılmak, gecikme birikmeye
            # başladığında fark edilmesi için kritik.
            print(
                f"[Uyarı] Pencere {window.window_index}: gecikme bütçesi aşıldı "
                f"({elapsed_s * 1000:.1f} ms > {self.max_allowed_latency_s * 1000:.1f} ms)."
            )

        return InferenceResult(
            label=predicted_label,
            window_index=window.window_index,
            inference_latency_s=elapsed_s,
            within_latency_budget=within_budget,
        )

    def reset(self) -> None:
        """Yeni bir kayıt oturumu / yeni bir bitkiye geçerken çağrılmalı.
        preprocessing.py'deki reset() ile aynı gerekçe: filtre/pencere
        durumunun önceki oturumdan sızmaması için."""
        self.pipeline.reset()


# ---------------------------------------------------------------------------
# 3) Simülasyon yardımcı fonksiyonu (donanım olmadan test/demo için)
# ---------------------------------------------------------------------------
def simulate_realtime_stream(
    classifier: RealtimeClassifier, signal: np.ndarray
) -> Iterator[InferenceResult]:
    """Elimizde zaten kaydedilmiş bir sinyal varken (örn. test/demo
    amaçlı), onu SANKİ canlı geliyormuş gibi örnek-örnek classifier'a
    besleyen bir jeneratör.

    Bu fonksiyon donanım (gerçek ADC) OLMADAN inference.py'nin davranışını
    doğrulamamızı sağlıyor - `data/labeling_protocol.py`'nin offline
    doğrulama demosuna paralel bir mantık.

    NOT: Burada gerçek zaman gecikmesi (time.sleep) EKLEMİYORUZ - amaç
    "gerçek saatte" çalıştırmak değil, pipeline'ın DOĞRU sırayla ve doğru
    sonuçlarla çalıştığını hızlıca göstermek. Gerçek donanımda örnekler
    zaten kendi doğal hızında (örn. 100 Hz) gelecek.
    """
    for sample in signal:
        result = classifier.process_sample(float(sample))
        if result is not None:
            yield result


if __name__ == "__main__":
    # Uçtan uca demo: sentetik etiketli pencerelerle HIZLICA bir model
    # eğitip (gerçek veri yerine), sonra o modeli canlı bir sinyal akışını
    # sınıflandırmak için kullanıyoruz. Bu, projenin "özgünlük ekseni"
    # iddiasının gerçekten çalıştığının kanıtı: aynı sistem hem offline
    # eğitilebiliyor hem online tahmin üretebiliyor.
    from models.classical_models import train_final_model

    fs = CONFIG.acquisition.event_sampling_rate_hz  # 100 Hz
    rng = np.random.default_rng(0)

    # --- Adım 1: Hızlı bir eğitim veri seti (statistical_features.py demosundaki
    # gibi, ama burada gerçek pencereler üzerinden extract ediyoruz) ---
    def make_labeled_feature(is_touch: bool) -> np.ndarray:
        window = rng.normal(0, 0.02, CONFIG.window.window_size)
        if is_touch:
            window[CONFIG.window.window_size // 2] += 3.0  # keskin spike
        return extract_statistical_features(window)

    X_train = np.stack(
        [make_labeled_feature(is_touch=False) for _ in range(40)]
        + [make_labeled_feature(is_touch=True) for _ in range(40)]
    )
    y_train = np.array(["baseline"] * 40 + ["mechanical_touch"] * 40)
    trained_model = train_final_model("random_forest", X_train, y_train)

    # --- Adım 2: "Canlı" bir sinyal akışı simüle et (30 saniyelik kayıt,
    # ortasında bir dokunma uyaranı) ---
    duration_s = 30
    n_samples = int(duration_s * fs)
    live_signal = rng.normal(0, 0.02, n_samples)
    touch_onset_sample = n_samples // 2
    live_signal[touch_onset_sample] += 3.0  # gerçek bir dokunma anı

    classifier = RealtimeClassifier(trained_model, sampling_rate_hz=fs)

    print(f"Pencere süresi: {classifier.window_duration_s:.2f} s")
    print(f"Gecikme bütçesi: {classifier.max_allowed_latency_s * 1000:.2f} ms\n")
    print("Canlı akış simülasyonu başlıyor...\n")

    for result in simulate_realtime_stream(classifier, live_signal):
        budget_marker = "OK" if result.within_latency_budget else "BÜTÇE AŞILDI"
        print(
            f"  Pencere {result.window_index:2d} -> '{result.label:17s}' "
            f"(gecikme: {result.inference_latency_s * 1000:5.2f} ms) [{budget_marker}]"
        )