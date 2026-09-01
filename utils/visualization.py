"""
utils/visualization.py
=========================
Sinyal grafikleri, özellik önemi ve CANLI GÖSTERGE (real-time monitor).

Bu dosya, prompt'un gerçek zamanlı sistem gereksiniminin doğrudan bir
karşılığı: "inference.py, canlı sinyali okuyup her pencerede bir
sınıflandırma sonucu üretmeli ve bunu GECİKMESİZ ŞEKİLDE GÖRSELLEŞTİRMELİ
(ör. canlı grafik + anlık tahmin etiketi)".

Dürüstlük notu - "canlı" ne demek burada:
--------------------------------------------
Bu kod headless (ekransız) bir konteynerde çalışıyor, bu yüzden GERÇEK
bir ekranda saniye saniye güncellenen bir pencere GÖSTEREMİYORUZ. Bunun
yerine `LiveMonitor` sınıfı, tam olarak GERÇEK bir GUI ortamında
(masaüstü, Jupyter, vb.) `matplotlib.animation.FuncAnimation` ile
kullanılabilecek şekilde tasarlandı: `update_samples()` / `update_result()`
durumu güncelliyor, `render_snapshot()` o anki durumun bir "kare"sini
üretiyor. Gerçek bir ekranda, `render_snapshot()`'ı bir zamanlayıcıyla
(örn. her 200ms'de bir) tekrar tekrar çağırmak GERÇEK canlı görselleştirmeyi
verir - burada demo amaçlı sadece son kareyi PNG olarak kaydediyoruz.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless ortamda ekran olmadan dosyaya çizim için
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from config import StimulusLabel

# Her sınıf için TUTARLI bir renk - eğitim/demo/rapor arasında aynı
# sınıfın hep aynı renkte görünmesi, okunabilirlik için önemli.
_LABEL_COLORS: dict[str, str] = {
    StimulusLabel.BASELINE.value: "#8c8c8c",
    StimulusLabel.MECHANICAL_TOUCH.value: "#d62728",
    StimulusLabel.LIGHT_TRANSITION.value: "#ffbf00",
    StimulusLabel.DROUGHT_STRESS.value: "#8c564b",
    StimulusLabel.CHEMICAL_SALT_STRESS.value: "#2ca02c",
    StimulusLabel.TEMPERATURE_SHOCK.value: "#1f77b4",
}


def _color_for_label(label: str) -> str:
    """Bilinen bir sınıf için sabit rengi, bilinmeyen bir etiket için
    (örn. ileride yeni bir sınıf eklenirse) nötr bir gri döner - grafik
    hiçbir zaman renk eksikliğinden dolayı hata vermemeli."""
    return _LABEL_COLORS.get(label, "#444444")


# ---------------------------------------------------------------------------
# 1) Tek bir pencerenin zaman-domeni grafiği
# ---------------------------------------------------------------------------
def plot_signal_window(
    window: np.ndarray, sampling_rate_hz: float, title: Optional[str] = None
) -> Figure:
    """Bir pencereyi (1024 örnek) zaman ekseninde (saniye) çizer.

    Neden saniye, örnek indeksi değil? Farklı örnekleme hızlarında
    (10 Hz baseline vs 100 Hz event) toplanan pencereleri KARŞILAŞTIRILABİLİR
    kılmak için - örnek indeksi, örnekleme hızından bağımsız, yanıltıcı
    bir eksen olurdu.
    """
    window = np.asarray(window, dtype=np.float64)
    t = np.arange(window.size) / sampling_rate_hz

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, window, color="#1f77b4", linewidth=0.8)
    ax.set_xlabel("Zaman (s)")
    ax.set_ylabel("Genlik")
    ax.set_title(title or "Sinyal penceresi")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2) Özellik önemi grafiği
# ---------------------------------------------------------------------------
def plot_feature_importance(
    feature_names: Sequence[str], importances: Sequence[float], title: Optional[str] = None
) -> Figure:
    """Ağaç-tabanlı modellerin (RandomForest, XGBoost, DecisionTree...)
    `.feature_importances_` çıktısını, en önemliden en az önemliye
    sıralanmış bir yatay çubuk grafik olarak çizer.

    Neden bu grafik önemli: "model %92 doğru" demek yeterli değil -
    "model HANGİ özelliğe dayanarak karar veriyor" sorusu, hem
    modelin sağlığını (örn. sadece 'skewness'e mi bakıyor, yoksa
    dengeli mi kullanıyor) hem de bilimsel yorumlanabilirliği
    (örn. 'kurtosis'in dokunma sınıfı için baskın çıkması, keskin
    spike hipotezini destekler) doğrudan gösteriyor.
    """
    order = np.argsort(importances)  # artan sırada, yatay çubukta en önemli üstte görünsün diye
    sorted_names = np.array(feature_names)[order]
    sorted_importances = np.array(importances)[order]

    fig, ax = plt.subplots(figsize=(6, max(2, 0.4 * len(feature_names))))
    ax.barh(sorted_names, sorted_importances, color="#1f77b4")
    ax.set_xlabel("Önem skoru")
    ax.set_title(title or "Özellik önemi")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3) Karışıklık matrisi ısı haritası
# ---------------------------------------------------------------------------
def plot_confusion_matrix_heatmap(cm_df: pd.DataFrame, title: Optional[str] = None) -> Figure:
    """utils/metrics.py'deki confusion_matrix_report()'un ürettiği
    DataFrame'i ısı haritası olarak görselleştirir - her hücrede
    hem renk (yoğunluk) hem de sayısal değer birlikte gösteriliyor.
    """
    fig, ax = plt.subplots(figsize=(1.2 * len(cm_df.columns) + 2, 1.2 * len(cm_df.index) + 1))
    im = ax.imshow(cm_df.values, cmap="Blues")

    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)

    # Her hücreye sayısal değeri yaz - koyu hücrelerde beyaz, açık
    # hücrelerde siyah metin kullanarak okunabilirliği garanti et.
    max_val = cm_df.values.max() if cm_df.values.size else 0
    for i in range(len(cm_df.index)):
        for j in range(len(cm_df.columns)):
            value = cm_df.values[i, j]
            text_color = "white" if value > max_val / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color)

    ax.set_title(title or "Karışıklık matrisi")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4) Canlı gösterge (real-time monitor)
# ---------------------------------------------------------------------------
class LiveMonitor:
    """Gerçek zamanlı akıştaki ham sinyali ve anlık tahminleri biriktirip,
    her an "şu ana kadar ne oldu" görünümünü bir figürde özetleyen sınıf.

    inference.py'deki RealtimeClassifier ile birlikte kullanım:

        monitor = LiveMonitor(sampling_rate_hz=100.0, display_seconds=15)
        while True:
            sample = source.read_sample()
            monitor.update_samples([sample])
            result = classifier.process_sample(sample)
            if result is not None:
                monitor.update_result(result)
            if <bir GUI event loop'undaysanız>:
                monitor.render_snapshot()   # ekranı güncelle
    """

    def __init__(
        self, sampling_rate_hz: float, display_seconds: float = 15.0, max_predictions: int = 20
    ) -> None:
        self.sampling_rate_hz = sampling_rate_hz
        self.display_seconds = display_seconds

        max_raw_samples = int(display_seconds * sampling_rate_hz)
        # deque(maxlen=...) - tampon dolduğunda EN ESKİ örnek otomatik
        # atılır; bu, sınırsız büyüyen bir bellek yerine sabit boyutlu
        # bir "kayan pencere" görünümü sağlıyor (preprocessing.py'deki
        # SlidingWindower'ın tamponuyla aynı desen, farklı amaç).
        self._raw_samples: deque[float] = deque(maxlen=max_raw_samples)
        self._predictions: deque[tuple[int, str]] = deque(maxlen=max_predictions)
        self._latest_label: Optional[str] = None

    def update_samples(self, samples: Sequence[float]) -> None:
        """Yeni gelen ham örnekleri tampona ekler (üst sınır aşılırsa
        en eskiler otomatik düşer)."""
        self._raw_samples.extend(samples)

    def update_result(self, result) -> None:  # inference.InferenceResult (döngüsel import'tan kaçınmak için type hint'siz)
        """Yeni bir tahmin sonucu geldiğinde geçmişe ekler ve
        'şu anki' etiketi günceller (grafikte büyük başlıkla gösterilecek)."""
        self._predictions.append((result.window_index, result.label))
        self._latest_label = result.label

    def render_snapshot(self, save_path: Optional[str] = None) -> Figure:
        """Şu ana kadar biriken durumun bir "kare"sini üretir: üstte ham
        sinyal, altta son tahminlerin renkli zaman çizelgesi, başlıkta
        en güncel tahmin.

        save_path verilirse PNG olarak diske kaydeder (headless ortamda
        canlı görüntüleme yerine kullanılan yöntem).
        """
        fig, (ax_signal, ax_timeline) = plt.subplots(
            2, 1, figsize=(9, 4), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Üst panel: ham sinyal
        if len(self._raw_samples) > 0:
            samples = np.array(self._raw_samples)
            t = np.arange(len(samples)) / self.sampling_rate_hz
            ax_signal.plot(t, samples, color="#1f77b4", linewidth=0.6)
        ax_signal.set_ylabel("Genlik")
        ax_signal.set_xlabel("Zaman (son pencere, s)")
        ax_signal.grid(alpha=0.3)

        title_label = self._latest_label or "bekleniyor..."
        title_color = _color_for_label(self._latest_label) if self._latest_label else "#444444"
        ax_signal.set_title(f"Anlık tahmin: {title_label}", color=title_color, fontsize=13, fontweight="bold")

        # Alt panel: son N tahminin renkli zaman çizelgesi
        if len(self._predictions) > 0:
            for i, (window_idx, label) in enumerate(self._predictions):
                ax_timeline.barh(0, 1, left=i, color=_color_for_label(label), edgecolor="white")
            ax_timeline.set_xlim(0, len(self._predictions))
        ax_timeline.set_yticks([])
        ax_timeline.set_xlabel("Pencere sırası (en yeni sağda)")

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=120)
        return fig


if __name__ == "__main__":
    # Uçtan uca demo: inference.py'deki senaryonun aynısını çalıştırıp,
    # sonucu bir PNG "kare"si olarak kaydediyoruz.
    from features.statistical_features import extract_statistical_features
    from inference import RealtimeClassifier
    from models.classical_models import train_final_model

    fs = 100.0
    rng = np.random.default_rng(0)

    def make_labeled_feature(is_touch: bool) -> np.ndarray:
        window = rng.normal(0, 0.02, 1024)
        if is_touch:
            window[512] += 3.0
        return extract_statistical_features(window)

    X_train = np.stack(
        [make_labeled_feature(False) for _ in range(40)]
        + [make_labeled_feature(True) for _ in range(40)]
    )
    y_train = np.array(["baseline"] * 40 + ["mechanical_touch"] * 40)
    trained_model = train_final_model("random_forest", X_train, y_train)
    classifier = RealtimeClassifier(trained_model, sampling_rate_hz=fs)

    monitor = LiveMonitor(sampling_rate_hz=fs, display_seconds=30.0)

    duration_s = 30
    n_samples = int(duration_s * fs)
    live_signal = rng.normal(0, 0.02, n_samples)
    live_signal[n_samples // 2] += 3.0

    for sample in live_signal:
        monitor.update_samples([float(sample)])
        result = classifier.process_sample(float(sample))
        if result is not None:
            monitor.update_result(result)
            print(f"  Pencere {result.window_index} -> '{result.label}'")

    import os

    os.makedirs("/tmp/viz_demo", exist_ok=True)
    monitor.render_snapshot(save_path="/tmp/viz_demo/live_monitor_snapshot.png")
    print("\nAnlık gösterge kare görüntüsü kaydedildi: /tmp/viz_demo/live_monitor_snapshot.png")