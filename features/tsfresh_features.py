"""
features/tsfresh_features.py
===============================
tsfresh tabanlı, genişletilmiş zaman serisi özellik çıkarımı (Buss ve
ark., 2025/2026 yaklaşımı).

KRİTİK MİMARİ KARAR - bu modül BİLEREK inference.py'ye BAĞLANMIYOR:
------------------------------------------------------------------------
statistical_features.py (6 özellik, mikrosaniyeler) ve mfcc_features.py
(13 katsayı, milisaniyeler) hem offline hem online modda kullanılabilecek
kadar UCUZ. tsfresh ise (varsayılan ayarlarla bile) yüzlerce özellik
hesaplayabiliyor ve tek bir pencere için saniyeler mertebesinde sürebiliyor
- bu, projenin "pencere başına sınıflandırma süresi, pencere uzunluğundan
kısa olmalı" performans hedefini (bkz. config.py RealtimeConfig) doğrudan
ihlal eder.

Bu yüzden tsfresh_features.py, SADECE offline keşif/karşılaştırma amaçlı:
"6 basit istatistiksel özellik yerine yüzlerce özellik kullansaydık,
offline doğruluk NE KADAR artardı?" sorusuna cevap arıyoruz - eğer cevap
"çok az" ise, gerçek zamanlı sistemde ucuz özellik setini kullanmaya
devam etmenin GEREKÇESİ güçleniyor. Eğer cevap "çok fazla" ise, bu bir
mühendislik değiş-tokuşu (trade-off) tartışmasını başlatıyor: doğruluk
kazancı, gerçek zamanlı performans kaybına değer mi?

Bu ayrım, projenin özgünlük ekseninin (gerçek zamanlı çalışma) bilinçli
bir mühendislik kısıtı olduğunu, kazara bir eksiklik olmadığını gösteriyor.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from data.labeling_protocol import LabeledWindow


def _windows_to_tsfresh_long_format(labeled_windows: Sequence[LabeledWindow]) -> pd.DataFrame:
    """tsfresh, "long format" adı verilen özel bir DataFrame yapısı
    bekler: her satır TEK bir (pencere_id, zaman_adımı, değer) üçlüsü.

    Örnek: 3 pencere, her biri 1024 örnek olsun. Bu fonksiyon,
    3 * 1024 = 3072 satırlık bir DataFrame üretir - her pencere kendi
    `id` değeriyle işaretlenir, tsfresh bu id'ye göre pencereleri
    birbirinden AYIRARAK özellik çıkarır (yani pencereler arası veri
    karışması olmaz - her pencerenin özellik vektörü sadece kendi
    örneklerinden hesaplanır).

    Neden `id` olarak listedeki SIRA İNDEKSİNİ kullanıyoruz, trial_id
    değil? Çünkü aynı trial_id'den birden fazla pencere üretilebiliyor
    (preprocessing.py'deki örtüşmeli pencereleme nedeniyle) - id'nin
    HER PENCERE İÇİN BENZERSİZ olması gerekiyor, trial_id bu benzersizliği
    garanti etmez.
    """
    frames = []
    for window_position, lw in enumerate(labeled_windows):
        n = len(lw.samples)
        frames.append(
            pd.DataFrame(
                {
                    "id": window_position,
                    "time": np.arange(n),
                    "value": lw.samples,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def extract_tsfresh_feature_matrix(
    labeled_windows: Sequence[LabeledWindow],
    use_efficient_parameters: bool = False,
    n_jobs: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """tsfresh ile TÜM pencerelerden genişletilmiş özellik matrisini çıkarır.

    Parametreler
    ------------
    use_efficient_parameters: False ise `MinimalFCParameters` (pencere
        başına ~10 hızlı özellik: mean, length, standard_deviation vb.)
        kullanılır - keşif amaçlı hızlı bir ilk bakış için yeterli. True
        ise `EfficientFCParameters` (pencere başına yüz+ özellik, çok
        daha yavaş) kullanılır - "gerçekten daha fazla özellik yardımcı
        oluyor mu?" sorusunu CİDDİ şekilde test etmek istediğinde.
        VARSAYILAN OLARAK False: bu fonksiyonu ilk kez çalıştıran birinin
        beklemediği şekilde dakikalarca beklemesini istemiyoruz.
    n_jobs: tsfresh'in paralel işlemci sayısı. 0 = paralellik kapalı
        (tek çekirdek) - küçük veri setlerinde paralellik başlatma
        maliyeti (overhead), kazancından fazla olabiliyor; büyük veri
        setlerinde bu değeri artırman önerilir.

    Dönüş değeri: (X, y, groups, feature_names) - statistical_features.py
    ve mfcc_features.py'deki build_feature_matrix() fonksiyonlarıyla AYNI
    (X, y, groups) sözleşmesine ek olarak, tsfresh'in özellik isimlerinin
    (örn. "value__mean", "value__standard_deviation") DEĞİŞKEN olabilmesi
    nedeniyle feature_names de döndürülüyor - diğer iki modülde isimler
    SABİT olduğu için orada ayrıca döndürmemize gerek yoktu.
    """
    if len(labeled_windows) == 0:
        raise ValueError("Boş LabeledWindow listesinden özellik matrisi üretilemez.")

    long_df = _windows_to_tsfresh_long_format(labeled_windows)

    fc_parameters = EfficientFCParameters() if use_efficient_parameters else MinimalFCParameters()

    features_df = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=fc_parameters,
        n_jobs=n_jobs,
        disable_progressbar=True,
    )

    # tsfresh bazı özelliklerde (örn. sabit sinyalde tanımsız istatistikler)
    # NaN/inf üretebilir - kendi statistical_features.py'mizdeki NaN
    # korumasının tsfresh eşdeğeri: impute() bu değerleri median/0 gibi
    # güvenli varsayılanlarla dolduruyor (tsfresh'in resmi önerisi).
    impute(features_df)

    # extract_features()'ın çıktısı, `id` değerine göre SIRALANMIŞ olabilir
    # (orijinal liste sırasıyla aynı olmayabilir) - bu yüzden orijinal
    # labeled_windows sırasına göre YENİDEN hizalıyoruz, aksi halde y/groups
    # ile X arasında sessiz bir sıra kayması (data corruption) oluşabilir.
    features_df = features_df.reindex(range(len(labeled_windows)))

    X = features_df.values
    y = np.array([lw.label.value for lw in labeled_windows])
    groups = np.array([lw.plant_id for lw in labeled_windows])
    feature_names = list(features_df.columns)

    return X, y, groups, feature_names


if __name__ == "__main__":
    # Demo: statistical_features.py (6 özellik) ile tsfresh (Minimal, ~10
    # özellik) arasında offline doğruluk karşılaştırması yaparak, "daha
    # fazla özellik gerçekten yardımcı oluyor mu" sorusuna somut bir
    # cevap üretiyoruz.
    from config import StimulusLabel
    from features.statistical_features import build_feature_matrix
    from models.classical_models import evaluate_models_grouped_cv, summarize_cv_results

    rng = np.random.default_rng(0)
    n = 1024
    windows: list[LabeledWindow] = []

    for plant_idx in range(6):
        plant_id = f"plant_{plant_idx}"
        for _ in range(10):
            sig = rng.normal(0, 0.05, n)  # baseline: düşük varyans
            windows.append(
                LabeledWindow(sig, StimulusLabel.BASELINE, plant_id, "demo_trial", 0)
            )
        for _ in range(10):
            sig = rng.normal(0, 0.05, n)
            sig[n // 2] += 2.0  # touch: keskin spike
            windows.append(
                LabeledWindow(sig, StimulusLabel.MECHANICAL_TOUCH, plant_id, "demo_trial", 0)
            )

    print(f"Toplam {len(windows)} pencere ile karşılaştırma yapılıyor...\n")

    print("--- statistical_features.py (6 özellik) ---")
    X_stat, y_stat, groups_stat = build_feature_matrix(windows)
    results_stat = evaluate_models_grouped_cv(
        X_stat, y_stat, groups_stat, model_names=["decision_tree", "random_forest"], n_splits=5
    )
    print(summarize_cv_results(results_stat))

    print("\n--- tsfresh (MinimalFCParameters, hızlı keşif seti) ---")
    X_ts, y_ts, groups_ts, feature_names = extract_tsfresh_feature_matrix(windows)
    print(f"tsfresh özellik sayısı: {len(feature_names)} -> {feature_names}")
    results_ts = evaluate_models_grouped_cv(
        X_ts, y_ts, groups_ts, model_names=["decision_tree", "random_forest"], n_splits=5
    )
    print(summarize_cv_results(results_ts))