"""
features/statistical_features.py
===================================
Bir sinyal penceresinden (1024 örneklik numpy array) klasik istatistiksel
özellik vektörü çıkaran modül.

Bu dosya, pipeline'da preprocessing/labeling katmanı ile models katmanı
ARASINDAKİ köprü: `data/labeling_protocol.py`'nin ürettiği `LabeledWindow`
listesini alıp, `models/classical_models.py`'nin (RF, SVC, XGBoost, KNN...)
doğrudan `.fit(X, y)` ile eğitebileceği sayısal bir matrise (X) çeviriyor.

Neden bu 6 özellik? (prompt'taki literatür referansı)
--------------------------------------------------------
Bunlar "klasik istatistiksel özellikler" olarak literatürde en sık
kullanılan, hesaplama maliyeti en düşük, yorumlanabilirliği en yüksek
özellik setidir. Zaman serisinin şeklini kabaca özetlerler:
    - mean, min, max      -> sinyalin genlik seviyesi ve aralığı
    - variance             -> sinyalin ne kadar "hareketli" olduğu
    - skewness (çarpıklık) -> dağılımın simetrik olup olmadığı (ani, tek
                               yönlü bir aksiyon potansiyeli sinyali
                               genelde çarpık bir dağılım üretir)
    - kurtosis (basıklık)  -> dağılımın "sivriliği" (keskin, ani spike'lar
                               yüksek kurtosis üretir - dokunma/yaralanma
                               tepkisi için özellikle ayırt edici olması
                               beklenir)

Bu modül BİLEREK sadece bu 6 özellikle sınırlı tutuldu (prompt'un
architecture bölümünde tsfresh_features.py ve mfcc_features.py AYRI
dosyalar olarak planlanmış) - yani "daha kapsamlı özellik seti"
sorumluluğu bilerek başka modüllere bırakıldı, burada tekrarlanmadı.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import skew as _scipy_skew

from data.labeling_protocol import LabeledWindow

# ---------------------------------------------------------------------------
# Özellik isimleri - SABİT SIRA
# ---------------------------------------------------------------------------
# Bu tuple, extract_statistical_features()'ın döndürdüğü array'deki
# sütunların isim-sırasını tanımlar. Neden bu kadar önem veriyorum?
# Çünkü bir ML pipeline'ında "özellik sırası" sessiz bir sözleşmedir:
# eğitimde 3. sütun "skewness" ise, inference.py'de de 3. sütun
# "skewness" OLMAK ZORUNDADIR - aksi halde model, tamamen anlamsız bir
# girdiyle çalışır ama HİÇBİR HATA VERMEZ (shape uyumlu olduğu için).
# Bu isim listesini burada TEK YERDE tanımlayıp her yerde buradan
# import ederek bu riski ortadan kaldırıyoruz.
STATISTICAL_FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "variance",
    "min",
    "max",
    "skewness",
    "kurtosis",
)


def extract_statistical_features(window: np.ndarray) -> np.ndarray:
    """Tek bir penceyeden (1D numpy array) STATISTICAL_FEATURE_NAMES
    sırasıyla 6 elemanlı bir özellik vektörü çıkarır.

    Kenar durumu - sıfır varyanslı (sabit) sinyal:
    -------------------------------------------------
    Eğer pencerenin tamamı sabit bir değerse (örn. sensör kopmuşsa ya da
    gerçekten hiç değişim yoksa), skewness ve kurtosis'in matematiksel
    tanımı 0/0 belirsizliğine düşer. scipy bu durumda NaN döndürür
    (bunu bilerek test ettim). Eğitim verisine NaN sızması sklearn/xgboost
    gibi kütüphanelerde ya sessizce yanlış sonuca ya da anlaşılması güç
    bir hataya yol açar. Bu yüzden varyans sıfıra çok yakınsa (pratik bir
    epsilon ile), skewness ve kurtosis'i açıkça 0.0 olarak tanımlıyoruz -
    bu matematiksel olarak da savunulabilir: "şekli olmayan" bir sinyalin
    ne çarpıklığı ne basıklığı vardır, nötr kabul edilir.
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 1:
        raise ValueError(f"1 boyutlu pencere bekleniyor, gelen şekil: {window.shape}")
    if window.size == 0:
        raise ValueError("Boş pencereden özellik çıkarılamaz.")

    mean = float(np.mean(window))
    variance = float(np.var(window, ddof=0))
    minimum = float(np.min(window))
    maximum = float(np.max(window))

    # ddof=0 (popülasyon varyansı) kullanıyoruz çünkü pencere boyutu
    # sabit (1024) ve elimizdeki örnekler "tüm popülasyon" gibi ele
    # alınıyor - örnek/popülasyon farkı burada pratik bir etki yaratmaz,
    # ama tutarlılık için açıkça belirtmek önemli.

    variance_epsilon = 1e-12
    if variance < variance_epsilon:
        skewness = 0.0
        kurt = 0.0
    else:
        # bias=True: popülasyon momenti tanımını kullanıyoruz (ddof=0 ile
        # tutarlı olsun diye). fisher=True -> "excess kurtosis" (normal
        # dağılımın kurtosis'i 0 kabul edilir), literatürde standart tercih.
        skewness = float(_scipy_skew(window, bias=True))
        kurt = float(_scipy_kurtosis(window, fisher=True, bias=True))

    return np.array([mean, variance, minimum, maximum, skewness, kurt], dtype=np.float64)


def extract_statistical_features_dict(window: np.ndarray) -> dict[str, float]:
    """extract_statistical_features()'ın isim->değer sözlüğü döndüren,
    insan-okunur versiyonu. Debug ve keşifsel analiz (EDA) için kullanışlı;
    eğitim pipeline'ında performans için array versiyonu tercih edilmeli."""
    values = extract_statistical_features(window)
    return dict(zip(STATISTICAL_FEATURE_NAMES, values))


def build_feature_matrix(
    labeled_windows: Sequence[LabeledWindow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bir LabeledWindow listesini, sklearn uyumlu (X, y, groups) üçlüsüne
    çevirir. Bu fonksiyon, labeling_protocol.py ile models/ katmanı
    arasındaki DOĞRUDAN köprüdür.

    Dönüş değerleri:
        X:      (n_windows, 6) şeklinde özellik matrisi
        y:      (n_windows,) şeklinde etiket dizisi (string değerler,
                örn. "baseline", "mechanical_touch")
        groups: (n_windows,) şeklinde plant_id dizisi - ileride
                sklearn.model_selection.GroupKFold(groups=groups) ile
                BİTKİ-BAZLI cross-validation yapmak için (aynı bitkiden
                gelen pencerelerin hem train hem test setine düşmesini
                önlemek için) kullanılacak.

    y ve groups'un AYNI SIRADA döndürülmesi kritik: X[i] örneğinin
    etiketi y[i], hangi bitkiden geldiği groups[i]'dir. Bu üçlüyü
    ayrı ayrı üretip birleştirmek yerine TEK bir döngüde birlikte
    üretiyoruz ki sıra kayması (off-by-one) riski olmasın.
    """
    if len(labeled_windows) == 0:
        raise ValueError("Boş LabeledWindow listesinden özellik matrisi üretilemez.")

    features: list[np.ndarray] = []
    labels: list[str] = []
    plant_ids: list[str] = []

    for lw in labeled_windows:
        features.append(extract_statistical_features(lw.samples))
        labels.append(lw.label.value)
        plant_ids.append(lw.plant_id)

    X = np.stack(features, axis=0)
    y = np.array(labels)
    groups = np.array(plant_ids)
    return X, y, groups


if __name__ == "__main__":
    # Küçük bir self-check: farklı karakterde sentetik pencereler üretip
    # özelliklerin sezgisel olarak beklenen yönde davrandığını gösteriyoruz.
    rng = np.random.default_rng(0)

    quiet_baseline = rng.normal(0, 0.02, 1024)               # düşük varyans
    spiky_touch = rng.normal(0, 0.02, 1024)
    spiky_touch[500] += 3.0                                   # tek, keskin spike

    flat_signal = np.full(1024, 1.5)                          # sabit sinyal (kenar durum)

    for name, sig in [
        ("sakin baseline", quiet_baseline),
        ("keskin spike (dokunma benzeri)", spiky_touch),
        ("tamamen sabit (sensör kopmuş gibi)", flat_signal),
    ]:
        feats = extract_statistical_features_dict(sig)
        print(f"\n[{name}]")
        for k, v in feats.items():
            print(f"  {k:10s}: {v:.4f}")