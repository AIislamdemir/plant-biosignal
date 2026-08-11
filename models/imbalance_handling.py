"""
models/imbalance_handling.py
===============================
SMOTE (Synthetic Minority Over-sampling Technique) tabanlı sınıf
dengesizliği yönetimi (Chawla ve ark., 2002; PLOS One 2023 çalışmasında
çok sınıflı kimyasal uyaran sınıflandırmasında kullanılmış).

Bu modül, models/classical_models.py'de evaluate_models_grouped_cv()
fonksiyonunun `resampler` parametresine TAKILMAK üzere tasarlandı - o
dosyada bilerek bıraktığımız "hook"u burada dolduruyoruz.

Neden sınıf dengesizliği bu projede kaçınılmaz?
--------------------------------------------------
`baseline` etiketi HER trial'da var (her deneyde uyaran-öncesi kayıt
alınıyor), ama örneğin `chemical_salt_stress` sadece o sınıfın
uygulandığı trial'larda var. Bu yapısal asimetri, veri toplama
protokolü ne kadar özenli olursa olsun ortadan kalkmaz - baseline
pencere sayısı doğal olarak diğer sınıflardan fazla olacaktır.

Neden SMOTE'u GroupKFold'un İÇİNDE, train fold'a uyguluyoruz (dışında değil)?
--------------------------------------------------------------------------------
Bu, sınıf dengesizliği literatüründe EN SIK yapılan metodolojik hatanın
tam panzehiri: SMOTE'u tüm veri setine BİR KEZ uygulayıp SONRA train/test
ayırmak, sentetik (yapay) örneklerin test setine sızmasına yol açar.
Test setinde gerçek olmayan, interpolasyonla üretilmiş örnekler olursa,
raporlanan performans gerçek dünyada elde edilebilecekten daha iyimser
çıkar. Bu yüzden SMOTE HER ZAMAN, SADECE train fold'una, GroupKFold
ayrımı yapıldıktan SONRA uygulanmalı - classical_models.py'deki
resampler hook'u bunu yapısal olarak garanti ediyor.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE


def get_class_distribution(y: np.ndarray) -> dict:
    """Bir etiket dizisindeki sınıf başına örnek sayısını döndürür.

    Herhangi bir işlem yapmadan ÖNCE ve SONRA bu fonksiyonu çağırıp
    karşılaştırmak, dengesizliğin ne kadar ciddi olduğunu ve resampling
    sonrası dağılımın gerçekten dengelendiğini gözle görmeyi sağlar.
    """
    counts = Counter(y.tolist() if isinstance(y, np.ndarray) else y)
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def smote_resample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE ile azınlık sınıflarını çoğunluk sınıfının seviyesine
    kadar sentetik örneklerle tamamlar.

    Bu fonksiyon, classical_models.evaluate_models_grouped_cv()'nin
    `resampler=...` parametresine DOĞRUDAN geçirilebilecek imzaya sahip:
    (X_train, y_train) -> (X_resampled, y_resampled).

    Kenar durumu - çok küçük azınlık sınıfı:
    --------------------------------------------
    SMOTE, her sentetik örneği "en yakın k komşunun arasını enterpole
    ederek" üretir. Bu yüzden bir sınıfın en az k_neighbors+1 örneğe
    sahip olması matematiksel olarak GEREKLİDİR. Bitki-bazlı CV'de
    bazı fold'larda azınlık sınıfın örnek sayısı çok düşük kalabilir
    (örn. sadece 3 bitkiden biri o fold'un train'inde kaldıysa).

    Bu durumu SESSİZCE hata vermesine izin vermek yerine:
      - Eğer azınlık sınıfın örnek sayısı >= 2 ise: k_neighbors'ı
        otomatik olarak (azınlık_sayısı - 1)'e düşürüyoruz (SMOTE'un
        çalışabileceği en düşük geçerli değer).
      - Eğer azınlık sınıfın örnek sayısı == 1 ise: SMOTE hiç
        çalışamaz (enterpolasyon için en az 2 nokta gerekir). Bu
        durumda RandomOverSampler'a (basitçe var olan örneği
        çoğaltır, enterpolasyon yapmaz) otomatik olarak düşüyoruz -
        SESSİZCE değil, bir uyarı yazdırarak.
    """
    class_counts = Counter(y_train.tolist() if isinstance(y_train, np.ndarray) else y_train)
    minority_count = min(class_counts.values())

    if minority_count < 2:
        print(
            f"[Uyarı] En küçük sınıfta sadece {minority_count} örnek var - "
            f"SMOTE enterpolasyon için en az 2 örnek gerektirir. "
            f"RandomOverSampler'a (enterpolasyonsuz, basit çoğaltma) düşülüyor."
        )
        sampler = RandomOverSampler(random_state=random_state)
    else:
        effective_k = min(k_neighbors, minority_count - 1)
        if effective_k < k_neighbors:
            print(
                f"[Uyarı] k_neighbors={k_neighbors} istenmişti, ama en küçük sınıfta "
                f"{minority_count} örnek var. k_neighbors={effective_k} olarak düşürüldü."
            )
        sampler = SMOTE(k_neighbors=effective_k, random_state=random_state)

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def make_smote_resampler(k_neighbors: int = 5, random_state: int = 42):
    """smote_resample()'ı sabit k_neighbors/random_state ile önceden
    yapılandırılmış, TEK ARGÜMANLI (X_train, y_train) bir closure'a
    çevirir.

    Neden bu sarmalayıcıya ihtiyaç var?
    classical_models.evaluate_models_grouped_cv()'nin `resampler`
    parametresi tam olarak `(X_train, y_train) -> (X, y)` imzası
    bekliyor - k_neighbors/random_state gibi ek argümanları taşıyamaz.
    Bu fonksiyon, o ek argümanları "kapatıp" (closure) beklenen imzada
    bir fonksiyon üretir. Kullanım:

        resampler = make_smote_resampler(k_neighbors=3)
        results = evaluate_models_grouped_cv(X, y, groups, resampler=resampler)
    """

    def _resampler(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return smote_resample(X_train, y_train, k_neighbors=k_neighbors, random_state=random_state)

    return _resampler


def compute_class_weights(y: np.ndarray) -> dict:
    """SMOTE'a ALTERNATİF, daha hafif bir dengesizlik yönetimi yöntemi:
    veri setini değiştirmek yerine, modele "azınlık sınıfının hatalarını
    daha ağır cezalandır" talimatını veren sınıf ağırlıkları üretir.

    sklearn'ün çoğu sınıflandırıcısı (RandomForest, SVC, DecisionTree...)
    `class_weight=...` parametresini doğrudan kabul eder - bu durumda
    SMOTE'un ürettiği sentetik örneklere hiç ihtiyaç duymadan, veri
    setinin boyutunu ARTIRMADAN dengesizliği ele alabilirsin.

    Formül: weight[sınıf] = n_toplam / (n_sınıf_sayısı * sınıf_örnek_sayısı)
    Bu, sklearn'ün 'balanced' modunun kullandığı standart formüldür -
    burada elle uyguluyoruz ki hangi sınıfın ne kadar ağırlık aldığını
    açıkça görebilesin (debug/rapor için kullanışlı).
    """
    counts = Counter(y.tolist() if isinstance(y, np.ndarray) else y)
    n_total = sum(counts.values())
    n_classes = len(counts)
    return {label: n_total / (n_classes * count) for label, count in counts.items()}


if __name__ == "__main__":
    # Demo: kasıtlı olarak dengesiz bir veri setinde SMOTE öncesi/sonrası
    # dağılımı ve class_weight alternatifini karşılaştırıyoruz.
    rng = np.random.default_rng(0)

    # Kasıtlı dengesizlik: 100 baseline, 60 mekanik dokunma, sadece 8 kimyasal stres
    y_imbalanced = np.array(
        ["baseline"] * 100 + ["mechanical_touch"] * 60 + ["chemical_salt_stress"] * 8
    )
    X_imbalanced = rng.normal(size=(len(y_imbalanced), 6))

    print("SMOTE ÖNCESİ dağılım:", get_class_distribution(y_imbalanced))

    X_resampled, y_resampled = smote_resample(X_imbalanced, y_imbalanced, k_neighbors=5)
    print("SMOTE SONRASI dağılım:", get_class_distribution(y_resampled))

    print("\nAlternatif: class_weight (veri boyutu değişmez):")
    weights = compute_class_weights(y_imbalanced)
    for label, weight in weights.items():
        print(f"  {label:22s}: {weight:.3f}")

    print("\nEkstrem kenar durum: bir sınıfta sadece 1 örnek varken:")
    y_extreme = np.array(["baseline"] * 20 + ["temperature_shock"] * 1)
    X_extreme = rng.normal(size=(len(y_extreme), 6))
    X_res, y_res = smote_resample(X_extreme, y_extreme)
    print("Sonuç dağılımı:", get_class_distribution(y_res))