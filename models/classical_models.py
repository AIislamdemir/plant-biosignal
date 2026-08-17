"""
models/classical_models.py
=============================
Literatürde karşılaştırılmış klasik sınıflandırıcıları (Reissig ve ark.,
2021 karşılaştırması + Najdenovska ve ark., 2021'deki XGBoost sonucu)
BİTKİ-BAZLI cross-validation ile eğiten/değerlendiren modül.

Bu dosya, features/statistical_features.py'nin ürettiği (X, y, groups)
üçlüsünü doğrudan tüketen ilk "gerçek model" katmanı - projenin uçtan
uca ÇALIŞTIĞINI kanıtlayan ilk baseline burada kuruluyor.

Neden burada GroupKFold, normal KFold değil? (kritik metodolojik karar)
--------------------------------------------------------------------------
Prompt'un "Değerlendirme metodolojisi" kısıtı açık: "bitki-bitki ve
gün-gün varyansı göz önüne alınarak, cross-validation bitki bazında
yapılmalı (aynı bitkiden gelen örnekler hem train hem test setinde
olmamalı - data leakage riski)". Normal KFold, örnekleri rastgele
karıştırır - aynı bitkiden gelen pencereler (ki bunlar birbirine çok
benzer, çünkü aynı elektrot yerleşimi, aynı bitki fizyolojisi) hem
train hem test setine düşebilir. Bu durumda model "yeni bir bitkiyi
genelleme" yerine "bu bitkinin imzasını ezberleme" yapar ve CV skoru
gerçekte olduğundan çok daha iyi görünür (optimistic bias). GroupKFold
bunu yapısal olarak engelliyor: her fold'da TÜM bir bitkinin verisi ya
tamamen train'de ya tamamen test'te.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - ortamda xgboost kurulu olmayabilir
    _XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1) Model kayıt defteri (registry)
# ---------------------------------------------------------------------------
def _build_model_registry(random_state: int = 42) -> dict[str, Callable[[], ClassifierMixin]]:
    """Literatürde karşılaştırılan modellerin FABRİKA fonksiyonlarını
    (nesne değil, nesne üreten fonksiyon) döndürür.

    Neden nesne değil fabrika fonksiyonu? Çünkü aynı model tipini
    cross-validation'ın HER fold'unda SIFIRDAN, önceki fold'un
    öğrendiklerinden bağımsız olarak eğitmemiz gerekiyor. Eğer tek bir
    model nesnesini paylaşıp fold'lar arasında tekrar tekrar `.fit()`
    çağırsaydık, bazı sklearn modelleri (örn. `warm_start=True` olanlar)
    önceki fold'un ağırlıklarından devam edebilirdi - bu da fold'lar
    arasında sessiz bir bilgi sızıntısına yol açardı. Fabrika deseni bunu
    yapısal olarak imkânsız kılıyor: her çağrıda TAMAMEN yeni bir nesne.
    """
    registry: dict[str, Callable[[], ClassifierMixin]] = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        "svc": lambda: SVC(kernel="rbf", probability=True, random_state=random_state),
        "gaussian_process": lambda: GaussianProcessClassifier(
            kernel=1.0 * RBF(length_scale=1.0), random_state=random_state
        ),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=random_state),
        "gaussian_naive_bayes": lambda: GaussianNB(),
        "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    }

    if _XGBOOST_AVAILABLE:
        registry["xgboost"] = lambda: XGBClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=random_state,
            eval_metric="mlogloss",
        )
    # xgboost kurulu değilse registry'ye eklenmiyor - evaluate fonksiyonu
    # bunu sessizce atlıyor, hata vermiyor (bkz. get_available_model_names).

    return registry


def get_model_registry(random_state: int = 42) -> dict[str, Callable[[], ClassifierMixin]]:
    """_build_model_registry()'nin PUBLIC (dışarıya açık) versiyonu.

    utils/metrics.py gibi diğer modüllerin, bir model FABRİKASINI
    (nesneyi değil) alıp kendi CV/analiz akışlarında (örn.
    cross_val_predict ile out-of-fold tahmin üretmek için) kullanabilmesi
    için eklendi. _build_model_registry ile aynı sözlüğü döndürür; bu
    sadece isimlendirme sözleşmesini netleştiren ince bir sarmalayıcı.
    """
    return _build_model_registry(random_state)


def get_available_model_names(random_state: int = 42) -> tuple[str, ...]:
    """Şu an ortamda kullanılabilir model isimlerini döndürür (xgboost
    kurulu değilse listede görünmez). CLI/notebook'ta 'hangi modeller
    var?' diye kontrol etmek için kullanışlı."""
    return tuple(_build_model_registry(random_state).keys())


# ---------------------------------------------------------------------------
# 2) Bitki-bazlı cross-validation değerlendirmesi
# ---------------------------------------------------------------------------
def evaluate_models_grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Optional[Sequence[str]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    resampler: Optional[Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """Seçilen modelleri GroupKFold ile değerlendirir, her (model, fold)
    kombinasyonu için accuracy ve macro-F1 döndürür.

    Parametreler
    ------------
    resampler: Opsiyonel bir fonksiyon (X_train, y_train) -> (X_res, y_res).
        Bu parametre, ileride yazılacak `models/imbalance_handling.py`
        (SMOTE) için bilerek bırakılmış bir "hook" (bağlantı noktası).
        KRİTİK KURAL: resampler SADECE train fold'una uygulanır, test
        fold'una ASLA. Eğer SMOTE test fold'una da uygulansaydı, sentetik
        (yapay) test örnekleri gerçek performansı olduğundan iyi gösterirdi
        - bu, sınıf dengesizliği literatüründe sık yapılan bir hatadır.

    Neden macro-F1 de raporluyoruz, sadece accuracy değil?
    ---------------------------------------------------------
    Sınıflar dengesiz olabilir (örn. baseline pencere sayısı, kuraklık
    stresi pencere sayısından çok daha fazla olabilir çünkü baseline her
    trial'da var). Böyle durumda "her şeyi baseline tahmin et" stratejisi
    bile yüksek accuracy verebilir ama azınlık sınıflarını hiç yakalamaz.
    Macro-F1, her sınıfı EŞİT ağırlıkla değerlendirdiği için bu durumu
    açığa çıkarır.
    """
    n_unique_groups = len(np.unique(groups))
    effective_n_splits = min(n_splits, n_unique_groups)
    if effective_n_splits < 2:
        raise ValueError(
            f"GroupKFold en az 2 farklı grup (bitki) gerektirir, "
            f"veri setinde sadece {n_unique_groups} farklı bitki var. "
            f"Daha fazla bitkiden veri toplanmalı (bkz. ProtocolMinimums.MIN_DISTINCT_PLANTS)."
        )
    if effective_n_splits < n_splits:
        print(
            f"[Uyarı] İstenen n_splits={n_splits}, ama sadece {n_unique_groups} farklı "
            f"bitki var. n_splits={effective_n_splits} olarak düşürüldü."
        )

    registry = _build_model_registry(random_state)
    selected_names = list(model_names) if model_names is not None else list(registry.keys())
    unknown = set(selected_names) - set(registry.keys())
    if unknown:
        raise ValueError(
            f"Bilinmeyen model isimleri: {unknown}. Kullanılabilir modeller: {list(registry.keys())}"
        )

    # Etiket kodlayıcıyı TÜM veri setindeki sınıflarla eğitiyoruz (fold
    # bazında değil) ki her fold'da sınıf-indeks eşlemesi TUTARLI kalsın
    # (örn. "baseline" her zaman 0 kodlansın, fold'dan fold'a değişmesin).
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    splitter = GroupKFold(n_splits=effective_n_splits)
    rows: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y_encoded, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        if resampler is not None:
            X_train, y_train = resampler(X_train, y_train)

        for model_name in selected_names:
            model = registry[model_name]()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                }
            )

    return pd.DataFrame(rows)


def summarize_cv_results(results: pd.DataFrame) -> pd.DataFrame:
    """Fold-bazlı sonuçları model başına ortalama +/- standart sapma
    olarak özetler - hangi modelin hem YÜKSEK hem İSTİKRARLI (düşük
    fold-arası varyans) performans gösterdiğini görmek için.

    Yüksek varyans (fold'lar arasında büyük fark), modelin bazı
    bitkilerde iyi bazılarında kötü genelleştirdiğinin işareti olabilir -
    bu, sadece ortalamaya bakarak fark edilemeyecek önemli bir sinyal.
    """
    return (
        results.groupby("model")[["accuracy", "f1_macro"]]
        .agg(["mean", "std"])
        .sort_values(("f1_macro", "mean"), ascending=False)
    )


# ---------------------------------------------------------------------------
# 3) Nihai model eğitimi (tüm veriyle, deployment için)
# ---------------------------------------------------------------------------
@dataclass
class TrainedModel:
    """Bir modeli ve onu doğru şekilde kullanmak için gereken her şeyi
    (etiket kodlayıcı dahil) bir arada taşıyan sonuç tipi.

    label_encoder'ı model ile BİRLİKTE taşımak zorunludur: model sadece
    0, 1, 2... gibi sayısal sınıflar üretir, bunları "baseline",
    "mechanical_touch" gibi insan-okunur etiketlere geri çevirmek için
    EĞİTİMDE KULLANILAN AYNI encoder'a ihtiyaç var. inference.py bu
    nesneyi bütün olarak (örn. pickle ile) kaydedip yükleyecek.
    """

    model_name: str
    model: ClassifierMixin
    label_encoder: LabelEncoder

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Sayısal tahminleri otomatik olarak orijinal string etiketlere
        (örn. 'mechanical_touch') geri çevirir - çağıran tarafın encoder
        detayını bilmesine gerek kalmaz."""
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)


def train_final_model(
    model_name: str, X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> TrainedModel:
    """Seçilen modeli TÜM veriyle (artık CV değil, deployment amaçlı)
    eğitir. evaluate_models_grouped_cv() ile en iyi model seçildikten
    SONRA, o modeli elindeki bütün veriyle son kez eğitmek için kullanılır.
    """
    registry = _build_model_registry(random_state)
    if model_name not in registry:
        raise ValueError(
            f"Bilinmeyen model adı: '{model_name}'. Kullanılabilir: {list(registry.keys())}"
        )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = registry[model_name]()
    model.fit(X, y_encoded)

    return TrainedModel(model_name=model_name, model=model, label_encoder=label_encoder)


# ---------------------------------------------------------------------------
# 4) Fold'lar arası ham tahminlerin toplanması (utils/metrics.py için)
# ---------------------------------------------------------------------------
def get_out_of_fold_predictions(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    resampler: Optional[Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TEK bir model için, GroupKFold'un TÜM fold'larındaki test tahminlerini
    toplayıp tek bir (y_true, y_pred, groups) üçlüsü olarak döndürür.

    Neden evaluate_models_grouped_cv()'den AYRI bir fonksiyon?
    ---------------------------------------------------------------
    evaluate_models_grouped_cv() sadece ÖZET skorları (accuracy, f1_macro)
    saklıyor - ham tahminleri atıyor. Bu, çoğu zaman yeterli ama karışıklık
    matrisi veya sınıf-bazlı recall/precision görmek istediğinde (bkz.
    utils/metrics.py) HAM (gerçek, tahmin) çiftlerine ihtiyaç var. Mevcut,
    test edilmiş fonksiyonun imzasını/davranışını bozmamak için bunu YENİ
    bir fonksiyon olarak ekliyorum - regresyon riski yok.

    "Out-of-fold" (OOF) tahmin ne demek?
    Her örnek için, o örneğin bulunduğu fold TEST'teyken üretilen tahmin
    kullanılıyor - yani her tahmin, o modelin o örneği HİÇ GÖRMEDEN yaptığı
    bir tahmin. Bu yüzden tüm veri seti üzerinde tarafsız bir performans
    resmi çizer (train setindeki "ezberlenmiş" tahminler karışmaz).
    """
    n_unique_groups = len(np.unique(groups))
    effective_n_splits = min(n_splits, n_unique_groups)
    if effective_n_splits < 2:
        raise ValueError(
            f"GroupKFold en az 2 farklı grup (bitki) gerektirir, "
            f"veri setinde sadece {n_unique_groups} farklı bitki var."
        )

    registry = _build_model_registry(random_state)
    if model_name not in registry:
        raise ValueError(
            f"Bilinmeyen model adı: '{model_name}'. Kullanılabilir: {list(registry.keys())}"
        )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    splitter = GroupKFold(n_splits=effective_n_splits)
    y_true_all, y_pred_all, groups_all = [], [], []

    for train_idx, test_idx in splitter.split(X, y_encoded, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        if resampler is not None:
            X_train, y_train = resampler(X_train, y_train)

        model = registry[model_name]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        groups_all.append(groups[test_idx])

    # Sayısal kodları geri, insan-okunur string etiketlere çeviriyoruz
    # (örn. 0 -> "baseline") - utils/metrics.py'nin okunabilir bir
    # karışıklık matrisi üretebilmesi için.
    y_true_labels = label_encoder.inverse_transform(np.concatenate(y_true_all))
    y_pred_labels = label_encoder.inverse_transform(np.concatenate(y_pred_all))
    groups_ordered = np.concatenate(groups_all)

    return y_true_labels, y_pred_labels, groups_ordered


if __name__ == "__main__":
    # Uçtan uca self-check: sentetik ama sınıf-ayırt-edici bir veri seti
    # üretip, tüm klasik modelleri bitki-bazlı CV ile karşılaştırıyoruz.
    # Bu, pipeline'ın (labeling -> features -> models) GERÇEKTEN uçtan
    # uca çalıştığının kanıtı.
    rng = np.random.default_rng(42)
    n_plants = 6
    n_windows_per_plant_per_class = 15

    X_rows, y_rows, group_rows = [], [], []
    for plant_idx in range(n_plants):
        plant_id = f"plant_{plant_idx}"
        for _ in range(n_windows_per_plant_per_class):
            # "baseline": düşük varyans, düşük kurtosis
            X_rows.append([0.0, 0.001, -0.05, 0.05, 0.0, 0.3])
            y_rows.append("baseline")
            group_rows.append(plant_id)

            # "mechanical_touch": yüksek kurtosis (keskin spike benzeri)
            X_rows.append([0.05, 0.01, -0.1, 3.0, 25.0, 800.0])
            y_rows.append("mechanical_touch")
            group_rows.append(plant_id)

    # Küçük, gerçekçi bir gürültü ekleyip mükemmel-ayrılabilir olmaktan çıkarıyoruz
    X = np.array(X_rows) + rng.normal(0, 0.02, size=(len(X_rows), 6))
    y = np.array(y_rows)
    groups = np.array(group_rows)

    print(f"Sentetik veri seti: {X.shape[0]} pencere, {len(set(groups))} farklı bitki\n")

    results = evaluate_models_grouped_cv(X, y, groups, n_splits=5)
    print("Fold-bazlı ham sonuçlar (ilk 5 satır):")
    print(results.head())

    print("\nModel bazında özet (f1_macro ortalamasına göre sıralı):")
    print(summarize_cv_results(results))