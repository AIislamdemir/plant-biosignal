"""
tests/test_imbalance_handling.py
===================================
models/imbalance_handling.py için birim testleri.

En kritik testler:
  - test_smote_never_applied_to_test_fold: SMOTE'un classical_models.py
    ile entegre kullanıldığında test fold'una asla sızmadığını kanıtlar
  - test_extreme_minority_falls_back_gracefully: 1 örneklik bir sınıfla
    karşılaşıldığında sistemin çökmediğini kanıtlar
"""

from __future__ import annotations

import numpy as np
import pytest

from models.classical_models import evaluate_models_grouped_cv
from models.imbalance_handling import (
    compute_class_weights,
    get_class_distribution,
    make_smote_resampler,
    smote_resample,
)


def make_imbalanced_dataset(seed: int = 0):
    """100 baseline, 60 mechanical_touch, 8 chemical_salt_stress ile
    kasıtlı olarak dengesiz bir veri seti üretir."""
    rng = np.random.default_rng(seed)
    y = np.array(
        ["baseline"] * 100 + ["mechanical_touch"] * 60 + ["chemical_salt_stress"] * 8
    )
    X = rng.normal(size=(len(y), 6))
    return X, y


# ---------------------------------------------------------------------------
# get_class_distribution
# ---------------------------------------------------------------------------
class TestGetClassDistribution:
    def test_counts_are_correct(self):
        X, y = make_imbalanced_dataset()
        dist = get_class_distribution(y)
        assert dist["baseline"] == 100
        assert dist["mechanical_touch"] == 60
        assert dist["chemical_salt_stress"] == 8

    def test_sorted_descending_by_count(self):
        X, y = make_imbalanced_dataset()
        dist = get_class_distribution(y)
        counts = list(dist.values())
        assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# smote_resample - temel davranış
# ---------------------------------------------------------------------------
class TestSmoteResample:
    def test_all_classes_balanced_after_resampling(self):
        """SMOTE sonrası TÜM sınıflar, en büyük sınıfın örnek sayısına
        eşit olmalı (varsayılan 'auto' stratejisi)."""
        X, y = make_imbalanced_dataset()
        X_res, y_res = smote_resample(X, y, k_neighbors=5)
        dist = get_class_distribution(y_res)
        assert len(set(dist.values())) == 1  # tüm sınıflar eşit sayıda
        assert list(dist.values())[0] == 100  # en büyük sınıfın (baseline) sayısına eşitlendi

    def test_feature_dimensionality_preserved(self):
        """Sentetik örnekler de aynı özellik sayısına (6) sahip olmalı."""
        X, y = make_imbalanced_dataset()
        X_res, _ = smote_resample(X, y)
        assert X_res.shape[1] == X.shape[1] == 6

    def test_extreme_minority_falls_back_gracefully(self):
        """Bir sınıfta sadece 1 örnek varsa (SMOTE'un matematiksel olarak
        çalışamayacağı durum), sistem hata FIRLATMAMALI, RandomOverSampler'a
        düşerek çalışmaya devam etmeli."""
        rng = np.random.default_rng(2)
        y = np.array(["baseline"] * 20 + ["temperature_shock"] * 1)
        X = rng.normal(size=(len(y), 6))

        X_res, y_res = smote_resample(X, y)  # hata fırlatmamalı
        dist = get_class_distribution(y_res)
        assert dist["baseline"] == dist["temperature_shock"] == 20

    def test_k_neighbors_auto_reduced_for_small_minority_class(self):
        """Azınlık sınıfın örnek sayısı k_neighbors'tan azsa, sistem
        k_neighbors'ı otomatik düşürüp yine de çalışmalı (hata vermemeli)."""
        rng = np.random.default_rng(3)
        # azınlık sınıfta sadece 3 örnek var, k_neighbors=5 istenmiş
        y = np.array(["baseline"] * 20 + ["drought_stress"] * 3)
        X = rng.normal(size=(len(y), 6))

        X_res, y_res = smote_resample(X, y, k_neighbors=5)  # hata vermemeli
        dist = get_class_distribution(y_res)
        assert dist["baseline"] == dist["drought_stress"] == 20


# ---------------------------------------------------------------------------
# make_smote_resampler - classical_models.py entegrasyonu
# ---------------------------------------------------------------------------
class TestMakeSmoteResampler:
    def test_returned_closure_has_correct_signature(self):
        """Üretilen closure, evaluate_models_grouped_cv()'nin beklediği
        (X_train, y_train) -> (X, y) imzasıyla uyumlu olmalı."""
        X, y = make_imbalanced_dataset()
        resampler = make_smote_resampler(k_neighbors=3)
        X_res, y_res = resampler(X, y)
        assert X_res.shape[0] == y_res.shape[0]

    def test_smote_never_applied_to_test_fold(self):
        """PROJENİN TEMEL METODOLOJİK GARANTİSİ: classical_models.py ile
        entegre kullanıldığında, SMOTE'un test fold boyutunu ASLA
        değiştirmediğini (yani test setine hiç sızmadığını) kanıtlar.
        """
        rng = np.random.default_rng(4)
        X_rows, y_rows, group_rows = [], [], []
        for plant_idx in range(5):
            plant_id = f"plant_{plant_idx}"
            for _ in range(15):
                X_rows.append([0.0, 0.001, -0.05, 0.05, 0.0, 0.3])
                y_rows.append("baseline")
                group_rows.append(plant_id)
            for _ in range(3):  # kasıtlı azınlık
                X_rows.append([0.05, 0.01, -0.1, 3.0, 25.0, 800.0])
                y_rows.append("mechanical_touch")
                group_rows.append(plant_id)

        X = np.array(X_rows) + rng.normal(0, 0.02, size=(len(X_rows), 6))
        y = np.array(y_rows)
        groups = np.array(group_rows)

        no_resample = evaluate_models_grouped_cv(
            X, y, groups, model_names=["decision_tree"], n_splits=5
        )
        with_smote = evaluate_models_grouped_cv(
            X, y, groups, model_names=["decision_tree"], n_splits=5,
            resampler=make_smote_resampler(k_neighbors=2),
        )

        # n_test (test fold boyutu) SMOTE'tan TAMAMEN bağımsız olmalı
        assert list(no_resample["n_test"]) == list(with_smote["n_test"])
        # n_train ise SMOTE sayesinde artmış veya eşit olmalı (asla azalmaz)
        assert all(with_smote["n_train"] >= no_resample["n_train"])


# ---------------------------------------------------------------------------
# compute_class_weights - SMOTE'a alternatif
# ---------------------------------------------------------------------------
class TestComputeClassWeights:
    def test_minority_class_gets_highest_weight(self):
        """En az örnekli sınıf (chemical_salt_stress, 8 örnek), en fazla
        ağırlığı almalı - modele 'bu sınıfın hatasını daha ağır cezalandır'
        talimatı bu şekilde veriliyor."""
        X, y = make_imbalanced_dataset()
        weights = compute_class_weights(y)
        assert weights["chemical_salt_stress"] == max(weights.values())
        assert weights["baseline"] == min(weights.values())

    def test_weights_are_positive(self):
        X, y = make_imbalanced_dataset()
        weights = compute_class_weights(y)
        assert all(w > 0 for w in weights.values())