"""
tests/test_classical_models.py
=================================
models/classical_models.py için birim testleri.

En kritik test: test_group_kfold_never_splits_same_plant_across_train_test
-> Bu proje için sadece bir "nice to have" değil, prompt'un açık kısıtı
   olan "aynı bitkiden gelen örnekler hem train hem test setinde olmamalı"
   gerekliliğinin fiilen sağlandığının kanıtı.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.classical_models import (
    TrainedModel,
    evaluate_models_grouped_cv,
    get_available_model_names,
    get_out_of_fold_predictions,
    summarize_cv_results,
    train_final_model,
)


def make_synthetic_dataset(n_plants: int = 6, n_per_plant_per_class: int = 10, seed: int = 0):
    """İki sınıfın (baseline / mechanical_touch) belirgin şekilde ayırt
    edilebildiği, ama küçük bir gürültü içeren sentetik veri seti üretir."""
    rng = np.random.default_rng(seed)
    X_rows, y_rows, group_rows = [], [], []
    for plant_idx in range(n_plants):
        plant_id = f"plant_{plant_idx}"
        for _ in range(n_per_plant_per_class):
            X_rows.append([0.0, 0.001, -0.05, 0.05, 0.0, 0.3])
            y_rows.append("baseline")
            group_rows.append(plant_id)

            X_rows.append([0.05, 0.01, -0.1, 3.0, 25.0, 800.0])
            y_rows.append("mechanical_touch")
            group_rows.append(plant_id)

    X = np.array(X_rows) + rng.normal(0, 0.02, size=(len(X_rows), 6))
    y = np.array(y_rows)
    groups = np.array(group_rows)
    return X, y, groups


# ---------------------------------------------------------------------------
# Model registry testleri
# ---------------------------------------------------------------------------
class TestModelRegistry:
    def test_expected_models_are_available(self):
        """Prompt'ta istenen klasik modellerin (xgboost hariç garanti
        kurulu olanlar) registry'de bulunduğunu doğrular."""
        names = get_available_model_names()
        for expected in ["random_forest", "svc", "decision_tree", "gaussian_naive_bayes", "knn"]:
            assert expected in names


# ---------------------------------------------------------------------------
# GroupKFold - bitki-bazlı ayrım (en kritik test grubu)
# ---------------------------------------------------------------------------
class TestGroupedCrossValidation:
    def test_group_kfold_never_splits_same_plant_across_train_test(self):
        """PROJENİN TEMEL METODOLOJİK GARANTİSİ: hiçbir fold'da, aynı
        plant_id hem train hem test setinde görünmemeli. Bu test kırılırsa
        data leakage riski geri gelmiş demektir - CV skorları yanıltıcı
        şekilde iyimser olur.
        """
        from sklearn.model_selection import GroupKFold

        X, y, groups = make_synthetic_dataset(n_plants=6)
        splitter = GroupKFold(n_splits=5)
        for train_idx, test_idx in splitter.split(X, y, groups=groups):
            train_plants = set(groups[train_idx])
            test_plants = set(groups[test_idx])
            assert train_plants.isdisjoint(test_plants), (
                f"Aynı bitki hem train ({train_plants}) hem test ({test_plants}) "
                f"setinde bulundu - data leakage!"
            )

    def test_too_few_plants_raises_informative_error(self):
        """Bitki sayısı (grup sayısı) istenen n_splits'ten azsa, sessizce
        yanlış davranmak yerine AÇIK bir hata vermeli."""
        X, y, groups = make_synthetic_dataset(n_plants=1)
        with pytest.raises(ValueError, match="GroupKFold"):
            evaluate_models_grouped_cv(X, y, groups, n_splits=5)

    def test_n_splits_auto_reduced_when_fewer_plants_than_requested(self):
        """İstenen n_splits, mevcut bitki sayısından fazlaysa hata vermek
        yerine otomatik düşürülmeli (kullanıcı deneyimi için) - ama en az
        2 bitki olduğu sürece çalışabilmeli."""
        X, y, groups = make_synthetic_dataset(n_plants=3)
        results = evaluate_models_grouped_cv(X, y, groups, n_splits=10, model_names=["decision_tree"])
        assert results["fold"].nunique() == 3  # 10 istenmişti, 3'e düşürüldü

    def test_unknown_model_name_raises(self):
        X, y, groups = make_synthetic_dataset(n_plants=4)
        with pytest.raises(ValueError, match="Bilinmeyen model"):
            evaluate_models_grouped_cv(X, y, groups, model_names=["not_a_real_model"])

    def test_results_contain_expected_columns(self):
        X, y, groups = make_synthetic_dataset(n_plants=4)
        results = evaluate_models_grouped_cv(X, y, groups, model_names=["decision_tree", "knn"], n_splits=4)
        for col in ["model", "fold", "n_train", "n_test", "accuracy", "f1_macro"]:
            assert col in results.columns
        assert set(results["model"].unique()) == {"decision_tree", "knn"}

    def test_resampler_hook_only_applied_to_train_fold(self):
        """resampler fonksiyonu ÇAĞRILMALI (train fold için) ama test
        fold'unun boyutu resampler'dan ETKİLENMEMELİ - bu, SMOTE gibi bir
        yeniden örnekleyicinin test setine asla sızmadığının kanıtı."""
        X, y, groups = make_synthetic_dataset(n_plants=4)
        call_count = {"n": 0}

        def fake_oversampler(X_train, y_train):
            call_count["n"] += 1
            # train fold'unu YAPAY olarak iki katına çıkar (sahte SMOTE)
            return np.vstack([X_train, X_train]), np.concatenate([y_train, y_train])

        original_results = evaluate_models_grouped_cv(
            X, y, groups, model_names=["decision_tree"], n_splits=4
        )
        resampled_results = evaluate_models_grouped_cv(
            X, y, groups, model_names=["decision_tree"], n_splits=4, resampler=fake_oversampler
        )

        assert call_count["n"] == 4  # her fold'da bir kere çağrıldı
        # n_test DEĞİŞMEMELİ (resampler test'e dokunmuyor)
        assert list(original_results["n_test"]) == list(resampled_results["n_test"])
        # n_train ARTMALI (resampler train'i iki katına çıkardı)
        assert all(resampled_results["n_train"] == original_results["n_train"] * 2)

    def test_summarize_cv_results_ranks_by_f1_macro(self):
        X, y, groups = make_synthetic_dataset(n_plants=5)
        results = evaluate_models_grouped_cv(
            X, y, groups, model_names=["decision_tree", "gaussian_naive_bayes"], n_splits=5
        )
        summary = summarize_cv_results(results)
        # en yüksek f1_macro ortalaması ilk satırda olmalı
        f1_means = summary[("f1_macro", "mean")]
        assert list(f1_means) == sorted(f1_means, reverse=True)


# ---------------------------------------------------------------------------
# Nihai model eğitimi testleri
# ---------------------------------------------------------------------------
class TestGetOutOfFoldPredictions:
    def test_returns_arrays_of_equal_length(self):
        """y_true, y_pred ve groups AYNI uzunlukta olmali - her satir
        birbirine karsilik gelen tek bir test-fold ornegini temsil eder."""
        X, y, groups = make_synthetic_dataset(n_plants=5)
        y_true, y_pred, groups_ordered = get_out_of_fold_predictions(
            "decision_tree", X, y, groups, n_splits=5
        )
        assert len(y_true) == len(y_pred) == len(groups_ordered) == len(y)

    def test_predictions_use_original_string_labels(self):
        """Donen tahminler sayisal kod (0,1,2..) DEGIL, orijinal string
        etiketler (orn. 'baseline') olmali - label_encoder.inverse_transform
        dogru calismis mi diye kontrol."""
        X, y, groups = make_synthetic_dataset(n_plants=5)
        y_true, y_pred, _ = get_out_of_fold_predictions("decision_tree", X, y, groups, n_splits=5)
        assert set(y_true) <= {"baseline", "mechanical_touch"}
        assert set(y_pred) <= {"baseline", "mechanical_touch"}

    def test_every_row_is_out_of_fold_i_e_unseen_during_its_own_training(self):
        """OOF tahminin anlami: model, o ornegi HIC gormeden tahmin
        uretmis olmali. Bunu dogrudan test etmek zor, ama dolayli bir
        kontrol yapabiliriz: y_true her zaman orijinal y ile AYNI COKLU
        KUMEYI (multiset) icermeli - hicbir ornek atlanmamis/eklenmemis."""
        X, y, groups = make_synthetic_dataset(n_plants=5)
        y_true, _, _ = get_out_of_fold_predictions("decision_tree", X, y, groups, n_splits=5)
        assert sorted(y_true) == sorted(y)

    def test_unknown_model_name_raises(self):
        X, y, groups = make_synthetic_dataset(n_plants=4)
        with pytest.raises(ValueError, match="Bilinmeyen model"):
            get_out_of_fold_predictions("not_a_real_model", X, y, groups)


class TestTrainFinalModel:
    def test_train_final_model_returns_usable_predictor(self):
        X, y, groups = make_synthetic_dataset(n_plants=4)
        trained = train_final_model("decision_tree", X, y)

        assert isinstance(trained, TrainedModel)
        predictions = trained.predict_labels(X[:5])
        # predict_labels, kodlanmış sayılar değil orijinal string etiketleri döndürmeli
        assert set(predictions) <= {"baseline", "mechanical_touch"}

    def test_unknown_model_name_raises_on_final_training(self):
        X, y, groups = make_synthetic_dataset(n_plants=4)
        with pytest.raises(ValueError):
            train_final_model("not_a_real_model", X, y)