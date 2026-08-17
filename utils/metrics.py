"""
utils/metrics.py
===================
Doğruluk, F1, karışıklık matrisi ve sınıf-bazlı performans raporlaması.

Bu modül, classical_models.py'nin ürettiği ÖZET skorların (accuracy,
f1_macro) yeterli olmadığı durumlar için var: "Model genel olarak
%92 doğru, ama HANGİ sınıfta zorlanıyor?" sorusuna cevap veriyor.

Neden bu özellikle imbalance_handling.py'den SONRA anlamlı?
----------------------------------------------------------------
SMOTE veya class_weight uyguladığında, toplam accuracy çok az değişebilir
(çünkü zaten çoğunluk sınıfı - baseline - kolay tahmin ediliyor), ama
asıl amaç azınlık sınıfların (örn. kimyasal stres) YAKALANMA oranını
(recall) artırmaktı. Bu modüldeki compare_recall_before_after()
fonksiyonu, tam olarak bu etkiyi sınıf bazında görünür kılıyor - SMOTE'un
gerçekten işe yarayıp yaramadığının kanıtı burada.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tek bir tahmin setinin toplu (sınıf bazlı olmayan) özetini döndürür.

    f1_macro: her sınıfı EŞİT ağırlıklandırır (azınlık sınıf da çoğunluk
    sınıf kadar önemli sayılır) - dengesiz veri setinde asıl bakılması
    gereken metrik budur, sadece accuracy değil.
    f1_weighted: her sınıfı örnek SAYISINA göre ağırlıklandırır - "genel
    pratik performans" sorusuna daha yakın bir cevap verir.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def confusion_matrix_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Okunabilir, satır/sütun isimleri "true_X" / "pred_X" şeklinde
    etiketlenmiş bir karışıklık matrisi (confusion matrix) döndürür.

    Satırlar GERÇEK sınıfı, sütunlar TAHMİN edilen sınıfı temsil eder.
    Köşegen dışındaki büyük sayılar, modelin hangi sınıfları birbirine
    KARIŞTIRDIĞINI gösterir - örn. "true_drought_stress" satırında
    "pred_baseline" sütununda yüksek bir sayı, modelin yavaş kuraklık
    sinyalini baseline'la karıştırdığını gösterir (beklenebilir bir
    hata: kuraklık sinyali baseline'a benzer şekilde yavaş ve gürültülü).
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Her sınıf için precision, recall, f1-score ve support (o sınıftan
    kaç gerçek örnek olduğu) değerlerini bir DataFrame olarak döndürür.

    precision: "model 'mechanical_touch' dediğinde ne kadar sıklıkla
                haklı çıkıyor" (yanlış alarm oranının tersi)
    recall:    "gerçekte 'mechanical_touch' olan pencerelerin ne kadarını
                model gerçekten yakalıyor" (kaçırma oranının tersi) -
                azınlık sınıflarda bu metrik özellikle önemli
    """
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report_dict).transpose()
    # "accuracy" satırı sınıf-bazlı değil, skaler bir değer - per-class
    # tabloda satır olarak görünmesi kafa karıştırır, bu yüzden çıkarıyoruz
    # (isteyen zaten overall_metrics() ile ayrıca alabilir).
    df = df.drop(index="accuracy", errors="ignore")
    return df


def compare_recall_before_after(
    y_true: np.ndarray,
    y_pred_before: np.ndarray,
    y_pred_after: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    before_label: str = "recall_before",
    after_label: str = "recall_after",
) -> pd.DataFrame:
    """İki farklı tahmin setinin (örn. SMOTE'suz vs SMOTE'lu) sınıf-bazlı
    recall değerlerini yan yana koyar ve farkı (delta) hesaplar.

    y_true AYNI olmalı (aynı veri setinin iki farklı modelle/ayarla
    tahmin edilmiş hali) - bu fonksiyon iki farklı deney SONUCUNU
    karşılaştırmak için, iki farklı veri setini değil.

    En yüksek delta'ya sahip sınıf en üstte görünür - "SMOTE hangi sınıfı
    en çok iyileştirdi" sorusuna doğrudan cevap.
    """
    if labels is None:
        labels = sorted(set(y_true))

    report_before = classification_report(
        y_true, y_pred_before, labels=labels, output_dict=True, zero_division=0
    )
    report_after = classification_report(
        y_true, y_pred_after, labels=labels, output_dict=True, zero_division=0
    )

    rows = [
        {
            "label": label,
            before_label: report_before[label]["recall"],
            after_label: report_after[label]["recall"],
            "delta": report_after[label]["recall"] - report_before[label]["recall"],
        }
        for label in labels
    ]
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Uctan uca demo: kasitli dengesiz bir veri setinde, SMOTE'suz ve
    # SMOTE'lu modelin azinlik sinif recall'unu karsilastiriyoruz - bu,
    # imbalance_handling.py'nin GERCEKTEN ise yaradiginin kaniti.
    from models.classical_models import get_out_of_fold_predictions
    from models.imbalance_handling import make_smote_resampler

    rng = np.random.default_rng(5)
    X_rows, y_rows, group_rows = [], [], []
    for plant_idx in range(6):
        plant_id = f"plant_{plant_idx}"
        for _ in range(20):
            X_rows.append([0.0, 0.001, -0.05, 0.05, 0.0, 0.3])
            y_rows.append("baseline")
            group_rows.append(plant_id)
        for _ in range(12):
            X_rows.append([0.05, 0.01, -0.1, 3.0, 20.0, 700.0])
            y_rows.append("mechanical_touch")
            group_rows.append(plant_id)
        for _ in range(3):  # kasitli ciddi azinlik sinif, baseline'a YAKIN (zor sinif)
            X_rows.append([0.01, 0.0015, -0.06, 0.07, 0.3, 0.6])
            y_rows.append("chemical_salt_stress")
            group_rows.append(plant_id)

    # Daha yuksek gurultu -> siniflar arasinda gercekci bir ortusme olusuyor,
    # boylece SMOTE'un azinlik sinif recall'una etkisi GORULEBILIR hale gelir.
    X = np.array(X_rows) + rng.normal(0, 0.35, size=(len(X_rows), 6))
    y = np.array(y_rows)
    groups = np.array(group_rows)

    y_true, y_pred_no_smote, _ = get_out_of_fold_predictions("decision_tree", X, y, groups, n_splits=5)
    _, y_pred_with_smote, _ = get_out_of_fold_predictions(
        "decision_tree", X, y, groups, n_splits=5, resampler=make_smote_resampler(k_neighbors=2)
    )

    print("SMOTE'SUZ genel metrikler:", overall_metrics(y_true, y_pred_no_smote))
    print("SMOTE'LU  genel metrikler:", overall_metrics(y_true, y_pred_with_smote))

    print("\nSMOTE'suz karisiklik matrisi:")
    print(confusion_matrix_report(y_true, y_pred_no_smote))

    print("\nSinif-bazli recall karsilastirmasi (SMOTE oncesi vs sonrasi):")
    print(compare_recall_before_after(y_true, y_pred_no_smote, y_pred_with_smote))