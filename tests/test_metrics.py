"""
tests/test_metrics.py
========================
utils/metrics.py icin birim testleri.

En kritik test: test_compare_recall_before_after_detects_improvement
-> SMOTE'un azinlik sinif recall'unu gercekten iyilestirdigini olcup
   olcemedigimizi kanitlar (imbalance_handling.py ile bu modulun
   birlikte anlamli calistigi kanit).
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.metrics import (
    compare_recall_before_after,
    confusion_matrix_report,
    overall_metrics,
    per_class_report,
)


# ---------------------------------------------------------------------------
# overall_metrics
# ---------------------------------------------------------------------------
class TestOverallMetrics:
    def test_perfect_predictions_score_one(self):
        y_true = np.array(["baseline", "mechanical_touch", "baseline", "mechanical_touch"])
        y_pred = y_true.copy()
        metrics = overall_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["f1_weighted"] == pytest.approx(1.0)

    def test_all_wrong_predictions_score_zero_accuracy(self):
        y_true = np.array(["baseline", "baseline", "mechanical_touch"])
        y_pred = np.array(["mechanical_touch", "mechanical_touch", "baseline"])
        metrics = overall_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# confusion_matrix_report
# ---------------------------------------------------------------------------
class TestConfusionMatrixReport:
    def test_diagonal_holds_correct_predictions(self):
        y_true = np.array(["baseline"] * 5 + ["mechanical_touch"] * 3)
        y_pred = np.array(["baseline"] * 5 + ["mechanical_touch"] * 3)
        cm = confusion_matrix_report(y_true, y_pred)
        assert cm.loc["true_baseline", "pred_baseline"] == 5
        assert cm.loc["true_mechanical_touch", "pred_mechanical_touch"] == 3

    def test_off_diagonal_captures_confusion(self):
        y_true = np.array(["drought_stress"] * 3)
        y_pred = np.array(["baseline", "baseline", "drought_stress"])
        cm = confusion_matrix_report(y_true, y_pred)
        assert cm.loc["true_drought_stress", "pred_baseline"] == 2
        assert cm.loc["true_drought_stress", "pred_drought_stress"] == 1

    def test_custom_label_order_respected(self):
        y_true = np.array(["baseline", "mechanical_touch"])
        y_pred = np.array(["baseline", "mechanical_touch"])
        cm = confusion_matrix_report(y_true, y_pred, labels=["mechanical_touch", "baseline"])
        assert list(cm.index) == ["true_mechanical_touch", "true_baseline"]


# ---------------------------------------------------------------------------
# per_class_report
# ---------------------------------------------------------------------------
class TestPerClassReport:
    def test_accuracy_row_is_excluded(self):
        y_true = np.array(["baseline", "mechanical_touch"])
        y_pred = np.array(["baseline", "mechanical_touch"])
        df = per_class_report(y_true, y_pred)
        assert "accuracy" not in df.index

    def test_support_column_matches_true_label_counts(self):
        y_true = np.array(["baseline"] * 7 + ["mechanical_touch"] * 3)
        y_pred = np.array(["baseline"] * 5 + ["mechanical_touch"] * 5)
        df = per_class_report(y_true, y_pred)
        assert df.loc["baseline", "support"] == 7
        assert df.loc["mechanical_touch", "support"] == 3


# ---------------------------------------------------------------------------
# compare_recall_before_after
# ---------------------------------------------------------------------------
class TestCompareRecallBeforeAfter:
    def test_compare_recall_before_after_detects_improvement(self):
        y_true = np.array(["baseline"] * 6 + ["chemical_salt_stress"] * 4)
        y_pred_before = np.array(
            ["baseline"] * 6 + ["baseline", "baseline", "baseline", "chemical_salt_stress"]
        )
        y_pred_after = np.array(
            ["baseline"] * 6
            + ["chemical_salt_stress", "chemical_salt_stress", "chemical_salt_stress", "baseline"]
        )

        comparison = compare_recall_before_after(y_true, y_pred_before, y_pred_after)
        minority_row = comparison[comparison["label"] == "chemical_salt_stress"].iloc[0]

        assert minority_row["recall_before"] == pytest.approx(0.25)
        assert minority_row["recall_after"] == pytest.approx(0.75)
        assert minority_row["delta"] == pytest.approx(0.5)
        assert comparison.iloc[0]["label"] == "chemical_salt_stress"

    def test_no_change_gives_zero_delta(self):
        y_true = np.array(["baseline", "mechanical_touch"])
        y_pred = np.array(["baseline", "mechanical_touch"])
        comparison = compare_recall_before_after(y_true, y_pred, y_pred)
        assert (comparison["delta"] == 0).all()