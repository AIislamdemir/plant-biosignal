# Bitki Biyoelektrik Sinyalinden Uyaran Sınıflandırma Sistemi

Bir bitkinin yaprak/gövdesinden ölçülen zayıf elektrofizyolojik sinyalleri
(mikrovolt-milivolt seviyesinde) **gerçek zamanlı** olarak toplayıp,
uygulanan dış uyaranı (dokunma, ışık değişimi, kuraklık/tuz/sıcaklık
stresi) anında sınıflandıran bir makine öğrenmesi sistemi.

Sistem bilimsel olarak savunulabilir olacak şekilde tasarlandı: bitkilere
insan duygusu atfeden dil (mutlu/üzgün/korkuyor vb.) kullanılmıyor, bunun
yerine "stres tepkisi", "aksiyon potansiyeli", "fizyolojik değişim" gibi
terminoloji tercih edildi.

## Özgünlük ekseni

Literatürdeki mevcut çalışmaların (Najdenovska 2021, Reissig 2021,
Sai/Sood/Saini 2022, Buss 2023/2025) tamamı **offline/batch**
sınıflandırma yapıyor: veri önce toplanıyor, sonra laboratuvarda ayrı bir
analiz aşamasında sınıflandırılıyor. Bu proje, **canlı akan sinyali
pencereleyip anında sınıflandıran ve sonucu gerçek zamanlı üreten**
uçtan uca bir sistem olması bakımından farklılaşıyor.

Bunu teknik olarak mümkün kılan temel karar: `data/preprocessing.py`'deki
filtreleme ve pencereleme mantığı, offline (eğitim) ve online (canlı
inference) modları arasında **aynı kod yolunu** paylaşıyor — ayrı ayrı
implementasyonlar değil. Bu, iki mod arasında zamanla oluşabilecek sessiz
bir tutarsızlığı (train/serve skew) yapısal olarak imkânsız kılıyor.
`inference.py` bu yüzden bir ek özellik değil, projenin ana teknik
iddiasıdır.

## Mimari

```
plant_biosignal/
├── config.py                      # Tüm parametrelerin tek gerçek kaynağı
├── requirements.txt
│
├── data/
│   ├── preprocessing.py           # IIR yüksek geçiren filtre + sliding window
│   │                               # (offline/online ortak çekirdek)
│   ├── labeling_protocol.py       # Zaman-bazlı, guard-band'li otomatik etiketleme
│   └── signal_acquisition.py      # Donanım (Plant SpikerBox/seri port) + simülasyon kaynağı
│
├── features/
│   ├── statistical_features.py    # mean, variance, min, max, skewness, kurtosis
│   ├── tsfresh_features.py        # [planlanan] Genişletilmiş özellik seti
│   └── mfcc_features.py           # Mel-frekans kepstral katsayıları (elle, librosa'sız)
│
├── models/
│   ├── classical_models.py        # RF, SVC, GP, DT, GaussianNB, KNN, XGBoost
│   │                               # + bitki-bazlı GroupKFold cross-validation
│   ├── imbalance_handling.py      # SMOTE + class_weight
│   ├── deep_model.py              # [planlanan] CNN sınıflandırıcı
│   └── automl_pipeline.py         # [planlanan] AutoML
│
├── utils/
│   ├── metrics.py                 # Karışıklık matrisi, sınıf-bazlı rapor
│   └── visualization.py           # [planlanan] Canlı gösterge
│
├── inference.py                   # Gerçek zamanlı sınıflandırma motoru
│
└── tests/                         # pytest birim testleri (68 test)
```

## Bilimsel zemin

| Konu | Referans | Bu projede kullanımı |
|---|---|---|
| Veri toplama / cihaz | Najdenovska ve ark., 2021 — 36 domates bitkisi, PhytlSigns cihazı, kuraklık/besin eksikliği/böcek istilası | `data/labeling_protocol.py` etiket seti ve protokol asgari değerleri |
| Sinyal ön işleme | Sai, Sood, Saini, 2022 — yüksek geçiren IIR filtre + sabit boyutlu (1024 örnek) pencereleme | `data/preprocessing.py` |
| Klasik model karşılaştırması | Reissig ve ark., 2021 — RF, SVC, GP, DT, GaussianNB, KNN | `models/classical_models.py` |
| XGBoost performansı | Najdenovska ve ark., 2021 — 1 dakikalık pencerede ~%85 doğruluk | `models/classical_models.py` |
| Sınıf dengesizliği | Chawla ve ark., 2002 (SMOTE); PLOS One, 2023 — çok sınıflı kimyasal uyaran sınıflandırması | `models/imbalance_handling.py` |
| Uzun-vadeli izleme donanımı | Buss ve ark., 2025 — PhytoNode, 10 Hz, güneş panelli, Bluetooth | `config.py` — `baseline_sampling_rate_hz` |
| Genişletilmiş özellik çıkarımı | Buss ve ark., 2025/2026 — tsfresh tabanlı yaklaşım | `features/tsfresh_features.py` (planlanan) |

**Şeffaflık notu:** `config.py`'deki filtre kesim frekansı (0.5 Hz) gibi
bazı sayısal parametreler, kaynak makalelerde birebir raporlanmadığı için
mühendislik pratiğine dayalı başlangıç değerleridir — makalelerden alınmış
kesin sayılar olarak sunulmamıştır, gerçek deney verisiyle
tune edilmesi gerekir.

## Bilimsel kısıtlar (nasıl uygulandı)

- **Antropomorfik dil yasağı** — `config.py`'deki `StimulusLabel` enum'u
  ve tüm kod tabanı, insan duygusu içeren terimler yerine fizyolojik
  terminoloji kullanıyor.
- **Ground truth netliği** — `data/labeling_protocol.py`'deki
  `label_window()` fonksiyonu SADECE deneyci tarafından kaydedilen uyaran
  zamanına bakar, sinyalin şekline bakarak yorum YAPMAZ. Uyaran geçiş
  anının etrafına bilinçli bir "guard band" konularak belirsiz pencereler
  veri setinden tamamen çıkarılıyor.
- **Bitki-bazlı değerlendirme** — `models/classical_models.py`'deki
  `evaluate_models_grouped_cv()`, `sklearn.GroupKFold` kullanarak aynı
  bitkiden gelen örneklerin hem train hem test setine düşmesini
  (data leakage) engelliyor.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Ön işleme (offline ve online mod aynı pipeline'ı paylaşır)

```python
from data.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(sampling_rate_hz=100.0)

# Offline (eğitim verisi hazırlarken):
windows = pipeline.process_offline(raw_signal_array)

# Online (canlı akışta, her yeni örnekte):
window = pipeline.process_sample(new_sample)  # None veya PreprocessedWindow döner
```

### Etiketleme

```python
from data.labeling_protocol import Trial, build_labeled_dataset
from config import StimulusLabel

trial = Trial(
    trial_id="plant0_touch_rep1",
    plant_id="plant_0",
    label=StimulusLabel.MECHANICAL_TOUCH,
    sampling_rate_hz=100.0,
    stimulus_onset_time_s=120.0,
)
labeled_windows = build_labeled_dataset([trial], {"plant0_touch_rep1": raw_signal})
```

### Özellik çıkarımı ve model eğitimi

```python
from features.statistical_features import build_feature_matrix
from models.classical_models import evaluate_models_grouped_cv, train_final_model

X, y, groups = build_feature_matrix(labeled_windows)
results = evaluate_models_grouped_cv(X, y, groups, n_splits=5)  # bitki-bazlı CV

trained_model = train_final_model("random_forest", X, y)
```

### Gerçek zamanlı sınıflandırma

```python
from inference import RealtimeClassifier

classifier = RealtimeClassifier(trained_model, sampling_rate_hz=100.0)

result = classifier.process_sample(new_sample)
if result is not None:
    print(result.label, result.inference_latency_s)
```

## Testler

```bash
pytest tests/ -v
```

## Durum

Uçtan uca zincir (ham sinyal → etiketleme → özellik çıkarımı → model →
gerçek zamanlı tahmin) çalışır durumda ve `inference.py`'deki demo ile
doğrulandı. `data/signal_acquisition.py`, Backyard Brains Plant
SpikerBox'ı (seri port üzerinden) ve donanımsız test/geliştirme için bir
simülasyon kaynağını aynı `SignalSource` arayüzü altında sunuyor - gerçek
donanım geldiğinde `inference.py` hiç değişmeden çalışmaya devam edecek.
Henüz eklenmemiş modüller yukarıdaki mimari şemada `[planlanan]` olarak
işaretlendi.