# 🎙️ VoiceOrigin AI

> **DSP + Machine Learning pipeline for voice analysis, synthetic voice detection, and macro-regional origin estimation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()

---

## 📌 Objectif

**VoiceOrigin AI** analyse un signal vocal brut pour répondre à trois questions :

| Étape | Question | Méthode |
|-------|----------|---------|
| 🛑 Filtre | Est-ce une voix humaine ou synthétique ? | Jitter & Shimmer (Praat) |
| 🗣️ Classification | Quelle est la langue maternelle ? | Random Forest (F1, F2, MFCC1) |
| 📍 Géolocalisation | Quelle est l'origine macro-régionale ? | Distance acoustique euclidienne |

Le projet adopte une approche **légère et interprétable** basée sur le DSP classique et le ML traditionnel, sans modèles lourds de type LLM ou transformer.

---

## 🏗️ Architecture du pipeline

```
Signal audio (.wav)
        │
        ▼
┌───────────────────────────────┐
│  MODULE 1 : Extraction DSP    │  ← librosa + Praat (parselmouth)
│  F0, F1, F2, MFCC, Jitter,   │
│  Shimmer                      │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  ÉTAPE 3 : Filtre Anti-Robot  │  ← Jitter < 0.2% AND Shimmer < 0.5%
│  → REJET si voix synthétique  │
└──────────────┬────────────────┘
               │ humain validé
               ▼
┌───────────────────────────────┐
│  ÉTAPE 4 : Classification     │  ← Random Forest (100 arbres)
│  Langue prédite               │    Features : F1, F2, MFCC1
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│  ÉTAPE 5 : Géolocalisation    │  ← Distance euclidienne
│  Macro-région estimée         │    vs profils régionaux
└───────────────────────────────┘
```

---

## 📊 Résultats actuels

### Validation croisée (5-Fold)
```
Cross-Validation Accuracy : 75.8% (+/- 2.3%)
```

### Matrice de Confusion — Random Forest (200 samples test)

![Matrice de Confusion](assetsconfusion_matrix_rf.png.png)

| Langue | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| Anglais | ~0.67 | ~0.68 | ~0.67 |
| Arabe | ~0.82 | ~0.77 | ~0.79 |
| Français | ~0.82 | ~0.83 | ~0.82 |

### Séparabilité acoustique des formants

![Séparabilité Acoustique](assetsseparabilite_acoustique.png.png)

### Distribution MFCC1 par langue

![Distribution MFCC1](assetsdistribution_mfcc1.png.png)

---

## 🔧 Technologies

| Librairie | Rôle |
|-----------|------|
| `librosa` | Extraction MFCC, chargement audio |
| `praat-parselmouth` | Pitch, formants F1/F2, Jitter, Shimmer |
| `scikit-learn` | Random Forest, validation croisée |
| `scipy` | Distance euclidienne |
| `pandas / numpy` | Manipulation des données |
| `seaborn / matplotlib` | Visualisation |

---

## 🌍 Régions couvertes (9 macro-régions)

```
Arabe    → Afrique du Nord (Maghreb) | Moyen-Orient (Levant) | Golfe Arabique
Français → Europe du Sud | Amérique du Nord (Québec) | Afrique Subsaharienne
Anglais  → Amérique du Nord (USA/Canada) | Europe du Nord (UK/Irlande) | Océanie
```

---

## 🚀 Installation & Utilisation

### 1. Cloner le repo
```bash
git clone https://github.com/taiebdabbech-ops/VoiceOrigin-AI.git
cd VoiceOrigin-AI
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer le pipeline
```bash
python voice_origin_pipeline.py
```

### 4. Tester avec ton propre fichier audio
```python
from voice_origin_pipeline import VoiceAnalysisPipelineV2

pipeline = VoiceAnalysisPipelineV2()
pipeline.fit_dataset(n_samples=1000)
pipeline.execute_pipeline("mon_audio.wav")
```

---

## ⚠️ Limites actuelles & Prochaines étapes

### Limites honnêtes
- Dataset **simulé statistiquement** (pas de vraies voix annotées)
- 75.8% de précision → acceptable pour un prototype, insuffisant en production
- Pas de gestion du bruit de fond ni du multilinguisme

### Roadmap v2
- [ ] Intégration **Mozilla Common Voice** (vraies voix multilingues)
- [ ] Tests avec **XGBoost** et **SVM** (comparaison de classifieurs)
- [ ] Augmentation de données (time-stretching, pitch-shift, bruit additif)
- [ ] Interface web simple (FastAPI + React)
- [ ] Export du modèle entraîné (`.pkl` via joblib)

---

## 📁 Structure du projet

```
VoiceOrigin-AI/
├── voice_origin_pipeline.py   # Pipeline principal (DSP + ML)
├── requirements.txt           # Dépendances Python
├── .gitignore
├── README.md
└── assets/
    ├── dashboard.png              # Dashboard complet généré
    ├── separabilite_acoustique.png
    ├── distribution_mfcc1.png
    └── confusion_matrix_rf.png
```

---

## 👤 Auteur

**Taieb DABBECH**
Étudiant Ingénieur ICT — SUP'COM Tunis 

[![GitHub](https://img.shields.io/badge/GitHub-taiebdabbech--ops-black?logo=github)](https://github.com/taiebdabbech-ops)
[![Email](https://img.shields.io/badge/Email-taiebdabbech%40gmail.com-red?logo=gmail)](mailto:taiebdabbech@gmail.com)

---

## 📄 Licence

Ce projet est sous licence **MIT** — libre d'utilisation et de modification avec attribution.
