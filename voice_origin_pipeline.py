"""
VoiceOrigin AI - Pipeline d'analyse vocale DSP + Machine Learning
Auteur : Taieb DABBECH | SUP'COM Tunis - GDG AI/Web
GitHub : https://github.com/taiebdabbech-ops

Objectif :
    1. Détecter les voix synthétiques / deepfake (Jitter & Shimmer)
    2. Identifier la langue maternelle (Random Forest)
    3. Estimer l'origine macro-régionale (distance acoustique euclidienne)
"""

import os
import urllib.request
import warnings
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
import seaborn as sns
from parselmouth.praat import call
from scipy.spatial import distance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Configuration
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# =====================================================================
# 📥 MODULE 1 : PREPARATION DES FICHIERS AUDIO PHYSIQUES
# =====================================================================

def preparer_fichiers_audio():
    """Télécharge les voix humaines réelles et génère le signal robot synthétique."""
    print("📥 [DSP] Étape préliminaire : Préparation des fichiers physiques...")

    fichiers_sources = {
        "audio_humain_1.wav": "https://raw.githubusercontent.com/pdx-cs-sound/wavs/main/voice.wav",
        "audio_humain_2.wav": "https://raw.githubusercontent.com/pdx-cs-sound/wavs/main/voice-note.wav"
    }

    for nom, url in fichiers_sources.items():
        if not os.path.exists(nom):
            try:
                urllib.request.urlretrieve(url, nom)
                print(f"  ✅ '{nom}' téléchargé avec succès.")
            except Exception as e:
                print(f"  ❌ Échec du téléchargement pour {nom}: {e}")
        else:
            print(f"  ℹ️ '{nom}' déjà présent localement.")

    # Génération d'une voix robotique pure (aucun tremblement glottique)
    nom_robot = "audio_robot.wav"
    if not os.path.exists(nom_robot):
        sr, duration = 16000, 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.25 * np.sin(2 * np.pi * 300 * t)
        signal = (signal * 32767).astype(np.int16)
        import scipy.io.wavfile as wav
        wav.write(nom_robot, sr, signal)
        print(f"  ✅ '{nom_robot}' (Voix synthétique pure) généré avec succès.")


# =====================================================================
# 🛠️ MODULE 2 : PIPELINE DE PRODUCTION SPEECH-ANALYSIS (CLASSE POO)
# =====================================================================

class VoiceAnalysisPipelineV2:
    """
    Pipeline d'analyse vocale en cascade combinant :
        - DSP (extraction de features acoustiques via Praat/librosa)
        - Filtre anti-robot (Jitter/Shimmer)
        - Classification ML (Random Forest)
        - Géolocalisation acoustique (distance euclidienne)
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.macro_regions = {
            "Arabe":   ["Afrique du Nord (Maghreb)", "Moyen-Orient (Levant)", "Golfe Arabique"],
            "Français":["Europe du Sud", "Amérique du Nord (Québec)", "Afrique Subsaharienne"],
            "Anglais": ["Amérique du Nord (USA/Canada)", "Europe du Nord (UK/Irlande)", "Océanie"]
        }
        self.profiles  = {}
        self.df_voix   = None
        self.X_test    = None
        self.y_test    = None

    # ------------------------------------------------------------------
    # ÉTAPES 1 & 2 : EXTRACTION DE FEATURES (DSP)
    # ------------------------------------------------------------------

    def extract_features(self, audio_path: str) -> dict:
        """
        Extrait les signatures physiques critiques du signal audio.
        Features extraites :
            - F0  : fréquence fondamentale (pitch)
            - F1/F2 : formants (Burg algorithm via Praat)
            - Jitter / Shimmer : micro-fluctuations (anti-robot)
            - MFCC1 : 1er coefficient cepstral mel-fréquence
        """
        y, sr = librosa.load(audio_path, sr=16000)

        # MFCC
        mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=400, hop_length=160)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Parselmouth / Praat
        snd     = parselmouth.Sound(audio_path)
        pitch   = snd.to_pitch()
        f0      = call(pitch, "Get mean", 0, 0, "Hertz")
        formants= snd.to_formant_burg(time_step=0.010, window_length=0.025)
        f1      = call(formants, "Get mean", 1, 0, 0, "Hertz")
        f2      = call(formants, "Get mean", 2, 0, 0, "Hertz")

        # Jitter & Shimmer
        pp      = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter  = call(pp,       "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pp],"Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # Nettoyage NaN
        f0      = 150.0 if np.isnan(f0)      else f0
        f1      = 550.0 if np.isnan(f1)      else f1
        f2      = 1650.0 if np.isnan(f2)     else f2
        jitter  = 0.0   if np.isnan(jitter)  else jitter
        shimmer = 0.0   if np.isnan(shimmer) else shimmer

        return {"F0": f0, "F1": f1, "F2": f2,
                "Jitter": jitter, "Shimmer": shimmer, "MFCC1": mfccs_mean[1]}

    # ------------------------------------------------------------------
    # ENTRAÎNEMENT + VALIDATION CROISÉE
    # ------------------------------------------------------------------

    def fit_dataset(self, n_samples: int = 1000):
        """
        Génère un dataset simulé de n_samples profils vocaux
        et entraîne le classifieur Random Forest.
        Effectue une validation croisée 5-fold pour vérifier la robustesse.

        Note : les données sont simulées statistiquement à partir de la littérature
        phonétique (Hillenbrand et al., 1995 ; Fougeron & Smith, 1999).
        L'intégration de Common Voice et VoxCeleb est prévue en v2.
        """
        np.random.seed(42)
        data   = []
        langues = np.random.choice(list(self.macro_regions.keys()), size=n_samples)
        regions = [np.random.choice(self.macro_regions[l]) for l in langues]

        for lang, reg in zip(langues, regions):
            if   reg == "Afrique du Nord (Maghreb)":       f1, f2 = np.random.normal(590,40), np.random.normal(1420,80)
            elif reg == "Europe du Sud":                    f1, f2 = np.random.normal(500,35), np.random.normal(1800,75)
            elif reg == "Amérique du Nord (USA/Canada)":   f1, f2 = np.random.normal(720,50), np.random.normal(1550,90)
            elif reg == "Amérique du Nord (Québec)":       f1, f2 = np.random.normal(540,40), np.random.normal(1700,80)
            elif reg == "Europe du Nord (UK/Irlande)":     f1, f2 = np.random.normal(630,45), np.random.normal(1650,95)
            else:                                          f1, f2 = np.random.normal(600,50), np.random.normal(1600,100)

            mfcc1 = np.random.normal(13 if lang=="Arabe" else (23 if lang=="Français" else 18), 4)
            data.append([f1, f2, mfcc1, lang, reg])

        self.df_voix = pd.DataFrame(data, columns=["F1","F2","MFCC1","Langue","Macro_Region"])
        X, y = self.df_voix[["F1","F2","MFCC1"]], self.df_voix["Langue"]

        scores_cv = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        print(f"\n🔄 [Validation] Cross-Validation (5-Fold) : {scores_cv.mean()*100:.1f}% (+/- {scores_cv.std()*100:.1f}%)")

        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.profiles = self.df_voix.groupby("Macro_Region")[["F1","F2","MFCC1"]].mean().to_dict(orient="index")
        print(f"🧠 [ML] Random Forest entraîné sur {n_samples} profils.")

    # ------------------------------------------------------------------
    # PIPELINE DE DÉCISION EN CASCADE (ÉTAPES 3, 4, 5)
    # ------------------------------------------------------------------

    def execute_pipeline(self, audio_path: str):
        """
        Prend un fichier audio en entrée et exécute :
            Étape 3 → Filtre anti-robot (Jitter/Shimmer)
            Étape 4 → Classification langue (Random Forest)
            Étape 5 → Géolocalisation (distance acoustique euclidienne)
        """
        if not os.path.exists(audio_path):
            print(f"\n⚠️  Fichier absent : '{audio_path}'. Ignoré.")
            return

        print(f"\n⚡ Pipeline → {audio_path}")
        feat = self.extract_features(audio_path)

        j_pct, s_pct = feat['Jitter'] * 100, feat['Shimmer'] * 100
        if j_pct < 0.2 and s_pct < 0.5:
            print(f"  ❌ Jitter={j_pct:.3f}% | Shimmer={s_pct:.3f}%")
            print("  🛑 REJET : Robot / Synthèse vocale détecté !")
            return

        print(f"  ✅ Humain validé (Jitter={j_pct:.3f}% | Shimmer={s_pct:.3f}%)")

        X_pred = pd.DataFrame([{"F1": feat['F1'], "F2": feat['F2'], "MFCC1": feat['MFCC1']}])
        langue_predite = self.model.predict(X_pred)[0]
        print(f"  🗣️  Langue prédite : {langue_predite}")

        min_dist, meilleure_region = float('inf'), None
        v_reel = np.array([feat['F1'], feat['F2'], feat['MFCC1']])
        for reg in self.macro_regions[langue_predite]:
            v_profil = np.array([self.profiles[reg]["F1"], self.profiles[reg]["F2"], self.profiles[reg]["MFCC1"]])
            dist = distance.euclidean(v_reel, v_profil)
            if dist < min_dist:
                min_dist, meilleure_region = dist, reg

        print(f"  📍 Origine estimée : {meilleure_region}")

    # ------------------------------------------------------------------
    # DASHBOARD DE VISUALISATION
    # ------------------------------------------------------------------

    def generate_dashboard(self, save_assets: bool = True):
        """
        Génère le tableau de bord d'évaluation :
            1. Séparabilité acoustique (scatter F1/F2)
            2. Distribution MFCC1 par langue (boxplot)
            3. Matrice de confusion Random Forest
        """
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        fig.suptitle("VoiceOrigin AI — Rapport d'Évaluation", fontsize=15, fontweight='bold')

        sns.scatterplot(ax=axes[0], data=self.df_voix, x="F1", y="F2",
                        hue="Macro_Region", alpha=0.7, palette="tab10")
        axes[0].set_title("1. Séparabilité Acoustique", fontweight='bold')
        axes[0].set_xlabel("F1 (Hz)")
        axes[0].set_ylabel("F2 (Hz)")

        sns.boxplot(ax=axes[1], data=self.df_voix, x="Langue", y="MFCC1", palette="Set3")
        axes[1].set_title("2. Distribution Discriminante MFCC1", fontweight='bold')
        axes[1].set_xlabel("Langue")
        axes[1].set_ylabel("MFCC1")

        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=self.model.classes_,
                    yticklabels=self.model.classes_,
                    ax=axes[2], cbar=False)
        axes[2].set_title("3. Matrice de Confusion (Random Forest)", fontweight='bold')
        axes[2].set_xlabel("Predicted")
        axes[2].set_ylabel("True")

        plt.tight_layout()

        if save_assets:
            os.makedirs("assets", exist_ok=True)
            plt.savefig("assets/dashboard.png", dpi=150, bbox_inches="tight")
            print("\n💾 Dashboard sauvegardé → assets/dashboard.png")

        plt.show()

        print("\n📝 RAPPORT MÉTRIQUE :")
        print("-" * 60)
        print(classification_report(self.y_test, y_pred))
        print("-" * 60)


# =====================================================================
# 🔥 POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":
    preparer_fichiers_audio()

    pipeline = VoiceAnalysisPipelineV2()
    pipeline.fit_dataset(n_samples=1000)

    fichiers_test = [
        "audio_humain_1.wav",
        "audio_humain_2.wav",
        "audio_robot.wav",
        "vrai_audio_arabe.wav",
        "vrai_audio_francais.wav"
    ]

    for fichier in fichiers_test:
        pipeline.execute_pipeline(fichier)

    pipeline.generate_dashboard()
