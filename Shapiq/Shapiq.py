"""
uncertainty_shapiq_pipeline.py
================================================================================
Gradient-Boosting-Ensemble zur Unsicherheitsquantifizierung (angelehnt an
'probly') mit Shapley-basierter Erklärbarkeit via 'shapiq'.

Pipeline:
  1. Laden/Erzeugen des Datensatzes (X_1..X_10 -> Y)
  2. Training eines LightGBM-Ensembles:
       - mehrere bootstrap-trainierte Modelle -> Punktvorhersage + EPISTEMISCHE
         Unsicherheit (Varianz über Ensemble-Mitglieder)
       - Quantils-Modelle (z.B. 10%/90%) -> ALEATORISCHE Unsicherheit
         (datengetriebenes Rauschen, aus Quantilsabstand geschätzt)
  3. Eigene shapiq.Game-Subklassen für: Vorhersage, aleatorische und
     epistemische Unsicherheit, jeweils mit Mean-Imputation für
     out-of-coalition Features.
  4. Berechnung von Shapley-Werten für ausgewählte (besonders unsichere)
     Testpunkte.
  5. Visuelle Gegenüberstellung von Vorhersage, Unsicherheitszerlegung und
     Feature-Attributionen als PNG-Dateien.

Ausführen:
    python uncertainty_shapiq_pipeline.py
================================================================================
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns98.59
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import shapiq

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Globale Konfiguration
# --------------------------------------------------------------------------- #

RANDOM_STATE: int = 42
DATA_PATH: str = "synthetic_uncertainty_dataset.csv"
N_FEATURES: int = 10
FEATURE_NAMES: List[str] = [f"X_{i}" for i in range(1, N_FEATURES + 1)]
OUTPUT_DIR: str = "plots_uncertainty"

np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", context="talk")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 1. Datensatz laden / erzeugen
# =============================================================================

def generate_synthetic_dataset(path: str, n_samples: int = 2000) -> pd.DataFrame:
    """Erzeugt einen synthetischen Datensatz mit nichtlinearem Signal und
    heteroskedastischem (aleatorischem) Rauschen, falls keine CSV vorliegt.

    Args:
        path: Zielpfad der CSV-Datei.
        n_samples: Anzahl zu erzeugender Beobachtungen.

    Returns:
        Erzeugter DataFrame mit Spalten X_1..X_10 und Y.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.uniform(-3, 3, size=(n_samples, N_FEATURES))

    # Nichtlineare Signalkomponente: nur ein Teil der Features ist relevant
    signal = (
        2.0 * np.sin(X[:, 0])
        + 0.5 * X[:, 1] ** 2 * np.sign(X[:, 1])
        + 1.0 * X[:, 2] * X[:, 3]
        - 0.8 * np.abs(X[:, 4])
    )

    # Heteroskedastisches Rauschen: Streuung hängt von X_6 ab
    # -> erzeugt feature-abhängige ALEATORISCHE Unsicherheit
    noise_std = 0.3 + 0.9 * np.abs(X[:, 5])
    noise = rng.normal(0.0, noise_std)

    Y = signal + noise

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["Y"] = Y
    df.to_csv(path, index=False)
    print(f"[Info] Synthetischer Datensatz gespeichert unter '{path}'.")
    return df


def load_dataset(path: str) -> pd.DataFrame:
    """Lädt den Unsicherheits-Datensatz oder generiert ihn bei Bedarf."""
    if not os.path.exists(path):
        print(f"[Info] '{path}' nicht gefunden -> generiere synthetischen Ersatz.")
        return generate_synthetic_dataset(path)
    df = pd.read_csv(path)
    missing = set(FEATURE_NAMES + ["Y"]) - set(df.columns)
    if missing:
        raise ValueError(f"Datensatz fehlen Spalten: {missing}")
    return df



# =============================================================================
# 2. Gradient-Boosting-Ensemble für Unsicherheitsquantifizierung
# =============================================================================

@dataclass
class UncertaintyEnsemble:
    """Ensemble aus Gradient-Boosting-Modellen zur Zerlegung der
    Vorhersageunsicherheit in epistemische und aleatorische Komponenten.

    Epistemische Unsicherheit:
        Varianz der Punktvorhersagen über mehrere, auf Bootstrap-Stichproben
        trainierte LightGBM-Modelle ("Deep Ensemble"-Prinzip).

    Aleatorische Unsicherheit:
        Aus zwei Quantils-Modellen (z.B. 10%/90%-Quantil) abgeleitete
        Streuungsschätzung, die inhärentes Datenrauschen abbildet.
    """

    n_estimators: int = 15
    quantile_lower: float = 0.1
    quantile_upper: float = 0.9
    random_state: int = RANDOM_STATE
    lgb_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )
    )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UncertaintyEnsemble":
        """Trainiert alle Ensemble-Mitglieder sowie die Quantilsmodelle."""
        rng = np.random.default_rng(self.random_state)
        n = len(X)

        # --- Ensemble für Punktvorhersage + epistemische Unsicherheit ---
        self.members_: List[lgb.LGBMRegressor] = []
        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 1_000_000))
            idx = rng.integers(0, n, size=n)  # Bootstrap-Resampling
            X_boot, y_boot = X.iloc[idx], y.iloc[idx]
            model = lgb.LGBMRegressor(**self.lgb_params, random_state=seed)
            model.fit(X_boot, y_boot)
            self.members_.append(model)

        # --- Quantilsmodelle für aleatorische Unsicherheit ---
        self.model_lower_ = lgb.LGBMRegressor(
            **{**self.lgb_params, "objective": "quantile", "alpha": self.quantile_lower},
            random_state=self.random_state,
        )
        self.model_lower_.fit(X, y)

        self.model_upper_ = lgb.LGBMRegressor(
            **{**self.lgb_params, "objective": "quantile", "alpha": self.quantile_upper},
            random_state=self.random_state,
        )
        self.model_upper_.fit(X, y)

        self.feature_names_: List[str] = list(X.columns)
        return self

    def predict_members(self, X: pd.DataFrame) -> np.ndarray:
        """Vorhersagen aller Ensemble-Mitglieder. Shape: (n_members, n_samples)."""
        return np.stack([m.predict(X) for m in self.members_], axis=0)

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Berechnet Punktvorhersage, epistemische, aleatorische und
        Gesamtunsicherheit für gegebene Eingaben.

        Returns:
            Dict mit Keys 'mean', 'epistemic_std', 'aleatoric_std', 'total_std'.
        """
        member_preds = self.predict_members(X)
        mean_pred = member_preds.mean(axis=0)
        epistemic_std = member_preds.std(axis=0)

        q_lower = self.model_lower_.predict(X)
        q_upper = self.model_upper_.predict(X)

        # Umrechnung des Quantilsabstands in eine Standardabweichung unter
        # Normalverteilungsannahme: (q_upper - q_lower) = z * sigma
        z = norm.ppf(self.quantile_upper) - norm.ppf(self.quantile_lower)
        aleatoric_std = np.clip((q_upper - q_lower) / z, a_min=1e-6, a_max=None)

        total_std = np.sqrt(epistemic_std**2 + aleatoric_std**2)

        return {
            "mean": mean_pred,
            "epistemic_std": epistemic_std,
            "aleatoric_std": aleatoric_std,
            "total_std": total_std,
        }


# =============================================================================
# 3. shapiq-Wertfunktionen (eigene Game-Subklassen)
# =============================================================================

class BaseUncertaintyGame(shapiq.Game):
    """Basisklasse für kooperative Spiele über Feature-Koalitionen S.

    Imputationsstrategie: Für Features *außerhalb* der Koalition S werden
    Werte aus einem Hintergrund-Datensatz eingesetzt (Marginal-Erwartungswert
    über den Hintergrund), Features *innerhalb* von S behalten den Wert des
    zu erklärenden Datenpunkts x_explain. Der Spielwert ist der über den
    Hintergrund gemittelte Ensemble-Output für die gewählte Zielgröße.
    """

    def __init__(
        self,
        ensemble: UncertaintyEnsemble,
        x_explain: pd.Series,
        background_data: pd.DataFrame,
        target: str,
        normalize: bool = True,
    ) -> None:
        self.ensemble = ensemble
        self.x_explain = x_explain
        self.feature_names = ensemble.feature_names_
        self.target = target  # "mean" | "epistemic_std" | "aleatoric_std"

        # Hintergrunddaten (z.B. Stichprobe der Trainingsdaten) für Imputation
        self.background = background_data[self.feature_names].reset_index(drop=True)

        super().__init__(n_players=len(self.feature_names), normalize=normalize)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Berechnet den Spielwert für jede übergebene Koalition.

        Args:
            coalitions: bool-Array der Form (n_coalitions, n_players); True
                bedeutet, dass das jeweilige Feature Teil der Koalition S ist.

        Returns:
            Array der Länge n_coalitions mit den zugehörigen Werten v(S).
        """
        n_coalitions = coalitions.shape[0]
        values = np.zeros(n_coalitions, dtype=float)

        for i in range(n_coalitions):
            coalition_mask = coalitions[i]
            batch = self.background.copy()

            # In-Coalition-Features auf den Wert des Erklärungspunkts setzen
            in_coalition_features = np.array(self.feature_names)[coalition_mask]
            for feat in in_coalition_features:
                batch[feat] = self.x_explain[feat]

            preds = self.ensemble.predict(batch)[self.target]
            values[i] = preds.mean()

        return values


class PredictionGame(BaseUncertaintyGame):
    """Spiel zur Erklärung der Punktvorhersage (Erwartungswert)."""

    def __init__(self, ensemble: UncertaintyEnsemble, x_explain: pd.Series,
                 background_data: pd.DataFrame) -> None:
        super().__init__(ensemble, x_explain, background_data, target="mean")


class EpistemicUncertaintyGame(BaseUncertaintyGame):
    """Spiel zur Erklärung der epistemischen (Modell-)Unsicherheit."""

    def __init__(self, ensemble: UncertaintyEnsemble, x_explain: pd.Series,
                 background_data: pd.DataFrame) -> None:
        super().__init__(ensemble, x_explain, background_data, target="epistemic_std")


class AleatoricUncertaintyGame(BaseUncertaintyGame):
    """Spiel zur Erklärung der aleatorischen (Daten-)Unsicherheit."""

    def __init__(self, ensemble: UncertaintyEnsemble, x_explain: pd.Series,
                 background_data: pd.DataFrame) -> None:
        super().__init__(ensemble, x_explain, background_data, target="aleatoric_std")


# =============================================================================
# 4. Shapley-Wert-Berechnung mit shapiq
# =============================================================================

def compute_shapley_values(game: shapiq.Game, n_players: int,
                            budget: int = 2**12) -> np.ndarray:
    """Berechnet Shapley-Werte erster Ordnung für ein shapiq.Game.

    Für <= ~12 Spieler wird exakt gerechnet; bei API-Abweichungen bzw. zu
    großem Suchraum wird auf eine Permutations-Approximation ausgewichen.

    Args:
        game: Instanz einer shapiq.Game-Subklasse.
        n_players: Anzahl der Features (Spieler).
        budget: Approximations-Budget als Fallback.

    Returns:
        Array der Länge n_players mit den Shapley-Werten je Feature.
    """
    try:
        computer = shapiq.ExactComputer(n_players=n_players, game=game)
        interaction_values = computer(index="SV", order=1)
    except Exception as exc:  # robust gegenüber kleineren API-Unterschieden
        print(f"[Hinweis] ExactComputer nicht nutzbar ({exc}); "
              f"verwende Permutations-Approximation.")
        approximator = shapiq.PermutationSamplingSV(
            n=n_players, random_state=RANDOM_STATE
        )
        interaction_values = approximator.approximate(budget=budget, game=game)

    values = np.array([interaction_values[(i,)] for i in range(n_players)])
    return values


# =============================================================================
# 5. Visualisierung
# =============================================================================

def plot_uncertainty_overview(results_df: pd.DataFrame, filename: str) -> None:
    """Gegenüberstellung von Punktvorhersage (mit Gesamtunsicherheitsband)
    und der Zerlegung in epistemische vs. aleatorische Unsicherheit."""
    sorted_df = results_df.sort_values("y_pred").reset_index(drop=True)
    x_axis = np.arange(len(sorted_df))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # --- Links: Vorhersage + Gesamtunsicherheit ---
    ax = axes[0]
    ax.scatter(x_axis, sorted_df["y_true"], s=8, alpha=0.3, color="gray",
               label="Wahre Werte")
    ax.plot(x_axis, sorted_df["y_pred"], color="C0", lw=1.5, label="Vorhersage")
    ax.fill_between(
        x_axis,
        sorted_df["y_pred"] - sorted_df["total_std"],
        sorted_df["y_pred"] + sorted_df["total_std"],
        color="C0", alpha=0.2, label="± 1 Gesamt-Std",
    )
    ax.set_title("Vorhersage mit Gesamtunsicherheit")
    ax.set_xlabel("Testpunkte (sortiert nach Vorhersage)")
    ax.set_ylabel("Y")
    ax.legend(loc="upper left", fontsize=10)

    # --- Rechts: Zerlegung epistemisch vs. aleatorisch ---
    ax = axes[1]
    ax.plot(x_axis, sorted_df["epistemic_std"], color="C1", lw=1.5,
             label="Epistemische Unsicherheit")
    ax.plot(x_axis, sorted_df["aleatoric_std"], color="C2", lw=1.5,
             label="Aleatorische Unsicherheit")
    ax.set_title("Zerlegung der Unsicherheit")
    ax.set_xlabel("Testpunkte (sortiert nach Vorhersage)")
    ax.set_ylabel("Standardabweichung")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot gespeichert] {out_path}")


def plot_shapley_attributions(
    feature_names: List[str],
    shap_mean: np.ndarray,
    shap_epistemic: np.ndarray,
    shap_aleatoric: np.ndarray,
    sample_id: int,
    filename: str,
) -> None:
    """Balkendiagramme der Shapley-Attributionen für Vorhersage, epistemische
    und aleatorische Unsicherheit, nebeneinander für einen Testpunkt."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    titles = ["Vorhersage (Erwartungswert)",
              "Epistemische Unsicherheit",
              "Aleatorische Unsicherheit"]
    data_list = [shap_mean, shap_epistemic, shap_aleatoric]
    colors = ["C0", "C1", "C2"]

    for ax, title, data, color in zip(axes, titles, data_list, colors):
        order = np.argsort(data)
        sns.barplot(
            x=data[order], y=np.array(feature_names)[order],
            ax=ax, color=color,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Shapley-Wert")

    fig.suptitle(f"Feature-Attributionen für Testpunkt #{sample_id}", fontsize=15)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot gespeichert] {out_path}")


def plot_shapley_heatmaps(
    all_results: List[Dict], feature_names: List[str], filename: str
) -> None:
    """Heatmap der Shapley-Werte für epistemische/aleatorische Unsicherheit
    über mehrere ausgewählte (besonders unsichere) Testpunkte hinweg."""
    sample_ids = [r["index"] for r in all_results]
    epi_matrix = np.array([r["epistemic"] for r in all_results])
    ale_matrix = np.array([r["aleatoric"] for r in all_results])

    # Größeres Layout und constrained_layout für saubere Abstände gewählt
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    sns.heatmap(
        epi_matrix,
        annot=True,
        fmt=".3f",
        cmap="rocket_r",
        xticklabels=feature_names,
        yticklabels=[f"#{i}" for i in sample_ids],
        ax=axes[0],
        cbar_kws={"label": "Shapley-Wert"},
        annot_kws={"size": 8},  # Kleinere Schriftgröße verhindert Überlappungen
        linewidths=0.5          # Dezente Trennlinien zwischen den Zellen
    )
    axes[0].set_title("Epistemische Unsicherheit – Attributionen")
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(
        ale_matrix,
        annot=True,
        fmt=".3f",
        cmap="mako_r",
        xticklabels=feature_names,
        yticklabels=[f"#{i}" for i in sample_ids],
        ax=axes[1],
        cbar_kws={"label": "Shapley-Wert"},
        annot_kws={"size": 8},  # Kleinere Schriftgröße verhindert Überlappungen
        linewidths=0.5          # Dezente Trennlinien zwischen den Zellen
    )
    axes[1].set_title("Aleatorische Unsicherheit – Attributionen")
    axes[1].tick_params(axis='x', rotation=45)

    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot gespeichert] {out_path}")


# =============================================================================
# 6. Hauptprogramm
# =============================================================================

def main() -> None:
    """Orchestriert Datenladung, Training, Erklärung und Visualisierung."""

    # --- 1. Daten laden und aufteilen ---
    df = load_dataset(DATA_PATH)
    X = df[FEATURE_NAMES]
    y = df["Y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # --- 2. Ensemble trainieren ---
    print("[Info] Trainiere Gradient-Boosting-Ensemble ...")
    ensemble = UncertaintyEnsemble(n_estimators=15)
    ensemble.fit(X_train, y_train)

    # --- 3. Vorhersagen + Unsicherheit für Testset ---
    preds = ensemble.predict(X_test)
    results_df = X_test.copy()
    results_df["y_true"] = y_test.values
    results_df["y_pred"] = preds["mean"]
    results_df["epistemic_std"] = preds["epistemic_std"]
    results_df["aleatoric_std"] = preds["aleatoric_std"]
    results_df["total_std"] = preds["total_std"]

    print("\n[Ergebnisübersicht – erste Zeilen]")
    print(results_df[["y_true", "y_pred", "epistemic_std",
                       "aleatoric_std", "total_std"]].head())

    # --- 4. Übersichtsplot (Vorhersage + Unsicherheitszerlegung) ---
    plot_uncertainty_overview(results_df, "01_uncertainty_overview.png")

    # --- 5. shapiq-Erklärungen für besonders unsichere Testpunkte ---
    background_sample = X_train.sample(
        n=min(50, len(X_train)), random_state=RANDOM_STATE
    )
    top_idx = results_df["total_std"].nlargest(3).index

    all_shapley_results: List[Dict] = []
    for idx in top_idx:
        x_explain = X_test.loc[idx]
        print(f"\n[Info] Berechne Shapley-Werte für Testpunkt #{idx} ...")

        shap_mean = compute_shapley_values(
            PredictionGame(ensemble, x_explain, background_sample), N_FEATURES
        )
        shap_epi = compute_shapley_values(
            EpistemicUncertaintyGame(ensemble, x_explain, background_sample),
            N_FEATURES,
        )
        shap_ale = compute_shapley_values(
            AleatoricUncertaintyGame(ensemble, x_explain, background_sample),
            N_FEATURES,
        )

        all_shapley_results.append({
            "index": idx, "mean": shap_mean,
            "epistemic": shap_epi, "aleatoric": shap_ale,
        })

        plot_shapley_attributions(
            feature_names=FEATURE_NAMES,
            shap_mean=shap_mean,
            shap_epistemic=shap_epi,
            shap_aleatoric=shap_ale,
            sample_id=idx,
            filename=f"02_shapley_attributions_sample_{idx}.png",
        )

    # --- 6. Zusammenfassende Heatmap über alle ausgewählten Punkte ---
    plot_shapley_heatmaps(
        all_shapley_results, FEATURE_NAMES, "03_shapley_heatmap_summary.png"
    )

    print(f"\n[Fertig] Alle Plots wurden im Ordner '{OUTPUT_DIR}/' gespeichert.")


if __name__ == "__main__":
    main()