import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error

class HealthAIApp:
    def __init__(self, root):
        self.root = root
        root.title("HealthAI - Analyse de données de santé")
        root.geometry("900x600")
        root.configure(bg="#f0f0f0")

        # Frame menu à gauche avec fond coloré
        self.frame_menu = tk.Frame(root, bg="#2c3e50", width=220)
        self.frame_menu.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Titre menu
        title = tk.Label(self.frame_menu, text="Menu Principal", font=("Arial", 16, "bold"), bg="#2c3e50", fg="white")
        title.grid(row=0, column=0, pady=20, padx=10)

        # Boutons colorés dans le menu, avec couleur et hover simplifié
        button_specs = [
            ("Régression Linéaire", self.run_regression, "#1abc9c"),
            ("Clustering K-means", self.run_clustering, "#3498db"),
            ("ARIMA (Séries temporelles)", self.run_arima, "#9b59b6"),
            ("Forêt Aléatoire", self.run_random_forest, "#e67e22"),
            ("1.3.6 Validation Croisée", self.run_cross_validation, "#e74c3c"),
            ("Quitter", root.quit, "#c0392b")
        ]

        for i, (text, cmd, color) in enumerate(button_specs, start=1):
            btn = tk.Button(self.frame_menu, text=text, command=cmd, bg=color, fg="white",
                            activebackground="#34495e", font=("Arial", 12), bd=0, relief="raised")
            btn.grid(row=i, column=0, sticky="ew", padx=10, pady=8)

        # Zone graphique à droite
        self.frame_graph = tk.Frame(root, bg="white")
        self.frame_graph.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def clear_frame(self):
        for widget in self.frame_graph.winfo_children():
            widget.destroy()

    def _display_figure(self, fig):
        self.clear_frame()
        canvas = FigureCanvasTkAgg(fig, master=self.frame_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_regression(self):
        np.random.seed(0)
        ages = np.random.randint(20, 70, 100)
        cholest = 0.5 * ages + np.random.normal(0, 5, 100)

        model = LinearRegression()
        model.fit(ages.reshape(-1, 1), cholest)
        pred = model.predict(ages.reshape(-1, 1))

        mse = mean_squared_error(cholest, pred)

        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(ages, cholest, label="Données réelles", c="#16a085", alpha=0.7)
        ax.plot(ages, pred, color="#e74c3c", linewidth=2, label="Régression linéaire")
        ax.set_title(f"Régression linéaire : âge vs cholestérol\nMSE = {mse:.2f}", fontsize=16, color="#2c3e50")
        ax.set_xlabel("Âge", fontsize=12)
        ax.set_ylabel("Cholestérol", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        self._display_figure(fig)

        messagebox.showinfo("Résultat MSE", f"Erreur quadratique moyenne (MSE) : {mse:.2f}")

    def run_clustering(self):
        np.random.seed(1)
        imc = np.random.normal(25, 4, 100)
        tension = np.random.normal(130, 15, 100)
        data = np.column_stack((imc, tension))
        kmeans = KMeans(n_clusters=3, random_state=1)
        clusters = kmeans.fit_predict(data)

        fig, ax = plt.subplots(figsize=(6,5))
        scatter = ax.scatter(imc, tension, c=clusters, cmap="Set2", alpha=0.85, edgecolor='k')
        ax.set_title("Clustering K-means : IMC et Tension", fontsize=16, color="#2c3e50")
        ax.set_xlabel("IMC", fontsize=12)
        ax.set_ylabel("Tension artérielle", fontsize=12)
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        ax.grid(True, linestyle='--', alpha=0.4)

        self._display_figure(fig)

    def run_arima(self):
        np.random.seed(2)
        data = np.cumsum(np.random.normal(0, 1, 50)) + 100
        model = ARIMA(data, order=(2,1,2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)

        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(range(len(data)), data, label="Glycémie historique", color="#2980b9", linewidth=2)
        ax.plot(range(len(data), len(data)+10), forecast, label="Prévision ARIMA", color="#c0392b", linestyle='--', linewidth=2)
        ax.set_title("Prévision glycémie avec ARIMA", fontsize=16, color="#2c3e50")
        ax.set_xlabel("Temps (jours)", fontsize=12)
        ax.set_ylabel("Glycémie", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)

        self._display_figure(fig)

    def run_random_forest(self):
        np.random.seed(3)
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(200)*0.1 > 0).astype(int)

        clf = RandomForestClassifier(n_estimators=100, random_state=3)
        clf.fit(X, y)
        importances = clf.feature_importances_

        fig, ax = plt.subplots(figsize=(6,5))
        bars = ax.bar(range(len(importances)), importances, color=["#27ae60", "#2980b9", "#8e44ad", "#d35400", "#c0392b"])
        ax.set_title("Importance des variables - Forêt Aléatoire", fontsize=16, color="#2c3e50")
        ax.set_xlabel("Variables", fontsize=12)
        ax.set_ylabel("Importance", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        self._display_figure(fig)

    def run_cross_validation(self):
        np.random.seed(4)
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(200)*0.1 > 0).astype(int)

        clf = RandomForestClassifier(n_estimators=100, random_state=4)
        scores = cross_val_score(clf, X, y, cv=5)

        fig, ax = plt.subplots(figsize=(6,5))
        ax.bar(range(1, 6), scores, color="#2980b9")
        ax.set_title("Validation croisée - Forêt Aléatoire", fontsize=16, color="#2c3e50")
        ax.set_xlabel("Fold", fontsize=12)
        ax.set_ylabel("Précision", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        self._display_figure(fig)

# Ajouter à la fin du fichier
if __name__ == "__main__":
    root = tk.Tk()
    app = HealthAIApp(root)
    root.mainloop()