import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

class HealthAIApp:
    def __init__(self, root):
        self.root = root
        root.title("HealthAI - Analyse Intelligente de la Santé")
        root.geometry("1000x620")
        root.configure(bg="#ecf0f1")

        # Menu latéral
        self.menu = tk.Frame(root, bg="#2c3e50", width=220)
        self.menu.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Label(self.menu, text="HealthAI", font=("Arial", 20, "bold"),
                 fg="white", bg="#2c3e50").pack(pady=20)

        buttons = [
            ("Régression Linéaire", self.run_regression),
            ("Clustering K-means", self.run_clustering),
            ("Prévision ARIMA", self.run_arima),
            ("Forêt Aléatoire", self.run_random_forest),
            ("Validation croisée", self.run_prediction_bar),  # Nom mis à jour ici
            ("Quitter", root.quit)
        ]

        for txt, cmd in buttons:
            tk.Button(self.menu, text=txt, command=cmd,
                      bg="#1abc9c", fg="white", bd=0,
                      font=("Arial", 12), pady=10, padx=5,
                      activebackground="#16a085").pack(fill="x", pady=5, padx=10)

        # Formulaire utilisateur
        self.inputs = tk.Frame(root, bg="#ecf0f1")
        self.inputs.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.vars = {}
        for label in ["Âge", "IMC", "Tension", "Glycémie"]:
            frame = tk.Frame(self.inputs, bg="#ecf0f1")
            frame.pack(side=tk.LEFT, padx=10)
            tk.Label(frame, text=label, bg="#ecf0f1", font=("Arial", 11)).pack()
            var = tk.StringVar()
            entry = tk.Entry(frame, textvariable=var, width=10, font=("Arial", 11))
            entry.pack()
            self.vars[label] = var

        self.refresh_btn = tk.Button(self.inputs, text="Actualiser", command=self.refresh,
                                     bg="#2980b9", fg="white", font=("Arial", 11),
                                     padx=20, pady=5)
        self.refresh_btn.pack(side=tk.LEFT, padx=20)

        # Espace graphique
        self.graph = tk.Frame(root, bg="white")
        self.graph.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def clear_graph(self):
        for widget in self.graph.winfo_children():
            widget.destroy()

    def show_plot(self, fig):
        self.clear_graph()
        canvas = FigureCanvasTkAgg(fig, master=self.graph)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def get_user_inputs(self):
        try:
            age = float(self.vars["Âge"].get())
            imc = float(self.vars["IMC"].get())
            tension = float(self.vars["Tension"].get())
            glycemie = float(self.vars["Glycémie"].get())
            return age, imc, tension, glycemie
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez remplir tous les champs avec des valeurs valides.")
            return None

    def refresh(self):
        for var in self.vars.values():
            var.set("")

    def run_regression(self):
        data = self.get_user_inputs()
        if not data: return
        age, _, _, _ = data

        ages = np.random.randint(20, 70, 100)
        chol = 0.5 * ages + np.random.normal(0, 5, 100)

        model = LinearRegression()
        model.fit(ages.reshape(-1, 1), chol)
        pred = model.predict(ages.reshape(-1, 1))
        mse = mean_squared_error(chol, pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(ages, chol, c="#16a085", label="Données", alpha=0.6)
        ax.plot(ages, pred, c="#e74c3c", label="Régression")
        ax.set_title("Âge vs Cholestérol", fontsize=14)
        ax.set_xlabel("Âge")
        ax.set_ylabel("Cholestérol")
        ax.legend()
        ax.grid(True)

        self.show_plot(fig)
        messagebox.showinfo("MSE", f"Erreur quadratique moyenne : {mse:.2f}")

    def run_clustering(self):
        data = self.get_user_inputs()
        if not data: return
        _, imc, tension, _ = data

        imc_data = np.random.normal(imc, 3, 100)
        tension_data = np.random.normal(tension, 10, 100)
        X = np.column_stack((imc_data, tension_data))

        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(X)

        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap="Set2", edgecolor='k')
        ax.set_title("Clustering : IMC vs Tension")
        ax.set_xlabel("IMC")
        ax.set_ylabel("Tension artérielle")
        ax.grid(True)
        self.show_plot(fig)

    def run_arima(self):
        data = self.get_user_inputs()
        if not data: return
        _, _, _, gly = data

        series = np.cumsum(np.random.normal(0, 1, 50)) + gly
        model = ARIMA(series, order=(2, 1, 2))
        result = model.fit()
        forecast = result.forecast(10)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(series, label="Historique", c="#3498db")
        ax.plot(range(len(series), len(series)+10), forecast, label="Prévision", c="#e67e22")
        ax.set_title("Prévision Glycémie (ARIMA)")
        ax.set_xlabel("Temps (jours)")
        ax.set_ylabel("Glycémie")
        ax.legend()
        ax.grid(True)
        self.show_plot(fig)

    def run_random_forest(self):
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1]*0.5 + np.random.randn(200)*0.1 > 0).astype(int)
        model = RandomForestClassifier()
        model.fit(X, y)
        importances = model.feature_importances_

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(range(5), importances, color="#2ecc71")
        ax.set_title("Importance des variables")
        ax.set_xlabel("Variables")
        ax.set_ylabel("Importance")
        ax.grid(True)
        self.show_plot(fig)

    def run_prediction_bar(self):  # Remplacé par validation croisée
        data = self.get_user_inputs()
        if not data: return
        age, _, _, _ = data

        # Données simulées
        X = np.linspace(20, 70, 100).reshape(-1, 1)
        y = 0.6 * X.flatten() + np.random.normal(0, 5, 100)

        models = {
            "Régression Linéaire": LinearRegression(),
            "Forêt Aléatoire": RandomForestRegressor(n_estimators=100)
        }

        scores = {}
        for name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            mse_scores = -cv_scores
            scores[name] = np.mean(mse_scores)

        fig, ax = plt.subplots(figsize=(6, 5))
        names = list(scores.keys())
        values = list(scores.values())
        bars = ax.bar(names, values, color=["#3498db", "#2ecc71"])
        ax.set_ylabel("Erreur quadratique moyenne (MSE)")
        ax.set_title("Validation croisée : Régression vs Forêt")
        ax.grid(True, linestyle="--", alpha=0.5)

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.5,
                    f"{height:.2f}", ha='center', va='bottom', fontsize=10)

        self.show_plot(fig)

# Lancer l'application
if __name__ == "__main__":
    root = tk.Tk()
    app = HealthAIApp(root)
    root.mainloop()
