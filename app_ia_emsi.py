import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Palette avec contraste élevé et visibilité optimale
PALETTE = {
    "background": "#1F7470",        # teal foncé pour fond principal
    "card_bg": "#F0F8F1",           # vert très clair pour fenêtres modules
    "primary": "#005BBB",           # bleu vif saturé, boutons & accents
    "primary_dark": "#003D7A",      # bleu foncé pour hover et actif
    "text": "#0D0D0D",              # texte très sombre, presque noir
    "button_bg": "#005BBB",         # bleu vif pour boutons
    "button_hover": "#003D7A",      # bleu foncé sur hover
    "button_active": "#002651",     # bleu très foncé en clic
    "code_bg": "#E8EEF7",           # fond clair légèrement bleuté pour code
    "code_text": "#1B1B1B"          # texte sombre dans la zone code
}

FONT_TITLE = ("Segoe UI", 24, "bold")
FONT_SUBTITLE = ("Segoe UI", 14)
FONT_BUTTON = ("Segoe UI", 14, "bold")
FONT_CODE = ("Consolas", 11)

class ModuleWindow(tk.Toplevel):
    def __init__(self, title, parent):
        super().__init__(parent)
        self.title(title)
        self.geometry("700x600")
        self.configure(bg=PALETTE["card_bg"])
        
        tk.Label(self, text=title, font=FONT_TITLE, bg=PALETTE["card_bg"], fg=PALETTE["text"]).pack(pady=10)

        self.text_code = scrolledtext.ScrolledText(self, height=10, width=80, bg=PALETTE["code_bg"], fg=PALETTE["code_text"], font=FONT_CODE)
        self.text_code.pack(pady=10, padx=10, fill="x")

        self.btn_show_code = tk.Button(self, text="Afficher le code", command=self.show_code,
                                      bg=PALETTE["button_bg"], fg="white", font=FONT_BUTTON, activebackground=PALETTE["button_hover"], borderwidth=0)
        self.btn_show_code.pack(pady=5, ipadx=10, ipady=5)

        self.btn_run = tk.Button(self, text="Exécuter l'algorithme", command=self.run_algorithm,
                                 bg=PALETTE["button_bg"], fg="white", font=FONT_BUTTON, activebackground=PALETTE["button_hover"], borderwidth=0)
        self.btn_run.pack(pady=5, ipadx=10, ipady=5)

        self.label_result = tk.Label(self, text="", bg=PALETTE["card_bg"], fg=PALETTE["text"], font=FONT_SUBTITLE)
        self.label_result.pack(pady=10)

    def show_code(self):
        pass

    def run_algorithm(self):
        pass

class RegressionLineaireWindow(ModuleWindow):
    def __init__(self, parent):
        super().__init__("Régression Linéaire", parent)

    def show_code(self):
        code = '''import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.random.rand(200,1)*100
y = 3*X + 7 + np.random.randn(200,1)*10

model = LinearRegression()
model.fit(X, y)

print("Coef:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])'''
        self.text_code.delete(1.0, tk.END)
        self.text_code.insert(tk.END, code)

    def run_algorithm(self):
        X = np.random.rand(200,1)*100
        y = 3*X + 7 + np.random.randn(200,1)*10
        model = LinearRegression()
        model.fit(X, y)
        coef = model.coef_[0][0]
        intercept = model.intercept_[0]
        self.label_result.config(text=f"Coef: {coef:.2f}, Intercept: {intercept:.2f}")
        
        plt.scatter(X, y, label="Données")
        plt.plot(X, model.predict(X), color=PALETTE["primary_dark"], label="Modèle")
        plt.title("Régression Linéaire")
        plt.legend()
        plt.show()

class KMeansWindow(ModuleWindow):
    def __init__(self, parent):
        super().__init__("K-Means", parent)

    def show_code(self):
        code = '''from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0)
model = KMeans(n_clusters=3)
model.fit(X)

print("Centres:", model.cluster_centers_)'''
        self.text_code.delete(1.0, tk.END)
        self.text_code.insert(tk.END, code)

    def run_algorithm(self):
        X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0)
        model = KMeans(n_clusters=3)
        model.fit(X)
        centers = model.cluster_centers_
        self.label_result.config(text=f"Centres:\n{np.array2string(centers, precision=2)}")
        
        plt.scatter(X[:,0], X[:,1], c=model.labels_)
        plt.scatter(centers[:,0], centers[:,1], c='red', marker='x', s=100)
        plt.title("K-Means Clustering")
        plt.show()

class ARIMAWindow(ModuleWindow):
    def __init__(self, parent):
        super().__init__("ARIMA", parent)

    def show_code(self):
        code = '''import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

data = pd.Series([100 + i + np.random.randn() for i in range(200)])
model = ARIMA(data, order=(2,1,2))
result = model.fit()
print(result.summary())'''
        self.text_code.delete(1.0, tk.END)
        self.text_code.insert(tk.END, code)

    def run_algorithm(self):
        data = pd.Series([100 + i + np.random.randn() for i in range(200)])
        model = ARIMA(data, order=(2,1,2))
        result = model.fit()
        self.label_result.config(text="ARIMA exécuté, voir graphique diagnostic.")
        result.plot_diagnostics(figsize=(10,6))
        plt.show()

class RandomForestWindow(ModuleWindow):
    def __init__(self, parent):
        super().__init__("Random Forest", parent)

    def show_code(self):
        code = '''from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=200, n_features=4, n_classes=2)
model = RandomForestClassifier()
model.fit(X, y)

print(model.feature_importances_)'''
        self.text_code.delete(1.0, tk.END)
        self.text_code.insert(tk.END, code)

    def run_algorithm(self):
        X, y = make_classification(n_samples=200, n_features=4, n_classes=2)
        model = RandomForestClassifier()
        model.fit(X, y)
        importances = model.feature_importances_
        self.label_result.config(text="Importances: " + ", ".join(f"{imp:.2f}" for imp in importances))
        
        plt.bar(range(len(importances)), importances, color=PALETTE["primary"])
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importances - Random Forest")
        plt.show()

class ValidationCroiseeWindow(ModuleWindow):
    def __init__(self, parent):
        super().__init__("Validation Croisée", parent)

    def show_code(self):
        code = '''from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X, y, cv=5)

print("Scores:", scores)
print("Moyenne:", scores.mean())'''
        self.text_code.delete(1.0, tk.END)
        self.text_code.insert(tk.END, code)

    def run_algorithm(self):
        X, y = load_iris(return_X_y=True)
        model = LogisticRegression(max_iter=1000)
        scores = cross_val_score(model, X, y, cv=5)
        self.label_result.config(text=f"Scores: {np.round(scores,3)}\nMoyenne: {scores.mean():.3f}")
        
        plt.bar(range(1,6), scores, color=PALETTE["primary"])
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.title("Validation Croisée 5-fold")
        plt.ylim(0,1)
        plt.show()

class IAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Application IA - EMSI")
        self.geometry("420x430")
        self.configure(bg=PALETTE["background"])

        header = tk.Label(self, text="Sélectionnez un module", font=FONT_TITLE,
                          bg=PALETTE["background"], fg=PALETTE["text"])
        header.pack(pady=30)

        style = ttk.Style(self)
        style.theme_use('clam')

        style.configure("TButton",
                        background=PALETTE["button_bg"],
                        foreground="white",
                        font=FONT_BUTTON,
                        padding=10,
                        borderwidth=0)
        style.map("TButton",
                  background=[("active", PALETTE["button_hover"]), ("pressed", PALETTE["button_active"])])

        btns = [
            ("Régression Linéaire", RegressionLineaireWindow),
            ("K-Means", KMeansWindow),
            ("ARIMA", ARIMAWindow),
            ("Random Forest", RandomForestWindow),
            ("Validation Croisée", ValidationCroiseeWindow)
        ]

        for text, cls in btns:
            b = ttk.Button(self, text=text, command=lambda c=cls: c(self))
            b.pack(fill='x', padx=50, pady=10)

if __name__ == "__main__":
    app = IAApp()
    app.mainloop()
