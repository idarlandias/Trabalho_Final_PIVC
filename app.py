"""
Aplicação GUI para Classificação de Pneumonia em Raio-X de Tórax.
Utiliza o modelo DenseNet121 (Experimento 3) treinado com Transfer Learning.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = ["NORMAL", "PNEUMONIA"]

if getattr(sys, "frozen", False):
    # PyInstaller --onedir: _MEIPASS aponta para a pasta _internal
    BASE_DIR = os.path.dirname(sys.executable)
    _INTERNAL = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _INTERNAL = BASE_DIR

# Procura o modelo em vários locais possíveis
_candidates = [
    os.path.join(_INTERNAL, "results", "exp3", "best_model.pth"),
    os.path.join(BASE_DIR, "results", "exp3", "best_model.pth"),
]
MODEL_PATH = None
for _p in _candidates:
    if os.path.exists(_p):
        MODEL_PATH = _p
        break
if MODEL_PATH is None:
    MODEL_PATH = _candidates[0]  # fallback para mensagem de erro


# =============================================================================
# MODELO
# =============================================================================
def load_model(device):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    tensor = get_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()
    pred_idx = 1 if prob >= 0.5 else 0
    return CLASSES[pred_idx], prob


# =============================================================================
# INTERFACE GRÁFICA
# =============================================================================
BG = "#f0f2f5"
WHITE = "#ffffff"
HEADER_BG = "#1a5276"
GREEN = "#27ae60"
RED = "#e74c3c"
BLUE = "#2980b9"
GRAY = "#7f8c8d"
PREVIEW_SIZE = 300


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Classificador de Pneumonia - Raio-X de Torax")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.image_path = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._photo = None

        self._build_ui()

        # Centralizar na tela
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

        self._load_model_async()

    def _build_ui(self):
        # ===== HEADER =====
        header = tk.Frame(self, bg=HEADER_BG)
        header.pack(fill="x", ipady=12)
        tk.Label(
            header, text="Classificador de Pneumonia em Raio-X",
            font=("Segoe UI", 15, "bold"), fg="white", bg=HEADER_BG,
        ).pack()
        tk.Label(
            header, text="DenseNet121  |  Transfer Learning  |  Trabalho Final PIVC",
            font=("Segoe UI", 9), fg="#aed6f1", bg=HEADER_BG,
        ).pack()

        # ===== CORPO =====
        body = tk.Frame(self, bg=BG, padx=25, pady=15)
        body.pack()

        # --- Imagem ---
        img_frame = tk.LabelFrame(
            body, text=" Imagem de Raio-X ", font=("Segoe UI", 10),
            bg=WHITE, padx=10, pady=10,
        )
        img_frame.pack(fill="x")

        # Canvas com tamanho fixo em pixels
        self.img_canvas = tk.Canvas(
            img_frame, width=PREVIEW_SIZE, height=PREVIEW_SIZE,
            bg="#e8ecef", highlightthickness=0,
        )
        self.img_canvas.pack(pady=5)
        self.img_canvas.create_text(
            PREVIEW_SIZE // 2, PREVIEW_SIZE // 2,
            text="Nenhuma imagem carregada\n\nClique em 'Carregar Imagem' abaixo",
            font=("Segoe UI", 10), fill=GRAY, justify="center",
        )

        # --- Botões ---
        btn_frame = tk.Frame(body, bg=BG, pady=10)
        btn_frame.pack(fill="x")

        self.btn_load = tk.Button(
            btn_frame, text="  Carregar Imagem  ",
            font=("Segoe UI", 11, "bold"),
            bg=BLUE, fg="white", activebackground="#2471a3", activeforeground="white",
            bd=0, padx=25, pady=10, cursor="hand2",
            command=self._load_image,
        )
        self.btn_load.pack(side="left", padx=(0, 10), expand=True)

        self.btn_classify = tk.Button(
            btn_frame, text="  Classificar  ",
            font=("Segoe UI", 11, "bold"),
            bg=GREEN, fg="white", activebackground="#229954", activeforeground="white",
            bd=0, padx=25, pady=10, cursor="hand2",
            command=self._classify, state="disabled",
        )
        self.btn_classify.pack(side="left", padx=(10, 0), expand=True)

        # --- Resultado ---
        res_frame = tk.LabelFrame(
            body, text=" Resultado ", font=("Segoe UI", 10),
            bg=WHITE, padx=15, pady=12,
        )
        res_frame.pack(fill="x", pady=(5, 0))

        self.lbl_class = tk.Label(
            res_frame, text="---",
            font=("Segoe UI", 26, "bold"), bg=WHITE, fg=GRAY,
        )
        self.lbl_class.pack()

        self.lbl_prob = tk.Label(
            res_frame, text="Carregue uma imagem e clique em Classificar",
            font=("Segoe UI", 10), bg=WHITE, fg=GRAY,
        )
        self.lbl_prob.pack(pady=(0, 8))

        # Barra de confiança
        bar_labels = tk.Frame(res_frame, bg=WHITE)
        bar_labels.pack(fill="x", padx=30)
        tk.Label(bar_labels, text="NORMAL", font=("Segoe UI", 8, "bold"),
                 bg=WHITE, fg=GREEN).pack(side="left")
        tk.Label(bar_labels, text="PNEUMONIA", font=("Segoe UI", 8, "bold"),
                 bg=WHITE, fg=RED).pack(side="right")

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Conf.Horizontal.TProgressbar",
            troughcolor="#e8ecef", background=GRAY, thickness=20,
        )
        self.progress = ttk.Progressbar(
            res_frame, length=350, mode="determinate",
            style="Conf.Horizontal.TProgressbar",
        )
        self.progress.pack(padx=30, pady=(2, 5))

        # ===== STATUS BAR =====
        self.status = tk.Label(
            self, text="Carregando modelo DenseNet121...",
            font=("Segoe UI", 9), bg="#dfe6e9", fg="#636e72",
            anchor="w", padx=10, pady=5,
        )
        self.status.pack(fill="x", side="bottom")

    # ----- Carregar modelo -----
    def _load_model_async(self):
        def _load():
            try:
                self.model = load_model(self.device)
                dev = "GPU" if self.device.type == "cuda" else "CPU"
                self.after(0, self.status.config,
                           {"text": f"Modelo DenseNet121 pronto ({dev})"})
                if self.image_path:
                    self.after(0, self.btn_classify.config, {"state": "normal"})
            except Exception as e:
                self.after(0, self.status.config,
                           {"text": f"Erro ao carregar modelo: {e}"})

        threading.Thread(target=_load, daemon=True).start()

    # ----- Carregar imagem -----
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Selecionar imagem de Raio-X",
            filetypes=[
                ("Imagens", "*.jpeg *.jpg *.png *.bmp *.tiff"),
                ("Todos", "*.*"),
            ],
        )
        if not path:
            return

        self.image_path = path

        # Exibir no canvas
        img = Image.open(path).convert("RGB")
        img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        self._photo = ImageTk.PhotoImage(img)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(
            PREVIEW_SIZE // 2, PREVIEW_SIZE // 2,
            image=self._photo, anchor="center",
        )

        # Resetar resultado
        self.lbl_class.config(text="---", fg=GRAY)
        self.lbl_prob.config(text="Clique em 'Classificar'", fg=GRAY)
        self.progress["value"] = 0
        ttk.Style().configure("Conf.Horizontal.TProgressbar", background=GRAY)

        if self.model:
            self.btn_classify.config(state="normal")

        self.status.config(text=f"Imagem: {os.path.basename(path)}")

    # ----- Classificar -----
    def _classify(self):
        if not self.image_path or not self.model:
            return

        self.btn_classify.config(state="disabled")
        self.status.config(text="Classificando...")

        def _run():
            try:
                classe, prob = predict(self.model, self.image_path, self.device)
                self.after(0, self._show_result, classe, prob)
            except Exception as e:
                self.after(0, self._show_error, str(e))

        threading.Thread(target=_run, daemon=True).start()

    def _show_error(self, msg):
        self.lbl_class.config(text="ERRO", fg=RED)
        self.lbl_prob.config(text=msg, fg=RED)
        self.btn_classify.config(state="normal")
        self.status.config(text="Erro na classificacao")

    def _show_result(self, classe, prob):
        prob_pneu = prob * 100
        prob_norm = (1 - prob) * 100

        if classe == "PNEUMONIA":
            color = RED
            detail = f"Probabilidade de Pneumonia: {prob_pneu:.1f}%"
        else:
            color = GREEN
            detail = f"Probabilidade de Normal: {prob_norm:.1f}%"

        self.lbl_class.config(text=classe, fg=color)
        self.lbl_prob.config(text=detail, fg="#2c3e50")

        self.progress["value"] = prob_pneu
        bar_color = RED if prob_pneu >= 50 else GREEN
        ttk.Style().configure("Conf.Horizontal.TProgressbar", background=bar_color)

        self.btn_classify.config(state="normal")
        self.status.config(text=f"Resultado: {classe} ({prob_pneu:.1f}% pneumonia)")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()
