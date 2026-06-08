import tkinter as tk
from tkinter import ttk, messagebox
import json
import datetime
import os
import math

BG_DEEP    = "#0d1117"
BG_CARD    = "#161b22"
BG_HOVER   = "#1c2128"
BORDER     = "#30363d"
TEXT_PRI   = "#e6edf3"
TEXT_SEC   = "#8b949e"
ACCENT_RED = "#e05d5d"
GREEN_FG   = "#3fb950"
GREEN_BG   = "#0d2318"
RED_BG     = "#2d1010"
BLUE_FG    = "#58a6ff"
BLUE_BG    = "#0c1d32"
GOLD       = "#d29922"

# impelement logic manually
def _matmul_vec(W, b, x): #y= w*x+b
    return [sum(W[i][j] * x[j] for j in range(len(x))) + b[i]
            for i in range(len(W))]

def _relu(x):
    return [max(0.0, v) for v in x]

def _sigmoid(x):
    return [1.0 / (1.0 + math.exp(-v)) for v in x]

def _scale_row(row, mean_, scale_):
    return [(row[i] - mean_[i]) / scale_[i] for i in range(len(row))]


class PurePythonANN: #read and connect the data in model_weights.json
    def __init__(self, weights):
        self.W1 = weights["fc1_weight"]   
        self.b1 = weights["fc1_bias"]
        self.W2 = weights["fc2_weight"]   
        self.b2 = weights["fc2_bias"]

    def predict_proba(self, x):
        h = _relu(_matmul_vec(self.W1, self.b1, x))
        out = _matmul_vec(self.W2, self.b2, h)
        return _sigmoid(out)[0]


class ScalerData:#read mean and std from scaler_params.json
    def __init__(self, params):
        self.mean_  = params["mean_"]
        self.scale_ = params["scale_"]

    def transform(self, row):
        return _scale_row(row, self.mean_, self.scale_)


def load_model(base_dir="."): #read names, shapes, threshold from model_meta.json
    meta_path = os.path.join(base_dir, "model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "model_meta.json not found.\n"
            "Please run model.py first to train and save the model."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    weights_path = os.path.join(base_dir, "model_weights.json")
    scaler_path  = os.path.join(base_dir, "scaler_params.json")
    missing = [p for p in (weights_path, scaler_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Model files not found:\n" +
            "\n".join(f"  - {p}" for p in missing) +
            "\n\nPlease run model.py first to train and save the model."
        )
    with open(weights_path) as f:
        net = PurePythonANN(json.load(f))
    with open(scaler_path) as f:
        scaler = ScalerData(json.load(f))
    meta["backend"] = "pure_python"
    meta.setdefault("threshold", 0.35)
    return net, scaler, meta


def run_inference(model, scaler, feature_names, slider_values): # get, clean, scale user input
    row = [slider_values[f] for f in feature_names]

    scaled = scaler.transform(row)
    prob   = model.predict_proba(scaled) # use model to predict

    threshold = slider_values.get("_threshold", 0.35)
    safe = prob < threshold
    if safe: #output
        label = "System Operating Normally"
        rec   = ("Borderline readings — monitor closely over the next 24 h."
                 if prob > (threshold - 0.05) else
                 "Machine is operating normally. No action required.")
    else:
        label = "Maintenance Required"
        if slider_values.get("anomaly_flag", 0):
            rec = "Anomaly detected. Shut down and inspect immediately."
        elif slider_values.get("vibration", 0) > 70:
            rec = "Excessive vibration. Check bearings and alignment."
        elif slider_values.get("temperature", 0) > 85:
            rec = "High temperature. Check cooling system and lubricant."
        elif slider_values.get("pressure", 0) > 70:
            rec = "High pressure. Inspect valves and pressure relief system."
        elif slider_values.get("energy_consumption", 0) > 200:
            rec = "High energy draw. Inspect motor and drive components."
        else:
            rec = "Multiple sensors elevated. Schedule maintenance soon."
    return label, round(prob * 100, 1), rec


class MaintenanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Predictive Maintenance System")
        self.configure(bg=BG_DEEP)
        self.resizable(True, True)
        self.geometry("840x870")  # رجعناها لحجم متناسق ومريح والـ Scroll هيرتب الباقي
        self.model = self.scaler = self.meta = None
        self.history = []
        self._build_ui()
        self._try_load_model()

    def _card(self, parent, **kw):
        return tk.Frame(parent, bg=BG_CARD,
                        highlightbackground=BORDER, highlightthickness=1, **kw)

    def _label(self, parent, text, size=13, color=TEXT_PRI, bold=False, **kw):
        return tk.Label(parent, text=text, bg=parent["bg"], fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

    def _slider_row(self, parent, label, var, frm, to, unit="", resolution=1, fmt="%d"):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=6)
        hdr = tk.Frame(row, bg=BG_CARD)
        hdr.pack(fill="x")
        self._label(hdr, label, 12, TEXT_SEC).pack(side="left")
        val_lbl = self._label(hdr, (fmt % var.get()) + (" " + unit if unit else ""), 12, ACCENT_RED, bold=True)
        val_lbl.pack(side="right")
        def on_move(v):
            val_lbl.config(text=(fmt % float(v)) + (" " + unit if unit else ""))
        ttk.Scale(row, from_=frm, to=to, orient="horizontal",
                  variable=var, command=on_move).pack(fill="x", pady=(4, 0))

    def _try_load_model(self):
        try:
            self.model, self.scaler, self.meta = load_model(".")
            n = len(self.meta["feature_names"])
            thr = self.meta.get("threshold", 0.35)
            self.status_bar.config(text=f"Model loaded  ({n} features)  |  threshold: {thr}", fg=GREEN_FG)
        except FileNotFoundError as e:
            self.status_bar.config(text="Model not found — run model.py first", fg=GOLD)
            messagebox.showwarning("Model not found", str(e))
        except Exception as e:
            self.status_bar.config(text=f"Load error: {e}", fg=ACCENT_RED)
            messagebox.showerror("Error loading model", str(e))

    def _build_ui(self):
        self._style()

        # scroll
        self.main_container = tk.Frame(self, bg=BG_DEEP)
        self.main_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.main_container, bg=BG_DEEP, bd=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        
        # فرام داخلي هيركب جواه كل محتوى الشاشة القديم
        outer = tk.Frame(self.canvas, bg=BG_DEEP, padx=24, pady=20)
        
        # update scroll container
        outer.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=outer, anchor="nw", width=820)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # شكل الscroll bar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # keep same widgets after updating
        hdr = tk.Frame(outer, bg=BG_DEEP)
        hdr.pack(fill="x", pady=(0, 16))
        self._label(hdr, "Smart Predictive Maintenance System", 17, TEXT_PRI, bold=True).pack(anchor="w")
        self._label(hdr, "Enter machine sensor data to predict its status", 12, TEXT_SEC).pack(anchor="w", pady=(4, 0))
        self.status_bar = self._label(hdr, "Loading model...", 11, TEXT_SEC)
        self.status_bar.pack(anchor="w", pady=(6, 0))

        # input sliders
        inp_card = self._card(outer)
        inp_card.pack(fill="x", pady=(0, 12))
        inp_inner = tk.Frame(inp_card, bg=BG_CARD, padx=18, pady=14)
        inp_inner.pack(fill="x")
        self._label(inp_inner, "Sensor Inputs", 12, TEXT_SEC).pack(anchor="w", pady=(0, 10))

        self.v_temp     = tk.DoubleVar(value=75)
        self.v_pressure = tk.DoubleVar(value=3.0)
        self.v_vib      = tk.DoubleVar(value=50)
        self.v_energy   = tk.DoubleVar(value=2.75)
        self.v_humidity = tk.DoubleVar(value=55)

        grid = tk.Frame(inp_inner, bg=BG_CARD)
        grid.pack(fill="x")
        left  = tk.Frame(grid, bg=BG_CARD)
        right = tk.Frame(grid, bg=BG_CARD)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        right.grid(row=0, column=1, sticky="nsew")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self._slider_row(left,  "Temperature (C)",           self.v_temp,      35,   122,  "C",    1,   "%d")
        self._slider_row(right, "Pressure (1-5 scale)",      self.v_pressure,   1.0,   5.0, "",    0.1, "%.1f")
        self._slider_row(left,  "Vibration (mm/s)",          self.v_vib,        0,    114,  "mm/s", 1,   "%d")
        self._slider_row(right, "Energy Consumption (0.5-5)",self.v_energy,     0.5,   5.0, "",    0.1, "%.1f")
        self._slider_row(left,  "Humidity (%)",              self.v_humidity,   30,    80,  "%",    1,   "%d")

        flag_frame = tk.Frame(right, bg=BG_CARD)
        flag_frame.pack(fill="x", pady=6)
        self._label(flag_frame, "Anomaly Flag", 12, TEXT_SEC).pack(anchor="w")
        self.v_anomaly = tk.StringVar(value="0")
        ttk.Combobox(flag_frame, textvariable=self.v_anomaly, values=["0", "1"],
                     state="readonly", width=10).pack(anchor="w", pady=(6, 0))
        
        btn_bar = tk.Frame(inp_card, bg=BG_CARD)
        btn_bar.pack(anchor="w", padx=18, pady=(0, 14))
        self.btn = tk.Button(btn_bar, text="Predict Machine Status", #predict button
                             command=self._run_predict,
                             bg=BG_HOVER, fg=TEXT_PRI,
                             activebackground="#2a2f3a", activeforeground=TEXT_PRI,
                             relief="flat", bd=0, padx=14, pady=8,
                             font=("Segoe UI", 12, "bold"), cursor="hand2",
                             highlightbackground=BORDER, highlightthickness=1)
        self.btn.pack(side="left", padx=(0, 10))
        tk.Button(btn_bar, text="Reload Model", #reload button
                  command=self._try_load_model,
                  bg=BG_DEEP, fg=TEXT_SEC,
                  activebackground=BG_HOVER, activeforeground=TEXT_PRI,
                  relief="flat", bd=0, padx=10, pady=8,
                  font=("Segoe UI", 11), cursor="hand2",
                  highlightbackground=BORDER, highlightthickness=1).pack(side="left")

        res_card = self._card(outer) # prediction result section
        res_card.pack(fill="x", pady=(0, 12))
        res_inner = tk.Frame(res_card, bg=BG_CARD, padx=18, pady=14)
        res_inner.pack(fill="x")
        self._label(res_inner, "Prediction Result", 14, TEXT_PRI, bold=True).pack(anchor="w", pady=(0, 8))
        self.result_box = tk.Frame(res_inner, bg=GREEN_BG,
                                   highlightbackground=GREEN_FG, highlightthickness=1)
        self.result_box.pack(fill="x")
        box_inner = tk.Frame(self.result_box, bg=GREEN_BG, padx=14, pady=10)
        box_inner.pack(fill="x")
        self.result_icon = self._label(box_inner, "v", 13, GREEN_FG, bold=True)
        self.result_icon.pack(side="left", padx=(0, 8))
        self.result_text = self._label(box_inner, "System Operating Normally", 13, GREEN_FG, bold=True) #message after prediction
        self.result_text.pack(side="left")
        self.result_prob = self._label(box_inner, "(prob: --)", 11, TEXT_SEC)
        self.result_prob.pack(side="right")

        rec_card = self._card(outer) # recommendation section
        rec_card.pack(fill="x", pady=(0, 12))
        rec_inner = tk.Frame(rec_card, bg=BG_CARD, padx=18, pady=14)
        rec_inner.pack(fill="x")
        self._label(rec_inner, "Recommendation", 14, TEXT_PRI, bold=True).pack(anchor="w", pady=(0, 8))
        self.rec_box = tk.Frame(rec_inner, bg=BLUE_BG,
                                highlightbackground=BLUE_FG, highlightthickness=1)
        self.rec_box.pack(fill="x")
        rec_b = tk.Frame(self.rec_box, bg=BLUE_BG, padx=14, pady=10)
        rec_b.pack(fill="x")
        self._label(rec_b, "v", 13, BLUE_FG, bold=True).pack(side="left", padx=(0, 8))
        self.rec_text = self._label(rec_b, "Press 'Predict' to get a recommendation.", 12, BLUE_FG) #message after recommendation
        self.rec_text.pack(side="left")

        hist_card = self._card(outer) # history section padding
        hist_card.pack(fill="x")
        hist_inner = tk.Frame(hist_card, bg=BG_CARD, padx=18, pady=14)
        hist_inner.pack(fill="x")
        self._label(hist_inner, "Prediction History (Last 5)", 14, TEXT_PRI, bold=True).pack(anchor="w", pady=(0, 8))
        cols  = ("Time", "Status", "Temp C", "Vibration", "Anomaly", "Prob %")
        col_w = (90, 190, 72, 80, 72, 72)
        hdr_row = tk.Frame(hist_inner, bg="#1c2128")
        hdr_row.pack(fill="x")
        for c, w in zip(cols, col_w): # loop for creating each iteration in the history section
            tk.Label(hdr_row, text=c, bg="#1c2128", fg=TEXT_SEC,
                     font=("Segoe UI", 11), width=w // 7,
                     anchor="w").pack(side="left", padx=(8, 0), pady=4)
        self.hist_frame = tk.Frame(hist_inner, bg=BG_CARD)
        self.hist_frame.pack(fill="x")
        self._label(self.hist_frame, "No predictions yet.", 11, TEXT_SEC).pack(anchor="w", padx=8, pady=6)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TScale", background=BG_CARD, troughcolor="#21262d",
                    slidercolor=ACCENT_RED, bordercolor=BG_CARD,
                    lightcolor=ACCENT_RED, darkcolor=ACCENT_RED)
        s.configure("TCombobox", fieldbackground="#161b22", background="#161b22",
                    foreground=TEXT_PRI, selectbackground="#21262d",
                    selectforeground=TEXT_PRI, bordercolor=BORDER, arrowcolor=TEXT_SEC)
        s.map("TCombobox",
              fieldbackground=[("readonly", "#161b22")],
              foreground=[("readonly", TEXT_PRI)])

    def _run_predict(self):
        if self.model is None:
            messagebox.showwarning("Model not loaded",
                "The trained model could not be found.\n"
                "Please run model.py first, then click 'Reload Model'.")
            return
        threshold = self.meta.get("threshold", 0.35) if self.meta else 0.35
        slider_values = {
            "temperature":        self.v_temp.get(),
            "pressure":           self.v_pressure.get(),
            "vibration":          self.v_vib.get(),
            "energy_consumption": self.v_energy.get(),
            "humidity":           self.v_humidity.get(),
            "anomaly_flag":       float(self.v_anomaly.get()),
            "_threshold":         threshold,
        }
        try:
            label, prob, rec = run_inference(
                self.model, self.scaler, self.meta["feature_names"], slider_values)
        except Exception as e:
            messagebox.showerror("Inference error", str(e))
            return
        safe = label == "System Operating Normally"
        r_bg = GREEN_BG if safe else RED_BG
        r_fg = GREEN_FG if safe else ACCENT_RED
        icon = "v" if safe else "x"
        for w in (self.result_box, self.result_box.winfo_children()[0]):
            w.config(bg=r_bg, highlightbackground=r_fg)
        for child in (self.result_icon, self.result_text, self.result_prob):
            child.config(bg=r_bg)
        self.result_icon.config(text=icon, fg=r_fg)
        self.result_text.config(text=label, fg=r_fg)
        self.result_prob.config(text=f"(prob: {prob}%)")
        rec_fg = BLUE_FG if safe else GOLD
        rec_bg = BLUE_BG if safe else "#1e1608"
        for w in (self.rec_box, self.rec_box.winfo_children()[0]):
            w.config(bg=rec_bg, highlightbackground=rec_fg)
        for child in self.rec_box.winfo_children()[0].winfo_children():
            child.config(fg=rec_fg, bg=rec_bg)
        self.rec_text.config(text=rec, fg=rec_fg, bg=rec_bg)
        self.history.insert(0, {
            "time":    datetime.datetime.now().strftime("%H:%M:%S"),
            "status":  "Safe" if safe else "Maintenance",
            "temp":    int(self.v_temp.get()),
            "vib":     int(self.v_vib.get()),
            "anomaly": int(self.v_anomaly.get()),
            "prob":    prob, "safe": safe,
        })
        self.history = self.history[:5]
        self._render_history()

    def _render_history(self):
        for w in self.hist_frame.winfo_children():
            w.destroy()
        col_w = (90, 190, 72, 80, 72, 72)
        for i, h in enumerate(self.history):
            row_bg = BG_CARD if i % 2 == 0 else "#1a1f27"
            row = tk.Frame(self.hist_frame, bg=row_bg)
            row.pack(fill="x")
            st_fg = GREEN_FG if h["safe"] else ACCENT_RED
            cells = [(h["time"], TEXT_SEC), (h["status"], st_fg),
                     (str(h["temp"]), TEXT_PRI), (str(h["vib"]), TEXT_PRI),
                     (str(h["anomaly"]), TEXT_PRI), (f"{h['prob']}%", GOLD)]
            for (txt, fg), w in zip(cells, col_w):
                tk.Label(row, text=txt, bg=row_bg, fg=fg,
                         font=("Segoe UI", 11), width=w // 7,
                         anchor="w").pack(side="left", padx=(8, 0), pady=4)


if __name__ == "__main__":
    app = MaintenanceApp()
    app.mainloop()