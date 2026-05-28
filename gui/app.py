import customtkinter as ctk
from tkinter import filedialog
from PIL import Image

from skin_analysis import analisis_kulit

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("900x700")
app.title("Skin Analysis AI")

selected_image = None


# =========================
# PILIH GAMBAR
# =========================
def upload_image():
    global selected_image

    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if path:
        selected_image = path

        image = ctk.CTkImage(
            light_image=Image.open(path),
            dark_image=Image.open(path),
            size=(300, 300)
        )

        image_label.configure(image=image, text="")


# =========================
# ANALISIS
# =========================
def run_analysis():

    if selected_image is None:
        result_label.configure(text="Pilih gambar dulu!")
        return

    result = analisis_kulit(selected_image)

    text = f"""
Skin Score : {result['score']:.2f}

Oil Level  : {result['oil']:.2f}%
Texture    : {result['texture']:.2f}%
Color Var  : {result['color']:.2f}
"""

    result_label.configure(text=text)


# =========================
# TITLE
# =========================
title = ctk.CTkLabel(
    app,
    text="Skin Analysis Application",
    font=("Arial", 28, "bold")
)
title.pack(pady=20)


# =========================
# IMAGE PREVIEW
# =========================
image_label = ctk.CTkLabel(
    app,
    text="Belum ada gambar",
    width=300,
    height=300
)
image_label.pack(pady=20)


# =========================
# BUTTON UPLOAD
# =========================
upload_btn = ctk.CTkButton(
    app,
    text="Upload Gambar",
    command=upload_image
)
upload_btn.pack(pady=10)


# =========================
# BUTTON ANALISIS
# =========================
analyze_btn = ctk.CTkButton(
    app,
    text="Analisis Kulit",
    command=run_analysis
)
analyze_btn.pack(pady=10)


# =========================
# HASIL
# =========================
result_label = ctk.CTkLabel(
    app,
    text="Hasil analisis akan muncul di sini",
    font=("Arial", 18)
)
result_label.pack(pady=20)


app.mainloop()