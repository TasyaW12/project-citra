import cv2
import customtkinter as ctk
import tkinter as tk
import os

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from skin_analysis import analisis_kulit


# ==========================================
# TEMA
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ==========================================
# WINDOW
# ==========================================
app = ctk.CTk()

app.geometry("1700x950")

app.title("Skin Analysis AI")


# ==========================================
# SCROLLABLE FRAME
# ==========================================
main_frame = ctk.CTkScrollableFrame(
    app,
    width=1650,
    height=900
)

main_frame.pack(
    fill="both",
    expand=True
)


selected_image = None


# ==========================================
# UPLOAD GAMBAR
# ==========================================
def upload_image():

    global selected_image
    global preview_img

    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if path:

        selected_image = path

        image = Image.open(path)

        image = image.resize((320, 320))

        preview_img = ImageTk.PhotoImage(image)

        image_label.configure(
            image=preview_img
        )

        image_label.image = preview_img


# ==========================================
# CLEAR IMAGE
# ==========================================
def clear_image():

    global selected_image

    selected_image = None

    image_label.configure(image="")
    image_label.image = None

    result_label.configure(text="")

    face_result.configure(image="")
    face_result.image = None

    oil_result.configure(image="")
    oil_result.image = None

    texture_result.configure(image="")
    texture_result.image = None

    color_result.configure(image="")
    color_result.image = None

    heatmap_result.configure(image="")
    heatmap_result.image = None

    result_frame.pack_forget()

    score_indicator.configure(text="")

    status_label.configure(
        text="Status : Siap"
    )

    progress_bar.set(0)


# ==========================================
# SAVE RESULT
# ==========================================
def save_result():

    if selected_image is None:

        messagebox.showwarning(
            "Warning",
            "Belum ada hasil analisis!"
        )

        return

    if not os.path.exists("outputs"):

        os.makedirs("outputs")

    result = analisis_kulit(selected_image)

    cv2.imwrite(
        "outputs/face_result.jpg",
        result["face"]
    )

    cv2.imwrite(
        "outputs/oil_result.jpg",
        result["oily_mask"]
    )

    cv2.imwrite(
        "outputs/texture_result.jpg",
        result["edges"]
    )

    cv2.imwrite(
        "outputs/color_result.jpg",
        cv2.cvtColor(
            result["color_result"],
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.imwrite(
        "outputs/heatmap_result.jpg",
        cv2.cvtColor(
            result["heatmap"],
            cv2.COLOR_RGB2BGR
        )
    )

    messagebox.showinfo(
        "Success",
        "Hasil berhasil disimpan di folder outputs/"
    )


# ==========================================
# OPEN CAMERA
# ==========================================
def open_camera():

    global selected_image
    global preview_img

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow(
            "Tekan SPACE untuk Capture",
            frame
        )

        key = cv2.waitKey(1)

        # SPACE
        if key == 32:

            selected_image = "camera_capture.jpg"

            cv2.imwrite(
                selected_image,
                frame
            )

            image = Image.open(selected_image)

            image = image.resize((320, 320))

            preview_img = ImageTk.PhotoImage(image)

            image_label.configure(
                image=preview_img
            )

            image_label.image = preview_img

            break

        # ESC
        elif key == 27:
            break

    cap.release()

    cv2.destroyAllWindows()


# ==========================================
# ANALISIS
# ==========================================
def run_analysis():

    global face_img
    global oil_img
    global texture_img
    global color_img
    global heatmap_img

    if selected_image is None:

        result_label.configure(
            text="Pilih gambar dulu!"
        )

        return

    try:

        # ==========================================
        # STATUS ANALISIS
        # ==========================================
        status_label.configure(
            text="Status : Sedang menganalisis..."
        )

        progress_bar.set(0.2)

        app.update()

        result = analisis_kulit(selected_image)

        progress_bar.set(0.7)

        app.update()

        if result is None:

            result_label.configure(
                text="Wajah tidak terdeteksi!"
            )

            return

        result_frame.pack(pady=30)

        # ==========================================
        # FACE IMAGE
        # ==========================================
        face_rgb = cv2.cvtColor(
            result["face"],
            cv2.COLOR_BGR2RGB
        )

        face_pil = Image.fromarray(face_rgb)

        face_pil = face_pil.resize((240, 240))

        face_img = ImageTk.PhotoImage(face_pil)

        face_result.configure(
            image=face_img
        )

        face_result.image = face_img

        # ==========================================
        # OIL IMAGE
        # ==========================================
        oil_pil = Image.fromarray(
            result["oily_mask"]
        )

        oil_pil = oil_pil.convert("RGB")

        oil_pil = oil_pil.resize((240, 240))

        oil_img = ImageTk.PhotoImage(oil_pil)

        oil_result.configure(
            image=oil_img
        )

        oil_result.image = oil_img

        # ==========================================
        # TEXTURE IMAGE
        # ==========================================
        texture_pil = Image.fromarray(
            result["edges"]
        )

        texture_pil = texture_pil.convert("RGB")

        texture_pil = texture_pil.resize((240, 240))

        texture_img = ImageTk.PhotoImage(texture_pil)

        texture_result.configure(
            image=texture_img
        )

        texture_result.image = texture_img

        # ==========================================
        # COLOR IMAGE
        # ==========================================
        color_pil = Image.fromarray(
            result["color_result"]
        )

        color_pil = color_pil.resize((240, 240))

        color_img = ImageTk.PhotoImage(color_pil)

        color_result.configure(
            image=color_img
        )

        color_result.image = color_img

        # ==========================================
        # HEATMAP IMAGE
        # ==========================================
        heatmap_pil = Image.fromarray(
            result["heatmap"]
        )

        heatmap_pil = heatmap_pil.resize((240, 240))

        heatmap_img = ImageTk.PhotoImage(heatmap_pil)

        heatmap_result.configure(
            image=heatmap_img
        )

        heatmap_result.image = heatmap_img

        # ==========================================
        # SKIN SCORE INDICATOR
        # ==========================================
        score = result["score"]

        if score >= 80:

            warna = "green"

        elif score >= 60:

            warna = "yellow"

        else:

            warna = "red"

        score_indicator.configure(
            text=f"SKIN SCORE : {score:.2f}",
            text_color=warna
        )

        # ==========================================
        # HASIL TEXT
        # ==========================================
        result_text = f"""
Skin Score : {result['score']:.2f} / 100

Oil Level  : {result['oil']:.2f}%
Texture    : {result['texture']:.2f}%
Color Var  : {result['color']:.2f}%
"""

        result_label.configure(
            text=result_text
        )

        # ==========================================
        # STATUS SELESAI
        # ==========================================
        progress_bar.set(1)

        status_label.configure(
            text="Status : Analisis selesai"
        )

    except Exception as e:

        result_label.configure(
            text=f"ERROR:\n{e}"
        )

        status_label.configure(
            text="Status : Gagal"
        )

        progress_bar.set(0)

# ==========================================
# TITLE
# ==========================================
title = ctk.CTkLabel(
    main_frame,
    text="Skin Analysis Application",
    font=("Arial", 42, "bold")
)

title.pack(pady=20)
# ==========================================
# STATUS ANALISIS
# ==========================================
status_label = ctk.CTkLabel(
    main_frame,
    text="Status : Siap",
    font=("Arial",18)
)

status_label.pack()

# ==========================================
# PROGRESS BAR
# ==========================================
progress_bar = ctk.CTkProgressBar(
    main_frame,
    width=500
)

progress_bar.set(0)

progress_bar.pack(
    pady=10
)

# ==========================================
# SKIN SCORE INDICATOR
# ==========================================
score_indicator = ctk.CTkLabel(
    main_frame,
    text="",
    font=("Arial",32,"bold")
)

score_indicator.pack(
    pady=10
)


# ==========================================
# PREVIEW IMAGE
# ==========================================
image_label = tk.Label(
    main_frame,
    bg="#2b2b2b"
)

image_label.pack(pady=20)


# ==========================================
# BUTTON FRAME
# ==========================================
button_frame = ctk.CTkFrame(
    main_frame,
    fg_color="transparent"
)

button_frame.pack(pady=15)


# ==========================================
# UPLOAD BUTTON
# ==========================================
upload_btn = ctk.CTkButton(
    button_frame,
    text="Upload Gambar",
    command=upload_image,
    width=220,
    height=45
)

upload_btn.grid(
    row=0,
    column=0,
    padx=10
)

# ==========================================
# CAMERA BUTTON
# ==========================================
camera_btn = ctk.CTkButton(
    button_frame,
    text="Buka Kamera",
    command=open_camera,
    width=220,
    height=45
)

camera_btn.grid(
    row=0,
    column=1,
    padx=10
)

# ==========================================
# ANALYZE BUTTON
# ==========================================
analyze_btn = ctk.CTkButton(
    button_frame,
    text="Analisis Kulit",
    command=run_analysis,
    width=220,
    height=45
)

analyze_btn.grid(
    row=0,
    column=2,
    padx=10
)


# ==========================================
# SAVE BUTTON
# ==========================================
save_btn = ctk.CTkButton(
    button_frame,
    text="Simpan Hasil",
    command=save_result,
    width=220,
    height=45,
    fg_color="green"
)

save_btn.grid(
    row=0,
    column=3,
    padx=10
)




# ==========================================
# RESULT TEXT
# ==========================================
result_label = ctk.CTkLabel(
    main_frame,
    text="",
    font=("Arial", 26)
)

result_label.pack(pady=30)

# ==========================================
# CLEAR BUTTON
# ==========================================
clear_btn = ctk.CTkButton(
    button_frame,
    text="Bersihkan",
    command=clear_image,
    width=220,
    height=45,
    fg_color="red"
)

clear_btn.grid(
    row=0,
    column=4,
    padx=10
)


# ==========================================
# RESULT FRAME
# ==========================================
result_frame = ctk.CTkFrame(
    main_frame,
    fg_color="transparent"
)

result_frame.pack_forget()


# ==========================================
# FACE
# ==========================================
face_title = ctk.CTkLabel(
    result_frame,
    text="Gambar Asli",
    font=("Arial", 20, "bold")
)

face_title.grid(
    row=0,
    column=0,
    pady=(0, 15)
)

face_result = tk.Label(
    result_frame,
    bg="#2b2b2b"
)

face_result.grid(
    row=1,
    column=0,
    padx=15
)


# ==========================================
# OIL
# ==========================================
oil_title = ctk.CTkLabel(
    result_frame,
    text="Analisis Minyak",
    font=("Arial", 20, "bold")
)

oil_title.grid(
    row=0,
    column=1,
    pady=(0, 15)
)

oil_result = tk.Label(
    result_frame,
    bg="#2b2b2b"
)

oil_result.grid(
    row=1,
    column=1,
    padx=15
)


# ==========================================
# TEXTURE
# ==========================================
texture_title = ctk.CTkLabel(
    result_frame,
    text="Analisis Tekstur",
    font=("Arial", 20, "bold")
)

texture_title.grid(
    row=0,
    column=2,
    pady=(0, 15)
)

texture_result = tk.Label(
    result_frame,
    bg="#2b2b2b"
)

texture_result.grid(
    row=1,
    column=2,
    padx=15
)


# ==========================================
# COLOR
# ==========================================
color_title = ctk.CTkLabel(
    result_frame,
    text="Analisis Warna",
    font=("Arial", 20, "bold")
)

color_title.grid(
    row=0,
    column=3,
    pady=(0, 15)
)

color_result = tk.Label(
    result_frame,
    bg="#2b2b2b"
)

color_result.grid(
    row=1,
    column=3,
    padx=15
)


# ==========================================
# HEATMAP
# ==========================================
heatmap_title = ctk.CTkLabel(
    result_frame,
    text="Peta Sebaran Warna",
    font=("Arial", 20, "bold")
)

heatmap_title.grid(
    row=0,
    column=4,
    pady=(0, 15)
)

heatmap_result = tk.Label(
    result_frame,
    bg="#2b2b2b"
)

heatmap_result.grid(
    row=1,
    column=4,
    padx=15
)


# ==========================================
# RUN APP
# ==========================================
app.mainloop()