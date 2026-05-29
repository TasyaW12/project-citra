import cv2
import numpy as np


def analisis_kulit(image_path):

    # ==========================================
    # VALIDASI GAMBAR
    # ==========================================
    if image_path is None:
        return None

    image = cv2.imread(image_path)

    if image is None:
        print("[ERROR] Gambar gagal dibaca.")
        return None

    # ==========================================
    # RESIZE GAMBAR
    # ==========================================
    height, width = image.shape[:2]

    new_width = 500
    new_height = int((new_width / width) * height)

    image = cv2.resize(
        image,
        (new_width, new_height)
    )

    # ==========================================
    # DETEKSI WAJAH
    # ==========================================
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        print("[ERROR] Wajah tidak ditemukan.")
        return None

    # ==========================================
    # AMBIL WAJAH
    # ==========================================
    (x, y, w, h) = faces[0]

    face = image[y:y+h, x:x+w]

    face_gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # ==========================================
    # MASK AREA WAJAH
    # ==========================================
    mask = np.zeros_like(face_gray)

    center = (w // 2, h // 2)

    axes = (
        int(w * 0.38),
        int(h * 0.48)
    )

    cv2.ellipse(
        mask,
        center,
        axes,
        0,
        0,
        360,
        255,
        -1
    )

    # ==========================================
    # HILANGKAN AREA MATA
    # ==========================================
    eye_y1 = int(h * 0.22)
    eye_y2 = int(h * 0.45)

    # Mata kiri
    cv2.rectangle(
        mask,
        (int(w * 0.18), eye_y1),
        (int(w * 0.42), eye_y2),
        0,
        -1
    )

    # Mata kanan
    cv2.rectangle(
        mask,
        (int(w * 0.58), eye_y1),
        (int(w * 0.82), eye_y2),
        0,
        -1
    )

    # ==========================================
    # HILANGKAN AREA MULUT
    # ==========================================
    cv2.rectangle(
        mask,
        (int(w * 0.32), int(h * 0.68)),
        (int(w * 0.68), int(h * 0.90)),
        0,
        -1
    )

    # ==========================================
    # ANALISIS KADAR MINYAK
    # ==========================================
    blur = cv2.GaussianBlur(
        face_gray,
        (7, 7),
        0
    )

    mean_brightness = np.mean(
        blur[mask > 0]
    )

    threshold_value = mean_brightness + 30

    _, oily_mask = cv2.threshold(
        blur,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    oily_mask = cv2.bitwise_and(
        oily_mask,
        mask
    )

    kernel = np.ones((5, 5), np.uint8)

    oily_mask = cv2.morphologyEx(
        oily_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    oily_mask = cv2.morphologyEx(
        oily_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    face_pixels = np.sum(mask == 255)

    oily_pixels = np.sum(oily_mask == 255)

    oily_percentage = (
        oily_pixels / face_pixels
    ) * 100

    brightness_score = 100 - oily_percentage

    # ==========================================
    # ANALISIS TEKSTUR
    # ==========================================
    edges = cv2.Canny(
        face_gray,
        15,
        45
    )

    edges = cv2.bitwise_and(
        edges,
        mask
    )

    texture_pixels = np.sum(edges == 255)

    texture_percentage = (
        texture_pixels / face_pixels
    ) * 100

    texture_score = 100 - texture_percentage

    # ==========================================
    # ANALISIS VARIASI WARNA
    # ==========================================
    smooth = cv2.bilateralFilter(
        face_rgb,
        9,
        75,
        75
    )

    hsv = cv2.cvtColor(
        smooth,
        cv2.COLOR_RGB2HSV
    )

    lower_skin = np.array([0, 20, 50])
    upper_skin = np.array([25, 180, 255])

    skin_mask = cv2.inRange(
        hsv,
        lower_skin,
        upper_skin
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    skin_mask = cv2.morphologyEx(
        skin_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ==========================================
    # REMOVE EDGE AREA
    # ==========================================
    margin = 20

    skin_mask[:margin, :] = 0
    skin_mask[-margin:, :] = 0
    skin_mask[:, :margin] = 0
    skin_mask[:, -margin:] = 0

    # ==========================================
    # LAB COLOR
    # ==========================================
    lab = cv2.cvtColor(
        smooth,
        cv2.COLOR_RGB2LAB
    )

    L, A, B = cv2.split(lab)

    # ==========================================
    # CLAHE
    # ==========================================
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    A = clahe.apply(A)
    B = clahe.apply(B)

    # ==========================================
    # BASE SKIN COLOR
    # ==========================================
    baseA = cv2.GaussianBlur(
        A,
        (71, 71),
        0
    )

    baseB = cv2.GaussianBlur(
        B,
        (71, 71),
        0
    )

    # ==========================================
    # COLOR DIFFERENCE
    # ==========================================
    diffA = cv2.absdiff(A, baseA)

    diffB = cv2.absdiff(B, baseB)

    color_diff = cv2.addWeighted(
        diffA,
        0.5,
        diffB,
        0.5,
        0
    )

    # ==========================================
    # APPLY SKIN MASK
    # ==========================================
    color_diff = cv2.bitwise_and(
        color_diff,
        color_diff,
        mask=skin_mask
    )

    # ==========================================
    # SMOOTH RESULT
    # ==========================================
    color_diff = cv2.GaussianBlur(
        color_diff,
        (5, 5),
        0
    )

    # ==========================================
    # NORMALIZATION
    # ==========================================
    color_diff = cv2.normalize(
        color_diff,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # ==========================================
    # THRESHOLD
    # ==========================================
    valid_pixels = color_diff[
        skin_mask > 0
    ]

    if len(valid_pixels) == 0:
        return None

    mean_val = np.mean(valid_pixels)

    threshold_value = mean_val + 20

    variation_mask = cv2.threshold(
        color_diff,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )[1]

    variation_mask = variation_mask.astype(
        np.uint8
    )

    # ==========================================
    # MORPHOLOGY
    # ==========================================
    kernel_small = np.ones(
        (3, 3),
        np.uint8
    )

    variation_mask = cv2.morphologyEx(
        variation_mask,
        cv2.MORPH_OPEN,
        kernel_small
    )

    variation_mask = cv2.morphologyEx(
        variation_mask,
        cv2.MORPH_CLOSE,
        kernel_small
    )

    # ==========================================
    # CONTOUR FILTER
    # ==========================================
    contours, _ = cv2.findContours(
        variation_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_mask = np.zeros_like(
        variation_mask
    )

    color_result = face_rgb.copy()

    total_area = 0

    face_h, face_w = face_rgb.shape[:2]

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if 80 < area < 1800:

            x, y, w, h = cv2.boundingRect(cnt)

            ratio = w / h

            extent = area / (w * h)

            if extent < 0.35:
                continue

            if ratio > 3.5 or ratio < 0.30:
                continue

            if x < 10 or x+w > face_w-10:
                continue

            if y < face_h * 0.10:
                continue

            total_area += area

            cv2.drawContours(
                clean_mask,
                [cnt],
                -1,
                255,
                -1
            )

            cv2.rectangle(
                color_result,
                (x, y),
                (x+w, y+h),
                (255, 0, 0),
                2
            )

    variation_mask = clean_mask

    # ==========================================
    # HEATMAP
    # ==========================================
    heatmap = cv2.applyColorMap(
        color_diff.astype(np.uint8),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # ==========================================
    # SCORE WARNA
    # ==========================================
    face_area = np.count_nonzero(
        skin_mask
    )

    if face_area == 0:
        percentage = 0

    else:
        percentage = (
            total_area / face_area
        ) * 100

    percentage = min(
        percentage,
        100
    )

    color_variation = percentage

    color_score = max(
        0,
        100 - percentage
    )

    # ==========================================
    # SKIN SCORE FINAL
    # ==========================================
    skin_score = (
        (brightness_score * 0.40) +
        (texture_score * 0.35) +
        (color_score * 0.25)
    )

    skin_score = max(
        0,
        min(100, skin_score)
    )

    # ==========================================
    # RETURN HASIL
    # ==========================================
    return {

        "oil": oily_percentage,
        "texture": texture_percentage,
        "color": color_variation,
        "score": skin_score,

        "face": face,
        "oily_mask": oily_mask,
        "edges": edges,

        "variation_mask": variation_mask,
        "heatmap": heatmap,
        "color_result": color_result
    }