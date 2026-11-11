"""Пайплайн: фото товара -> вырезка фона -> студийная сцена -> карточка с инфографикой."""
import base64
import io
import json
import logging
import os
import random
import re

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

log = logging.getLogger("pipeline")

# Формат 3:4. Ширина настраивается: 900 -> 900x1200, 1200 -> 1200x1600.
CARD_W = int(os.getenv("CARD_WIDTH", "1200"))
CARD_H = CARD_W * 4 // 3
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
REMBG_MODEL = os.getenv("REMBG_MODEL", "isnet-general-use")

# Готовые студийные сцены (градиент + мягкий свет). glow - сила подсветки за товаром.
STYLES = {
    "white": dict(label="⚪ Белая студия", top=(251, 251, 253), bottom=(212, 217, 226), glow=90),
    "warm": dict(label="🟠 Тёплый беж", top=(250, 240, 226), bottom=(220, 190, 152), glow=85),
    "pastel": dict(label="🌸 Пастель", top=(226, 236, 255), bottom=(255, 222, 238), glow=80),
    "dark": dict(label="⚫ Тёмный премиум", top=(66, 71, 84), bottom=(14, 15, 19), glow=60),
    "ai": dict(label="✨ AI-сцена (бета)"),
}

AI_SCENE_PROMPTS = [
    "minimalist product photography backdrop, large empty white marble surface in the foreground, "
    "soft diffused window light, pastel beige wall, clean, empty, no objects, no text",
    "large empty light oak table in a bright modern kitchen, softly blurred background, natural daylight, "
    "product photography, empty, no objects, no text",
    "large empty dark slate surface, moody studio, soft rim light, subtle gradient wall, "
    "product photography, empty, no objects, no text",
]


def available_styles() -> dict:
    styles = dict(STYLES)
    if not os.getenv("FAL_KEY"):
        styles.pop("ai")
    return styles


# ---------- шрифты ----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BOLD = [
    os.getenv("FONT_BOLD", ""),
    os.path.join(_HERE, "fonts", "Inter-Bold.ttf"),
    os.path.join(_HERE, "fonts", "DejaVuSans-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
_REG = [
    os.getenv("FONT_REGULAR", ""),
    os.path.join(_HERE, "fonts", "Inter-Regular.ttf"),
    os.path.join(_HERE, "fonts", "DejaVuSans.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    for path in (_BOLD if bold else _REG):
        if path and os.path.exists(path):
            return ImageFont.truetype(path, size)
    log.warning("Не найден шрифт с кириллицей, положите TTF в fonts/ (см. README)")
    return ImageFont.load_default(size)


# ---------- 1. вырезка фона ----------
# local: rembg на самом сервере (нужно ~2 ГБ RAM); fal: внешний API, сервер остаётся лёгким
REMBG_BACKEND = os.getenv("REMBG_BACKEND", "local")
FAL_REMBG_MODEL = os.getenv("FAL_REMBG_MODEL", "fal-ai/imageutils/rembg")
_session = None


def _remove_local(img: Image.Image) -> Image.Image:
    """Маска считается на уменьшенной копии (мало памяти), потом растягивается на оригинал."""
    global _session
    from rembg import new_session, remove  # тяжёлый импорт, поэтому лениво

    if _session is None:
        _session = new_session(REMBG_MODEL)
    small = img.copy()
    small.thumbnail((1024, 1024))
    mask = remove(small, session=_session, only_mask=True, post_process_mask=True)
    out = img.convert("RGBA")
    out.putalpha(mask.resize(img.size, Image.LANCZOS))
    return out


def _remove_fal(img: Image.Image) -> Image.Image:
    import fal_client
    import httpx

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    result = fal_client.subscribe(FAL_REMBG_MODEL, arguments={"image_url": uri})
    return Image.open(io.BytesIO(httpx.get(result["image"]["url"], timeout=60).content))


def cut_out(data: bytes) -> Image.Image:
    """Вырезает товар. Возвращает RGBA, обрезанный по границам объекта."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((1600, 1600))
    out = (_remove_fal(img) if REMBG_BACKEND == "fal" else _remove_local(img)).convert("RGBA")
    # чуть подтягиваем край, чтобы не было светлого ореола от старого фона
    alpha = out.getchannel("A").filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
    out.putalpha(alpha)
    bbox = alpha.getbbox()
    return out.crop(bbox) if bbox else out


# ---------- 2. студийная сцена ----------
def studio_gradient(w: int, h: int, cfg: dict) -> Image.Image:
    t = np.linspace(0, 1, h)[:, None, None]
    arr = np.array(cfg["top"]) * (1 - t) + np.array(cfg["bottom"]) * t
    base = Image.fromarray(np.repeat(arr, w, axis=1).astype("uint8"), "RGB")
    glow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(glow).ellipse((int(w * 0.08), int(h * 0.12), int(w * 0.92), int(h * 0.62)), fill=cfg["glow"])
    glow = glow.filter(ImageFilter.GaussianBlur(w * 0.12))
    return Image.composite(Image.new("RGB", (w, h), (255, 255, 255)), base, glow)


def ai_background(w: int, h: int) -> Image.Image:
    import fal_client
    import httpx

    result = fal_client.subscribe(
        "fal-ai/flux/schnell",
        arguments={
            "prompt": random.choice(AI_SCENE_PROMPTS),
            "image_size": {"width": 896, "height": 1152},
            "num_images": 1,
        },
    )
    url = result["images"][0]["url"]
    img = Image.open(io.BytesIO(httpx.get(url, timeout=60).content)).convert("RGB")
    return ImageOps.fit(img, (w, h), Image.LANCZOS)


def make_background(style: str, w: int, h: int) -> Image.Image:
    if style == "ai":
        try:
            return ai_background(w, h)
        except Exception:
            log.exception("AI-сцена не получилась, беру тёплый градиент")
            style = "warm"
    return studio_gradient(w, h, STYLES.get(style, STYLES["white"]))


def place_product(bg: Image.Image, product: Image.Image, area: tuple) -> Image.Image:
    """Ставит товар на «пол» области: контактная тень + мягкое отражение."""
    bg = bg.convert("RGBA")
    ax0, ay0, ax1, ay1 = area
    scale = min((ax1 - ax0) / product.width, (ay1 - ay0) / product.height)
    p = product.resize((max(1, int(product.width * scale)), max(1, int(product.height * scale))), Image.LANCZOS)
    x = ax0 + ((ax1 - ax0) - p.width) // 2
    y = ay1 - p.height
    cx = x + p.width // 2

    # тень
    sh_h = max(16, int(p.height * 0.07))
    shadow = Image.new("L", bg.size, 0)
    ImageDraw.Draw(shadow).ellipse((cx - int(p.width * 0.48), ay1 - sh_h, cx + int(p.width * 0.48), ay1 + sh_h), fill=150)
    shadow = shadow.filter(ImageFilter.GaussianBlur(sh_h * 0.8))
    black = Image.new("RGBA", bg.size, (0, 0, 0, 255))
    black.putalpha(shadow)
    bg = Image.alpha_composite(bg, black)

    # отражение
    rh = max(1, int(p.height * 0.16))
    refl = p.transpose(Image.FLIP_TOP_BOTTOM).crop((0, 0, p.width, rh))
    fade = Image.linear_gradient("L").resize((p.width, rh)).transpose(Image.FLIP_TOP_BOTTOM)
    fade = fade.point(lambda v: int(v * 0.30))
    refl.putalpha(ImageChops.multiply(refl.getchannel("A"), fade))
    layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    layer.paste(refl, (x, ay1))
    bg = Image.alpha_composite(bg, layer)

    bg.alpha_composite(p, (x, y))
    return bg


# ---------- 3. инфографика ----------
def _wrap(draw, text, fnt, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _fit(draw, text, size, min_size, max_w, max_lines, bold=True):
    while True:
        fnt = font(size, bold)
        lines = _wrap(draw, text, fnt, max_w)
        if len(lines) <= max_lines or size <= min_size:
            return fnt, lines[:max_lines]
        size -= 2


def _scrim(overlay, w, height, color, max_alpha):
    mask = Image.linear_gradient("L").resize((w, height)).transpose(Image.FLIP_TOP_BOTTOM)
    mask = mask.point(lambda v: int(v * max_alpha / 255))
    layer = Image.new("RGBA", (w, height), color + (255,))
    layer.putalpha(mask)
    overlay.paste(layer, (0, 0))


def compose_card(scene: Image.Image, copy: dict) -> Image.Image:
    W, H = scene.size
    k = W / 900
    s = lambda v: int(v * k)  # noqa: E731  (все размеры заданы для ширины 900)

    dark = ImageStat.Stat(scene.convert("L")).mean[0] < 135
    fg = (255, 255, 255) if dark else (24, 28, 36)
    fg_soft = (225, 228, 236) if dark else (84, 90, 104)
    panel = (16, 18, 24, 178) if dark else (255, 255, 255, 220)
    accent = (255, 196, 64) if dark else (255, 77, 46)
    check = (25, 25, 25) if dark else (255, 255, 255)

    benefits = [b for b in copy.get("benefits", []) if b][:4]
    margin, gap, ch, top = s(60), s(20), s(112), s(880)
    cw = (W - 2 * margin - gap) // 2

    def slot(i):
        row, col = divmod(i, 2)
        full = i == len(benefits) - 1 and len(benefits) % 2 == 1
        x = margin if full else margin + col * (cw + gap)
        return x, top + row * (ch + gap), (W - 2 * margin if full else cw)

    # слой 1: подложки (полупрозрачные)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    _scrim(overlay, W, s(330), (0, 0, 0) if dark else (255, 255, 255), 150)
    od = ImageDraw.Draw(overlay)
    for i in range(len(benefits)):
        x, y, w = slot(i)
        od.rounded_rectangle((x, y, x + w, y + ch), radius=s(26), fill=panel)
    card = Image.alpha_composite(scene.convert("RGBA"), overlay)

    # слой 2: текст и иконки
    d = ImageDraw.Draw(card)
    y = s(48)
    title = copy.get("title", "")
    for size in range(s(66), s(46) - 1, -2):  # сначала пробуем уместить в одну строку
        title_font = font(size)
        title_lines = _wrap(d, title, title_font, W - 2 * margin)
        if len(title_lines) == 1:
            break
    else:
        title_font, title_lines = _fit(d, title, s(60), s(40), W - 2 * margin, 2)
    for line in title_lines:
        d.text((W / 2, y), line, font=title_font, fill=fg, anchor="ma")
        y += int(title_font.size * 1.14)
    sub = copy.get("subtitle", "")
    if sub:
        sub_font, sub_lines = _fit(d, sub, s(30), s(22), W - 2 * margin, 2, bold=False)
        y += s(6)
        for line in sub_lines:
            d.text((W / 2, y), line, font=sub_font, fill=fg_soft, anchor="ma")
            y += int(sub_font.size * 1.25)
    y += s(14)
    d.rounded_rectangle((W / 2 - s(60), y, W / 2 + s(60), y + s(8)), radius=s(4), fill=accent)

    icon = s(56)
    for i, text in enumerate(benefits):
        x, by, w = slot(i)
        ix, iy = x + s(24), by + (ch - icon) // 2
        d.ellipse((ix, iy, ix + icon, iy + icon), fill=accent)
        d.line(
            [(ix + icon * 0.27, iy + icon * 0.52), (ix + icon * 0.43, iy + icon * 0.68), (ix + icon * 0.73, iy + icon * 0.34)],
            fill=check, width=s(6), joint="curve",
        )
        tx = ix + icon + s(20)
        fnt, lines = _fit(d, text, s(31), s(22), x + w - s(22) - tx, 2)
        lh = int(fnt.size * 1.2)
        ty = by + (ch - lh * len(lines)) // 2
        for line in lines:
            d.text((tx, ty), line, font=fnt, fill=fg)
            ty += lh
    return card.convert("RGB")


# ---------- 4. тексты через ИИ ----------
SYSTEM_PROMPT = """Ты копирайтер карточек товаров для маркетплейсов. По фото товара и подсказке продавца
составь тексты для инфографики. Верни ТОЛЬКО JSON без markdown и пояснений:
{"title": "...", "subtitle": "...", "benefits": ["...", "..."]}
Правила:
- title: название товара, до 32 символов.
- subtitle: короткая строка-выгода, до 45 символов.
- benefits: 3-4 преимущества, каждое до 24 символов.
- Используй только то, что видно на фото или написано продавцом. Не выдумывай размеры, материалы,
  цифры, сертификаты, гарантии и не используй слова «№1», «лучший», «хит».
- Язык: русский, если продавец не написал иначе."""


def _fallback_copy(hint: str) -> dict:
    lines = [ln.strip(" -•\t") for ln in hint.splitlines() if ln.strip()]
    return {"title": lines[0][:40] if lines else "Товар", "subtitle": "", "benefits": [ln[:28] for ln in lines[1:5]]}


def _jpeg_b64(data: bytes) -> str:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_copy(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    parsed = json.loads(raw)
    return {
        "title": str(parsed.get("title", ""))[:40],
        "subtitle": str(parsed.get("subtitle", ""))[:60],
        "benefits": [str(x)[:30] for x in parsed.get("benefits", [])][:4],
    }


def _copy_gemini(data: bytes, hint: str) -> dict:
    import httpx

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": _jpeg_b64(data)}},
            {"text": f"Подсказка продавца: {hint or 'нет'}"},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
    }
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return _parse_copy("".join(p.get("text", "") for p in parts))


def _copy_claude(data: bytes, hint: str) -> dict:
    import anthropic

    msg = anthropic.Anthropic().messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg_b64(data)}},
            {"type": "text", "text": f"Подсказка продавца: {hint or 'нет'}"},
        ]}],
    )
    return _parse_copy("".join(b.text for b in msg.content if b.type == "text"))


def generate_copy(data: bytes, hint: str) -> dict:
    """Gemini (бесплатный ключ) -> Claude (если есть ключ) -> просто текст из подписи продавца."""
    try:
        if os.getenv("GEMINI_API_KEY"):
            return _copy_gemini(data, hint)
        if os.getenv("ANTHROPIC_API_KEY"):
            return _copy_claude(data, hint)
    except Exception:
        log.exception("Не удалось сгенерировать тексты, использую текст продавца")
    return _fallback_copy(hint)


# ---------- сборка ----------
def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def build_card(photo: bytes, style: str = "white", hint: str = "") -> dict:
    product = cut_out(photo)
    copy = generate_copy(photo, hint)
    k = CARD_W / 900
    bg = make_background(style, CARD_W, CARD_H)
    scene = place_product(bg, product, (int(90 * k), int(270 * k), CARD_W - int(90 * k), int(765 * k)))
    card = compose_card(scene, copy)
    return {
        "card": _png(card),
        "scene": _png(scene.convert("RGB")),  # чистая сцена без текста, вторая фотка в галерею
        "cutout": _png(product),
        "copy": copy,
    }
