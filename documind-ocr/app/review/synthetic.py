"""Seeded synthetic receipts/invoices with known labels (for tests, the demo and evaluation).

Everything here is generated, so numbers measured on it describe *rendered text on a plain
background with controlled noise*, not real scans.
"""

import random
from datetime import date, timedelta
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VENDORS = [
    "Green Garden Restaurant",
    "Burger Barn",
    "Spice Route Kitchen",
    "Harbor Fresh Seafood",
    "Maple Leaf Bakery",
    "Urban Wok Express",
    "Sunrise Cafe",
    "Blue Door Catering",
    "Copper Pot Diner",
    "Olive and Thyme Bistro",
    "Noodle Junction",
    "Taco Corner Grill",
]
ITEMS = [
    "Paneer Wrap",
    "Cold Coffee",
    "Veg Biryani",
    "Garlic Bread",
    "Fish Curry",
    "Fries",
    "Green Salad",
    "Masala Dosa",
    "Lemon Soda",
    "Brownie",
    "Chicken Roll",
    "Mango Lassi",
]
LEVELS = ("clean", "noisy", "rotated", "degraded", "severe")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def make_label(rng: random.Random) -> Dict[str, str]:
    d = date(2024, 1, 1) + timedelta(days=rng.randrange(0, 365))
    items = [(rng.choice(ITEMS), round(rng.uniform(2, 40), 2)) for _ in range(rng.randint(2, 5))]
    sub = round(sum(p for _, p in items), 2)
    tax = round(sub * 0.05, 2)
    return {
        "vendor_name": rng.choice(VENDORS),
        "invoice_date": d.isoformat(),
        "total_amount": f"{sub + tax:.2f}",
        "_items": ";".join(f"{n}:{p:.2f}" for n, p in items),
        "_subtotal": f"{sub:.2f}",
        "_tax": f"{tax:.2f}",
        "_date_style": rng.choice(["iso", "us", "long"]),
        "doc_type": rng.choice(["invoice", "receipt"]),
    }


def _date_text(iso: str, style: str) -> str:
    d = date.fromisoformat(iso)
    return {"iso": iso, "us": d.strftime("%m/%d/%Y"), "long": d.strftime("%b %d, %Y")}[style]


def render(label: Dict[str, str], level: str = "clean", seed: int = 0) -> np.ndarray:
    """Draw a document and apply the degradation ``level``; returns a BGR image."""
    rng = np.random.default_rng(seed)
    lines: List[Tuple[str, int]] = [(label["vendor_name"], 40), ("", 20)]
    title = "INVOICE" if label["doc_type"] == "invoice" else "RECEIPT"
    lines += [(title, 30), (f"Invoice Date: {_date_text(label['invoice_date'], label['_date_style'])}", 28)]
    for entry in label["_items"].split(";"):
        n, p = entry.split(":")
        lines.append((f"{n}   {p}", 26))
    lines += [
        (f"Subtotal: ${label['_subtotal']}", 26),
        (f"Tax: ${label['_tax']}", 26),
        (f"Total: ${label['total_amount']}", 30),
        ("Thank you", 24),
    ]
    h = sum(sz + 18 for _, sz in lines) + 60
    img = Image.new("RGB", (900, h), "white")
    draw, y = ImageDraw.Draw(img), 30
    for text, size in lines:
        draw.text((40, y), text, fill="black", font=_font(size))
        y += size + 18
    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    if level == "noisy":
        arr = np.clip(arr + rng.normal(0, 28, arr.shape), 0, 255).astype(np.uint8)
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
    elif level == "rotated":
        m = cv2.getRotationMatrix2D((arr.shape[1] / 2, arr.shape[0] / 2), rng.uniform(-3, 3), 1.0)
        arr = cv2.warpAffine(arr, m, (arr.shape[1], arr.shape[0]), borderValue=(255, 255, 255))
    elif level == "degraded":
        arr = cv2.convertScaleAbs(arr, alpha=0.45, beta=95)  # low contrast, washed out
        arr = np.clip(arr + rng.normal(0, 22, arr.shape), 0, 255).astype(np.uint8)
        arr = cv2.GaussianBlur(arr, (5, 5), 0)
        arr = cv2.resize(arr, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    elif level == "severe":
        arr = cv2.convertScaleAbs(arr, alpha=0.4, beta=100)
        arr = np.clip(arr + rng.normal(0, 30, arr.shape), 0, 255).astype(np.uint8)
        arr = cv2.GaussianBlur(arr, (5, 5), 0)
        arr = cv2.resize(arr, None, fx=0.42, fy=0.42, interpolation=cv2.INTER_AREA)
    elif level != "clean":
        raise ValueError(f"unknown level {level!r}")
    return arr


def generate(n: int, seed: int = 0) -> List[Tuple[Dict[str, str], str, np.ndarray]]:
    """``n`` documents cycling through the degradation levels: (label, level, image)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        label, level = make_label(rng), LEVELS[i % len(LEVELS)]
        out.append((label, level, render(label, level, seed * 100003 + i)))
    return out
