#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter
import re

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
OUTPUT_FILE = BASE_DIR / "art_periods_data.json"

N_BARV = 6
VELIKOST_SLIKE = (96, 96)

# -----------------------------------------
# OBDOBJA + KLJUČNE BESEDE
# -----------------------------------------

OBDOBJA = [
    {
        "naslov": "Prazgodovina",
        "leto": "40000–3000 pr.n.št.",
        "opis": "Jamske poslikave in zemeljski pigmenti.",
        "keywords": [
            "lascaux", "altamira", "chauvet", "bhimbetka",
            "willendorf", "cueva", "niaux", "capivara"
        ]
    },
    {
        "naslov": "Stari Egipt",
        "leto": "3100–30 pr.n.št.",
        "opis": "Zlati toni in lapis lazuli.",
        "keywords": [
            "nefertiti", "tutank", "egypt", "giza",
            "sphinx", "nebamun", "ramzes", "amenhotep",
            "narmer"
        ]
    },
    {
        "naslov": "Srednji vek",
        "leto": "500–1500",
        "opis": "Religiozna umetnost in bogate simbolne barve.",
        "keywords": [
            "kells", "rubljev", "rublev", "giotto",
            "bayeux", "justinian", "chartres",
            "maesta", "limbourg"
        ]
    },
    {
        "naslov": "Renesansa",
        "leto": "1400–1600",
        "opis": "Bogati zemeljski toni in chiaroscuro.",
        "keywords": [
            "mona", "leonardo", "michelangelo",
            "botticelli", "raphael", "titian",
            "van_eyck", "arnolfini", "primavera"
        ]
    },
    {
        "naslov": "Impresionizem",
        "leto": "1870–1900",
        "opis": "Svetle in vibrantne barve.",
        "keywords": [
            "monet", "renoir", "degas",
            "pissarro", "caillebotte",
            "morisot", "cassatt"
        ]
    },
    {
        "naslov": "Bauhaus",
        "leto": "1919–1933",
        "opis": "Geometrija in primarne barve.",
        "keywords": [
            "kandinsky", "mondrian",
            "bauhaus", "gropius",
            "klee", "bayer"
        ]
    },
    {
        "naslov": "Pop Art",
        "leto": "1955–1975",
        "opis": "Močne nasičene barve pop kulture.",
        "keywords": [
            "warhol", "lichtenstein",
            "hockney", "jasper",
            "marilyn", "campbell"
        ]
    },
    {
        "naslov": "Digitalna doba",
        "leto": "1990–danes",
        "opis": "Digitalne in neonske barve.",
        "keywords": [
            "beeple", "teamlab", "anadol",
            "eliasson", "paik", "arcangel",
            "reas", "digital"
        ]
    }
]

# -----------------------------------------
# BARVNA IMENA
# -----------------------------------------

BARVNA_TABELA = [
    ("Rdeča", (255, 0, 0)),
    ("Modra", (0, 0, 255)),
    ("Zelena", (0, 128, 0)),
    ("Rumena", (255, 255, 0)),
    ("Oranžna", (255, 165, 0)),
    ("Vijolična", (128, 0, 128)),
    ("Rjava", (165, 42, 42)),
    ("Bež", (245, 245, 220)),
    ("Siva", (128, 128, 128)),
    ("Črna", (0, 0, 0)),
    ("Bela", (255, 255, 255)),
]

# -----------------------------------------
# POMOŽNE
# -----------------------------------------

def poimenuj_barvo(hex_barva):
    rgb = tuple(int(hex_barva[i:i+2], 16) for i in (1,3,5))

    best_name = "Nevtralna"
    best_distance = float('inf')

    for name, sample in BARVNA_TABELA:
        d = sum((a-b)**2 for a,b in zip(rgb, sample))

        if d < best_distance:
            best_distance = d
            best_name = name

    return best_name


def nalozi_piksle(path):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(VELIKOST_SLIKE)

        arr = np.array(img).reshape(-1, 3)

        svetlost = arr.mean(axis=1)

        arr = arr[
            (svetlost > 10) &
            (svetlost < 245)
        ]

        if len(arr) < 20:
            return None

        return arr

    except Exception as e:
        print(f"[NAPAKA] {path.name}: {e}")
        return None


def ekstrahiraj_barve(vsi_piksli, n_barv=N_BARV):

    if vsi_piksli is None or len(vsi_piksli) < n_barv:
        return ["#808080"] * n_barv

    if len(vsi_piksli) > 30000:
        idx = np.random.choice(
            len(vsi_piksli),
            30000,
            replace=False
        )

        vsi_piksli = vsi_piksli[idx]

    model = MiniBatchKMeans(
        n_clusters=n_barv,
        random_state=42,
        n_init=10
    )

    model.fit(vsi_piksli)

    counts = Counter(model.labels_)

    ordered = [
        i for i, _ in counts.most_common(n_barv)
    ]

    colors = []

    for idx in ordered:
        r, g, b = [
            int(v)
            for v in model.cluster_centers_[idx]
        ]

        colors.append(f"#{r:02X}{g:02X}{b:02X}")

    return colors


# -----------------------------------------
# RAZPOREDI SLIKE PO OBDOBJIH
# -----------------------------------------

def poisci_obdobje(filename):

    ime = filename.lower()

    for obdobje in OBDOBJA:

        for keyword in obdobje["keywords"]:

            if keyword in ime:
                return obdobje

    return None


# -----------------------------------------
# GLAVNA LOGIKA
# -----------------------------------------

def generiraj_json():

    slike = list(IMAGES_DIR.glob("*"))

    dovoljene = {
        ".jpg", ".jpeg", ".png",
        ".webp", ".gif", ".svg"
    }

    rezultat = []

    for obdobje in OBDOBJA:

        print(f"\nObdelujem: {obdobje['naslov']}")

        obdobje_slike = []

        for slika in slike:

            if slika.suffix.lower() not in dovoljene:
                continue

            najdeno = poisci_obdobje(slika.name)

            if najdeno and najdeno["naslov"] == obdobje["naslov"]:
                obdobje_slike.append(slika)

        print(f"Najdenih slik: {len(obdobje_slike)}")

        vsi_piksli = []

        dela = []

        for slika in obdobje_slike:

            piksli = nalozi_piksle(slika)

            if piksli is not None:
                vsi_piksli.append(piksli)

            title = re.sub(r'[_-]+', ' ', slika.stem)

            dela.append({
                "title": title,
                "artist": "",
                "img": f"images/{slika.name}"
            })

        vsi_piksli = (
            np.vstack(vsi_piksli)
            if vsi_piksli else None
        )

        paleta = ekstrahiraj_barve(vsi_piksli)

        rezultat.append({
            "naslov": obdobje["naslov"],
            "leto": obdobje["leto"],
            "opis": obdobje["opis"],
            "paleta": paleta,
            "imena": [
                poimenuj_barvo(c)
                for c in paleta
            ],
            "dela": dela
        })

    OUTPUT_FILE.write_text(
        json.dumps(
            rezultat,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nKončano.")
    print("JSON shranjen v:", OUTPUT_FILE)


# -----------------------------------------

if __name__ == "__main__":

    if not IMAGES_DIR.exists():
        print("Mapa images ne obstaja:")
        print(IMAGES_DIR)

    else:
        generiraj_json()