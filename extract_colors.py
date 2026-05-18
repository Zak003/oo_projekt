#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import random
import os
import sys
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import threading

# ── Preveri odvisnosti ────────────────────────────────────────────
def _preveri_in_uvozi():
    manjkajoci = []
    for paket, ime in [("PIL", "Pillow"), ("numpy", "numpy"),
                        ("sklearn", "scikit-learn"), ("requests", "requests")]:
        try:
            __import__(paket)
        except ImportError:
            manjkajoci.append(ime)
    if manjkajoci:
        print(f"Manjkajoci paketi: {', '.join(manjkajoci)}")
        print("   Namestite z: pip install -r requirements.txt")
        sys.exit(1)

_preveri_in_uvozi()

import requests
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

# ── Konstante ─────────────────────────────────────────────────────
CILJ_SLIK_NA_DOBO = 10     # Ciljno stevilo uspesno obdelanih slik
VZPOREDNI_PRENOSI = 20      # Vzporedni HTTP prenosi
TIMEOUT_PRENOS    = 2        # Sekund za posamezen prenos
VELIKOST_SLIKE    = (48, 48)
N_BARV            = 6
CHECKPOINT_DAT    = "checkpoint.json"
IZHOD_DAT         = "art_periods_data.json"

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {'User-Agent': 'ArtColorAnalyzer/3.0 (educational; color-palette-research)'}

TISK_LOCK = threading.Lock()

# ── Barvno poimenovanje (LAB prostor) ─────────────────────────────
BARVNA_TABELA = [
    ("Crnina",          0,   0,   0),
    ("Temno siva",     64,  64,  64),
    ("Siva",          128, 128, 128),
    ("Svetlo siva",   192, 192, 192),
    ("Belina",        255, 255, 255),
    ("Slonokoščena",  255, 255, 224),
    ("Kremna",        255, 253, 208),
    ("Bez",           245, 245, 220),
    ("Pescena",       194, 178, 128),
    ("Okrasta",       204, 119,  34),
    ("Zlata",         212, 175,  55),
    ("Rumena",        255, 255,   0),
    ("Svetlo rumena", 255, 255, 153),
    ("Jantarna",      255, 191,   0),
    ("Oranzna",       255, 165,   0),
    ("Temno oranzna", 255, 140,   0),
    ("Cinober",       227,  66,  52),
    ("Rdeca",         255,   0,   0),
    ("Temno rdeca",   139,   0,   0),
    ("Karmin",        150,   0,  24),
    ("Roza",          255, 182, 193),
    ("Skrlatna",      128,   0,   0),
    ("Rjava",         165,  42,  42),
    ("Temno rjava",   101,  67,  33),
    ("Sienna",        160,  82,  45),
    ("Umbra",          99,  81,  71),
    ("Konjak",        152, 105,  96),
    ("Violicna",      128,   0, 128),
    ("Indigo",         75,   0, 130),
    ("Lavanda",       230, 230, 250),
    ("Modra",           0,   0, 255),
    ("Pruska modra",    0,  49,  83),
    ("Temno modra",     0,   0, 139),
    ("Kobaltna",        0,  71, 171),
    ("Lapislazuli",    15,  82, 186),
    ("Nebska",        135, 206, 235),
    ("Svetlo modra",  173, 216, 230),
    ("Turkizna",        0, 206, 209),
    ("Cyan",            0, 255, 255),
    ("Zelena",          0, 128,   0),
    ("Neon zelena",     0, 255,   0),
    ("Temno zelena",    0, 100,   0),
    ("Olivna",        128, 128,   0),
    ("Smaragdna",       0, 201,  87),
    ("Mintna",        152, 255, 152),
    ("Lesna",          34, 139,  34),
]

def _rgb_v_lab(r, g, b):
    rgb = np.array([r, g, b], dtype=float) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] /= 12.92
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = M @ rgb
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    eps, kap = 0.008856, 903.3
    f = np.where(xyz > eps, xyz ** (1/3), (kap * xyz + 16) / 116)
    return 116*f[1]-16, 500*(f[0]-f[1]), 200*(f[1]-f[2])

_TABELA_LAB = [(ime, _rgb_v_lab(r, g, b)) for ime, r, g, b in BARVNA_TABELA]

def poimenuj_barvo(hex_barva: str) -> str:
    h = hex_barva.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    L, a, bv = _rgb_v_lab(r, g, b)
    best_ime, best_d = "Nevtralna", float('inf')
    for ime, (Lr, ar, br) in _TABELA_LAB:
        d = (L-Lr)**2 + (a-ar)**2 + (bv-br)**2
        if d < best_d:
            best_d, best_ime = d, ime
    return best_ime


# ── Wikimedia Commons — iskanje URL-jev (POPRAVLJENO) ─────────────

def _wikimedia_pridobi_url_slik(seja, iskalni_izraz: str,
                                  limit: int = 50,
                                  sroffset: int = 0):
    params = {
        'action':       'query',
        'format':       'json',
        'generator':    'search',
        'gsrsearch':    iskalni_izraz,
        'gsrnamespace': 6,        # Namespace 6 = File:
        'gsrlimit':     limit,
        'gsroffset':    sroffset,
        'prop':         'imageinfo',
        'iiprop':       'url|mime|size',
        'iiurlwidth':   200,      # Thumbnail = manjsi prenos
    }
    try:
        odg = seja.get(WIKIMEDIA_API, params=params, timeout=20)
        odg.raise_for_status()
        data = odg.json()
    except Exception:
        return [], None

    urls = []
    for page in data.get('query', {}).get('pages', {}).values():
        for ii in page.get('imageinfo', []):
            mime = ii.get('mime', '')
            url  = ii.get('thumburl') or ii.get('url', '')
            size = ii.get('size', 0)
            ext  = url.lower().split('?')[0]
            if (url
                    and 'image' in mime
                    and (ext.endswith('.jpg') or ext.endswith('.jpeg'))
                    and 2_000 < size < 30_000_000):
                urls.append(url)

    # gsroffset za naslednjo stran (vrnjeno v 'continue')
    naslednji = data.get('continue', {}).get('gsroffset')
    return urls, naslednji


def zberi_url_iz_wikimedie(seja, iskalni_izrazi: list,
                             cilj_url: int = 2000) -> list:
    vse_url: set = set()

    for izraz in iskalni_izrazi:
        if len(vse_url) >= cilj_url:
            break

        with TISK_LOCK:
            print(f"    Iscemo: '{izraz}' ...", flush=True)

        offset = 0
        for stran in range(8):  # Do 8 x 50 = 400 URL-jev na izraz
            if len(vse_url) >= cilj_url:
                break
            urls, naslednji = _wikimedia_pridobi_url_slik(seja, izraz,
                                                           limit=50,
                                                           sroffset=offset)
            nove = [u for u in urls if u not in vse_url]
            vse_url.update(nove)

            with TISK_LOCK:
                print(f"      stran {stran+1}: +{len(nove)} URL-jev "
                      f"(skupaj {len(vse_url)})", flush=True)

            if naslednji is None or not urls:
                break
            offset = naslednji
            time.sleep(0.3)

    rezultat = list(vse_url)
    random.shuffle(rezultat)
    return rezultat


# ── Prenos in obdelava slike ──────────────────────────────────────

def prenesi_in_obdelaj(seja, url: str):
    try:
        odg = seja.get(url, timeout=TIMEOUT_PRENOS, allow_redirects=True)
        odg.raise_for_status()
        vsebina = odg.content
        if len(vsebina) < 1_000:
            return None

        slika = Image.open(BytesIO(vsebina))

        # Pretvori v RGB
        if slika.mode == 'RGBA':
            ozadje = Image.new('RGB', slika.size, (255, 255, 255))
            ozadje.paste(slika, mask=slika.split()[3])
            slika = ozadje
        elif slika.mode == 'P':
            slika = slika.convert('RGBA').convert('RGB')
        elif slika.mode == 'CMYK':
            slika = slika.convert('RGB')
        elif slika.mode == 'L':
            slika = slika.convert('RGB')
        elif slika.mode != 'RGB':
            slika = slika.convert('RGB')

        slika.thumbnail(VELIKOST_SLIKE, Image.LANCZOS)
        arr = np.array(slika, dtype=np.float32)
        piksli = arr.reshape(-1, 3)

        # Odstrani skoraj-bele in skoraj-crne piksle
        svetlost = piksli.mean(axis=1)
        maska = (svetlost > 20) & (svetlost < 238)
        piksli = piksli[maska]

        return piksli if len(piksli) >= 30 else None

    except Exception:
        return None
    """ except Exception as e:
        print("NAPAKA:", type(e).__name__, str(e)[:120])
        return None """


def ekstrahiraj_barve_iz_pikslov(vsi_piksli: np.ndarray,
                                   n_barv: int = N_BARV) -> list:
    """K-means clustering -> hex barve."""
    if len(vsi_piksli) < n_barv * 10:
        return []

    if len(vsi_piksli) > 100_000:
        idx = np.random.choice(len(vsi_piksli), 600_000, replace=False)
        vsi_piksli = vsi_piksli[idx]

    n_clustrov = n_barv + 4
    n_clustrov = min(n_clustrov, len(vsi_piksli) // 5, 24)
    n_clustrov = max(n_clustrov, n_barv)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clustrov,
        n_init=5,
        batch_size=min(20_000, len(vsi_piksli)),
        random_state=42,
        max_iter=300
    )
    kmeans.fit(vsi_piksli)

    centroidi = kmeans.cluster_centers_
    stevci = Counter(kmeans.labels_)
    urejeno = sorted(stevci.items(), key=lambda x: x[1], reverse=True)

    barve = []
    for idx_c, _ in urejeno[:n_barv]:
        r, g, b = [int(v) for v in centroidi[idx_c].clip(0, 255)]
        barve.append(f'#{r:02X}{g:02X}{b:02X}')
    return barve


# ── Checkpoint sistem ─────────────────────────────────────────────

def nalozi_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_DAT):
        try:
            with open(CHECKPOINT_DAT, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def shrani_checkpoint(chk: dict):
    with open(CHECKPOINT_DAT, 'w', encoding='utf-8') as f:
        json.dump(chk, f, ensure_ascii=False, indent=2)


# ── Konfiguracija dob ─────────────────────────────────────────────
DOBE = [
    {
        "naslov": "Prazgodovina",
        "leto": "40000-3000 pr.n.st.",
        "opis": "Jamske poslikave in okrasti pigmenti. Umetniki so za pigmente uporabljali minerale, oglje in zivalske mascobe za belezenje zivljenja, lovskih ritualov in duhovnih simbolov.",
        "iskalni_izrazi": [
            "cave painting prehistoric art",
            "Lascaux cave art France",
            "rock art paleolithic ancient",
            "prehistoric ochre pigment painting",
            "Altamira cave painting Spain",
            "Chauvet cave art France",
            "Stone Age art ancient",
            "Upper Paleolithic art Europe",
            "prehistoric handprint cave wall",
            "Bhimbetka rock paintings India",
        ],
        "dela": [
            {"title": "Lascaux: Jamske poslikave", "artist": "Neznan umetnik, ~17000 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/1/1e/Lascaux_painting.jpg"},
            {"title": "Altamira: Bizon", "artist": "Neznan umetnik, ~14000 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/8/8b/9_Bisonte_Magdaleniense_pol%C3%ADcromo.jpg"},
            {"title": "Chauvet: Konji", "artist": "Neznan umetnik, ~32000 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Chauvet_Cave_Paintings_Horses.jpg/1280px-Chauvet_Cave_Paintings_Horses.jpg"},
            {"title": "Cueva de las Manos: Roke", "artist": "Neznan umetnik, ~9000 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/SantaCruz-CuevaManos-P2210651b.jpg/1280px-SantaCruz-CuevaManos-P2210651b.jpg"},
            {"title": "Willendorfska Venera", "artist": "Neznan umetnik, ~28000 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Venus_von_Willendorf_01.jpg/320px-Venus_von_Willendorf_01.jpg"},
            {"title": "Bhimbetka: Skalne poslikave", "artist": "Neznan umetnik, ~10000 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Bhimbetka_Cave_3.jpg/1280px-Bhimbetka_Cave_3.jpg"},
        ]
    },
    {
        "naslov": "Stari Egipt",
        "leto": "3100-30 pr.n.st.",
        "opis": "Ikonicni zlati toni, lapislazuli in okrasti odtenki. Barva je imela globok simbolni pomen — zlata za sonce in bogove, modra za Nil, zelena za plodnost in preporod.",
        "iskalni_izrazi": [
            "ancient Egyptian painting tomb",
            "Egyptian hieroglyphics wall art",
            "Egyptian tomb painting mural Luxor",
            "pharaoh Egyptian ancient art",
            "Tutankhamun gold artifact",
            "Egyptian papyrus painting Book of Dead",
            "ancient Egyptian sculpture",
            "Nefertiti Egyptian portrait bust",
            "Egyptian sarcophagus painted",
            "Ramesses temple art Egypt",
        ],
        "dela": [
            {"title": "Tutankamonova zlatna maska", "artist": "Neznan umetnik, ~1323 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/e/e1/Tutankhamun_mask_%28close-up%29_01.jpg"},
            {"title": "Nefertitina bista", "artist": "Thutmose, ~1345 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Nefertiti_30-01-2006.jpg/320px-Nefertiti_30-01-2006.jpg"},
            {"title": "Egipcanska knjiga mrtvih", "artist": "Neznan umetnik, ~1275 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/BD_Weighing_of_the_Heart.jpg/1280px-BD_Weighing_of_the_Heart.jpg"},
            {"title": "Poslikave v grobnici Nebamun", "artist": "Neznan umetnik, ~1350 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Maler_der_Grabkammer_des_Nebamun_001.jpg/1280px-Maler_der_Grabkammer_des_Nebamun_001.jpg"},
            {"title": "Amenhotep III", "artist": "Neznan umetnik",
             "img": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Amenhotep_III_head_with_pschent.jpg"},
            {"title": "Sfinga v Gizi", "artist": "Neznan umetnik, ~2500 pr.n.st.",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Great_Sphinx_of_Giza_-_20080716a.jpg/1280px-Great_Sphinx_of_Giza_-_20080716a.jpg"},
        ]
    },
    {
        "naslov": "Srednji vek",
        "leto": "500-1500 n.st.",
        "opis": "Zlata ozadja, intenzivne gemske barve in simbolicni odtenki. Barvna paleta je odrazala versko hierarhijo — modra za Devico Marijo, skrlatna za vladarje, zlata za bozansko.",
        "iskalni_izrazi": [
            "medieval illuminated manuscript painting",
            "Byzantine mosaic art gold",
            "Gothic cathedral stained glass",
            "medieval religious icon painting",
            "Romanesque church fresco art",
            "Book of Kells illumination",
            "medieval fresco church wall",
            "Byzantine icon painting gold background",
            "medieval tapestry Bayeux",
            "Giotto di Bondone fresco",
        ],
        "dela": [
            {"title": "Tres Riches Heures du Duc de Berry", "artist": "Bratje Limbourg, 1412-16",
             "img": "https://upload.wikimedia.org/wikipedia/commons/e/ec/Limbourg_bros_Tr%C3%A8s_Riches_Heures_Janvier.jpg"},
            {"title": "Andrej Rubljev: Trojica", "artist": "Andrej Rubljev, ~1411",
             "img": "https://upload.wikimedia.org/wikipedia/commons/9/9a/Andrei_Rublev_-_Trinity.jpg"},
            {"title": "Knjiga iz Kellsa", "artist": "Irski menihi, ~800",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/KellsFol032vChristEnthroned.jpg/320px-KellsFol032vChristEnthroned.jpg"},
            {"title": "Mozaik iz Ravenne: Justinijan I", "artist": "Neznan mojster, ~547",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Meister_von_San_Vitale_in_Ravenna.jpg/1280px-Meister_von_San_Vitale_in_Ravenna.jpg"},
            {"title": "Giotto: Lamentation (Pieta)", "artist": "Giotto di Bondone, ~1305",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Giotto_di_Bondone_-_No._36_Scenes_from_the_Life_of_Christ_-_20._The_Lamentation_of_Christ_-_WGA09280.jpg/1280px-Giotto_di_Bondone_-_No._36_Scenes_from_the_Life_of_Christ_-_20._The_Lamentation_of_Christ_-_WGA09280.jpg"},
            {"title": "Tapiserija iz Bayeuxa", "artist": "Neznan mojster, ~1070-80",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Bayeux_Tapestry_scene51_Eustace_Odo_William.jpg/1280px-Bayeux_Tapestry_scene51_Eustace_Odo_William.jpg"},
        ]
    },
    {
        "naslov": "Renesansa",
        "leto": "1400-1600",
        "opis": "Bogati zemeljski toni in subtilni prehodi svetlobe. Renesancni mojstri so uvedli perspektivo in realizem z izpopolnjenim chiaroscurom ter novimi pigmenti.",
        "iskalni_izrazi": [
            "Renaissance painting Italy oil",
            "Leonardo da Vinci painting artwork",
            "Michelangelo painting Sistine",
            "Sandro Botticelli painting",
            "Raphael Renaissance fresco painting",
            "Italian Renaissance portrait oil painting",
            "Titian Renaissance Venice painting",
            "Jan van Eyck Flemish painting",
            "Dürer German Renaissance",
            "Renaissance Madonna religious painting",
        ],
        "dela": [
            {"title": "Leonardo da Vinci: Mona Lisa", "artist": "Leonardo da Vinci, 1503-19",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"},
            {"title": "Botticelli: Rojstvo Venere", "artist": "Sandro Botticelli, ~1484-86",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg/1024px-Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg"},
            {"title": "Raphael: Sola v Atenah", "artist": "Raphael, 1509-11",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/%22The_School_of_Athens%22_by_Raffaello_Sanzio_da_Urbino.jpg/1280px-%22The_School_of_Athens%22_by_Raffaello_Sanzio_da_Urbino.jpg"},
            {"title": "Michelangelo: Stvaritev Adama", "artist": "Michelangelo, 1508-12",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg/1280px-Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg"},
            {"title": "Jan van Eyck: Arnolfinijeva poroka", "artist": "Jan van Eyck, 1434",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Van_Eyck_-_Arnolfini_Portrait.jpg/800px-Van_Eyck_-_Arnolfini_Portrait.jpg"},
            {"title": "Titian: Assumption of the Virgin", "artist": "Titian, 1516-18",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Assunta_Tiziano.jpg/800px-Assunta_Tiziano.jpg"},
        ]
    },
    {
        "naslov": "Impresionizem",
        "leto": "1870-1900",
        "opis": "Svetle, vibrantne barve nanesene z vidnimi potezami copiča. Impresionisti so lovili trenutno igro svetlobe in atmosfero prizora, zavracajoc akademski realizem.",
        "iskalni_izrazi": [
            "Impressionist painting Monet landscape",
            "Claude Monet water lilies painting",
            "Pierre-Auguste Renoir Impressionism",
            "Edgar Degas ballet dancer painting",
            "Camille Pissarro Impressionism landscape",
            "Alfred Sisley river landscape painting",
            "Berthe Morisot Impressionism woman",
            "Impressionist light atmosphere painting",
            "Mary Cassatt Impressionism",
            "Gustave Caillebotte Paris Impressionism",
        ],
        "dela": [
            {"title": "Monet: Impression, Soleil Levant", "artist": "Claude Monet, 1872",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Monet_-_Impression%2C_Sunrise.jpg/1280px-Monet_-_Impression%2C_Sunrise.jpg"},
            {"title": "Renoir: Bal du Moulin de la Galette", "artist": "Pierre-Auguste Renoir, 1876",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Renoir%2C_Pierre-Auguste_-_Dance_at_Le_Moulin_de_la_Galette%2C_1876.jpg/1280px-Renoir%2C_Pierre-Auguste_-_Dance_at_Le_Moulin_de_la_Galette%2C_1876.jpg"},
            {"title": "Degas: Plesalke v roza", "artist": "Edgar Degas, ~1867-68",
             "img": "https://upload.wikimedia.org/wikipedia/commons/f/f1/Edgar_Degas%2C_1867c_-_Ballerinas_in_Pink.jpg"},
            {"title": "Monet: Lokvanjske lise (1906)", "artist": "Claude Monet, 1906",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg/1280px-Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg"},
            {"title": "Pissarro: Boulevard Montmartre ponocí", "artist": "Camille Pissarro, 1897",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Camille_Pissarro_-_The_Boulevard_Montmartre_at_Night.jpg/1280px-Camille_Pissarro_-_The_Boulevard_Montmartre_at_Night.jpg"},
            {"title": "Caillebotte: Pariška ulica v dezju", "artist": "Gustave Caillebotte, 1877",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Gustave_Caillebotte_-_Paris_Street%3B_Rainy_Day_-_Google_Art_Project.jpg/1280px-Gustave_Caillebotte_-_Paris_Street%3B_Rainy_Day_-_Google_Art_Project.jpg"},
        ]
    },
    {
        "naslov": "Bauhaus",
        "leto": "1919-1933",
        "opis": "Primarne barve z geometrijsko natancnostjo. Bauhaus je zdruzil umetnost in obrt z revolucionarno estetiko funkcionalnosti — forma sledi funkciji.",
        "iskalni_izrazi": [
            "Bauhaus art design Dessau",
            "Wassily Kandinsky abstract composition painting",
            "Paul Klee painting abstract",
            "Piet Mondrian composition primary colors",
            "Bauhaus typography poster graphic design",
            "Herbert Bayer graphic design",
            "Bauhaus geometric abstract art",
            "Laszlo Moholy-Nagy composition",
            "Oskar Schlemmer Bauhaus",
            "Josef Albers color theory",
        ],
        "dela": [
            {"title": "Kandinsky: Kompozicija VIII", "artist": "Wassily Kandinsky, 1923",
             "img": "https://upload.wikimedia.org/wikipedia/commons/a/ad/Wassily_Kandinsky_Composition_VIII.jpg"},
            {"title": "Mondrian: Kompozicija II", "artist": "Piet Mondrian, 1930",
             "img": "https://upload.wikimedia.org/wikipedia/commons/f/fb/Mondrian_-_Composition_no_II%2C_1929.jpg"},
            {"title": "Paul Klee: Ceste in stranpoti", "artist": "Paul Klee, 1929",
             "img": "https://upload.wikimedia.org/wikipedia/commons/0/0d/Highway-byways.png"},
            {"title": "Kandinsky: Rumeno-rdece-modro", "artist": "Wassily Kandinsky, 1925",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Vassily_Kandinsky%2C_1925_-_Yellow-Red-Blue.jpg/1280px-Vassily_Kandinsky%2C_1925_-_Yellow-Red-Blue.jpg"},
            {"title": "Herbert Bayer: Bauhaus razstava poster", "artist": "Herbert Bayer, 1923",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Bauhaus_1923_Exhibition_Poster.jpg/640px-Bauhaus_1923_Exhibition_Poster.jpg"},
            {"title": "Bauhaus stavba Dessau", "artist": "Walter Gropius, 1925-26",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Bauhaus-Dessau.jpg/1280px-Bauhaus-Dessau.jpg"},
        ]
    },
    {
        "naslov": "Pop Art",
        "leto": "1955-1975",
        "opis": "Zarece, nasicene barve iz popularne kulture in mnozicnih medijev. Pop Art je zavrnil tradicijo visoke umetnosti in objel estetiko potrosnistva, reklam in stripov.",
        "iskalni_izrazi": [
            "Pop Art Andy Warhol silkscreen",
            "Roy Lichtenstein comic dot painting",
            "Pop Art bright bold colors poster",
            "Jasper Johns flag painting",
            "Robert Rauschenberg combine painting",
            "American Pop Art 1960s",
            "Pop Art advertisement parody",
            "David Hockney California Pop Art",
            "Claes Oldenburg Pop Art",
            "Tom Wesselmann Pop Art",
        ],
        "dela": [
            {"title": "Andy Warhol: Marilyn Monroe", "artist": "Andy Warhol, 1962",
             "img": "https://upload.wikimedia.org/wikipedia/en/thumb/8/8d/Warhol-Marilyn.jpg/800px-Warhol-Marilyn.jpg"},
            {"title": "Roy Lichtenstein: Whaam!", "artist": "Roy Lichtenstein, 1963",
             "img": "https://media.tate.org.uk/art/images/work/T/T00/T00897_10.jpg"},
            {"title": "David Hockney: A Bigger Splash", "artist": "David Hockney, 1967",
             "img": "https://upload.wikimedia.org/wikipedia/en/8/87/A_Bigger_Splash_David_Hockney.jpg"},
            {"title": "Jasper Johns: Flag", "artist": "Jasper Johns, 1954-55",
             "img": "https://upload.wikimedia.org/wikipedia/en/thumb/1/17/Flag_by_Jasper_Johns.jpg/1280px-Flag_by_Jasper_Johns.jpg"},
            {"title": "Andy Warhol: Campbell's Soup Cans", "artist": "Andy Warhol, 1962",
             "img": "https://upload.wikimedia.org/wikipedia/en/thumb/8/82/Campbells_Soup_Cans_MOMA.jpg/1024px-Campbells_Soup_Cans_MOMA.jpg"},
            {"title": "Roy Lichtenstein: Drowning Girl", "artist": "Roy Lichtenstein, 1963",
             "img": "https://upload.wikimedia.org/wikipedia/en/thumb/4/4e/Roy_Lichtenstein_Drowning_Girl.jpg/800px-Roy_Lichtenstein_Drowning_Girl.jpg"},
        ]
    },
    {
        "naslov": "Digitalna doba",
        "leto": "1990-danes",
        "opis": "Digitalni sijaj in neonske barve RGB zaslonov. Digitalna umetnost je razprla meje med resnicnim in virtualnim, ustvarila nove estetike od piksene umetnosti do NFT-jev.",
        "iskalni_izrazi": [
            "digital art concept art fantasy illustration",
            "digital painting fantasy art",
            "neon cyberpunk futuristic art",
            "3D render abstract digital art",
            "contemporary digital illustration",
            "pixel art retro game",
            "glitch art digital abstract",
            "generative art algorithm computer",
            "light installation contemporary art",
            "new media art digital installation",
        ],
        "dela": [
            {"title": "Beeple: Everydays — 5000 dni", "artist": "Beeple (Mike Winkelmann), 2021",
             "img": "https://upload.wikimedia.org/wikipedia/en/d/d4/Everydays%2C_the_First_5000_Days.jpg"},
            {"title": "Refik Anadol: Unsupervised (MoMA)", "artist": "Refik Anadol, 2022",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Refik_Anadol_MoMA_Unsupervised.jpg/640px-Refik_Anadol_MoMA_Unsupervised.jpg"},
            {"title": "James Turrell: Roden Crater", "artist": "James Turrell, 1977-danes",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Roden_Crater_aerial_view.jpg/640px-Roden_Crater_aerial_view.jpg"},
            {"title": "TeamLab: Cvetlicni gozd", "artist": "TeamLab, 2017",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/TeamLab_Planets_TOKYO_2019.jpg/640px-TeamLab_Planets_TOKYO_2019.jpg"},
            {"title": "Olafur Eliasson: The Weather Project", "artist": "Olafur Eliasson, 2003",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Olafur_Eliasson%2C_The_Weather_Project%2C_2003_%28cropped%29.jpg/640px-Olafur_Eliasson%2C_The_Weather_Project%2C_2003_%28cropped%29.jpg"},
            {"title": "Casey Reas: Process umetnost", "artist": "Casey Reas, 2004-",
             "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Processing-logo.svg/640px-Processing-logo.svg.png"},
        ]
    },
]


# ── Glavni razred ─────────────────────────────────────────────────

class AnalitikBarv:
    def __init__(self):
        self.seja = requests.Session()
        self.seja.headers.update(HEADERS)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=VZPOREDNI_PRENOSI,
            pool_maxsize=VZPOREDNI_PRENOSI * 2,
            max_retries=requests.adapters.Retry(total=2, backoff_factor=0.3)
        )
        self.seja.mount('https://', adapter)
        self.seja.mount('http://', adapter)

    def obdelaj_dobo(self, doba: dict, checkpoint: dict) -> dict:
        naslov = doba['naslov']

        if naslov in checkpoint:
            print(f"  Preskakujem '{naslov}' (ze v checkpointu)")
            return checkpoint[naslov]

        print(f"\n{'='*60}")
        print(f"DOBA: {naslov} ({doba['leto']})")
        print(f"{'='*60}")
        cas_start = time.time()

        # 1. Zberemo URL-je
        print(f"\n  [1/3] Zbiram URL-je slik iz Wikimedia Commons...")
        url_seznam = zberi_url_iz_wikimedie(
            self.seja, doba['iskalni_izrazi'],
            #cilj_url=CILJ_SLIK_NA_DOBO * 5
            cilj_url=CILJ_SLIK_NA_DOBO * 2
        )
        print(f"  Skupaj URL-jev: {len(url_seznam)}")

        if not url_seznam:
            return self._privzeta_doba(doba)

        # 2. Vzporedno prenasamo slike
        print(f"\n  [2/3] Prenasam slike ({VZPOREDNI_PRENOSI} vzporednih)...")
        vsi_piksli = []
        obdelano = 0
        napake = 0
        url_za_analizo = url_seznam[:CILJ_SLIK_NA_DOBO * 4]

        with ThreadPoolExecutor(max_workers=VZPOREDNI_PRENOSI) as executor:
            futures = {executor.submit(prenesi_in_obdelaj, self.seja, u): u
                       for u in url_za_analizo}
            for fut in as_completed(futures):
                piksli = fut.result()
                if piksli is not None and len(piksli) > 0:
                    vsi_piksli.append(piksli)
                    obdelano += 1
                else:
                    napake += 1

                skupaj = obdelano + napake
                if skupaj % 30 == 0 or skupaj == len(url_za_analizo):
                    with TISK_LOCK:
                        print(f"    OK: {obdelano} | Napake: {napake} "
                              f"| Skupaj: {skupaj}/{len(url_za_analizo)}", flush=True)

                if obdelano >= CILJ_SLIK_NA_DOBO:
                    #executor.shutdown(wait=False, cancel_futures=True)
                    break

        print(f"  Uspesno obdelanih slik: {obdelano}")

        if not vsi_piksli:
            return self._privzeta_doba(doba)

        # 3. K-means clustering
        skupaj_px = sum(len(p) for p in vsi_piksli)
        print(f"\n  [3/3] K-means clustering na {skupaj_px:,} pikslih...")
        vsi_skupaj = np.concatenate(vsi_piksli, axis=0)
        paleta = ekstrahiraj_barve_iz_pikslov(vsi_skupaj, n_barv=N_BARV)

        if not paleta:
            paleta = ["#808080"] * N_BARV

        imena = [poimenuj_barvo(b) for b in paleta]

        cas_trajanje = time.time() - cas_start
        print(f"\n  Paleta za '{naslov}': {' | '.join(paleta)}")
        print(f"  Trajanje: {cas_trajanje:.0f}s ({obdelano} slik)")

        return {
            "leto":   doba['leto'],
            "naslov": naslov,
            "paleta": paleta,
            "imena":  imena,
            "opis":   doba['opis'],
            "dela":   doba.get('dela', []),
        }

    def _privzeta_doba(self, doba: dict) -> dict:
        return {
            "leto":   doba['leto'],
            "naslov": doba['naslov'],
            "paleta": ["#808080"] * N_BARV,
            "imena":  ["Nevtralna"] * N_BARV,
            "opis":   doba['opis'],
            "dela":   doba.get('dela', []),
        }

    def shrani_v_json(self, podatki: list, pot: str = IZHOD_DAT):
        with open(pot, 'w', encoding='utf-8') as f:
            json.dump(podatki, f, ensure_ascii=False, indent=2)

    def zazeni(self):
        print("EKSTRAHIRANJE BARV IZ UMETNOSTNIH DEL - POPRAVLJENA VERZIJA")
        print("=" * 60)
        print(f"  Cilj: {CILJ_SLIK_NA_DOBO}+ slik per dobo")
        print(f"  Vzporedni prenosi: {VZPOREDNI_PRENOSI}")
        print(f"  Checkpoint: {CHECKPOINT_DAT}")
        print(f"  Izhod: {IZHOD_DAT}")
        print("=" * 60)

        checkpoint = nalozi_checkpoint()
        if checkpoint:
            print(f"\n  Checkpoint: {len(checkpoint)} dob ze obdelanih.")

        skupni_start = time.time()
        rezultati = []

        for doba in DOBE:
            rezultat = self.obdelaj_dobo(doba, checkpoint)
            rezultati.append(rezultat)
            checkpoint[doba['naslov']] = rezultat
            shrani_checkpoint(checkpoint)
            self.shrani_v_json(rezultati)

        skupno = time.time() - skupni_start
        print(f"\n{'='*60}")
        print(f"ANALIZA KONCANA! Trajanje: {skupno/60:.1f} minut")
        print(f"{'='*60}")
        print(f"  Podatki shranjeni v: {IZHOD_DAT}")
        print(f"\n  ✅ Odpri OO_projekt.html v brskalniku!")

        if os.path.exists(CHECKPOINT_DAT):
            os.remove(CHECKPOINT_DAT)

        return rezultati


# ── Vstopna tocka ─────────────────────────────────────────────────
if __name__ == "__main__":
    analitik = AnalitikBarv()
    analitik.zazeni()