## 🎨 Projekt: Spremembe barvnih palet skozi čas

Projekt preučuje evolucijo barvnih palet skozi različne umetnostne dobe, od prazgodovine do digitalne dobes.

**Ključna lastnost:** Analiza je narejena na osnovi **10+ slik na dobo** za neprimerno bolj točne rezultate!

## 📋 Struktura projekta

```
.
├── OO_projekt.html        # Interaktivni frontend
├── extract_colors.py      # Pridobivanje slik + ekstrakcija barv
├── requirements.txt       # Python odvisnosti
├── art_periods_data.json  # Generirani podatki (izhod)
├── checkpoint.json        # Vmesno shranjevanje napredka
└── README.md              # Dokumentacija projekta                       # Ta datoteka
```

## 🚀 Kako začeti?

### 1. Namestitev Python odvisnosti

```bash
pip install -r requirements.txt
```

### 2. Ekstrahiranje barv iz 10+ slik

Zaženi glavni script:

```bash
python extract_colors.py
```

**Kaj se zgodi:**
-Pridobivanje slik iz Wikimedia Commons.
-Ekstrakcija dominantnih barv (K-means).
-Agregacija rezultatov po dobah.
-Zapis v art_periods_data.json.

### 3. Zagon projekta

Za pravilen prikaz barv odprite projekt preko lokalnega strežnika:

```bash
python -m http.server 8080
```

Nato odprite v brskalniku:
```bash
http://localhost:8080/OO_projekt.html
```

## 📊 Kaj projekt prikazuje

- 8 umetnostnih obdobij
- 6 dominantnih barv na posamezno obdobje
- barve, izračunane iz realnih umetniških del
- primeri umetnin za vsako dobo
- interaktiven časovni trak (timeline)

---

## ⚙️ Algoritem

### Pridobivanje slik
- Wikimedia Commons API
- fallback URL-ji

### Obdelava slik
- resize (sprememba velikosti) za večjo hitrost obdelave
- RGB standardizacija

### Ekstrakcija barv
- K-means clustering (n=6)
- izbor dominantnih barv

### Agregacija
- združevanje podatkov iz 100+ slik na dobo
- rangiranje barv po frekvenci pojavljanja

---

## 📦 Ključne datoteke

- **extract_colors.py**  
  Glavni cevovod (pipeline): prenos slik, ekstrakcija barv in generiranje JSON izhoda

- **art_periods_data.json**  
  Končni podatki za frontend (avtomatsko generirano)

- **checkpoint.json**  
  Shranjevanje napredka procesa, kar omogoča nadaljevanje ob prekinitvi skripte

---

