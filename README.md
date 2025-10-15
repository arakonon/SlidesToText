# 🎯 SlidesToText

**Verwandle deine PDF-Folien in intelligenten Text - automatisch!** 

Extrahiert Text, beschreibt Bilder mit KI und räumt alles auf. Zwei Versionen verfügbar:

- 🌐 **API** (`slidesToText-API.py`) - Google Gemini (funktioniert überall)
- 🏠 **MLX** (`slidesToText-MLX.py`) - Lokal auf Apple Silicon

---

## 🚀 Quick Start

```bash
# Repo klonen
git clone <REPO-URL> && cd SlidesToText

# Python-Umgebung einrichten
python3 -m venv venv && source venv/bin/activate

# Abhängigkeiten installieren (wähle eine Version)
pip install -r requirements-API.txt    # Für Google Gemini
pip install -r requirements-MLX.txt    # Für Apple Silicon

# Los geht's!
python3 slidesToText-API.py deine-folien.pdf
```

---

## 🔑 API Setup (nur für API-Version)

1. Hole dir einen [Google AI Studio](https://aistudio.google.com/) API Key
2. `cp .env.example .env` 
3. Trage deinen Key in die `.env` ein

**Oder** setze einfach: `export GOOGLE_API_KEY="dein-key"`

---

## ✨ Was passiert?

1. **📄 Text rausziehen** - Extrahiert alles aus dem PDF
2. **🧹 Aufräumen** - Entfernt Kopfzeilen, Fußzeilen, doppelte Zeilen
3. **🖼️ Bilder finden** - Sammelt alle Bilder (ignoriert Duplikate)
4. **🤖 KI beschreibt** - Jedes Bild wird intelligent beschrieben
5. **📝 Zusammenfügen** - Text + Bildbeschreibungen = fertig!
6. **🎉 Auto-öffnen** - Deine `outcome_DD.MM.YY_HH:MM.txt` öffnet sich

### 🎮 Interaktiv (API-Version)
- Ordner öffnet sich automatisch → lösche ungewollte Bilder → Enter drücken → fertig!

---

## 🥊 API vs MLX

| | 🌐 API | 🏠 MLX |
|---|---|---|
| **Wo läuft's?** | Überall | Nur Apple Silicon |
| **Internet?** | Ja | Nein |
| **Kosten?** | ~Cents pro PDF | Einmalig Hardware |
| **Speed** | 🚀 | 🐌 |
| **Privatsphäre** | Google sieht's | 100% lokal |

---

## 🛠️ Troubleshooting

**💥 Import Error?** → `pip install -r requirements-XXX.txt`

**🔑 API Key Error?** → Check deine `.env` Datei

**🧠 Out of Memory (MLX)?** → Kleinere PDFs oder weniger Bilder

**🌍 Netzwerk Error?** → Internet-Verbindung prüfen

---

## 🎯 Pro-Tipps

- **Große PDFs?** Dauert ein paar Minuten - chill einfach ☕
- **Viele gleiche Logos?** Werden automatisch ignoriert 
- **Schlechte Bilder?** Lösch sie einfach aus dem `images/` Ordner
- **Alte Outputs?** Landen automatisch in `Legacy Outcomes/`

---

## 📁 Was entsteht?

```
📂 SlidesToText/
├── 📄 outcome_15.10.25_11:16.txt    ← Dein fertiger Text
├── 📁 Legacy Outcomes/              ← Alle alten Versionen
└── 🔐 .env                         ← Dein API Key
```

---

**Made with ❤️ by Konrad Czernohous • 2025**

---

*🤖 Dieses README wurde mit KI generiert und ist cooler als deine Folien*