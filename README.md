# Assistant IA – analyse embryonnaire (prototype)

Entraînement : Google Colab. Tout le reste : en local (VS Code).

## Installation (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # puis change les mots de passe
docker compose up -d db     # MySQL seulement
```

## Lancer
```powershell
uvicorn api.main:app --reload          # terminal 1  -> http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py     # terminal 2  -> http://localhost:8501
```
Test rapide : ouvre http://127.0.0.1:8000/health

## Modèle
Après l'entraînement sur Colab, dépose dans `models/` :
- `modele.pt` (state_dict d'un ResNet18)
- `classes.json` (liste des phases, dans l'ordre des sorties du modèle)

## Structure
- `notebooks/` : notebooks Colab (exploration, entraînement)
- `api/` : FastAPI    - `app/` : Streamlit    - `db/` : schéma MySQL
- `models/` : modèle entraîné (non versionné)    - `data/` : données (non versionnées)
