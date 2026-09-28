"""API FastAPI : /health, /predict, /history."""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker
from torchvision import models, transforms

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "modele.pt"
CLASSES_PATH = ROOT / "models" / "classes.json"
MODEL_VERSION = "v0.1"
FRAME_STEP = 5  # on analyse 1 image sur 5 (à ajuster)

# ---------- Base de données ----------
DB_URL = (
    f"postgresql+psycopg://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST', '127.0.0.1')}:{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB')}"
)
engine = create_engine(DB_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Analyse(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True)
    nom_video = Column(String(255), nullable=False)
    date_analyse = Column(DateTime, default=datetime.utcnow)
    resultat = Column(JSONB, nullable=False)
    confiance_moyenne = Column(Float)
    version_modele = Column(String(50))


# ---------- Modèle ----------
_model = None
_classes = None
_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    """Charge le modèle une seule fois. Attend models/modele.pt + models/classes.json."""
    global _model, _classes
    if _model is not None:
        return _model, _classes
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise HTTPException(503, "Modèle pas encore entraîné (models/modele.pt et classes.json manquants).")
    _classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(_classes))
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    _model = m
    return _model, _classes


def predict_video(path: str):
    model, classes = load_model()
    cap = cv2.VideoCapture(path)
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % FRAME_STEP == 0:
            frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    if not frames:
        raise HTTPException(400, "Impossible de lire des images dans cette vidéo.")
    out = []
    with torch.no_grad():
        for i, rgb in frames:
            probs = torch.softmax(model(_tf(rgb).unsqueeze(0)), dim=1)[0]
            conf, k = probs.max(dim=0)
            out.append({"frame": i, "phase": classes[int(k)], "confiance": float(conf)})
    return out


app = FastAPI(title="Assistant IA – analyse embryonnaire (prototype)")


@app.on_event("startup")
def startup():
    Base.metadata.create_all(engine)


@app.get("/health")
def health():
    return {"status": "ok", "modele_present": MODEL_PATH.exists()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.avi").suffix or ".avi"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        preds = predict_video(tmp_path)
    finally:
        os.unlink(tmp_path)
    conf = float(np.mean([p["confiance"] for p in preds]))
    with Session() as s:
        row = Analyse(nom_video=file.filename, resultat=preds, confiance_moyenne=conf, version_modele=MODEL_VERSION)
        s.add(row)
        s.commit()
        new_id = row.id
    return {"id": new_id, "nom_video": file.filename, "confiance_moyenne": conf, "predictions": preds}


@app.get("/history")
def history(limit: int = 50):
    with Session() as s:
        rows = s.query(Analyse).order_by(Analyse.id.desc()).limit(limit).all()
        return [
            {"id": r.id, "nom_video": r.nom_video, "date_analyse": r.date_analyse.isoformat(),
             "confiance_moyenne": r.confiance_moyenne, "version_modele": r.version_modele}
            for r in rows
        ]
