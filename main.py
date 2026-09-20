import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

app = FastAPI(title="Indian Bird Bioacoustic API")

# Enable CORS for React frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Vercel domain in production if desired
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Cornell BirdNET Analyzer...")
analyzer = Analyzer()

# Whitelist: 10 Common Indian Birds with metadata
TARGET_INDIAN_BIRDS = {
    "Pavo cristatus_Indian Peafowl": {
        "common": "Indian Peafowl",
        "scientific": "Pavo cristatus",
        "iucn": "Least Concern (LC)",
        "habitat": "Forests, scrublands & open fields"
    },
    "Eudynamys scolopaceus_Asian Koel": {
        "common": "Asian Koel",
        "scientific": "Eudynamys scolopaceus",
        "iucn": "Least Concern (LC)",
        "habitat": "Canopy trees, urban parks & suburban gardens"
    },
    "Athene brama_Spotted Owlet": {
        "common": "Spotted Owlet",
        "scientific": "Athene brama",
        "iucn": "Least Concern (LC)",
        "habitat": "Tree cavities, old buildings & orchards"
    },
    "Corvus splendens_House Crow": {
        "common": "House Crow",
        "scientific": "Corvus splendens",
        "iucn": "Least Concern (LC)",
        "habitat": "Towns, coastal plains & urban centers"
    },
    "Acridotheres tristis_Common Myna": {
        "common": "Common Myna",
        "scientific": "Acridotheres tristis",
        "iucn": "Least Concern (LC)",
        "habitat": "Urban areas, cultivations & open forests"
    },
    "Psittacula krameri_Rose-ringed Parakeet": {
        "common": "Rose-ringed Parakeet",
        "scientific": "Psittacula krameri",
        "iucn": "Least Concern (LC)",
        "habitat": "Woodlands, orchards & city gardens"
    },
    "Pycnonotus cafer_Red-vented Bulbul": {
        "common": "Red-vented Bulbul",
        "scientific": "Pycnonotus cafer",
        "iucn": "Least Concern (LC)",
        "habitat": "Scrub, agricultural lands & garden shrubs"
    },
    "Psilopogon haemacephalus_Coppersmith Barbet": {
        "common": "Coppersmith Barbet",
        "scientific": "Psilopogon haemacephalus",
        "iucn": "Least Concern (LC)",
        "habitat": "Forest edges, parks & roadside canopies"
    },
    "Milvus migrans_Black Kite": {
        "common": "Black Kite",
        "scientific": "Milvus migrans",
        "iucn": "Least Concern (LC)",
        "habitat": "Cities, harbors, wetlands & waste dump margins"
    },
    "Copsychus saularis_Oriental Magpie-Robin": {
        "common": "Oriental Magpie-Robin",
        "scientific": "Copsychus saularis",
        "iucn": "Least Concern (LC)",
        "habitat": "Low woodland, parks, cultivation & plantations"
    }
}

@app.get("/species")
def get_supported_species():
    """Returns the list of 10 calibrated Indian species."""
    return list(TARGET_INDIAN_BIRDS.values())

@app.post("/predict")
async def predict_bird_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Supported audio formats: WAV, MP3, OGG, M4A, FLAC.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        recording = Recording(analyzer, temp_path, min_conf=0.1)
        recording.analyze()

        raw_detections = {}
        for d in recording.detections:
            key = f"{d['scientific_name']}_{d['common_name']}"
            conf = float(d['confidence'])
            if key not in raw_detections or conf > raw_detections[key]:
                raw_detections[key] = conf

        matched_predictions = []
        for key, meta in TARGET_INDIAN_BIRDS.items():
            if key in raw_detections:
                matched_predictions.append({
                    "common": meta["common"],
                    "scientific": meta["scientific"],
                    "confidence": round(raw_detections[key] * 100, 2),
                    "iucn": meta["iucn"],
                    "habitat": meta["habitat"]
                })

        # Sort descending by confidence score
        matched_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        if not matched_predictions:
            outside_detection = None
            if raw_detections:
                top_raw = max(raw_detections.items(), key=lambda x: x[1])
                outside_detection = {
                    "raw_label": top_raw[0].split('_')[-1],
                    "confidence": round(top_raw[1] * 100, 2)
                }
            return {
                "detected": False,
                "outside_detection": outside_detection,
                "matches": []
            }

        return {
            "detected": True,
            "top_match": matched_predictions[0],
            "matches": matched_predictions
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
