import io
import pandas as pd
import uuid
import time
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from sqlalchemy.orm import Session
import datetime

from model import EnergyTheftPredictor
from database import engine, get_db
import db_models

db_models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Sudarshan Core API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

predictor = EnergyTheftPredictor()

BATCH_THRESHOLD = 500

VALID_CLASSES = [
    col.replace("Class_", "")
    for col in predictor.feature_columns
    if col.startswith("Class_")
]

# --- SCHEMAS ---

class PredictionInput(BaseModel):
    fans_electricity: float
    cooling_electricity: float
    heating_electricity: float
    interior_lights_electricity: float
    interior_equipment_electricity: float
    gas_facility: float
    heating_gas: float
    interior_equipment_gas: float
    water_heater_gas: float
    class_type: str = Field(..., alias="class")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "fans_electricity": 1.5,
                "cooling_electricity": 850.9,
                "heating_electricity": 0.8,
                "interior_lights_electricity": 2.2,
                "interior_equipment_electricity": 5.0,
                "gas_facility": 0.3,
                "heating_gas": 0.2,
                "interior_equipment_gas": 0.1,
                "water_heater_gas": 0.2,
                "class": "SmallOffice"
            }
        }
    }

class SinglePredictionRequest(BaseModel):
    tenant_slug: str = "demo-utility"
    prediction_name: str = "Manual Single Audit"
    data: PredictionInput
    
class ClientCreate(BaseModel):
    name: str
    slug: str
    region: str = "Global"
    status: str = "PROVISIONING"



# ─── Domain Rule Tiers ────────────────────────────────────────────────────────

ZERO_IMPOSSIBLE = {
    "Hospital", "OutPatient", "SuperMarket", "LargeHotel", "SmallHotel"
}
ZERO_SUSPICIOUS = {
    "FullServiceRestaurant", "QuickServiceRestaurant", "LargeOffice",
    "MediumOffice", "SmallOffice", "PrimarySchool", "SecondarySchool",
    "StripMall", "Stand-aloneRetail"
}
ZERO_PLAUSIBLE = {
    "MidriseApartment", "Warehouse"
}

NUMERIC_FEATURES = [
    "fans_electricity", "cooling_electricity", "heating_electricity",
    "interior_lights_electricity", "interior_equipment_electricity",
    "gas_facility", "heating_gas", "interior_equipment_gas", "water_heater_gas"
]

def apply_domain_rules(result: dict, features: dict) -> dict:
    building_class = features.get("class", "")
    all_zero = all(float(features.get(f, 0)) == 0 for f in NUMERIC_FEATURES)

    if not all_zero:
        return result  # no zero-consumption case, pass through unchanged

    if building_class in ZERO_IMPOSSIBLE:
        result["domain_flag"] = "CRITICAL_ANOMALY"
        result["domain_note"] = (
            f"{building_class} facilities cannot have zero consumption — "
            "complete shutdown is operationally impossible. "
            "High confidence meter tampering."
        )
        result["domain_tier"] = "impossible"

    elif building_class in ZERO_SUSPICIOUS:
        result["domain_flag"] = "ZERO_CONSUMPTION_WARNING"
        result["domain_note"] = (
            f"Zero consumption is atypical for {building_class} — "
            "possible extended closure, but warrants manual verification."
        )
        result["domain_tier"] = "suspicious"

    elif building_class in ZERO_PLAUSIBLE:
        result["domain_flag"] = "POSSIBLE_VACANCY"
        result["domain_note"] = (
            f"Zero consumption for {building_class} may indicate legitimate vacancy. "
            "Model flagged theft based on statistical pattern — "
            "recommend field verification before action."
        )
        result["domain_tier"] = "plausible"

    return result


# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {
        "message": "Sudarshan Core Engine Online",
        "status": "healthy",
        "model": "Random Forest",
        "version": "1.2.0"
    }

@app.get("/model/classes")
def get_valid_classes():
    return {"valid_classes": VALID_CLASSES}

@app.post("/predict/single")
def predict_single(
    request: SinglePredictionRequest,
    db: Session = Depends(get_db)
):
    start_time = time.time()
    try:
        input_dict = request.data.model_dump(by_alias=True)

        # validate class
        if input_dict["class"] not in VALID_CLASSES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid building class '{input_dict['class']}'. Valid options: {VALID_CLASSES}"
            )

        result = predictor.predict(input_dict)
        result = apply_domain_rules(result, input_dict) 
        exec_time = f"{round(time.time() - start_time, 3)}s"
        scan_id = f"PRD-{str(uuid.uuid4())[:8].upper()}"
        is_theft = result["prediction"] == 1

        theft_predictions = []
        if is_theft:
            theft_predictions.append({
                "record_index": 0,
                "features": input_dict,
                "prediction": result["prediction"],
                "prediction_label": result["prediction_label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "tree_votes": result.get("tree_votes"),
                "anomaly_flags": result.get("anomaly_flags", []),
                "domain_flag": result.get("domain_flag"),
                "domain_note": result.get("domain_note"),
                "domain_tier": result.get("domain_tier")
            })

        db_record = db_models.PredictionRecord(
            id=scan_id,
            tenant_slug=request.tenant_slug,
            prediction_name=request.prediction_name,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            mode="SINGLE",
            records_analyzed=1,
            theft_detected=1 if is_theft else 0,
            accuracy_score="99.61%",
            execution_time=exec_time,
            has_full_records=True,
            theft_predictions=theft_predictions,
            all_predictions=[{
                "record_index": 0,
                "features": input_dict,
                "prediction": result["prediction"],
                "prediction_label": result["prediction_label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "tree_votes": result.get("tree_votes"),
                "anomaly_flags": result.get("anomaly_flags", []),
                "domain_flag": result.get("domain_flag"),
                "domain_note": result.get("domain_note"),
                "domain_tier": result.get("domain_tier"),
            }]
        )
        db.add(db_record)
        db.commit()

        return {
            "success": True,
            "scan_id": scan_id,
            "data": result,
            "execution_time": exec_time
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    tenant_slug: str = Form("demo-utility"),
    prediction_name: str = Form("Batch Audit"),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # check required columns exist
        required_columns = [
            "fans_electricity", "cooling_electricity", "heating_electricity",
            "interior_lights_electricity", "interior_equipment_electricity",
            "gas_facility", "heating_gas", "interior_equipment_gas",
            "water_heater_gas", "class"
        ]
        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=422,
                detail=f"CSV is missing required columns: {missing_cols}"
            )

        # validate and separate bad class rows
        if "class" in df.columns:
            valid_mask = df["class"].isin(VALID_CLASSES)
            invalid_rows = df[~valid_mask]
            df_valid = df[valid_mask].copy()

            skipped_count = len(invalid_rows)
            skipped_classes = invalid_rows["class"].unique().tolist() if skipped_count > 0 else []
        else:
            df_valid = df.copy()
            skipped_count = 0
            skipped_classes = []

        if len(df_valid) == 0:
            raise HTTPException(
                status_code=422,
                detail=f"No valid records to process. All rows had invalid building classes. Valid options: {VALID_CLASSES}"
            )

        # extract area_id if present, then drop before prediction
        area_ids = df_valid["area_id"].tolist() if "area_id" in df_valid.columns else [None] * len(df_valid)
        df_for_model = df_valid.drop(columns=["area_id"], errors="ignore")

        records = df_for_model.to_dict(orient="records")
        results = predictor.predict(records)

        total_records = len(results)
        theft_predictions = []
        all_predictions = []
        theft_count = 0

        for i, res in enumerate(results):
            res = apply_domain_rules(res, records[i])  
            record_data = {
                "record_index": i,
                "area_id": area_ids[i],
                "features": records[i],
                **res
            }

            if res["prediction"] == 1:
                theft_count += 1
                theft_predictions.append(record_data)

            all_predictions.append(record_data)

        under_threshold = total_records <= BATCH_THRESHOLD
        exec_time = f"{round(time.time() - start_time, 3)}s"
        scan_id = f"PRD-{str(uuid.uuid4())[:8].upper()}"

        normal_count = total_records - theft_count
        compromise_rate = round((theft_count / total_records) * 100, 2) if total_records > 0 else 0

        # area breakdown if area_id present
        area_breakdown = {}
        if any(a is not None for a in area_ids):
            for i, res in enumerate(results):
                aid = str(area_ids[i]) if area_ids[i] is not None else "unknown"
                if aid not in area_breakdown:
                    area_breakdown[aid] = {"total": 0, "theft": 0, "normal": 0}
                area_breakdown[aid]["total"] += 1
                if res["prediction"] == 1:
                    area_breakdown[aid]["theft"] += 1
                else:
                    area_breakdown[aid]["normal"] += 1

        db_record = db_models.PredictionRecord(
            id=scan_id,
            tenant_slug=tenant_slug,
            prediction_name=prediction_name,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            mode="BATCH",
            records_analyzed=total_records,
            theft_detected=theft_count,
            accuracy_score="99.61%",
            execution_time=exec_time,
            has_full_records=under_threshold,
            theft_predictions=theft_predictions,
            all_predictions=all_predictions if under_threshold else []
        )
        db.add(db_record)
        db.commit()

        response = {
            "success": True,
            "scan_id": scan_id,
            "has_full_records": under_threshold,
            "group_stats": {
                "total_analyzed": total_records,
                "theft_detected": theft_count,
                "normal_count": normal_count,
                "compromise_rate": f"{compromise_rate}%",
                "theft_percentage": f"{compromise_rate}%",
                "normal_percentage": f"{round(100 - compromise_rate, 2)}%",
            },
            "area_breakdown": area_breakdown,
            "theft_predictions": theft_predictions,
            "execution_time": exec_time
        }

        if under_threshold:
            response["all_predictions"] = all_predictions

        if skipped_count > 0:
            response["warning"] = f"{skipped_count} rows were skipped due to invalid building classes: {skipped_classes}. Valid options: {VALID_CLASSES}"

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/{tenant_slug}")
def get_tenant_logs(
    tenant_slug: str,
    db: Session = Depends(get_db)
):
    records = db.query(db_models.PredictionRecord).filter(
        db_models.PredictionRecord.tenant_slug == tenant_slug
    ).order_by(db_models.PredictionRecord.timestamp.desc()).all()

    history = [
        {
            "id": r.id,
            "name": r.prediction_name,
            "date": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "type": r.mode,
            "records_analyzed": r.records_analyzed,
            "theft_detected": r.theft_detected,
            "has_full_records": r.has_full_records
        }
        for r in records
    ]
    return {"success": True, "history": history}


@app.get("/logs/{tenant_slug}/{scan_id}")
def get_scan_snapshot(
    tenant_slug: str,
    scan_id: str,
    db: Session = Depends(get_db)
):
    record = db.query(db_models.PredictionRecord).filter(
        db_models.PredictionRecord.tenant_slug == tenant_slug,
        db_models.PredictionRecord.id == scan_id
    ).first()

    print("all_predictions sample:", record.all_predictions[0] if record.all_predictions else "empty")

    
    if not record:
        raise HTTPException(status_code=404, detail="Scan not found")

    return {
        "success": True,
        "snapshot": {
            "id": record.id,
            "timestamp": record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": record.mode,
            "records_analyzed": record.records_analyzed,
            "theft_detected": record.theft_detected,
            "accuracy": record.accuracy_score,
            "execution_time": record.execution_time,
            "status": "FLAGGED" if record.theft_detected > 0 else "CLEAN",
            "has_full_records": record.has_full_records,
            "theft_predictions": record.theft_predictions,
            "all_predictions": record.all_predictions
        }
    }
    
@app.get("/clients")
def get_all_clients(db: Session = Depends(get_db)):
    clients = db.query(db_models.Client).order_by(db_models.Client.created_at.desc()).all()
    return {"success": True, "clients": [
        {
            "id": c.id,
            "name": c.name,
            "slug": c.slug,
            "region": c.region,
            "status": c.status,
            "created_at": c.created_at.strftime("%Y-%m-%d %H:%M:%S")
        } for c in clients
    ]}

@app.post("/clients")
def create_client(payload: ClientCreate, db: Session = Depends(get_db)):
    existing = db.query(db_models.Client).filter(db_models.Client.slug == payload.slug).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Slug '{payload.slug}' is already taken")
    
    client = db_models.Client(
        id=f"ORG-{str(uuid.uuid4())[:8].upper()}",
        name=payload.name,
        slug=payload.slug,
        region=payload.region,
        status=payload.status
    )
    db.add(client)
    db.commit()
    return {"success": True, "client": {"id": client.id, "name": client.name, "slug": client.slug}}

@app.delete("/clients/{slug}")
def delete_client(slug: str, db: Session = Depends(get_db)):
    client = db.query(db_models.Client).filter(db_models.Client.slug == slug).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    db.delete(client)
    db.commit()
    return {"success": True, "message": f"Client '{slug}' removed"}

@app.patch("/clients/{slug}")
def update_client_status(slug: str, status: str, db: Session = Depends(get_db)):
    client = db.query(db_models.Client).filter(db_models.Client.slug == slug).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    client.status = status
    db.commit()
    return {"success": True, "message": f"Status updated to {status}"}

@app.get("/logs")
def get_all_logs(db: Session = Depends(get_db)):
    records = db.query(db_models.PredictionRecord).order_by(
        db_models.PredictionRecord.timestamp.desc()
    ).all()

    return {"success": True, "logs": [
        {
            "id": r.id,
            "tenant_slug": r.tenant_slug,
            "prediction_name": r.prediction_name,
            "date": r.timestamp.strftime("%Y-%m-%d"),
            "time": r.timestamp.strftime("%H:%M:%S"),
            "mode": r.mode,
            "records_analyzed": r.records_analyzed,
            "theft_detected": r.theft_detected,
            "execution_time": r.execution_time,
            "status": "CRITICAL" if r.theft_detected > 0 else "SUCCESS"
        }
        for r in records
    ]}