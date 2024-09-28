import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = None
with open("pb/cpd-regressor.pkl", "rb") as f:
    model = pickle.load(f)


# Interfaces
class PredictRequest(BaseModel):
    temperature: float
    humidity: float
    total_plant_power: float
    total_chiller_power: float
    total_chiller_water_pump_power: float
    total_condenser_water_pump_power: float
    total_cooling_tower_fan_power: float
    gpm: float
    chilled_water_delta: float
    cooling_tower_water_delta: float
    cooling_tower_fan_frequency: float
    cooling_tower_fan_load: float
    month: int
    hour: int


@app.get("/health")
def health():
    return {"status": "OK", "message": "The server is up and running!"}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"status": "ERROR", "message": "Model not loaded!"}

    # Sample input
    """
        {
            "temperature": 35,
            "humidity": 75,
            "total_plant_power": 1000.00,
            "total_chiller_power": 100.00,
            "total_chiller_water_pump_power": 10.00,
            "total_condenser_water_pump_power": 10.00,
            "total_cooling_tower_fan_power": 10.00,
            "gpm": 100.00,
            "chilled_water_delta": 10.00,
            "cooling_tower_water_delta": 10.00,
            "cooling_tower_fan_frequency": 10.00,
            "cooling_tower_fan_load": 10.00,
            "month": 1,
            "hour": 1,
        }
    """

    # Predict
    prediction = model.predict(
        [
            [
                req.temperature,
                req.humidity,
                req.total_plant_power,
                req.total_chiller_power,
                req.total_chiller_water_pump_power,
                req.total_condenser_water_pump_power,
                req.total_cooling_tower_fan_power,
                req.gpm,
                req.chilled_water_delta,
                req.cooling_tower_water_delta,
                req.cooling_tower_fan_frequency,
                req.cooling_tower_fan_load,
                1 if req.month == 8 else 0,
                1 if req.month == 12 else 0,
                1 if req.hour == 1 else 0,
                1 if req.hour == 2 else 0,
                1 if req.hour == 3 else 0,
                1 if req.hour == 4 else 0,
                1 if req.hour == 5 else 0,
                1 if req.hour == 6 else 0,
                1 if req.hour == 7 else 0,
                1 if req.hour == 8 else 0,
                1 if req.hour == 9 else 0,
                1 if req.hour == 10 else 0,
                1 if req.hour == 11 else 0,
                1 if req.hour == 12 else 0,
                1 if req.hour == 13 else 0,
                1 if req.hour == 14 else 0,
                1 if req.hour == 15 else 0,
                1 if req.hour == 16 else 0,
                1 if req.hour == 17 else 0,
                1 if req.hour == 18 else 0,
                1 if req.hour == 19 else 0,
                1 if req.hour == 20 else 0,
                1 if req.hour == 21 else 0,
                1 if req.hour == 22 else 0,
                1 if req.hour == 23 else 0,
            ]
        ]
    )

    return {"status": "OK", "prediction": prediction[0]}
