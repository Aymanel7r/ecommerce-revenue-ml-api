from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


final_pipeline = joblib.load("revenue_prediction_pipeline.pkl")

app = FastAPI()


class RevenueInput(BaseModel):
    product_category: str
    region: str
    payment_method: str
    quantity: int
    unit_price: float
    discount: float
    delivery_days: int
    customer_rating: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: RevenueInput):
    order_df = pd.DataFrame([data.model_dump()])

    prediction = final_pipeline.predict(order_df)

    return {"predicted_revenue": prediction[0]}