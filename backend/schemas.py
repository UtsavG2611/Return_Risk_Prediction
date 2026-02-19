from pydantic import BaseModel


class OrderFeatures(BaseModel):
    product_category: str
    product_price: float
    order_quantity: int
    user_age: int
    user_gender: str
    payment_method: str
    shipping_method: str
    discount_applied: float


class PredictionResponse(BaseModel):
    prediction_label: str


class RegisterRequest(BaseModel):
    name: str
    userid: str
    password: str


class LoginRequest(BaseModel):
    userid: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    name: str | None = None
