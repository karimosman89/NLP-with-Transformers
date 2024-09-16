from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import logging
import sqlite3
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
import uvicorn
from typing import Optional
from io import BytesIO
from PIL import Image
import base64

# Initialize FastAPI app
app = FastAPI()

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth")
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Load models and configurations
with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
    tokenizer_config = json_file.read()
tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

text_model = tf.keras.models.load_model('models/best_lstm_model.keras')
image_model = tf.keras.models.load_model('models/best_vgg16_model.keras')

with open("models/best_weights.json", "r", encoding="utf-8") as json_file:
    best_weights = json.load(json_file)

with open("models/mapper.json", "r", encoding="utf-8") as json_file:
    mapper = json.load(json_file)

# Setup logging
logging.basicConfig(level=logging.INFO)

class ProductData(BaseModel):
    description: str
    image: str  # base64 encoded image string

class NewProductData(BaseModel):
    description: str
    image: str  # base64 encoded image string
    category: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Dummy user database
fake_users_db = {
    "admin": {
        "username": "admin",
        "password": "admin",
        "role": "admin"
    },
    "user": {
        "username": "user",
        "password": "kiko",
        "role": "user"
    }
}

def verify_password(plain_password, hashed_password):
    return plain_password == hashed_password

def get_user(db, username: str):
    return db.get(username)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user or not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    if current_user["role"] not in ["admin", "user"]:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=10, padding="post", truncating="post"
    )
    return padded_sequences

def preprocess_image(image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data)).resize((224, 224))
    img_array = keras.preprocessing.image.img_to_array(image)
    img_array = keras.applications.vgg16.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.post("/api/auth", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/predict")
def predict(data: ProductData, current_user: dict = Depends(get_current_active_user)):
    try:
        text_input = preprocess_text(data.description)
        image_input = preprocess_image(data.image)

        lstm_proba = text_model.predict(text_input)
        vgg16_proba = image_model.predict(image_input)

        concatenate_proba = (
            best_weights[0] * lstm_proba + best_weights[1] * vgg16_proba
        )
        final_predictions = np.argmax(concatenate_proba, axis=1)

        prediction = mapper[str(final_predictions[0])]
        
        # Save prediction to the database
        conn = get_db_connection()
        conn.execute("INSERT INTO predictions (product_id, description, image, prediction) VALUES (?, ?, ?, ?)",
                     (None, data.description, data.image, prediction))  # product_id is None for predictions
        conn.commit()
        conn.close()

        logging.info(f"Prediction made for user {current_user['username']}: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data-ingest")
def data_ingest(new_data: NewProductData, current_user: dict = Depends(get_current_active_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    try:
        conn = get_db_connection()
        conn.execute("INSERT INTO products (product_id, designation, description, image_id, prdtypecode) VALUES (?, ?, ?, ?, ?)",
                     (None, None, new_data.description, new_data.image, new_data.category))  # product_id, designation, image_id are None
        conn.commit()
        conn.close()
        logging.info(f"Data ingested by admin {current_user['username']}")
        return {"message": "Data ingested successfully"}
    except Exception as e:
        logging.error(f"Data ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitor")
def monitor(current_user: dict = Depends(get_current_active_user)):
    try:
        # Example: return model metrics and health status
        metrics = {
            "model_name": "best_lstm_model.keras",
            "accuracy": 0.95,  
            "status": "healthy"
        }
        logging.info(f"Monitor accessed by user {current_user['username']}")
        return metrics
    except Exception as e:
        logging.error(f"Monitoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
