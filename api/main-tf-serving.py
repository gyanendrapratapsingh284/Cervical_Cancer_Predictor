from fastapi import FastAPI,UploadFile,File
import requests
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
endpoint = "http://localhost:851/v1/models/Cancer_Model:predict"
CLASS_NAMES = ['Dyskeratotic cells: Can indicate cellular changes that may be precancerous or cancerous, but their presence alone does not definitively indicate cancer.','Koliocytotic cells (koilocytes): Strongly associated with high-risk HPV infection and a significant risk factor for cervical cancer.','Metaplastic cells: Generally not directly associated with cancer, but their presence might complicate the interpretation of Pap smear results.','Parabasal cells: Can be indicative of inflammation, infection, or hormonal changes, but not typically associated with cancer on their own','Superficial-intermediate cells: Normal part of cervical cell turnover, not directly associated with cervical cancer.']
# CLASS_NAMES1 = ['Dyskeratotic cells','Koliocytotic cells','Metaplastic cells','Parabasal cells','Superficial-intermediate cells']
def read_file_as_image(data) ->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    # image = Image.open(BytesIO(data))
    return image
@app.post("/predict")
async def predict(
    file : UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    # xx = (256,256)
    # img = image.resize(xx)
    img_batch = np.expand_dims(image,0)
    json_data = {
        "instances":img_batch.tolist()
    }
    response = requests.post(endpoint,json=json_data)
    prediction = response.json()["predictions"][0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    print(predicted_class)
    confidence = np.max(prediction)
    # confidence *= 100
    print(f'I am {confidence} % sure')
    return {
        "class":predicted_class,
        'confidence':float(confidence)
        }
# app.mount('/predict/frontend/',app = solara.server.fastapi.app)
if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=800)