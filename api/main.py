from fastapi import FastAPI,UploadFile,File
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
MODEL_Version1 = tf.keras.models.load_model("D:\\Cervical_Cancer_app\\models\\1")
MODEL_Version2 = tf.keras.models.load_model("D:\\Cervical_Cancer_app\\models\\2")
CLASS_NAMES = ['Dyskeratotic cells: Can indicate cellular changes that may be precancerous or cancerous, but their presence alone does not definitively indicate cancer.','Koliocytotic cells (koilocytes): Strongly associated with high-risk HPV infection and a significant risk factor for cervical cancer.','Metaplastic cells: Generally not directly associated with cancer, but their presence might complicate the interpretation of Pap smear results.','Parabasal cells: Can be indicative of inflammation, infection, or hormonal changes, but not typically associated with cancer on their own','Superficial-intermediate cells: Normal part of cervical cell turnover, not directly associated with cervical cancer.']
CLASS_NAMES1 = ['Dyskeratotic cells','Koliocytotic cells','Metaplastic cells','Parabasal cells','Superficial-intermediate cells']
def read_file_as_image(data) ->np.ndarray:

    # image = np.array(Image.open(BytesIO(data)))
    image = Image.open(BytesIO(data))
    return image
@app.post("/predict")
async def predict(
    file : UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    xx = (256,256)
    img = image.resize(xx)
    img_batch = np.expand_dims(img,0)
    predictions1 = MODEL_Version1.predict(img_batch)
    # prediction2 = MODEL_Version2.predict(img_batch)
    predicted_class1 = CLASS_NAMES1[np.argmax(predictions1)]
    # predicted_class2 = CLASS_NAMES[np.argmax(prediction2)]
    print(predicted_class1)
    confidence = np.max(predictions1[0])
   
    print(f"I am model {confidence} sure")
    # print(predicted_class2)
    # confidence2 = np.max(prediction2[0])
    # confidence2 *=100
    # print(f'I am pro model and I am {confidence2} sure')
    return {
        "class":predicted_class1,
        'confidence':float(confidence)
        # 'Cells2' : predicted_class2,
        # 'confidence2' : float(confidence2)
    }
if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=800)