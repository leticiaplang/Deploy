from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from utils import transform_dict_to_pandas
from model_tree import DecisionTreeModel
from typing import Optional


# instanciando objeto
app = FastAPI()

class Music(BaseModel): # payload
    loudness: float
    key: int
    explicit: int
    acousticness: float
    danceability: float
    energy: float
    valence: float
    name: str
    popularity: Optional[float]
    mode: Optional[int]


# fazendo as rotas (acoes por meio do @)
@app.get("/") #caminho default com barra
async def hello_word():
    return {'message':'hello word'}

#solicitar informações para a api
@app.post("/predict")
async def predict_pipe(music:Music):
    music_dict = music.dict()
    df = transform_dict_to_pandas(music_dict, ['acousticness', 'danceability', 'energy', 'valence', 'key', 'loudness', 'explicit'])
    #carregar arvore de decisao
    dt = DecisionTreeModel()
    predict_value = dt.predict(df)[0] ##devolve um array
    response_body = {}
    response_body['name'] = music.name
    response_body['predict'] = predict_value
    response_body['received_values'] = music_dict
    return response_body
