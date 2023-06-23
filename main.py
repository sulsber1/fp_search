import time
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import API_CPU_fp_search

# Preload when the API starts
smile_from_db, ref_fps = API_CPU_fp_search.startup()


class Structure(BaseModel):
    smiles: str


class Query_Response(BaseModel):
    result_set: list[dict]

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/CPU_fp/")
def get_fp(smileObj : Structure):
    now = time.perf_counter()
    results = API_CPU_fp_search.main(smile_from_db, ref_fps, smileObj.smiles)
    later = time.perf_counter()
    print(f"{later - now}")
    #return Query_Response(**results)
    return results