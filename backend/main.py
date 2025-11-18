from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import numpy as np
import uvicorn
import json


from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    import uvicorn
    from watchgod import watch
    uvicorn.run("main:app", host = "127.0.0.1", port = 8000,reload=True, workers=2)

