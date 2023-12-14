from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import mistral

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"health": "OK from chatbot!"}


app.include_router(mistral.router)

