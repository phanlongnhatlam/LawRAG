from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api.routes import chat, upload

app = FastAPI(
    title="LawRAG API",
    description="LawRAG API",
    version="1.0",
)

app.include_router(chat.router)
app.include_router(upload.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn gọi API
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các method (GET, POST, PUT, DELETE)
    allow_headers=["*"],
)
@app.get("/health")
async def health():
    return {"status": "ok"}
