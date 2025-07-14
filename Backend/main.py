#API Packages
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#Route Calling From routes.py
from routes import app as route

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include FAQ API routes
app.include_router(route)
