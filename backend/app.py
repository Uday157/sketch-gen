from pathlib import Path
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi import FastAPI, File, UploadFile, Form, Request
import cv2
import numpy as np
import uuid
import os

app = FastAPI()

# Root = sketch-generator/
BASE_DIR = Path(__file__).resolve().parent.parent

# Templates and static folders
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Home route
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Upload route
@app.post("/upload")
async def upload(file: UploadFile = File(...), intensity: str = Form("light")):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"error": "Invalid image file"}, status_code=400)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray

    # Adjust sketch based on intensity
    if intensity == "light":
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    elif intensity == "dark":
        blur = cv2.GaussianBlur(inv, (11, 11), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=180.0)
    elif intensity == "darker":
        blur = cv2.GaussianBlur(inv, (1, 1), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=100.0)
    else:
        return JSONResponse(content={"error": "Invalid intensity level"}, status_code=400)

    filename = f"{uuid.uuid4()}.png"
    output_dir = BASE_DIR / "static" / "sketches"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    cv2.imwrite(str(output_path), sketch)

    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
