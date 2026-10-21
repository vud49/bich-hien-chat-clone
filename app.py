import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Force UTF-8 on Windows
sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.brain import ChatBrain

app = FastAPI(title="Digital Clone Chat - Realistic Messenger Simulator")

import base64
import uuid

# Mount static folder
static_dir = WORKSPACE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
uploads_dir = static_dir / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize ChatBrain
brain = ChatBrain()


class ChatRequest(BaseModel):
    message: Optional[str] = ""
    history: Optional[List[Dict[str, Any]]] = []
    image_data: Optional[str] = None
    mime_type: Optional[str] = "image/jpeg"


class FeedbackRequest(BaseModel):
    user_message: str
    corrected_reply: str


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = WORKSPACE_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/info")
async def get_info():
    return {
        "target_name": "ngyn ngc bich hien",
        "target_display": "Bích Hiền",
        "target_username": "bchhniiee_",
        "user_name": "Vũ Đình Khoa",
        "user_username": "vud49",
        "total_pairs": len(brain.pairs),
        "total_sessions": len(brain.sessions),
        "metrics": brain.style_metrics.get("summary", {})
    }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    user_text = (req.message or "").strip()
    image_bytes = None
    saved_image_url = None
    mime_type = req.mime_type or "image/jpeg"

    if req.image_data:
        try:
            if "," in req.image_data:
                header, b64_str = req.image_data.split(",", 1)
                if "image/png" in header:
                    mime_type = "image/png"
                    ext = "png"
                elif "image/webp" in header:
                    mime_type = "image/webp"
                    ext = "webp"
                elif "image/gif" in header:
                    mime_type = "image/gif"
                    ext = "gif"
                else:
                    mime_type = "image/jpeg"
                    ext = "jpg"
            else:
                b64_str = req.image_data
                ext = "jpg"

            image_bytes = base64.b64decode(b64_str)
            filename = f"img_{uuid.uuid4().hex[:12]}.{ext}"
            filepath = uploads_dir / filename
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            saved_image_url = f"/static/uploads/{filename}"
        except Exception as e:
            print(f"[app] Lỗi giải mã ảnh: {e}")

    if not user_text and not image_bytes:
        raise HTTPException(status_code=400, detail="Vui lòng gửi tin nhắn hoặc hình ảnh")

    # Generate response from brain
    brain_out = brain.generate_response(user_text, req.history, image_bytes=image_bytes, mime_type=mime_type)

    if isinstance(brain_out, dict):
        raw_reply = brain_out.get("reply", "...")
        reaction = brain_out.get("reaction")
    else:
        raw_reply = str(brain_out)
        reaction = None

    # Split into bubbles by <br_msg>
    raw_bubbles = [b.strip() for b in raw_reply.split("<br_msg>")]
    bubbles = [b for b in raw_bubbles if b]
    if not bubbles:
        bubbles = ["..."]

    return {
        "raw_reply": raw_reply,
        "bubbles": bubbles,
        "reaction": reaction,
        "saved_image_url": saved_image_url
    }


@app.post("/api/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    if not req.user_message.strip() or not req.corrected_reply.strip():
        raise HTTPException(status_code=400, detail="Nội dung chỉnh sửa không hợp lệ")

    brain.add_feedback_pair(req.user_message, req.corrected_reply)
    return {
        "status": "success",
        "message": "Đã lưu phản hồi và cập nhật trí nhớ AI thành công!",
        "total_pairs": len(brain.pairs)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
