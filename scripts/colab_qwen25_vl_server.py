import base64
import io
import re
import subprocess
import threading
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

print(f"[*] Loading {MODEL_ID} onto Colab GPU...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()
print("[+] Qwen2.5-VL-3B-Instruct ready!")

app = FastAPI(title="SentinelEdge Cloud Qwen2.5-VL Bridge")


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[dict[str, Any]]
    max_tokens: int = 240
    temperature: float = 0.0


def decode_data_url_image(data_url: str) -> Image.Image:
    raw_b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    return Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "online", "model": MODEL_ID}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> dict[str, Any]:
    images: list[Image.Image] = []
    qwen_messages: list[dict[str, Any]] = []

    for msg in req.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for item in content:
                if item.get("type") == "image_url":
                    img = decode_data_url_image(item["image_url"]["url"])
                    images.append(img)
                    parts.append({"type": "image", "image": img})
                elif item.get("type") == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
            qwen_messages.append({"role": role, "content": parts})
        else:
            qwen_messages.append({"role": role, "content": str(content)})

    text_prompt = processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=images if images else None,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0.0
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    return {
        "id": f"sentinel-vlm-{int(time.time())}",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop"
            }
        ]
    }


def start_cloudflare_tunnel(port: int = 8001) -> None:
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    url_pattern = re.compile(r"(https://[a-z0-9-]+\.trycloudflare\.com)")
    assert proc.stdout is not None
    for line in proc.stdout:
        match = url_pattern.search(line)
        if match:
            public_url = match.group(1)
            print("=" * 72)
            print(" COPY THIS COMMAND INTO YOUR LAPTOP POWERSHELL:")
            print(f' $env:VLM_API_URL = "{public_url}/v1/chat/completions"')
            print("=" * 72)
            break


if __name__ == "__main__":
    threading.Thread(target=start_cloudflare_tunnel, args=(8001,), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8001)