from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(
    title="S Sanskar AI",
    version="1.0.0"
)

AI_NAME = "S Sanskar"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"{AI_NAME} is starting...")
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.eval()

print(f"{AI_NAME} is ready!")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "name": AI_NAME,
        "status": "online"
    }


@app.get("/v1/models")
def models():
    return {
        "models": [
            {
                "id": "s-sanskar",
                "name": AI_NAME,
                "type": "language-model"
            }
        ]
    }


@app.post("/v1/chat")
def chat(request: ChatRequest):

    messages = [
        {
            "role": "system",
            "content": (
                "You are S Sanskar, a helpful and friendly AI assistant. "
                "Answer clearly and naturally."
            )
        },
        {
            "role": "user",
            "content": request.message
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return {
        "success": True,
        "model": AI_NAME,
        "message": response.strip()
    }