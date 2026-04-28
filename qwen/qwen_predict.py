import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np


MODEL_PATH = "/home/ubuntu/stephen/02-weight/Qwen3-VL-4B-Instruct"


class Qwen3VL4BInferencer:
    def __init__(self, model_path: str = MODEL_PATH, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, image: Image.Image, prompt: str, **kwargs) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}],
                "text": prompt
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output_ids = self.model.generate(**inputs, **kwargs)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        return output_text


def preprocess_image(image: Image.Image, target_size: tuple = (448, 448)) -> torch.Tensor:
    image = image.resize(target_size, Image.BILINEAR)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    return image_tensor


if __name__ == "__main__":
    inferencer = Qwen3VL4BInferencer()
    image = Image.new("RGB", (448, 448), color="blue")
    result = inferencer.generate(image, "Describe this image")
    print(result)