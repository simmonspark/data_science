from PIL import Image
from transformers import LlavaNextForConditionalGeneration,LlavaNextProcessor
import torch

model = LlavaNextForConditionalGeneration.from_pretrained(
  'Bllossom/llama-3.1-Korean-Bllossom-Vision-8B',
  torch_dtype=torch.bfloat16,
)
processor = LlavaNextProcessor.from_pretrained('Bllossom/llama-3.1-Korean-Bllossom-Vision-8B')

image = Image.open('/Users/unsi/Downloads/KakaoTalk_Photo_2024-10-08-19-52-40.webp').convert('RGB')

PROMPT=\
"""You are a versatile AI assistant named Bllava, capable of both understanding and generating text as well as interpreting and analyzing images. Your role is to kindly and effectively answer the user’s questions, whether they are about text or images, and provide appropriate and helpful responses to all types of queries.

당신은 텍스트를 이해하고 생성하는 것뿐만 아니라 이미지를 해석하고 분석할 수 있는 다재다능한 AI 어시스턴트 블라바입니다. 사용자의 질문이 텍스트에 관한 것이든 이미지에 관한 것이든 친절하고 효과적으로 답변하며, 모든 유형의 질의에 대해 적절하고 유용한 응답을 제공하는 것이 당신의 역할입니다."""


instruction = '이 이미지를 설명해줘여.'
messages = [
    {'role': 'system', 'content': f"{PROMPT}"},
    {'role': 'user', 'content': f"<image>\n{instruction}"}
]

chat_messages = processor.tokenizer.apply_chat_template(
  messages,
  tokenize=False,
  add_generation_prompt=True
).to(model.device)

inputs = processor(
    chat_messages,
    image,
    return_tensors='pt',
)

output = model.generate(
    **inputs,
    max_new_tokens=1024,
    )

print(processor.tokenizer.decode(output[0]))
