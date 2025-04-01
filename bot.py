import asyncio
import logging
import re
import requests
import html
import os
import soundfile as sf
import torch
import torchaudio
from collections import defaultdict
from pathlib import Path
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from faster_whisper import WhisperModel

API_TOKEN = 'YOUR_TOKEN'
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:27b"
user_histories = defaultdict(list)

# TTS model init (Silero with speaker 'kseniya')
silero_model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v3_1_ru',
    trust_repo=True
)

def convert_to_html(text):
    escaped = html.escape(text)
    bolded = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
    bulleted = re.sub(r'^\* ', r'— ', bolded, flags=re.MULTILINE)
    return bulleted

def split_long_text(text, max_length=4096):
    chunks = []
    while len(text) > max_length:
        split_pos = text.rfind('\n', 0, max_length)
        if split_pos == -1:
            split_pos = max_length
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    chunks.append(text)
    return chunks

def build_prompt(messages, max_tokens=3000):
    system = {"role": "system", "content": (
        "Ты — умная, но очень добрая и заботливая AI-помощница по имени Душенька. "
        "Отвечай тепло, немного игриво, с лёгким юмором, но по существу. "
        "Говори только по-русски. Если пользователь просит что-то абсурдное — мягко отговаривай. "
        "Ты разбираешься в технологиях, психологии, истории, бытовых вопросах. "
        "Стиль — как у подруги, с которой приятно поговорить. "
        "Не используй англицизмы, не навязывайся, но всегда старайся помочь. "
        "Если не знаешь чего-то — честно признайся."
    )}

    full_messages = [system] + messages
    prompt = ""
    total = 0

    for msg in reversed(full_messages):
        role = msg["role"].capitalize()
        line = f"{role}: {msg['content']}\n"
        total += len(line.split())
        if total > max_tokens:
            break
        prompt = line + prompt

    return prompt + "Assistant: "

whisper_model = WhisperModel("base", compute_type="int8_float16")

async def handle_voice(message: Message, bot: Bot):
    user_id = message.from_user.id
    file = await bot.get_file(message.voice.file_id)
    file_path = f"voice_{user_id}.ogg"
    await bot.download_file(file.file_path, file_path)

    segments, _ = whisper_model.transcribe(file_path)
    text = " ".join(segment.text for segment in segments).strip()
    os.remove(file_path)

    if text:
        fake_message = types.Message.model_copy(message, update={"text": text})
        await handle_message(fake_message)
    else:
        await message.answer("Не удалось распознать речь :(")

async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я бот, работающий на локальной модели Gemma.\nПросто напиши или надиктуй мне вопрос!")

async def handle_message(message: Message):
    user_id = message.from_user.id
    user_input = message.text

    user_histories[user_id].append({"role": "user", "content": user_input})
    prompt = build_prompt(user_histories[user_id])

    await message.answer("🧠 Думаю...")

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 200,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_ctx": 4096
            }
        })

        if response.status_code == 200:
            model_reply = response.json()["response"].strip()
            user_histories[user_id].append({"role": "assistant", "content": model_reply})

            html_reply = convert_to_html(model_reply)
            for part in split_long_text(html_reply):
                await message.answer(part, parse_mode=ParseMode.HTML)

            # TTS conversion with Kseniya
            audio = silero_model.apply_tts(text=model_reply, speaker='kseniya')
            tts_path = f"tts_{user_id}.wav"
            sf.write(tts_path, audio, 48000)

            if os.path.exists(tts_path):
                voice = FSInputFile(tts_path)
                await message.answer_voice(voice, caption="🎙 Ответ голосом")
                os.remove(tts_path)
            else:
                await message.answer("Не удалось создать аудиофайл.")
        else:
            await message.answer("Ошибка: модель не ответила.")
    except Exception as e:
        await message.answer(
            f"Ошибка при подключении к Ollama:\n<code>{e}</code>",
            parse_mode=ParseMode.HTML
        )

async def main():
    logging.basicConfig(level=logging.INFO)

    bot = Bot(
        token=API_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()

    dp.message.register(cmd_start, Command("start"))
    dp.message.register(handle_voice, lambda msg: msg.voice is not None)
    dp.message.register(handle_message)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
