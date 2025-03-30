import asyncio
import logging
import re
import requests
import html
from collections import defaultdict
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# 🔐 Токен от BotFather
API_TOKEN = 'YOUR_TOKEN'

# ⚙️ Настройки Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:27b"

# 🧠 Храним истории по user_id
user_histories = defaultdict(list)

# 🎟️ Замена ** на <b> и *  на — для HTML
def convert_to_html(text):
    escaped = html.escape(text)
    bolded = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
    bulleted = re.sub(r'^\* ', r'—', bolded, flags=re.MULTILINE)
    return bulleted

# 🏗️ Собираем prompt из истории
def build_prompt(messages, max_tokens=3000):
    prompt = ""
    total = 0
    for msg in reversed(messages):
        role = "User" if msg["role"] == "user" else "Assistant"
        line = f"{role}: {msg['content']}\n"
        total += len(line.split())
        if total > max_tokens:
            break
        prompt = line + prompt
    return prompt + "Assistant: "

# /start — приветствие
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я бот, работающий на локальной модели Gemma.\nПросто напиши мне что-нибудь!")

# Обработка текстовых сообщений
async def handle_message(message: Message):
    user_id = message.from_user.id
    user_input = message.text

    user_histories[user_id].append({"role": "user", "content": user_input})
    prompt = build_prompt(user_histories[user_id])

    # 🧠 Думаю...
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
            await message.answer(html_reply, parse_mode=ParseMode.HTML)
        else:
            await message.answer("Ошибка: модель не ответила.")
    except Exception as e:
        await message.answer(
            f"Ошибка при подключении к Ollama:\n<code>{e}</code>",
            parse_mode=ParseMode.HTML
        )

# 🚀 Запуск бота
async def main():
    logging.basicConfig(level=logging.INFO)

    bot = Bot(
        token=API_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()

    dp.message.register(cmd_start, Command("start"))
    dp.message.register(handle_message)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
