import asyncio
import logging
import requests
from collections import defaultdict
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# 🔐 Токен от BotFather
API_TOKEN = 'YOUR_ROKEN'

# ⚙️ Настройки Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:27b"

# 🧠 Храним истории по user_id
user_histories = defaultdict(list)

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

# /reset — очистка истории
async def cmd_reset(message: Message):
    user_id = message.from_user.id
    user_histories[user_id] = []
    await message.answer("История очищена.")

# Обработка текстовых сообщений
async def handle_message(message: Message):
    user_id = message.from_user.id
    user_input = message.text

    # Добавляем в историю
    user_histories[user_id].append({"role": "user", "content": user_input})
    prompt = build_prompt(user_histories[user_id])

    # Запрос к Ollama
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        })

        if response.status_code == 200:
            model_reply = response.json()["response"].strip()
            user_histories[user_id].append({"role": "assistant", "content": model_reply})
            await message.answer(model_reply)
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
    dp.message.register(cmd_reset, Command("reset"))
    dp.message.register(handle_message)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
