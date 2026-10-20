
import os
import asyncio

import numpy as np
import torch
import torch.nn as nn
import joblib

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command


# === Загрузка модели и scaler ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "diabetes_pima_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_pima.pkl")

model = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scaler = joblib.load(SCALER_PATH)


# === Настройка бота ===
TOKEN = "8061599090:AAHwX-GVEBmCbB8map6ZY5Voltwmlf5a8Ec"

bot = Bot(token=TOKEN)
dp = Dispatcher()


# === Подсказка для пользователя ===
PREDICT_HINT = (
    "👋 Для прогноза вероятности диабета отправьте данные в формате:\n"
    "/predict <беременность> <глюкоза> <давление> <кожа> <инсулин> <BMI> <род_диабет> <возраст>\n\n"
    "ℹ Где:\n"
    "- беременность: число беременностей\n"
    "- глюкоза: уровень глюкозы в крови (mg/dL)\n"
    "- давление: артериальное давление (мм рт. ст.)\n"
    "- кожа: толщина кожной складки (мм)\n"
    "- инсулин: уровень инсулина (mU/L)\n"
    "- BMI: индекс массы тела (вес / рост²)\n"
    "- род_диабет: наследственность диабета (0.0 — нет, 1.0 — сильно выражена)\n"
    "- возраст: возраст (лет)\n\n"
    "Пример:\n"
    "/predict 2 130 70 20 85 32.5 0.4 45\n\n"
    "📏 Чтобы рассчитать BMI, используйте команду:\n"
    "/bmi <рост_в_см> <вес_в_кг>\n"
    "Пример: /bmi 175 68"
)


# === Команда /start ===
@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer(
        "👋 Привет! Я помогу спрогнозировать вероятность диабета.\n"
        + PREDICT_HINT
    )


# === Команда /predict ===
@dp.message(Command("predict"))
async def predict(message: types.Message):
    try:
        parts = message.text.split()

        if len(parts) != 9:
            await message.answer(
                "❗ Неверный формат.\n" + PREDICT_HINT
            )
            return

        values = np.array(
            [float(p) for p in parts[1:]],
            dtype=np.float32
        ).reshape(1, -1)

        if not np.isfinite(values).all():
            await message.answer(
                "⚠ Все значения должны быть конечными числами."
            )
            return

        values_scaled = scaler.transform(values)
        values_t = torch.tensor(
            values_scaled,
            dtype=torch.float32
        )

        with torch.no_grad():
            prob = model(values_t).item()
            prediction = (
                "✅ Диабет вероятен"
                if prob >= 0.5
                else "❌ Диабет маловероятен"
            )

        await message.answer(
            f"🏥 Вероятность диабета: {prob * 100:.2f}%\n"
            f"{prediction}"
        )

    except Exception as e:
        await message.answer(
            f"⚠ Ошибка обработки данных: {str(e)}"
        )


# === Команда /bmi ===
@dp.message(Command("bmi"))
async def bmi(message: types.Message):
    try:
        parts = message.text.split()

        if len(parts) != 3:
            await message.answer(
                "❗ Введите рост и вес: /bmi <рост_в_см> <вес_в_кг>\n"
                "Пример: /bmi 170 65"
            )
            return

        height_cm = float(parts[1])
        weight_kg = float(parts[2])

        if (
            not np.isfinite(height_cm)
            or not np.isfinite(weight_kg)
            or height_cm <= 0
            or weight_kg <= 0
        ):
            await message.answer(
                "⚠ Рост и вес должны быть положительными числами."
            )
            return

        height_m = height_cm / 100
        bmi_value = weight_kg / (height_m ** 2)

        await message.answer(
            f"📊 Ваш BMI: {bmi_value:.1f}"
        )

    except Exception as e:
        await message.answer(
            f"⚠ Ошибка при расчёте BMI: {str(e)}"
        )


# === Запуск ===
async def main():
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
