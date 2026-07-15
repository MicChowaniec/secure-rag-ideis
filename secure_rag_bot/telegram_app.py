from __future__ import annotations

import asyncio

from .factory import create_pipeline
from .evidence import is_study_domain_query


def main() -> None:
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
    except ImportError as exc:
        raise SystemExit("Zainstaluj zależności: pip install -e .") from exc

    pipeline = create_pipeline()
    if not pipeline.settings.telegram_bot_token:
        raise SystemExit("Ustaw TELEGRAM_BOT_TOKEN w środowisku.")
    if pipeline.store.count() == 0:
        raise SystemExit("Baza RAG jest pusta. Uruchom najpierw moduł ingest.")

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        context.chat_data.clear()
        await update.message.reply_text(
            "Jestem bezpiecznym asystentem regulaminu studiów. Nie wysyłaj danych osobowych."
        )

    async def health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        await update.message.reply_text(
            f"Bot działa. Fragmenty w bazie RAG: {pipeline.store.count()}. "
            f"Privacy model: {pipeline.settings.use_privacy_model}. "
            f"Bielik Guard: {pipeline.settings.use_bielik_guard}."
        )

    async def message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            return
        try:
            await update.message.chat.send_action("typing")
            history = list(context.chat_data.get("safe_history", []))
            previous = "\n".join(history)
            result = await asyncio.to_thread(pipeline.handle, update.message.text, previous)
            safe_current = result.privacy.masked_text
            if (
                result.status in {"answered", "insufficient_context", "grounding_fallback"}
                and is_study_domain_query(safe_current)
            ):
                if pipeline.is_followup_query(safe_current, previous):
                    history = (history + [safe_current])[-3:]
                else:
                    history = [safe_current]
                context.chat_data["safe_history"] = history
            await update.message.reply_text(result.text[:4096])
        except Exception as exc:
            print(f"Błąd obsługi wiadomości: {type(exc).__name__}: {exc}", flush=True)
            await update.message.reply_text(
                "Wystąpił błąd wewnętrzny. Wiadomość nie została przekazana do modelu."
            )

    app = Application.builder().token(pipeline.settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message))
    print(
        "Bot Telegram został uruchomiony i nasłuchuje. Wyślij /start lub /health. "
        "Aby wrócić do menu, naciśnij Ctrl+C.",
        flush=True,
    )
    try:
        app.run_polling(drop_pending_updates=True)
    finally:
        pipeline.store.close()


if __name__ == "__main__":
    main()
