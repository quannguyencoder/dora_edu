"""Zalo OA bot entry point (``dora-run-zalo-bot``)."""

from __future__ import annotations

import logging
import sys

from dora_edu.bot.core import TutorService
from dora_edu.bot.session import SessionStore
from dora_edu.bot.zalo.adapter import ZaloAdapter
from dora_edu.config import get_settings
from dora_edu.llm.generator import OpenAIAnswerGenerator
from dora_edu.rag_engine.retriever import Retriever

logger = logging.getLogger(__name__)


def main() -> int:
    """Start the DoraEdu Zalo OA webhook bot.

    Returns:
        ``0`` on a clean shutdown, ``1`` when the bot cannot start.
    """
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    try:
        retriever = Retriever(settings)
        generator = OpenAIAnswerGenerator(settings)
        tutor = TutorService(
            retriever=retriever,
            generator=generator,
            sessions=SessionStore(max_turns=settings.session_max_turns),
            settings=settings,
        )
        adapter = ZaloAdapter(
            tutor,
            access_token=settings.zalo_access_token or "",
            oa_secret=settings.zalo_oa_secret or "",
            host=settings.zalo_webhook_host,
            port=settings.zalo_webhook_port,
        )
    except (ValueError, RuntimeError) as exc:
        logger.error("Cannot start DoraEdu Zalo bot: %s", exc)
        logger.error("Run 'dora-ingest' first, and check the values in your .env file.")
        return 1

    try:
        adapter.run()
    except KeyboardInterrupt:
        logger.info("Shutting down on user request")
    return 0


if __name__ == "__main__":
    sys.exit(main())
