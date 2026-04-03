"""QQ channel implementation using botpy SDK."""

import asyncio
from collections import deque
from typing import TYPE_CHECKING

# --- [Evabot 架构适配] 替换日志库与核心模块 ---
from backend.core.log import get_logger, log_event
from backend.core.schemas import Message
from backend.app.gateway.gateway import Gateway
from backend.app.channels.base import BaseChannel
# 假设将来建立的 channel_config 中存放了 QQConfig 结构
from backend.app.channels.channel_config import QQConfig

logger = get_logger("channels.qq")

try:
    import botpy
    from botpy.message import C2CMessage, GroupMessage

    QQ_AVAILABLE = True
except ImportError:
    QQ_AVAILABLE = False
    botpy = None
    C2CMessage = None
    GroupMessage = None

if TYPE_CHECKING:
    from botpy.message import C2CMessage, GroupMessage


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(public_messages=True, direct_message=True)

    class _Bot(botpy.Client):
        def __init__(self):
            # Disable botpy's file log — nanobot uses loguru; default "botpy.log" fails on read-only fs
            super().__init__(intents=intents, ext_handlers=False)

        async def on_ready(self):
            # --- [Evabot 架构适配] 修正日志插值语法 ---
            logger.info(f"QQ bot ready: {self.robot.name}")

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_message(message, is_group=False)

        async def on_group_at_message_create(self, message: "GroupMessage"):
            await channel._on_message(message, is_group=True)

        async def on_direct_message_create(self, message):
            await channel._on_message(message, is_group=False)

    return _Bot


class QQChannel(BaseChannel):
    """QQ channel using botpy SDK with WebSocket connection."""

    name = "qq"

    # --- [Evabot 架构适配] 修改初始化方法使用 Gateway ---
    def __init__(self, config: QQConfig, gateway: Gateway):
        super().__init__(config, gateway)
        self.config: QQConfig = config
        self._client: "botpy.Client | None" = None
        self._processed_ids: deque = deque(maxlen=1000)
        self._msg_seq: int = 1  # 消息序列号，避免被 QQ API 去重
        self._chat_type_cache: dict[str, str] = {}

    async def start(self) -> None:
        """Start the QQ bot."""
        if not QQ_AVAILABLE:
            logger.error("QQ SDK not installed. Run: pip install qq-botpy")
            return

        if not self.config.app_id or not self.config.secret:
            logger.error("QQ app_id and secret not configured")
            return

        self._running = True
        BotClass = _make_bot_class(self)
        self._client = BotClass()
        logger.info("QQ bot started (C2C & Group supported)")
        await self._run_bot()

    async def _run_bot(self) -> None:
        """Run the bot connection with auto-reconnect."""
        while self._running:
            try:
                await self._client.start(appid=self.config.app_id, secret=self.config.secret)
            except Exception as e:
                logger.warning(f"QQ bot error: {e}")
            if self._running:
                logger.info("Reconnecting QQ bot in 5 seconds...")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the QQ bot."""
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        logger.info("QQ bot stopped")

    # --- [Evabot 架构适配] 出站逻辑重构 ---
    async def send(self, msg: Message) -> None:
        """Send a message through QQ."""
        if not self._client:
            logger.warning("QQ client not initialized")
            return

        try:
            # Evabot 的 Message 中，附加数据存放在 data 中
            msg_id = (msg.data or {}).get("message_id")
            self._msg_seq += 1
            msg_type = self._chat_type_cache.get(msg.receiver_id, "c2c")
            
            # QQ API 当前使用的是 markdown 纯文本结构
            # 如果存在附件，我们在文本末尾追加附件提示，进行降级处理
            final_content = msg.content or ""
            if msg.artifacts:
                media_texts = [f"[附件: {art.name}]" for art in msg.artifacts]
                final_content += "\n\n" + "\n".join(media_texts)

            if msg_type == "group":
                await self._client.api.post_group_message(
                    group_openid=msg.receiver_id,
                    msg_type=2,
                    markdown={"content": final_content},
                    msg_id=msg_id,
                    msg_seq=self._msg_seq,
                )
            else:
                await self._client.api.post_c2c_message(
                    openid=msg.receiver_id,
                    msg_type=2,
                    markdown={"content": final_content},
                    msg_id=msg_id,
                    msg_seq=self._msg_seq,
                )
        except Exception as e:
            logger.error(f"Error sending QQ message: {e}")

    async def _on_message(self, data: "C2CMessage | GroupMessage", is_group: bool = False) -> None:
        """Handle incoming message from QQ."""
        try:
            # Dedup by message ID
            if data.id in self._processed_ids:
                return
            self._processed_ids.append(data.id)

            content = (data.content or "").strip()
            if not content:
                return

            if is_group:
                chat_id = data.group_openid
                user_id = data.author.member_openid
                self._chat_type_cache[chat_id] = "group"
            else:
                chat_id = str(getattr(data.author, 'id', None) or getattr(data.author, 'user_openid', 'unknown'))
                user_id = chat_id
                self._chat_type_cache[chat_id] = "c2c"

            # 这里的 _handle_message 会调用 BaseChannel 中的新逻辑，自动组装向下分发的 Message
            await self._handle_message(
                sender_id=user_id,
                chat_id=chat_id,
                content=content,
                metadata={"message_id": data.id},
            )
        except Exception as e:
            logger.error(f"Error handling QQ message: {e}")