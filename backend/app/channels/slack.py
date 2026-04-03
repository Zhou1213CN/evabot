"""Slack channel implementation using Socket Mode."""

import asyncio
import re
from typing import Any

from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.websockets import SocketModeClient
from slack_sdk.web.async_client import AsyncWebClient
from slackify_markdown import slackify_markdown

# --- [Evabot 架构适配] 替换日志库与核心模块 ---
from backend.core.log import get_logger
from backend.core.schemas import Message
from backend.app.gateway.gateway import Gateway
from backend.app.channels.base import BaseChannel
# 假设将来在 channel_config 中存放了 SlackConfig 结构
from backend.app.channels.channel_config import SlackConfig

logger = get_logger("channels.slack")


class SlackChannel(BaseChannel):
    """Slack channel using Socket Mode."""

    name = "slack"

    # --- [Evabot 架构适配] 修改初始化方法使用 Gateway ---
    def __init__(self, config: SlackConfig, gateway: Gateway):
        super().__init__(config, gateway)
        self.config: SlackConfig = config
        self._web_client: AsyncWebClient | None = None
        self._socket_client: SocketModeClient | None = None
        self._bot_user_id: str | None = None

    async def start(self) -> None:
        """Start the Slack Socket Mode client."""
        if not self.config.bot_token or not self.config.app_token:
            logger.error("Slack bot/app token not configured")
            return
        if self.config.mode != "socket":
            logger.error(f"Unsupported Slack mode: {self.config.mode}")
            return

        self._running = True

        self._web_client = AsyncWebClient(token=self.config.bot_token)
        self._socket_client = SocketModeClient(
            app_token=self.config.app_token,
            web_client=self._web_client,
        )

        self._socket_client.socket_mode_request_listeners.append(self._on_socket_request)

        try:
            auth = await self._web_client.auth_test()
            self._bot_user_id = auth.get("user_id")
            logger.info(f"Slack bot connected as {self._bot_user_id}")
        except Exception as e:
            logger.warning(f"Slack auth_test failed: {e}")

        logger.info("Starting Slack Socket Mode client...")
        await self._socket_client.connect()

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Slack client."""
        self._running = False
        if self._socket_client:
            try:
                await self._socket_client.close()
            except Exception as e:
                logger.warning(f"Slack socket close failed: {e}")
            self._socket_client = None

    # --- [Evabot 架构适配] 出站逻辑重构 ---
    async def send(self, msg: Message) -> None:
        """Send a message through Slack."""
        if not self._web_client:
            logger.warning("Slack client not running")
            return
        try:
            msg_data = msg.data or {}
            slack_meta = msg_data.get("slack", {})
            thread_ts = slack_meta.get("thread_ts")
            channel_type = slack_meta.get("channel_type")
            # Slack DMs don't use threads; channel/group replies may keep thread_ts.
            thread_ts_param = thread_ts if thread_ts and channel_type != "im" else None

            # 从 Evabot Artifacts 中提取附件路径
            media_paths = [art.uri for art in msg.artifacts] if msg.artifacts else []

            # Slack rejects empty text payloads. Keep media-only messages media-only,
            # but send a single blank message when the bot has no text or files to send.
            if msg.content or not media_paths:
                await self._web_client.chat_postMessage(
                    channel=msg.receiver_id,
                    text=self._to_mrkdwn(msg.content) if msg.content else " ",
                    thread_ts=thread_ts_param,
                )

            for media_path in media_paths:
                try:
                    await self._web_client.files_upload_v2(
                        channel=msg.receiver_id,
                        file=media_path,
                        thread_ts=thread_ts_param,
                    )
                except Exception as e:
                    logger.error(f"Failed to upload file {media_path}: {e}")
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")

    async def _on_socket_request(
        self,
        client: SocketModeClient,
        req: SocketModeRequest,
    ) -> None:
        """Handle incoming Socket Mode requests."""
        if req.type != "events_api":
            return

        # Acknowledge right away
        await client.send_socket_mode_response(
            SocketModeResponse(envelope_id=req.envelope_id)
        )

        payload = req.payload or {}
        event = payload.get("event") or {}
        event_type = event.get("type")

        # Handle app mentions or plain messages
        if event_type not in ("message", "app_mention"):
            return

        sender_id = event.get("user")
        chat_id = event.get("channel")

        # Ignore bot/system messages (any subtype = not a normal user message)
        if event.get("subtype"):
            return
        if self._bot_user_id and sender_id == self._bot_user_id:
            return

        # Avoid double-processing: Slack sends both `message` and `app_mention`
        text = event.get("text") or ""
        if event_type == "message" and self._bot_user_id and f"<@{self._bot_user_id}>" in text:
            return

        # Debug: log basic event shape
        logger.debug(
            f"Slack event: type={event_type} subtype={event.get('subtype')} user={sender_id} channel={chat_id} channel_type={event.get('channel_type')} text={text[:80]}"
        )
        if not sender_id or not chat_id:
            return

        channel_type = event.get("channel_type") or ""

        if not self._is_allowed(sender_id, chat_id, channel_type):
            return

        if channel_type != "im" and not self._should_respond_in_channel(event_type, text, chat_id):
            return

        text = self._strip_bot_mention(text)

        thread_ts = event.get("thread_ts")
        if getattr(self.config, 'reply_in_thread', True) and not thread_ts:
            thread_ts = event.get("ts")
            
        try:
            if self._web_client and event.get("ts"):
                react_emoji = getattr(self.config, 'react_emoji', 'eyes')
                await self._web_client.reactions_add(
                    channel=chat_id,
                    name=react_emoji,
                    timestamp=event.get("ts"),
                )
        except Exception as e:
            logger.debug(f"Slack reactions_add failed: {e}")

        # Thread-scoped session key for channel/group messages
        session_key = f"slack:{chat_id}:{thread_ts}" if thread_ts and channel_type != "im" else None

        try:
            # 这里的 _handle_message 会调用 BaseChannel 中的逻辑，自动组装向下分发的 Message
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=text,
                metadata={
                    "slack": {
                        "event": event,
                        "thread_ts": thread_ts,
                        "channel_type": channel_type,
                    },
                },
                session_key=session_key,
            )
        except Exception as e:
            logger.error(f"Error handling Slack message from {sender_id}: {e}")

    def _is_allowed(self, sender_id: str, chat_id: str, channel_type: str) -> bool:
        if channel_type == "im":
            dm_config = getattr(self.config, 'dm', None)
            if dm_config:
                if not getattr(dm_config, 'enabled', True):
                    return False
                if getattr(dm_config, 'policy', 'open') == "allowlist":
                    return sender_id in getattr(dm_config, 'allow_from', [])
            return True

        group_policy = getattr(self.config, 'group_policy', 'mention')
        if group_policy == "allowlist":
            return chat_id in getattr(self.config, 'group_allow_from', [])
        return True

    def _should_respond_in_channel(self, event_type: str, text: str, chat_id: str) -> bool:
        group_policy = getattr(self.config, 'group_policy', 'mention')
        if group_policy == "open":
            return True
        if group_policy == "mention":
            if event_type == "app_mention":
                return True
            return self._bot_user_id is not None and f"<@{self._bot_user_id}>" in text
        if group_policy == "allowlist":
            return chat_id in getattr(self.config, 'group_allow_from', [])
        return False

    def _strip_bot_mention(self, text: str) -> str:
        if not text or not self._bot_user_id:
            return text
        return re.sub(rf"<@{re.escape(self._bot_user_id)}>\s*", "", text).strip()

    _TABLE_RE = re.compile(r"(?m)^\|.*\|$(?:\n\|[\s:|-]*\|$)(?:\n\|.*\|$)*")
    _CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
    _INLINE_CODE_RE = re.compile(r"`[^`]+`")
    _LEFTOVER_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
    _LEFTOVER_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    _BARE_URL_RE = re.compile(r"(?<![|<])(https?://\S+)")

    @classmethod
    def _to_mrkdwn(cls, text: str) -> str:
        if not text:
            return ""
        text = cls._TABLE_RE.sub(cls._convert_table, text)
        return cls._fixup_mrkdwn(slackify_markdown(text))

    @classmethod
    def _fixup_mrkdwn(cls, text: str) -> str:
        code_blocks: list[str] = []

        def _save_code(m: re.Match) -> str:
            code_blocks.append(m.group(0))
            return f"\x00CB{len(code_blocks) - 1}\x00"

        text = cls._CODE_FENCE_RE.sub(_save_code, text)
        text = cls._INLINE_CODE_RE.sub(_save_code, text)
        text = cls._LEFTOVER_BOLD_RE.sub(r"*\1*", text)
        text = cls._LEFTOVER_HEADER_RE.sub(r"*\1*", text)
        text = cls._BARE_URL_RE.sub(lambda m: m.group(0).replace("&amp;", "&"), text)

        for i, block in enumerate(code_blocks):
            text = text.replace(f"\x00CB{i}\x00", block)
        return text

    @staticmethod
    def _convert_table(match: re.Match) -> str:
        lines = [ln.strip() for ln in match.group(0).strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return match.group(0)
        headers = [h.strip() for h in lines[0].strip("|").split("|")]
        start = 2 if re.fullmatch(r"[|\s:\-]+", lines[1]) else 1
        rows: list[str] = []
        for line in lines[start:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells = (cells + [""] * len(headers))[: len(headers)]
            parts = [f"**{headers[i]}**: {cells[i]}" for i in range(len(headers)) if cells[i]]
            if parts:
                rows.append(" · ".join(parts))
        return "\n".join(rows)