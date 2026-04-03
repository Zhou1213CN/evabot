"""
基础渠道接口 (Base Channel Interface)
已适配 Evabot 架构：移除 MessageBus，改用 Gateway 和标准 Message 实体。
"""
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

# 替换原来的 loguru 为 Evabot 的统一 logger
from backend.core.log import get_logger, log_event
# 引入 Evabot 的核心结构字典
from backend.core.schemas import Message, Component, SendType, MessageRole, ArtifactRef
from backend.app.gateway.gateway import Gateway

logger = get_logger("channels.base")

class BaseChannel(ABC):
    """
    所有聊天渠道（如 Telegram, Feishu 等）的抽象基类。
    实现该接口即可接入 Evabot 的网关系统。
    """

    name: str = "base"

    def __init__(self, config: Any, gateway: Gateway):
        """
        初始化渠道
        :param config: 该渠道特有的配置信息
        :param gateway: Evabot 全局路由网关，用于下发和流转消息
        """
        self.config = config
        self.gateway = gateway
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """
        启动渠道监听。
        这是一个常驻的异步任务，负责连接平台、接收消息并调用 _handle_message()
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止渠道监听，清理资源"""
        pass

    @abstractmethod
    async def send(self, msg: Message) -> None:
        """
        发送出站消息（由系统回复给用户的消息）。
        :param msg: Evabot 内部标准的 Message 实体
        """
        pass

    def is_allowed(self, sender_id: str) -> bool:
        """检查用户权限（白名单机制）"""
        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list:
            logger.warning(f"{self.name}: allow_from 配置为空 — 将拒绝所有请求")
            return False
        if "*" in allow_list:
            return True
        return str(sender_id) in allow_list

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        session_key: Optional[str] = None,
    ) -> None:
        """
        处理来自外部平台的消息。将其转换为标准 Message 并抛给网关。
        """
        if not self.is_allowed(sender_id):
            logger.warning(f"权限拒绝：用户 {sender_id} 在渠道 {self.name} 上未被授权。")
            return

        # 1. 媒体附件转换：将字符串路径列表转换为 ArtifactRef 实体列表
        artifacts = []
        if media:
            for path in media:
                artifacts.append(ArtifactRef(
                    uri=path,
                    name=os.path.basename(path)
                ))

        # 2. 组装 Evabot 标准 Message
        # 在 Evabot 架构中，外部输入相当于 USER 身份向 Butler(第一层) 派发任务，
        # 故 send_type 为 DOWNWARD。我们将 chat_id 映射为 receiver_id(或者 sender_id 都可以，这里用 sender_id 作为会话 ID)
        msg = Message(
            sender_id=str(chat_id),       # 对于 Butler 层，这个 ID 就是记忆文件的上下文 ID
            sender=Component.USER,        # 发送者是外部用户
            send_type=SendType.DOWNWARD,  # 消息向下流转给 Butler
            content=content,
            message_role=MessageRole.USER,
            artifacts=artifacts,          # 挂载附件
            data=metadata or {},
            source_channel=self.name      # 标记渠道来源，出站时根据这个字段路由发回！
        )

        # 把原始的真实用户发信人 ID（如群组里的某个人）塞进 data 里备用
        msg.data["original_sender_id"] = str(sender_id)
        if session_key:
            msg.data["session_key"] = session_key

        # 3. 推送给网关，让 Butler 接管
        try:
            self.gateway.handle(msg)
            log_event(logger, "CHANNEL_INBOUND_SUCCESS", channel=self.name, chat_id=chat_id)
        except Exception as e:
            logger.error(f"网关处理外部消息失败 [{self.name}]: {e}")

    @property
    def is_running(self) -> bool:
        """检查渠道是否在运行"""
        return self._running