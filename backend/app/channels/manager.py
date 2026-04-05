"""
渠道管理器 (Channel Manager)
协调和管理多个外部聊天渠道的生命周期与消息路由。
已适配 Evabot 架构。
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from backend.core.log import get_logger
from backend.core.schemas import Message, MessageType
from backend.app.gateway.gateway import Gateway
from backend.app.channels.base import BaseChannel

logger = get_logger("channels.manager")

class ChannelManager:
    """
    维护所有渠道实例的初始化、启停以及出站消息(Outbound)的路由调度。
    """

    def __init__(self, config: Any, gateway: Gateway):
        self.config = config
        self.gateway = gateway
        self.channels: Dict[str, BaseChannel] = {}
        self._active_tasks = []

        # 之后我们会在 config 解析部分完成这里的映射
        self._init_channels()

    def _init_channels(self) -> None:
        from backend.app.channels.channel_config import ChannelRootConfig
        cfg = ChannelRootConfig.load()
        self.config = cfg

        # 动态映射表
        channel_classes = {
            "feishu": ("backend.app.channels.feishu", "FeishuChannel"),
            "telegram": ("backend.app.channels.telegram", "TelegramChannel"),
            "dingtalk": ("backend.app.channels.dingtalk", "DingTalkChannel"),
            "qq": ("backend.app.channels.qq", "QQChannel"),
            "discord": ("backend.app.channels.discord", "DiscordChannel"),
            "slack": ("backend.app.channels.slack", "SlackChannel"),
            "whatsapp": ("backend.app.channels.whatsapp", "WhatsAppChannel"),
            "email": ("backend.app.channels.email", "EmailChannel"),
            "weixin": ("backend.app.channels.weixin", "WeixinChannel"),
            "mochat": ("backend.app.channels.mochat", "MochatChannel"),
        }

        for ch_name, (module_path, class_name) in channel_classes.items():
            ch_config = getattr(cfg, ch_name, None)
            if ch_config and getattr(ch_config, 'enabled', False):
                try:
                    import importlib
                    module = importlib.import_module(module_path)
                    ChannelClass = getattr(module, class_name)
                    self.channels[ch_name] = ChannelClass(ch_config, self.gateway)
                    logger.info(f"外部通信渠道 [{ch_name}] 已成功挂载")
                except Exception as e:
                    logger.error(f"加载渠道 [{ch_name}] 失败: {e}")

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        """启动单个渠道并捕获异常"""
        try:
            await channel.start()
        except Exception as e:
            logger.error(f"启动渠道 {name} 失败: {e}")

    async def start_all(self) -> None:
        """启动所有在配置中启用(enabled: true)且已挂载的渠道"""
        if not self.channels:
            logger.warning("未检测到任何已挂载的外部通信渠道。")
            return

        tasks = []
        for name, channel in self.channels.items():
            # 获取该渠道对应的配置对象
            ch_config = getattr(self.config, name, None)
            
            # 二次校验：只有配置中 enabled 为 True 的才启动
            if ch_config and getattr(ch_config, 'enabled', False):
                logger.info(f"正在启动外部渠道: {name} ...")
                tasks.append(asyncio.create_task(self._start_channel(name, channel)))
            else:
                logger.info(f"跳过未启用的渠道: {name}")

        if tasks:
            self._active_tasks.extend(tasks)
        else:
            logger.warning("没有需要启动的已启用渠道。")

    async def stop_all(self) -> None:
        """停止所有渠道"""
        logger.info("正在停止所有外部渠道...")
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info(f"已停止渠道: {name}")
            except Exception as e:
                logger.error(f"停止渠道 {name} 时出错: {e}")

    async def dispatch_outbound(self, msg: Message) -> None:
        """核心路由：出站消息分发"""
        # 判断是否为系统生成的通知汇报类消息
        is_notification = 0#msg.message_type in (MessageType.REPORT, MessageType.HEARTBEAT)

        if is_notification:
            # 广播通知：读取你在 channel_config.yaml 设定的通知目标，比如 ['web', 'telegram']
            notification_targets = getattr(self.config, 'notification_channels', [])
            for target_channel in notification_targets:
                if target_channel in self.channels:
                    try:
                        await self.channels[target_channel].send(msg)
                    except Exception as e:
                        logger.error(f"渠道 {target_channel} 广播通知失败: {e}")
        else:
            # 普通对话：必须精准原路返回
            target_channel = getattr(msg, "source_channel", None)
            if target_channel and target_channel in self.channels:
                try:
                    await self.channels[target_channel].send(msg)
                except Exception as e:
                    logger.error(f"渠道 {target_channel} 回传消息失败: {e}")

    def get_channel(self, name: str) -> BaseChannel | None:
        """根据名称获取渠道实例"""
        return self.channels.get(name)

    @property
    def enabled_channels(self) -> List[str]:
        """获取已启用的渠道列表"""
        return list(self.channels.keys())
    

    async def reload_from_config(self, cfg: Any) -> None:
        """动态重载渠道配置"""
        self.config = cfg
        channel_classes = {
            "feishu": ("backend.app.channels.feishu", "FeishuChannel"),
            "telegram": ("backend.app.channels.telegram", "TelegramChannel"),
            "dingtalk": ("backend.app.channels.dingtalk", "DingTalkChannel"),
            "qq": ("backend.app.channels.qq", "QQChannel"),
            "discord": ("backend.app.channels.discord", "DiscordChannel"),
            "slack": ("backend.app.channels.slack", "SlackChannel"),
            "whatsapp": ("backend.app.channels.whatsapp", "WhatsAppChannel"),
            "email": ("backend.app.channels.email", "EmailChannel"),
            "weixin": ("backend.app.channels.weixin", "WeixinChannel"),
            "mochat": ("backend.app.channels.mochat", "MochatChannel"),
        }

        for ch_name, (module_path, class_name) in channel_classes.items():
            ch_config = getattr(cfg, ch_name, None)
            is_enabled = getattr(ch_config, 'enabled', False) if ch_config else False
            
            if is_enabled and ch_name not in self.channels:
                # 动态启动新启用的渠道
                try:
                    import importlib
                    module = importlib.import_module(module_path)
                    ChannelClass = getattr(module, class_name)
                    new_channel = ChannelClass(ch_config, self.gateway)
                    self.channels[ch_name] = new_channel
                    import asyncio
                    asyncio.create_task(self._start_channel(ch_name, new_channel))
                    logger.info(f"动态挂载并启动渠道 [{ch_name}]")
                except Exception as e:
                    logger.error(f"动态加载渠道 [{ch_name}] 失败: {e}")
            elif not is_enabled and ch_name in self.channels:
                # 动态卸载被禁用的渠道
                old_channel = self.channels.pop(ch_name)
                try:
                    import asyncio
                    await old_channel.stop()
                    logger.info(f"动态卸载渠道 [{ch_name}]")
                except Exception as e:
                    logger.error(f"停止渠道 [{ch_name}] 失败: {e}")
            elif is_enabled and ch_name in self.channels:
                # 渠道已经在运行，更新其内部配置对象
                self.channels[ch_name].config = ch_config
    