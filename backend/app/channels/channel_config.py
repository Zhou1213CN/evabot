"""
渠道配置加载器 (Channel Config)
负责从 channel_config.yaml 读取所有外部通讯渠道的配置，
并提供环境变量动态解析与强类型校验。支持自动生成默认文件和动态保存。
"""
import os
import yaml
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from backend.core.log import get_logger

logger = get_logger("core.channel_config")


def _resolve_env_vars(data: Any) -> Any:
    """
    递归解析配置字典中的环境变量。
    如果字符串以 'ENV:' 开头，则将其替换为系统环境变量的实际值。
    """
    if isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(v) for v in data]
    elif isinstance(data, str) and data.startswith("ENV:"):
        env_var_name = data.split(":", 1)[1].strip()
        return os.environ.get(env_var_name, "")
    return data


# ==========================================
# 1. 基础模型与具体渠道模型
# ==========================================

class BaseChannelConfig(BaseModel):
    """所有渠道的通用配置"""
    enabled: bool = False
    allow_from: List[str] = Field(default_factory=lambda: ["*"])


class FeishuConfig(BaseChannelConfig):
    app_id: str = ""
    app_secret: str = ""
    encrypt_key: Optional[str] = ""
    verification_token: Optional[str] = ""
    react_emoji: str = "THUMBSUP"


class TelegramConfig(BaseChannelConfig):
    token: str = ""
    proxy: Optional[str] = None
    group_policy: str = "mention"  # 'open', 'allowlist', 'mention'
    reply_to_message: bool = False


class QQConfig(BaseChannelConfig):
    app_id: str = ""
    secret: str = ""


class DingTalkConfig(BaseChannelConfig):
    client_id: str = ""
    client_secret: str = ""


class MochatConfig(BaseChannelConfig):
    claw_token: str = ""
    base_url: str = ""
    socket_url: Optional[str] = None
    socket_path: str = "/socket.io"
    sessions: List[str] = Field(default_factory=list)
    panels: List[str] = Field(default_factory=list)
    socket_disable_msgpack: bool = False
    max_retry_attempts: int = 10
    socket_reconnect_delay_ms: int = 1000
    socket_max_reconnect_delay_ms: int = 5000
    socket_connect_timeout_ms: int = 20000
    watch_limit: int = 50
    refresh_interval_ms: int = 60000
    watch_timeout_ms: int = 30000
    retry_delay_ms: int = 5000
    agent_user_id: Optional[str] = None
    reply_delay_mode: Optional[str] = None
    reply_delay_ms: int = 0


class DiscordConfig(BaseChannelConfig):
    token: str = ""
    gateway_url: str = "wss://gateway.discord.gg/?v=10&encoding=json"
    intents: int = 33280  # 默认的 intents 值，涵盖消息读取等
    group_policy: str = "mention"  # 'open', 'mention' 等


class SlackDmConfig(BaseModel):
    enabled: bool = True
    policy: str = "allowlist" # 'open' 或 'allowlist'
    allow_from: List[str] = Field(default_factory=list)


class SlackConfig(BaseChannelConfig):
    bot_token: str = ""
    app_token: str = ""
    mode: str = "socket"
    reply_in_thread: bool = True
    react_emoji: str = "eyes"
    group_policy: str = "mention"  # 'open', 'mention', 'allowlist'
    group_allow_from: List[str] = Field(default_factory=list)
    dm: SlackDmConfig = Field(default_factory=SlackDmConfig)


class WhatsAppConfig(BaseChannelConfig):
    bridge_url: str = "ws://localhost:3000"
    bridge_token: Optional[str] = None


class EmailConfig(BaseChannelConfig):
    consent_granted: bool = False
    poll_interval_seconds: int = 10
    auto_reply_enabled: bool = True
    subject_prefix: str = "Re: "
    max_body_chars: int = 10000
    mark_seen: bool = True
    
    imap_host: str = ""
    imap_port: int = 993
    imap_use_ssl: bool = True
    imap_username: str = ""
    imap_password: str = ""
    imap_mailbox: str = "INBOX"

    smtp_host: str = ""
    smtp_port: int = 465
    smtp_use_ssl: bool = True
    smtp_use_tls: bool = False
    smtp_username: str = ""
    smtp_password: str = ""
    from_address: str = ""


class WeixinConfig(BaseChannelConfig):
    """微信(个人)渠道专属配置"""
    base_url: str = "https://ilinkai.weixin.qq.com"
    cdn_base_url: str = "https://novac2c.cdn.weixin.qq.com/c2c"
    route_tag: Optional[str] = None
    token: str = ""  
    state_dir: str = "" 
    poll_timeout: int = 35 
    groq_api_key: Optional[str] = None


# ==========================================
# 2. 根配置加载器
# ==========================================

class ChannelRootConfig(BaseModel):
    """
    根配置模型，映射整个 channel_config.yaml
    当配置缺失时，将自动使用 default_factory 生成对应的默认空配置。
    """
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    qq: QQConfig = Field(default_factory=QQConfig)
    dingtalk: DingTalkConfig = Field(default_factory=DingTalkConfig)
    mochat: MochatConfig = Field(default_factory=MochatConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    whatsapp: WhatsAppConfig = Field(default_factory=WhatsAppConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    weixin: WeixinConfig = Field(default_factory=WeixinConfig) 

    # 遇到未强类型声明的边缘渠道配置时，先丢入字典，保证代码不报错
    others: Dict[str, Any] = Field(default_factory=dict)

    # 系统全局通知将推送到这些渠道（如 ['feishu', 'telegram']）
    notification_channels: List[str] = Field(default_factory=list)

    @classmethod
    def get_default_path(cls) -> str:
        """获取配置文件的默认绝对路径"""
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "channel_config.yaml")
        )

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "ChannelRootConfig":
        """
        加载并解析 YAML 配置文件。若文件不存在，则创建默认配置并保存。
        """
        path = config_path or cls.get_default_path()

        # 逻辑修改：如果文件不存在，生成空配置并立刻执行保存落盘
        if not os.path.exists(path):
            logger.info(f"渠道配置文件未找到: {path}，系统将自动生成默认的空配置并保存。")
            instance = cls()
            instance.save(path)
            return instance
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f) or {}

            # 解析 ENV: 语法的环境变量
            resolved_data = _resolve_env_vars(raw_data)

            # 剥离出已强类型声明的字段，剩下的统统塞进 others
            known_keys = {
                "feishu", "telegram", "qq", "dingtalk", 
                "mochat", "discord", "slack", "whatsapp", "email", "weixin",
                "notification_channels"
            }
            filtered_data = {}
            others_data = {}

            for k, v in resolved_data.items():
                if k in known_keys:
                    filtered_data[k] = v
                else:
                    others_data[k] = v

            filtered_data["others"] = others_data

            # 实例化并触发 Pydantic 严格校验
            return cls(**filtered_data)

        except Exception as e:
            logger.error(f"加载渠道配置失败 {path}: {e}")
            return cls()

    def save(self, config_path: Optional[str] = None) -> bool:
        """
        将当前配置写回 YAML 文件。
        注意：如果配置中原先使用了 'ENV:xxx' 的环境变量语法，
        由于加载时已经被解析为实际值，保存时会直接将实际密钥明文写入文件。
        """
        path = config_path or self.get_default_path()
        try:
            # 排除 None 值，导出字典
            data = self.model_dump(mode='json', exclude_none=True)
            
            # 将 others 里的未知配置提取到顶层，避免在 YAML 中出现嵌套的 "others: {}"
            others_data = data.pop("others", {})
            for k, v in others_data.items():
                if k not in data:
                    data[k] = v
            
            # 确保目标目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, indent=2)
            
            logger.info(f"渠道配置已成功保存至: {path}")
            return True
        except Exception as e:
            logger.error(f"保存渠道配置失败: {e}")
            return False