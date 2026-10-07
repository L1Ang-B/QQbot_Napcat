"""游戏名词自动解析插件 — 群聊中检测游戏术语并回复通俗解释。

策略：本地字典优先命中 → 完全没命中时走 AI（LiteLLM）回退。
"""

import asyncio
import time

from litellm.exceptions import RateLimitError

from ncatbot.plugin import NcatBotPlugin
from ncatbot.core import registrar
from ncatbot.event.qq import GroupMessageEvent

# —— 本地游戏名词字典 ——
# key: 游戏术语, value: 通俗解释（2-3 句话，适合群聊阅读）
GAME_DICT: dict[str, str] = {
    "原神": (
        "原神是米哈游开发的开放世界动作角色扮演游戏（ARPG）。"
        "主打探索+元素反应战斗，更新节奏约 6 周一版本。"
        "角色和武器靠抽卡获取，存在保底机制。"
    ),
    "艾尔登法环": (
        "艾尔登法环（Elden Ring）是 FromSoftware 开发的 3A 魂类开放世界游戏。"
        "以高难度探索和战斗著称，被视为魂系列集大成之作。"
        "玩家常俗称它为「老头环」。"
    ),
    "老头环": (
        "「老头环」是玩家对《艾尔登法环》（Elden Ring）的中文昵称。"
        "这是一款高难度魂类开放世界游戏，以硬核 Boss 和自由探索著称。"
    ),
    "黑神话悟空": (
        "黑神话：悟空是国产单机动作游戏，改编自西游记题材。"
        "由游戏科学工作室开发，以写实美术和硬核战斗闻名。"
        "2024 年发售，被誉为国产 3A 里程碑。"
    ),
    "塞尔达传说": (
        "塞尔达传说是任天堂旗下的经典动作冒险游戏系列。"
        "以开放世界探索、解谜和与林克的冒险为核心玩法。"
        "《王国之泪》《旷野之息》是近年来最具影响力的两部。"
    ),
    "英雄联盟": (
        "英雄联盟（LOL）是拳头游戏开发的多人在线战术竞技（MOBA）游戏。"
        "双方各 5 人，通过推掉对方水晶获胜。"
        "全球最热门的电竞项目之一。"
    ),
    "打野": (
        "打野是 MOBA 游戏里的一个位置（通常对应 LOL 的 Jungle）。"
        "玩家主要在野区刷怪升级，同时游走各路帮忙建立优势。"
        "打野需要有好的节奏和大局观。"
    ),
    "ADC": (
        "ADC 是 MOBA 游戏中的「物理核心输出」位置（Attack Damage Carry）。"
        "通常走下路，靠远程平 A 打出持续高额伤害。"
        "前期相对脆弱，非常依赖辅助保护和发育。"
    ),
    "二游": (
        "「二游」是二次元手游的圈内统称。"
        "通常指带有日式/国风二次元美术风格、以角色收集和剧情为卖点的手机游戏。"
        "常见例子：原神、崩坏、明日方舟、蔚蓝档案等。"
    ),
    "保底": (
        "保底是抽卡机制里的保障规则。"
        "当连续抽取一定次数都没出稀有物品时，第 N 次必出。"
        "这是厂商为缓解玩家抽卡焦虑而设计的机制。"
    ),
    "MOBA": (
        "MOBA 是多人在线战术竞技类游戏的英文缩写。"
        "通常是两队各 5 人在同一张地图上对抗，以摧毁对方主基地为目标。"
        "代表作品：英雄联盟、DOTA2、王者荣耀。"
    ),
    "魂类": (
        "「魂类」是 FromSoftware 开发的一类高难度动作 RPG 的统称。"
        "特点是战斗硬核、怪物攻击凶猛、死亡惩罚机制独特。"
        "代表作品：黑暗之魂、血源诅咒、艾尔登法环。"
    ),
}

# —— AI 回退提示词 ——
AI_PROMPT_TEMPLATE = (
    "从以下群聊文本中提取所有游戏专有名词（包括游戏名、角色名、黑话、装备名等）。"
    "如果没有游戏名词，只返回空字符串，不要输出任何其他文字。"
    "如果有，对每个名词给出 2-3 句话的简明解释，用日常语言，适合群聊场景。"
    "文本：{message}"
)

# AI 视为"空响应"的关键词（大模型可能不完全遵守指令，返回类似"无"/"没有"等）
AI_EMPTY_KEYWORDS = {"", "空", "无", "没有", "暂无", "未发现", "none", "null", "n/a"}

# 冷却时间（秒）：同一个群聊在此时间窗口内最多回复一次
COOLDOWN_SECONDS = 30

# AI 并发上限（允许多个请求排队，减少群友等待）
AI_MAX_CONCURRENCY = 2

# AI 单次调用硬超时（秒）—— 防卡死但给足响应时间
AI_TIMEOUT_SECONDS = 29

# RateLimitError 重试等待（秒）
RATE_LIMIT_RETRY_DELAY = 2.0


class GameTermPlugin(NcatBotPlugin):
    """游戏名词解析插件入口类。"""

    def __init__(self) -> None:
        super().__init__()
        # 冷却记录：group_id → 上次回复的时间戳
        self._last_reply_time: dict[str, float] = {}
        # AI 并发闸门，控制同时只有 1 个 AI 请求在飞
        self._semaphore = asyncio.Semaphore(AI_MAX_CONCURRENCY)

    async def on_load(self) -> None:
        self.logger.info(
            f"{self.name} 已加载，收录 {len(GAME_DICT)} 个游戏术语"
        )

    async def on_close(self) -> None:
        self.logger.info(f"{self.name} 已卸载")

    # ------------------------------------------------------------------
    #  主 handler
    # ------------------------------------------------------------------
    @registrar.on_group_message()
    async def on_group_message(self, event: GroupMessageEvent) -> None:
        """监听所有群消息：本地字典命中优先 → 无命中时走 AI 回退。"""
        group_id = event.group_id
        raw = event.raw_message or ""
        if not raw.strip():
            return

        now = time.time()

        # —— 冷却检查（最优先，省 AI token）——
        last_time = self._last_reply_time.get(group_id, 0)
        if now - last_time < COOLDOWN_SECONDS:
            self.logger.debug(
                f"群 {group_id} 冷却中，跳过 "
                f"（距上次 {now - last_time:.1f}s）"
            )
            return

        # —— 冷却计时从消息触发这一刻开始，而非 AI 回复完成 ——
        # 这样即使 AI 调了 29s 才超时，冷却也只剩 1s，下一个请求能快速进来
        self._last_reply_time[group_id] = now

        # —— 第一步：本地字典匹配 ——
        matched_terms = [term for term in GAME_DICT if term in raw]

        if matched_terms:
            await self._reply_from_dict(group_id, now, matched_terms)
            return

        # —— 第二步：无本地命中 → AI 回退 ——
        await self._reply_from_ai(group_id, now, raw)

    # ------------------------------------------------------------------
    #  本地字典回复
    # ------------------------------------------------------------------
    async def _reply_from_dict(
        self,
        group_id: str,
        now: float,
        matched_terms: list[str],
    ) -> None:
        """用本地字典解释命中的游戏术语，最多前 3 个。"""
        terms_to_explain = matched_terms[:3]

        lines = []
        for term in terms_to_explain:
            lines.append(f"【{term}】{GAME_DICT[term]}")
        if len(matched_terms) > 3:
            lines.append(f"（还有 {len(matched_terms) - 3} 个术语未展示）")

        reply_text = "\n\n".join(lines)

        try:
            await self.api.qq.send_group_text(group_id, reply_text)
            self.logger.info(
                f"群 {group_id} 本地命中 {matched_terms}，已回复"
            )
        except Exception:
            self.logger.exception(f"群 {group_id} 本地字典回复失败")

    # ------------------------------------------------------------------
    #  AI 回退回复
    # ------------------------------------------------------------------
    async def _reply_from_ai(
        self,
        group_id: str,
        now: float,
        raw_message: str,
    ) -> None:
        """本地未命中时调用 AI 适配器提取并解释游戏名词。"""
        # 检查 AI 适配器是否可用
        ai_client = getattr(self.api, "ai", None)
        if ai_client is None:
            self.logger.debug(
                f"群 {group_id} 未命中本地字典，且未配置 AI 适配器，跳过"
            )
            return

        prompt = AI_PROMPT_TEMPLATE.format(message=raw_message)

        # —— 并发闸门 + 硬超时 + 限流重试 ——
        # semaphore 保证同时最多 1 个 AI 请求在飞；
        # wait_for 强制 10s 超时，超时抛 TimeoutError，
        #       async with 正常退出 → semaphore 自动释放；
        # RateLimitError 限流：等 2s 只重试一次，再失败就静默退出；
        # 其他异常（网络/配置等）也静默退出，确保 semaphore 一定释放。
        async with self._semaphore:
            content = None
            try:
                content = await asyncio.wait_for(
                    ai_client.chat_text(prompt),
                    timeout=AI_TIMEOUT_SECONDS,
                )
            except RateLimitError:
                self.logger.warning(
                    f"群 {group_id} 触发 RateLimitError，"
                    f"{RATE_LIMIT_RETRY_DELAY:.0f}s 后重试一次"
                )
                await asyncio.sleep(RATE_LIMIT_RETRY_DELAY)
                try:
                    content = await asyncio.wait_for(
                        ai_client.chat_text(prompt),
                        timeout=AI_TIMEOUT_SECONDS,
                    )
                except RateLimitError:
                    self.logger.warning(
                        f"群 {group_id} 重试后仍 RateLimitError，放弃"
                    )
                    return
                except TimeoutError:
                    self.logger.warning(
                        f"群 {group_id} 重试超时（>{AI_TIMEOUT_SECONDS}s），放弃"
                    )
                    return
                except Exception:
                    self.logger.exception(
                        f"群 {group_id} 重试时发生未知异常，放弃"
                    )
                    return
            except TimeoutError:
                # 大模型响应过慢，semaphore 在 async with 退出时自动释放
                self.logger.warning(
                    f"群 {group_id} AI 调用超时（>{AI_TIMEOUT_SECONDS}s），放弃"
                )
                return
            except Exception:
                # 其他 AI 异常（网络/配置错误等）→ 静默退出
                self.logger.exception(f"群 {group_id} AI 调用失败")
                return

        # —— 严格空响应检测 ——
        if not content:
            self.logger.debug(f"群 {group_id} AI 返回 None/空，跳过")
            return

        stripped = content.strip()
        if not stripped:
            self.logger.debug(f"群 {group_id} AI 返回纯空白，跳过")
            return

        # 大模型有时不遵守指令，返回"无"/"没有"/"暂无"等
        normalized = stripped.strip("。.!！?？ \n\r\t")
        if normalized.lower() in AI_EMPTY_KEYWORDS:
            self.logger.debug(
                f"群 {group_id} AI 返回空响应关键词「{stripped}」，跳过"
            )
            return

        try:
            await self.api.qq.send_group_text(group_id, stripped)
            self.logger.info(f"群 {group_id} AI 回退回复成功")
        except Exception:
            # 发送失败也只记日志，不往群里补发任何报错
            self.logger.exception(f"群 {group_id} AI 回复发送失败")
