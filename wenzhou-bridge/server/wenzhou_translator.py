"""
Wenzhou Translator
==================
Mandarin text → Wenzhou dialect text converter + Wenzhou text → Mandarin.

Two-step for the reverse direction (Mandarin → Wenzhou):
  1. ASR: Mandarin speech → Mandarin text
  2. Translate: Mandarin text → Wenzhou dialect text
  3. TTS: Wenzhou text → speech (uses edge-tts Mandarin as fallback)

The translator uses a built-in lexical map for common Wenzhou phrases
plus a prompt template for an optional LLM-based translation pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import csv
import json
import re
import os


# ── Built-in Wenzhou ↔ Mandarin lexical map ──
# key = Mandarin, value = Wenzhou dialect written form
_MANDARIN_TO_WENZHOU: Dict[str, str] = {
    # Daily greetings
    "你好": "你好",
    "早上好": "天光好",
    "中午好": "日昼好",
    "晚上好": "黄昏好",
    "再见": "再会",
    "谢谢": "多谢",
    "对不起": "对勿住",
    "没关系": "冇关系",
    "请": "请",
    "吃饭": "吃天光/吃日昼/吃黄昏",
    "早饭": "天光",
    "午餐": "日昼",
    "晚餐": "黄昏",
    "睡觉": "困/睏",
    "起床": "爬起",
    "回家": "走归",
    "出去": "走出",
    "进来": "走入",
    "知道": "晓得",
    "不知道": "不晓得",
    "可以": "好",
    "不可以": "不好",
    "是的": "是",
    "不是": "不是/弗是",
    "没有": "冇",
    "有": "有",
    "很多": "多显",
    "一点点": "厘儿",
    "怎么了": "妆乜",
    "什么": "乜/何乜",
    "哪里": "阿垡/垡堂",
    "这里": "该垡/该里",
    "那里": "许垡/许里",
    "谁": "谁人/何乜人",
    "为什么": "为乜",
    "怎么样": "妆何",
    "多少": "几多",
    "多少钱": "几多铜钿",
    "很": "显",
    "非常": "显",
    "今天": "该日",
    "明天": "明朝",
    "昨天": "昨夜",
    "前天": "前日",
    "后天": "后日",
    "早上": "天光",
    "中午": "日昼",
    "下午": "晚界/下半日",
    "晚上": "黄昏/夜里",
    "现在": "该下/该下子",
    "等一下": "等下",
    "马上": "立刻",
    "已经": "已经",
    "还没有": "还未",
    "我们": "我侬/我俦",
    "你们": "你侬/你俦",
    "他们": "渠侬/渠俦",
    "我的": "我个",
    "你的": "你个",
    "他的": "渠个",
    "大家": "大家人",
    "小孩": "细儿/娒娒",
    "大人": "大人",
    "男人": "男子客",
    "女人": "女子客/老嬷客",
    "老人": "老人",
    "爸爸": "阿爸/爸爸",
    "妈妈": "阿妈/妈妈/姆妈",
    "哥哥": "阿哥/哥哥",
    "弟弟": "阿弟",
    "姐姐": "阿姐/姐姐",
    "妹妹": "阿妹",
    "爷爷": "阿爷",
    "奶奶": "阿婆/娘娘",
    "房子": "屋宅/屋",
    "家里": "屋里",
    "学校": "学堂",
    "医院": "医院",
    "超市": "超市",
    "商店": "店",
    "菜市场": "菜场",
    "银行": "银行",
    "公交站": "车站",
    "火车站": "火车站",
    "机场": "机场",
    "手机": "手机",
    "电脑": "电脑",
    "电视": "电视机",
    "电话": "电话",
    "钱": "钞票/铜钿",
    "便宜": "便宜/相因",
    "贵": "贵",
    "好吃": "好吃",
    "好喝": "好喝",
    "好看": "好看/好觑",
    "好玩": "好嬉",
    "开心": "开心/快活",
    "难过": "难过/烦恼",
    "生气": "生气/光火",
    "累了": "吃力",
    "忙": "忙",
    "有空": "有空",
    "等一下嘛": "等下一",
    "快一点": "快厘/快显",
    "慢一点": "慢厘/慢慢",
    "来": "来",
    "去": "去",
    "买": "买",
    "卖": "卖",
    "做": "做",
    "吃": "吃",
    "喝": "喝/呷",
    "走": "走",
    "跑": "跑/奔",
    "看": "看/觑",
    "听": "听",
    "说": "讲",
    "叫": "叫",
    "坐": "坐",
    "站": "隑",
    "想": "想/忖",
    "要": "要",
    "不要": "不要/弗要",
    "能": "会",
    "不能": "不会/弗会",
    "必须": "着/要",
    "因为": "因为",
    "所以": "所以",
    "但是": "但是",
    "如果": "如果/若果",
    "然后": "然后",
    "或者": "或者/亦",
    "和": "搭/跟",
    "从": "从/打",
    "到": "到/来",
    "在": "来/勒",
    "下雨了": "落雨",
    "天晴了": "天晴",
    "冷": "冷",
    "热": "热",
    "很冷": "冷显",
    "很热": "热显",
    "温州话": "温州话",
    "普通话": "普通话",
    "中文": "中文/汉语",
    "好的": "好个",
    "行": "好/着",
    "没问题": "没问题/冇问题",
    "加油": "鼓劲",
    "辛苦了": "辛苦了/吃力",
    "太棒了": "好显/赞显",
    "明白了": "懂了/晓得",
    "听不懂": "听弗懂",
    "再说一遍": "再讲一遍",
    "慢慢说": "慢慢讲",
    "我来帮你": "我来帮你",
    "没关系": "冇关系",
    "不用担心": "弗用担心/弗用愁",
    # Weather & nature
    "天气": "天气",
    "晴天": "晴天",
    "阴天": "乌阴天",
    "刮风": "打风/起风",
    "打雷": "响雷",
    "闪电": "闪光/爧",
    "下雪": "落雪",
    "结冰": "结冰",
    "雾": "雾",
    "台风": "风台",
    "太阳": "太阳/日头",
    "月亮": "月光/月",
    "星星": "星",
    "云": "云",
    "蓝天": "蓝天",
    "空气": "空气",
    "温度": "温度",
    "暖和": "暖/暖和",
    "凉快": "凉/凉快",
    "闷热": "烝热",
    "潮湿": "潮",
    "干燥": "燥",
    "春天": "春天",
    "夏天": "夏天/热天",
    "秋天": "秋天",
    "冬天": "冬天/冷天",
    # Time & dates
    "今年": "该年",
    "去年": "旧年",
    "明年": "明年",
    "前年": "前年",
    "后年": "后年",
    "这个月": "该个月",
    "上个月": "上个月",
    "下个月": "下个月",
    "这周": "该个礼拜",
    "上周": "上个礼拜",
    "下周": "下个礼拜",
    "周末": "礼拜末",
    "小时": "钟头",
    "上午": "上半日",
    "傍晚": "黄昏边",
    "半夜": "半夜三更",
    "整天": "成日",
    "每天": "每日/日日",
    "经常": "经常/时常",
    "偶尔": "偶尔",
    "从来": "从来",
    "从前": "从前/古时",
    "将来": "将来/以后",
    "以前": "以前/老早",
    "最近": "最近",
    "刚才": "刚刚",
    "一会儿": "一下",
    "很久": "长远",
    "星期几": "礼拜几",
    # Health & body
    "身体": "身体",
    "头": "头",
    "头发": "头发",
    "脸": "面",
    "眼睛": "眼",
    "耳朵": "耳",
    "鼻子": "鼻头",
    "嘴巴": "口",
    "牙齿": "牙齿",
    "手": "手",
    "脚": "脚",
    "肚子": "肚",
    "生病": "生病/弗爽",
    "感冒": "感冒/伤风",
    "发烧": "发烧/发热",
    "咳嗽": "嗽/咳嗽",
    "头痛": "头痛",
    "肚子痛": "肚痛",
    "打针": "打针",
    "吃药": "吃药",
    "手术": "开刀",
    "住院": "住院",
    "出院": "出院",
    "检查": "检查",
    "看病": "看病/望医生",
    "医生": "医生/先生",
    "护士": "护士",
    "病人": "病人",
    "药": "药",
    "预约": "预约/挂号",
    "舒服": "舒服/好过",
    "不舒服": "弗舒服",
    "健康": "健康",
    "锻炼": "锻炼/运动",
    # Food & drink
    "水": "水",
    "开水": "滚水",
    "茶": "茶",
    "咖啡": "咖啡",
    "牛奶": "牛奶",
    "米饭": "饭",
    "面条": "面",
    "包子": "包子",
    "饺子": "饺子",
    "面包": "面包",
    "鸡蛋": "鸡蛋",
    "猪肉": "猪肉",
    "牛肉": "牛肉",
    "鱼": "鱼",
    "虾": "虾",
    "蔬菜": "蔬菜/青菜",
    "水果": "水果",
    "苹果": "苹果",
    "香蕉": "香蕉",
    "西瓜": "西瓜",
    "葡萄": "葡萄",
    "橘子": "橘",
    "辣椒": "辣椒/番椒",
    "盐": "盐",
    "糖": "糖",
    "油": "油",
    "酱油": "酱油",
    "醋": "醋",
    "味道": "味道",
    "香": "香",
    "臭": "臭",
    "甜": "甜",
    "苦": "苦",
    "辣": "辣",
    "酸": "酸",
    "咸": "咸",
    "淡": "淡",
    "新鲜": "新鲜",
    "饿": "饿/肚饥",
    "饱": "饱",
    "渴": "渴/口干",
    # Transportation
    "车": "车",
    "汽车": "汽车",
    "出租车": "的士",
    "公交车": "公交车/巴士",
    "地铁": "地铁",
    "自行车": "脚踏车/单车",
    "飞机": "飞机",
    "开车": "开车",
    "坐车": "坐车",
    "上车": "上车",
    "下车": "落车",
    "路口": "路口",
    "红绿灯": "红绿灯",
    "停车": "停车",
    "堵车": "堵车/塞车",
    "导航": "导航",
    # Shopping
    "价格": "价钱",
    "打折": "打折/折扣",
    "免费": "免费/弗要钱",
    "发票": "发票",
    "现金": "现钞/现钱",
    "刷卡": "刷卡",
    "扫码": "扫码/扫",
    "微信支付": "微信支付",
    "支付宝": "支付宝",
    "找零": "找零",
    "退货": "退货",
    "换货": "换货",
    "保修": "保修",
    "颜色": "颜色",
    "红色": "红",
    "黄色": "黄",
    "蓝色": "蓝",
    "绿色": "绿",
    "白色": "白",
    "黑色": "黑/乌",
    # Family
    "儿子": "儿子",
    "女儿": "女儿",
    "孙子": "孙",
    "孙女": "孙女",
    "丈夫": "老公",
    "妻子": "老婆",
    "亲戚": "亲戚/亲眷",
    "朋友": "朋友",
    "邻居": "邻舍",
    "同事": "同事",
    "同学": "同学",
    "老师": "老师/先生",
    "学生": "学生",
    "老板": "老板/东家",
    "员工": "员工",
    "客人": "人客",
    # Work
    "工作": "工作/事干",
    "上班": "上班/去做生活",
    "下班": "下班/歇工",
    "加班": "加班",
    "请假": "请假",
    "辞职": "辞职/弗做",
    "面试": "面试",
    "工资": "工资/薪水",
    "开会": "开会",
    "出差": "出差",
    "公司": "公司",
    "工厂": "工厂",
    "合同": "合同",
    "签字": "签字/签名",
    # Education
    "读书": "读书",
    "考试": "考试",
    "作业": "作业/功课",
    "上课": "上课",
    "下课": "下课/落课",
    "放假": "放假",
    "暑假": "暑假",
    "寒假": "寒假",
    "毕业": "毕业",
    "报名": "报名",
    # Emotions
    "高兴": "欢喜",
    "快乐": "快活",
    "伤心": "伤心",
    "害怕": "怕",
    "担心": "愁",
    "放心": "放心/宽心",
    "着急": "急",
    "无聊": "无趣味",
    "有趣": "好嬉",
    "羡慕": "眼热",
    "感谢": "感谢",
    "讨厌": "讨厌/憎",
    "喜欢": "喜欢/中意",
    "爱": "爱",
    "期待": "期待/盼望",
    "失望": "失望",
    # Actions
    "打开": "打开/开",
    "关闭": "关",
    "开始": "开始/起头",
    "结束": "结束/完",
    "继续": "继续",
    "停止": "停止/歇",
    "忘记": "忘记/记弗得",
    "记得": "记得/记着",
    "找到": "寻着",
    "丢失": "失落",
    "等待": "等",
    "帮助": "帮助/帮凑",
    "使用": "用",
    "学习": "学/学习",
    "练习": "练",
    "准备": "准备",
    "整理": "整理/拾掇",
    "选择": "选/选择",
    "比较": "比",
    "决定": "决定/定",
    "允许": "准",
    "禁止": "禁用",
    "修理": "修",
    "打扫": "打扫/扫",
    "洗": "洗",
    "煮": "煮/烧",
    # Places
    "地方": "地方/垡堂",
    "城市": "城市/城",
    "农村": "乡下",
    "公园": "公园",
    "图书馆": "图书馆",
    "博物馆": "博物馆",
    "电影院": "电影院",
    "餐厅": "餐馆/饭店",
    "酒店": "酒店/旅馆",
    "厕所": "厕所/茅坑",
    "电梯": "电梯",
    "楼梯": "楼梯",
    # Directions
    "东": "东",
    "西": "西",
    "南": "南",
    "北": "北",
    "前": "前",
    "后": "后",
    "左": "左",
    "右": "右",
    "上": "上",
    "下": "下",
    "里面": "里面/里向",
    "外面": "外面/外头",
    "旁边": "旁边",
    "对面": "对面/对过",
    # Animals
    "狗": "狗",
    "猫": "猫",
    "鸟": "鸟",
    "马": "马",
    "牛": "牛",
    "猪": "猪",
    "鸡": "鸡",
    "鸭": "鸭",
    "鱼": "鱼",
    "兔子": "兔子",
    "老鼠": "老鼠",
    "蛇": "蛇",
    "老虎": "老虎",
    "猴子": "猴",
    "蚊子": "蚊虫",
    "蜜蜂": "蜜蜂",
    "蝴蝶": "蝴蝶",
    # Numbers
    "全部": "全部/统统",
    "每个": "每个",
    "一半": "一半",
    "第一": "第一",
    "最后": "最后/末尾",
    "一次": "一次",
    # Places
    "中国": "中国",
    "温州": "温州",
    "浙江": "浙江",
    "上海": "上海",
    "北京": "北京",
    "杭州": "杭州",
    # Internet
    "网络": "网络/网",
    "密码": "密码",
    "账号": "账号",
    "充电": "充电",
    "电池": "电池",
    "照片": "照片/照相",
    "视频": "视频",
    "微信": "微信",
    "短信": "短信/信息",
    # Common phrases
    "欢迎": "欢迎",
    "恭喜": "恭喜",
    "生日快乐": "生日快乐",
    "新年快乐": "新年快乐",
    "身体健康": "身体健康",
    "一路平安": "一路平安/顺风",
    "祝你": "祝你",
    "太好了": "好显",
    "当然": "当然",
    "可能": "可能",
    "一定": "一定/肯定",
    "真的": "真个",
    "确实": "确实",
    "差不多": "差不多/差弗多",
    "来不及": "来弗及",
    "来得及": "来得及",
    "算了": "算了/罢",
    "随便": "随便/随意",
    "小心": "小心/仔细",
    "注意": "注意/留心",
}

# Reverse map for Wenzhou text → Mandarin (handles 1:N mappings)
# Build from scratch: collect ALL unique wenzhou variants → mandarin
_WENZHOU_TO_MANDARIN: Dict[str, str] = {}
for mandarin, wenzhou in _MANDARIN_TO_WENZHOU.items():
    variants = [v.strip() for v in wenzhou.split("/")]
    for v in variants:
        # Only add if not already present (first seen wins)
        if v not in _WENZHOU_TO_MANDARIN:
            _WENZHOU_TO_MANDARIN[v] = mandarin

# Compile a regex-based tokenizer for phrase-level matching (longest-first)
def _build_phrase_regex(phrase_dict: Dict[str, str], longest_first: bool = True) -> re.Pattern:
    phrases = sorted(phrase_dict.keys(), key=len, reverse=longest_first)
    escaped = [re.escape(p) for p in phrases]
    return re.compile("|".join(escaped))


MANDARIN_PATTERN = _build_phrase_regex(_MANDARIN_TO_WENZHOU)
WENZHOU_PATTERN = _build_phrase_regex(_WENZHOU_TO_MANDARIN)

# Wenzhou dialect markers for auto-detection
_WENZHOU_MARKERS = set()
for v in _WENZHOU_TO_MANDARIN:
    # Add unique Wenzhou-only words (not shared with Mandarin)
    if len(v) <= 4 and v not in _MANDARIN_TO_WENZHOU:
        _WENZHOU_MARKERS.add(v)


# ── Prompt templates (for optional LLM integration) ──

MANDARIN_TO_WENZHOU_SYSTEM_PROMPT = """
你是温州话翻译助手。请将以下普通话语段翻译成地道、自然的温州话。

要求：
1. 保留原意不增减事实信息
2. 使用地道的温州话词汇和表达习惯
3. 输出温州话文本，不要解释
4. 保持自然口语风格
5. 如果无法确定某个词的温州话说法，保留普通话原词

直接输出温州话文本。
""".strip()

WENZHOU_TO_MANDARIN_SYSTEM_PROMPT = """
你是温州话翻译助手。请将以下温州话翻译成自然、流畅的普通话。

要求：
1. 保留原意不增减事实信息
2. 使用标准的现代普通话表达
3. 输出普通话文本，不要解释

直接输出普通话文本。
""".strip()


class WenzhouTranslator:
    """
    Bidirectional translator between Mandarin and Wenzhou dialect text.
    
    Uses a built-in lexical map for phrase-level translation,
    with an optional LLM pass for context-aware translation.
    """

    def __init__(self, lexicon_path: Optional[str | Path] = None, backend: str = "rules"):
        self.backend = backend
        self.extra_lexicon = self._load_lexicon(lexicon_path) if lexicon_path else {}

    # ── Public API ──

    def mandarin_to_wenzhou(self, text: str, context: str = "") -> str:
        """Translate Mandarin text to Wenzhou dialect text."""
        t = text.strip()
        t = self._apply_lexicon(t, _MANDARIN_TO_WENZHOU, MANDARIN_PATTERN)
        t = self._apply_extra_lexicon(t, forward=True)
        return t

    def wenzhou_to_mandarin(self, text: str, context: str = "") -> str:
        """Translate Wenzhou dialect text to Mandarin text."""
        t = text.strip()
        t = self._apply_lexicon(t, _WENZHOU_TO_MANDARIN, WENZHOU_PATTERN)
        t = self._apply_extra_lexicon(t, forward=False)
        return t

    # ── Language detection (auto-detect Wenzhou vs Mandarin) ──

    _WENZHOU_MARKERS_SORTED = sorted([m for m in _WENZHOU_MARKERS if len(m) >= 2], key=len, reverse=True)

    def detect_language(self, text: str) -> str | None:
        """
        Detect if `text` is Wenzhou dialect or Mandarin.
        Returns 'wenzhou_to_mandarin' or 'mandarin_to_wenzhou'
        Falls back to 'mandarin_to_wenzhou' when uncertain.
        """
        t = text.strip()
        if not t or len(t) < 2:
            return None
        # Score based on Wenzhou marker presence
        wenzhou_score = 0
        for marker in self._WENZHOU_MARKERS_SORTED:
            if marker in t:
                wenzhou_score += len(marker) ** 2  # quadratic weight for stronger signal
        # Mandarin-specific markers (exclusive to Mandarin, absent in Wenzhou)
        mandarin_markers = ["吗", "了", "的", "吧", "呢", "我们", "你们", "他们", "什么", "怎么"]
        mandarin_score = 0
        for m in mandarin_markers:
            if m in t:
                mandarin_score += 2
        # Heuristic
        if wenzhou_score > mandarin_score * 3 and wenzhou_score >= 4:
            return "wenzhou_to_mandarin"
        elif mandarin_score >= 2:
            return "mandarin_to_wenzhou"
        # Uncertain: let pipeline/ASR result guide
        return None

    def build_prompt(self, text: str, direction: str, context: str = "") -> str:
        """Build LLM prompt for context-aware translation."""
        if direction == "mandarin_to_wenzhou":
            prompt = MANDARIN_TO_WENZHOU_SYSTEM_PROMPT
        else:
            prompt = WENZHOU_TO_MANDARIN_SYSTEM_PROMPT
        return (
            f"{prompt}\n\n"
            f"场景/上下文：{context or '无'}\n"
            f"待翻译文本：{text}\n"
            f"译文："
        )

    # ── Internals ──

    def _load_lexicon(self, path: str | Path) -> Dict[str, str]:
        path = Path(path)
        if not path.exists():
            return {}
        mapping = {}
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = (row.get("mandarin") or row.get("source") or "").strip()
                tgt = (row.get("wenzhou") or "").strip()
                if src and tgt:
                    mapping[src] = tgt
        return mapping

    def _apply_lexicon(self, text: str, lexicon: Dict[str, str], pattern: re.Pattern) -> str:
        def replacer(m: re.Match) -> str:
            matched = m.group(0)
            return lexicon.get(matched, matched)
        return pattern.sub(replacer, text)

    def _apply_extra_lexicon(self, text: str, forward: bool) -> str:
        if not self.extra_lexicon:
            return text
        if forward:
            # extra lexicon: mandarin → wenzhou
            for src, tgt in sorted(self.extra_lexicon.items(), key=lambda x: len(x[0]), reverse=True):
                text = text.replace(src, tgt)
        else:
            # reverse: wenzhou → mandarin
            rev = {v: k for k, v in self.extra_lexicon.items()}
            for src, tgt in sorted(rev.items(), key=lambda x: len(x[0]), reverse=True):
                text = text.replace(src, tgt)
        return text


# ── CLI test ──
if __name__ == "__main__":
    import sys
    translator = WenzhouTranslator()

    if len(sys.argv) > 1 and sys.argv[1] == "--mw":
        text = sys.argv[2] if len(sys.argv) > 2 else "今天天气很好，我们一起吃饭吧。"
        print("Mandarin → Wenzhou:")
        print(f"  Input:  {text}")
        print(f"  Output: {translator.mandarin_to_wenzhou(text)}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--wm":
        text = sys.argv[2] if len(sys.argv) > 2 else "该日天光好显，我侬一起去吃天光。"
        print("Wenzhou → Mandarin:")
        print(f"  Input:  {text}")
        print(f"  Output: {translator.wenzhou_to_mandarin(text)}")
    else:
        # Demo both directions
        print("=" * 60)
        print("Demo: Mandarin → Wenzhou")
        print("=" * 60)
        tests_mw = [
            "今天天气很好，我们一起吃饭吧。",
            "你好，请问现在几点了？",
            "我不知道他在哪里，等他回来再说。",
            "这个多少钱？太贵了，可以便宜一点吗？",
            "谢谢你的帮助，辛苦了！",
        ]
        for t in tests_mw:
            print(f"  输入: {t}")
            print(f"  输出: {translator.mandarin_to_wenzhou(t)}")
            print()

        print("=" * 60)
        print("Demo: Wenzhou → Mandarin")
        print("=" * 60)
        tests_wm = [
            "该日天光好显，我侬一起去吃天光。",
            "你晓得渠勒阿垡不？",
            "该个几多铜钿？贵显，便宜厘好不？",
            "多谢你帮忙，吃力显！",
            "落雨，走快厘。",
        ]
        for t in tests_wm:
            print(f"  输入: {t}")
            print(f"  输出: {translator.wenzhou_to_mandarin(t)}")
