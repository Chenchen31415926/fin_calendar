"""
使用 akshare + DeepSeek 分析最新全球宏观财经新闻。

功能说明：
1. 通过 ak.news_economic_baidu() 获取最新宏观新闻；
2. 提取最新一条新闻内容；
3. 调用 DeepSeek（deepseek-chat）将新闻解析为严格 JSON；
4. 打印 DeepSeek 返回的 JSON 结果。
"""

import json
import os
from typing import Any, Dict

import akshare as ak
import requests


DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OUTPUT_FILE = "data.json"


def get_latest_macro_news() -> str:
    """
    获取最新一条全球宏观财经新闻文本。

    Returns:
        str: 最新新闻内容（尽量提取标题+内容）。

    Raises:
        RuntimeError: 当数据为空或格式异常时抛出。
    """
    try:
        df = ak.news_economic_baidu()
    except Exception as exc:
        raise RuntimeError(f"调用 ak.news_economic_baidu() 失败: {exc}") from exc

    if df is None or df.empty:
        raise RuntimeError("ak.news_economic_baidu() 返回为空，无法提取新闻。")

    # 通常最新数据在首行，若实际环境顺序相反可改为 df.iloc[-1]
    latest_row = df.iloc[0]

    # 常见字段可能有 “标题”、“内容”、“事件”、“地区”、“日期”等，这里做兼容提取
    title = ""
    content = ""
    for col in ("标题", "title", "事件", "event"):
        if col in df.columns and str(latest_row[col]).strip():
            title = str(latest_row[col]).strip()
            break

    for col in ("内容", "content", "公布值", "今值", "摘要", "解读"):
        if col in df.columns and str(latest_row[col]).strip():
            content = str(latest_row[col]).strip()
            break

    news_text = "；".join([x for x in (title, content) if x]).strip()

    # 若上面未能拿到可读字段，则退化为整行转字典
    if not news_text:
        news_text = json.dumps(latest_row.to_dict(), ensure_ascii=False)

    if not news_text:
        raise RuntimeError("最新新闻内容为空，无法继续分析。")

    return news_text


def call_deepseek_for_analysis(news_text: str, api_key: str) -> Dict[str, Any]:
    """
    调用 DeepSeek 对新闻进行金融分析，并返回 JSON 对象。

    Args:
        news_text (str): 新闻原文。
        api_key (str): DeepSeek API Key。

    Returns:
        Dict[str, Any]: DeepSeek 返回并成功解析后的 JSON。

    Raises:
        RuntimeError: 网络请求失败、响应不合法或 JSON 解析失败时抛出。
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 强约束输出格式：仅 JSON 且字段固定
    system_prompt = (
        "你是资深金融分析师。"
        "请严格只输出一个 JSON 对象，不要输出任何额外说明、前后缀、Markdown。"
        "JSON 格式必须完全为："
        '{"Tier":"1","Event":"新闻核心事件概括","Linked_Sector":"关联的A股行业板块","Impact":"利好/利空/中性"}。'
    )
    user_prompt = f"请解析以下新闻：\n{news_text}"

    payload: Dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"请求 DeepSeek API 失败: {exc}") from exc

    try:
        resp_json = response.json()
    except ValueError as exc:
        raise RuntimeError(f"DeepSeek 返回非 JSON 响应: {response.text}") from exc

    # 兼容 OpenAI 风格返回结构
    try:
        content = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"DeepSeek 响应结构异常: {resp_json}") from exc

    # 有些模型可能会包裹 ```json ... ```，做兜底清洗
    content = content.strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DeepSeek 返回内容不是合法 JSON: {content}") from exc

    required_keys = {"Tier", "Event", "Linked_Sector", "Impact"}
    if not required_keys.issubset(result.keys()):
        raise RuntimeError(f"DeepSeek JSON 缺少必要字段，实际返回: {result}")

    return result


def main() -> None:
    """
    主流程：抓取新闻 -> 调用 DeepSeek -> 打印标准 JSON。
    """
    try:
        latest_news = get_latest_macro_news()
        analysis_json = call_deepseek_for_analysis(latest_news, DEEPSEEK_API_KEY)

        # 按要求打印 DeepSeek 返回的 JSON 结果
        print(json.dumps(analysis_json, ensure_ascii=False))

        # 将结果保存到当前目录下的 data.json（不存在则创建，存在则覆盖）
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            json.dump(analysis_json, file, ensure_ascii=False, indent=4)
    except Exception as exc:
        print(f"执行失败: {exc}")


if __name__ == "__main__":
    main()
