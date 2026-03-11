"""
Chainlit Web GUI — Gemini 2.5 Flash 多模態聊天介面
- 支援圖片 (JPG/PNG)、PDF、純文字檔 (.txt) 上傳
- 具備多輪對話記憶
- 對話紀錄即時存儲為 JSON
"""

import json
import base64
import mimetypes
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import os

import chainlit as cl

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader

# ── 載入環境變數 ──────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("請在 .env 檔案中設定 GEMINI_API_KEY")

# ── 支援的檔案類型 ────────────────────────────────────────────
IMAGE_MIMES = {"image/jpeg", "image/png", "image/jpg"}
PDF_MIMES = {"application/pdf"}


# ── Chainlit 生命週期 ─────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """每個新 session 初始化 LLM、記憶和對話紀錄"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )
    history = InMemoryChatMessageHistory()
    conversation_log: list[dict] = []

    # 存入 session
    cl.user_session.set("llm", llm)
    cl.user_session.set("history", history)
    cl.user_session.set("conversation_log", conversation_log)

    await cl.Message(
        content=(
            "👋 歡迎使用 **Gemini 2.5 Flash** 多模態聊天機器人！\n\n"
            "你可以：\n"
            "- 💬 直接輸入文字對話\n"
            "- 🖼️ 上傳圖片 (JPG/PNG) 進行分析\n"
            "- 📄 上傳 PDF 文件進行摘要或問答\n"
            "- 📝 上傳 TXT 文字檔進行分析\n\n"
            "輸入 **exit** 可結束對話並儲存紀錄。"
        )
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    """處理每一則使用者訊息"""
    llm: ChatGoogleGenerativeAI = cl.user_session.get("llm")
    history: InMemoryChatMessageHistory = cl.user_session.get("history")
    conversation_log: list[dict] = cl.user_session.get("conversation_log")

    user_text = msg.content.strip()

    # ── 檢查 exit 指令 ────────────────────────────────────
    if user_text.lower() == "exit":
        _save_conversation(conversation_log)
        conversation_log.clear()
        await cl.Message(content="💾 對話紀錄已儲存！👋 再見！").send()
        return

    # ── 分析附件 ──────────────────────────────────────────
    elements = msg.elements or []
    images = [el for el in elements if el.mime and el.mime in IMAGE_MIMES]
    pdfs = [el for el in elements if el.mime and el.mime in PDF_MIMES]
    txts = [el for el in elements if el.name and el.name.lower().endswith(".txt")]

    has_files = bool(images or pdfs or txts)

    # ── 記錄使用者訊息 ────────────────────────────────────
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "role": "user",
        "content": user_text or "(僅上傳檔案)",
    }
    if has_files:
        log_entry["file"] = _build_file_log(images, pdfs, txts)
    conversation_log.append(log_entry)

    # ── 產生 AI 回覆 ──────────────────────────────────────
    ai_msg = cl.Message(content="")
    await ai_msg.send()

    try:
        if images:
            ai_text = await _handle_image(llm, history, images, user_text)
        elif pdfs:
            ai_text = await _handle_pdf(llm, history, pdfs, user_text)
        elif txts:
            ai_text = await _handle_txt(llm, history, txts, user_text)
        else:
            ai_text = await _handle_text(llm, history, user_text)
    except Exception as e:
        ai_text = f"❌ 錯誤：{e}"

    ai_msg.content = ai_text
    await ai_msg.update()

    # ── 記錄 AI 回覆 ──────────────────────────────────────
    conversation_log.append({
        "timestamp": datetime.now().isoformat(),
        "role": "ai",
        "content": ai_text,
    })


@cl.on_chat_end
async def on_chat_end():
    """對話結束時自動存檔"""
    conversation_log: list[dict] = cl.user_session.get("conversation_log")
    if conversation_log:
        _save_conversation(conversation_log)


# ── 處理函式 ──────────────────────────────────────────────────

async def _handle_text(llm, history, user_text: str) -> str:
    """純文字對話"""
    human_msg = HumanMessage(content=user_text)
    all_messages = history.messages + [human_msg]
    response = await cl.make_async(llm.invoke)(all_messages)
    history.add_message(human_msg)
    history.add_message(response)
    return response.content


async def _handle_image(llm, history, images: list, user_text: str) -> str:
    """圖片分析"""
    prompt_text = user_text if user_text else "請描述這張圖片的內容。"

    content_parts = [{"type": "text", "text": prompt_text}]

    for img_el in images:
        img_path = img_el.path
        mime_type = img_el.mime or "image/jpeg"

        with open(img_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
        })

    human_msg = HumanMessage(content=content_parts)
    all_messages = history.messages + [human_msg]
    response = await cl.make_async(llm.invoke)(all_messages)

    # 記憶中存純文字摘要（避免 base64 佔用記憶體）
    names = ", ".join(img.name for img in images)
    history.add_message(HumanMessage(content=f"{prompt_text} [圖片: {names}]"))
    history.add_message(response)
    return response.content


async def _handle_pdf(llm, history, pdfs: list, user_text: str) -> str:
    """PDF 文件分析"""
    all_text_parts = []

    for pdf_el in pdfs:
        loader = PyPDFLoader(pdf_el.path)
        pages = loader.load()
        page_text = "\n\n".join(
            f"--- 第 {i+1} 頁 ---\n{page.page_content}"
            for i, page in enumerate(pages)
        )
        all_text_parts.append(f"📄 {pdf_el.name}（共 {len(pages)} 頁）:\n{page_text}")

    full_text = "\n\n".join(all_text_parts)
    if not full_text.strip():
        full_text = "（PDF 中沒有可提取的文字內容）"

    prompt_text = user_text if user_text else "請摘要這份 PDF 文件的內容。"
    combined = f"以下是 PDF 文件的內容：\n\n{full_text}\n\n使用者的問題：{prompt_text}"

    human_msg = HumanMessage(content=combined)
    all_messages = history.messages + [human_msg]
    response = await cl.make_async(llm.invoke)(all_messages)

    history.add_message(human_msg)
    history.add_message(response)
    return response.content


async def _handle_txt(llm, history, txts: list, user_text: str) -> str:
    """純文字檔案分析"""
    all_text_parts = []

    for txt_el in txts:
        with open(txt_el.path, "r", encoding="utf-8") as f:
            content = f.read()
        all_text_parts.append(f"📝 {txt_el.name}（{len(content)} 字元）:\n{content}")

    full_text = "\n\n".join(all_text_parts)
    if not full_text.strip():
        full_text = "（檔案內容為空）"

    prompt_text = user_text if user_text else "請摘要這份文字檔案的內容。"
    combined = f"以下是文字檔案的內容：\n\n{full_text}\n\n使用者的問題：{prompt_text}"

    human_msg = HumanMessage(content=combined)
    all_messages = history.messages + [human_msg]
    response = await cl.make_async(llm.invoke)(all_messages)

    history.add_message(human_msg)
    history.add_message(response)
    return response.content


# ── 工具函式 ──────────────────────────────────────────────────

def _save_conversation(conversation_log: list[dict]):
    """將對話紀錄存成 JSON 檔案"""
    if not conversation_log:
        return
    filename = datetime.now().strftime("chat_%Y%m%d_%H%M%S.json")
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation_log, f, ensure_ascii=False, indent=2)
    print(f"💾 對話紀錄已儲存至：{filepath}")


def _build_file_log(images, pdfs, txts) -> list[dict]:
    """建立檔案紀錄列表"""
    files = []
    for img in images:
        files.append({"path": img.path, "type": "image", "name": img.name})
    for pdf in pdfs:
        files.append({"path": pdf.path, "type": "pdf", "name": pdf.name})
    for txt in txts:
        files.append({"path": txt.path, "type": "txt", "name": txt.name})
    return files
