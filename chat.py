"""
使用 LangChain + Gemini 2.0 Flash 的聊天程式
- 具備對話記憶功能
- 結束時將對話紀錄存成 JSON 檔案
- 支援圖片 (JPG/PNG)、PDF、純文字檔 (.txt) 輸入
"""

import json
import atexit
import base64
import mimetypes
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader

# ── 支援的檔案副檔名 ─────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEXT_EXTENSIONS = {".txt"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | TEXT_EXTENSIONS | PDF_EXTENSIONS

# ── 載入環境變數 ──────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("請在 .env 檔案中設定 GEMINI_API_KEY")

# ── 初始化 Gemini 模型 ────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7,
)

# ── 對話紀錄（用於匯出 JSON） ─────────────────────────────────
conversation_log: list[dict] = []

# ── 對話記憶 ──────────────────────────────────────────────────
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chain_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ── 檔案偵測 ──────────────────────────────────────────────────
def detect_file_path(user_input: str) -> tuple[str, str] | None:
    """
    檢查使用者輸入是否為有效的檔案路徑。
    回傳 (file_path, file_type) 或 None。
    file_type: "image", "pdf", "txt"
    """
    text = user_input.strip().strip('"').strip("'")
    path = Path(text)

    if not path.is_file():
        return None

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None

    if ext in IMAGE_EXTENSIONS:
        return (str(path), "image")
    elif ext in PDF_EXTENSIONS:
        return (str(path), "pdf")
    elif ext in TEXT_EXTENSIONS:
        return (str(path), "txt")
    return None


# ── 檔案處理 ──────────────────────────────────────────────────
def process_file(file_path: str, file_type: str, user_prompt: str) -> tuple[HumanMessage, str]:
    """
    根據檔案類型產生對應的 HumanMessage。
    回傳 (message, file_description)。
    """
    if file_type == "image":
        return _process_image(file_path, user_prompt)
    elif file_type == "pdf":
        return _process_pdf(file_path, user_prompt)
    elif file_type == "txt":
        return _process_txt(file_path, user_prompt)
    else:
        raise ValueError(f"不支援的檔案類型：{file_type}")


def _process_image(file_path: str, user_prompt: str) -> tuple[HumanMessage, str]:
    """處理圖片檔案：base64 編碼後透過 Gemini 多模態 API 傳送"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(file_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    prompt_text = user_prompt if user_prompt else "請描述這張圖片的內容。"
    file_desc = f"[圖片: {Path(file_path).name}]"

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
            },
        ]
    )
    return message, file_desc


def _process_pdf(file_path: str, user_prompt: str) -> tuple[HumanMessage, str]:
    """處理 PDF 檔案：使用 PyPDFLoader 提取文字"""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    full_text = "\n\n".join(
        f"--- 第 {i+1} 頁 ---\n{page.page_content}"
        for i, page in enumerate(pages)
    )

    if not full_text.strip():
        full_text = "（PDF 中沒有可提取的文字內容）"

    prompt_text = user_prompt if user_prompt else "請摘要這份 PDF 文件的內容。"
    file_desc = f"[PDF: {Path(file_path).name}, 共 {len(pages)} 頁]"

    combined_prompt = (
        f"以下是一份 PDF 文件的內容：\n\n{full_text}\n\n"
        f"使用者的問題：{prompt_text}"
    )
    message = HumanMessage(content=combined_prompt)
    return message, file_desc


def _process_txt(file_path: str, user_prompt: str) -> tuple[HumanMessage, str]:
    """處理純文字檔案"""
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()

    if not file_content.strip():
        file_content = "（檔案內容為空）"

    prompt_text = user_prompt if user_prompt else "請摘要這份文字檔案的內容。"
    file_desc = f"[TXT: {Path(file_path).name}, {len(file_content)} 字元]"

    combined_prompt = (
        f"以下是一份文字檔案的內容：\n\n{file_content}\n\n"
        f"使用者的問題：{prompt_text}"
    )
    message = HumanMessage(content=combined_prompt)
    return message, file_desc


# ── 存檔功能 ──────────────────────────────────────────────────
def save_conversation():
    """將對話紀錄存成 JSON 檔案"""
    if not conversation_log:
        return
    filename = datetime.now().strftime("chat_%Y%m%d_%H%M%S.json")
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation_log, f, ensure_ascii=False, indent=2)
    print(f"\n💾 對話紀錄已儲存至：{filename}")


# 註冊 atexit，確保程式意外結束時也能存檔
atexit.register(save_conversation)


# ── 主程式 ────────────────────────────────────────────────────
def main():
    session_id = "default"
    print("=" * 50)
    print("  🤖 Gemini 2.0 Flash 聊天機器人")
    print("  支援輸入：文字 / 圖片(JPG,PNG) / PDF / TXT")
    print("  輸入檔案路徑即可分析檔案內容")
    print("  輸入 'exit' 結束對話並儲存紀錄")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            break

        # ── 偵測是否為檔案路徑 ──────────────────────────────
        file_info = detect_file_path(user_input)

        if file_info:
            file_path, file_type = file_info
            type_label = {"image": "圖片", "pdf": "PDF", "txt": "文字檔"}[file_type]
            print(f"\n📎 偵測到{type_label}：{Path(file_path).name}")

            # 詢問使用者關於檔案的問題（可直接 Enter 使用預設問題）
            try:
                user_prompt = input("❓ 請輸入關於此檔案的問題（直接 Enter 使用預設）: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            # 處理檔案
            try:
                message, file_desc = process_file(file_path, file_type, user_prompt)
            except Exception as e:
                print(f"\n❌ 檔案處理失敗：{e}")
                continue

            # 記錄使用者訊息（含檔案資訊）
            log_content = user_prompt if user_prompt else f"（分析{type_label}）"
            conversation_log.append({
                "timestamp": datetime.now().isoformat(),
                "role": "user",
                "content": log_content,
                "file": {
                    "path": file_path,
                    "type": file_type,
                    "description": file_desc,
                },
            })

            # 呼叫模型
            try:
                if file_type == "image":
                    # 圖片使用多模態 content list，直接呼叫 llm
                    # 先取得歷史訊息以維持上下文
                    history = get_session_history(session_id)
                    history_messages = history.messages.copy()
                    all_messages = history_messages + [message]
                    response = llm.invoke(all_messages)
                    # 手動更新 session history
                    history.add_message(HumanMessage(content=log_content + " " + file_desc))
                    history.add_message(response)
                else:
                    # PDF / TXT 使用純文字 prompt，走 chain_with_history
                    response = chain_with_history.invoke(
                        {"input": [message]},
                        config={"configurable": {"session_id": session_id}},
                    )
                ai_text = response.content
            except Exception as e:
                ai_text = f"[錯誤] {e}"

        else:
            # ── 一般文字對話 ────────────────────────────────
            conversation_log.append({
                "timestamp": datetime.now().isoformat(),
                "role": "user",
                "content": user_input,
            })

            try:
                response = chain_with_history.invoke(
                    {"input": [HumanMessage(content=user_input)]},
                    config={"configurable": {"session_id": session_id}},
                )
                ai_text = response.content
            except Exception as e:
                ai_text = f"[錯誤] {e}"

        # 記錄 AI 回覆
        conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "role": "ai",
            "content": ai_text,
        })

        print(f"\n🤖 AI: {ai_text}")

    # 正常結束時存檔
    save_conversation()
    conversation_log.clear()
    print("\n👋 再見！")


if __name__ == "__main__":
    main()
