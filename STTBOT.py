import streamlit as st
import os
import requests
import io
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
from dotenv import load_dotenv

# --- config & env ---
load_dotenv()
# accept either env var name (fallback) — make sure the name you used in .env matches
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
if not API_KEY:
    st.error("API key missing. Please set GROQ_API_KEY (or GROK_API_KEY) in your .env")
    st.stop()

STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# initialize float UI features (if used)
float_init()

# session state for chat messages (optional)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How may I assist you today?"}]

st.title("Voice Helper Chatbot")


def detect_audio_file(bytes_data: bytes):
    
    if len(bytes_data) >= 4 and bytes_data[:4] == b"RIFF":
        return "speech.wav", "audio/wav"
    if len(bytes_data) >= 3 and bytes_data[:3] == b"ID3":
        return "speech.mp3", "audio/mpeg"
    
    if len(bytes_data) >= 2 and bytes_data[:2] == b"\xff\xfb":
        return "speech.mp3", "audio/mpeg"
    if len(bytes_data) >= 4 and bytes_data[:4] == b"OggS":
        return "speech.ogg", "audio/ogg"
    
    return "speech.webm", "audio/webm"


st.markdown("### Record your voice (mic) — press record, then stop to transcribe")
audio_bytes = audio_recorder()  

if audio_bytes:
    
    try:
        st.audio(audio_bytes)
    except Exception:
        
        if isinstance(audio_bytes, str):
            audio_bytes = audio_bytes.encode("utf-8")
        st.audio(io.BytesIO(audio_bytes))

    
    with st.spinner("🔄 Transcribing..."):
        
        filename, mime = detect_audio_file(audio_bytes)

        
        fileobj = io.BytesIO(audio_bytes)
        files = {"file": (filename, fileobj, mime)}
        data = {"model": "whisper-large-v3"}
        headers = {"Authorization": f"Bearer {API_KEY}"}

        try:
            st.write("📝 Sending audio to STT...")
            response = requests.post(STT_URL, headers=headers, files=files, data=data, timeout=120)
        except Exception as e:
            st.error(f"Network error sending STT request: {e}")
            st.stop()

        if response.status_code == 200:
            try:
                resp_json = response.json()
            except Exception:
                st.error("STT returned non-JSON response")
                st.write(response.text)
                st.stop()

            user_text = resp_json.get("text")
            if not user_text:
                st.error("STT returned empty transcription")
                st.write(resp_json)
            else:
                st.success(f"🗣️ You said: {user_text}")

                # Send to Chat LLM
                chat_payload = {
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [
                        {"role": "system", "content": "You are a helpful chatbot."},
                        {"role": "user", "content": user_text},
                    ],
                }
                chat_headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                }

                with st.spinner("🤖 Asking the chatbot..."):
                    try:
                        chat_response = requests.post(CHAT_URL, headers=chat_headers, json=chat_payload, timeout=60)
                    except Exception as e:
                        st.error(f"Network error calling chat API: {e}")
                        st.stop()

                if chat_response.status_code == 200:
                    try:
                        chat_json = chat_response.json()
                    except Exception:
                        st.error("Chat returned non-JSON response")
                        st.write(chat_response.text)
                        st.stop()

                    bot_reply = None
                    try:
                        bot_reply = chat_json["choices"][0]["message"]["content"]
                    except Exception:
                        
                        bot_reply = chat_json.get("text") or str(chat_json)

                    st.success(f"🤖 Bot: {bot_reply}")
                    
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                else:
                    st.error(f"Chat Error (status {chat_response.status_code}):")
                    
                    try:
                        st.write(chat_response.json())
                    except Exception:
                        st.write(chat_response.text)
        else:
            st.error(f"STT Error (status {response.status_code}):")
            try:
                st.write(response.json())
            except Exception:
                st.write(response.text)
