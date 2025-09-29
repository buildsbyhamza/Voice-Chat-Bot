# 🎙️ Voice Helper Chatbot

A Streamlit-based **voice-enabled chatbot** that:
1. Records audio from your microphone (using `audio_recorder_streamlit`).
2. Transcribes speech to text via **Groq’s Whisper API**.
3. Sends the transcribed text to **Groq’s LLM API** (Meta LLaMA models).
4. Displays the chatbot’s response in real time.

---

## 🚀 Features
- 🎤 Record voice directly in your browser.
- 📝 Automatic transcription (STT → Speech To Text).
- 🤖 Get AI-generated replies using Groq’s LLaMA models.
- 📜 Maintains chat history during session.
- 🌐 Simple web app built with **Streamlit**.

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/voice-helper-chatbot.git
cd voice-helper-chatbot
