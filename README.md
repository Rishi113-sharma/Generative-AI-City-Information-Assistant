# Generative-AI-City-Information-Assistant
Developed a **local Generative AI Assistant** using the **Qwen2.5-7B-Instruct** language model, optimized with **4-bit NF4 quantization** and CUDA GPU acceleration for efficient execution on an NVIDIA RTX 3060.

The project includes **two variants**:

**Base Version:**
A general-purpose conversational AI assistant built with Streamlit that maintains chat history and generates responses locally using Qwen2.5-7B-Instruct.

**City Information Version:**
An enhanced version that integrates external tools for real-time information. It uses **Open-Meteo** to retrieve current weather data and **Tavily** to retrieve the latest city-related news. The retrieved information is then provided to the language model to generate natural-language responses.

Both variants use Hugging Face Transformers for local model inference and provide a Streamlit-based conversational interface.

### Technologies Used

**Python • Generative AI • Qwen2.5-7B-Instruct • Hugging Face Transformers • Streamlit • LangChain • PyTorch • CUDA • BitsAndBytes • 4-bit NF4 Quantization • Tavily • Open-Meteo API • REST APIs • JSON • Prompt Engineering**

