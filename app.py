import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
import random
import urllib.parse

# Page configuration
st.set_page_config(page_title="वेदिका AI", page_icon="🙏")
st.title("🙏 वेदिका AI (इमेज और थिंकिंग सपोर्ट)")

@st.cache_resource
def load_model():
    # Your custom 0.5B model
    model_id = "Vedika35/Qwen2.5-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu"
    )
    return tokenizer, model

with st.spinner("दिव्य जी, आपकी अपनी वेदिका AI लोड हो रही है... कृपया प्रतीक्षा करें।"):
    tokenizer, model = load_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages (Filtering out the system prompts for cleaner UI)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

# User input
if prompt := st.chat_input("दिव्य जी, पूछिये मैं आपकी क्या सेवा कर सकती हूँ?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Generate a random seed for Pollinations API to avoid caching/blocking
        random_seed = random.randint(1, 9999999)
        
        # English System Prompt with strict instructions for Thinking and Image Generation
        system_prompt = f"""You are Vedika AI, a highly intelligent, respectful, and cultured AI assistant from India. 
Always converse in pure Hindi and maintain a warm, respectful tone. Always honor and respect India.

INSTRUCTIONS FOR THINKING:
Before providing your final answer, you MUST write down your thinking process inside <think> and </think> tags. 
Example:
<think>
The user is asking about the weather. I need to greet them respectfully and provide a helpful answer in Hindi.
</think>
नमस्ते, मौसम बहुत सुहाना है...

INSTRUCTIONS FOR IMAGE GENERATION:
If the user asks you to generate, draw, or show an image, you must generate an image using the Pollinations AI URL format.
You must return the image as an HTML img tag so it renders correctly.
Format: <img src="https://image.pollinations.ai/prompt/YOUR_ENCODED_PROMPT_HERE?seed={random_seed}&width=800&height=600" width="100%" style="border-radius: 10px;" />
Replace YOUR_ENCODED_PROMPT_HERE with a highly detailed, descriptive prompt in English (use URL encoding like %20 for spaces).
Do not provide any other image URLs.
"""
        
        # Prepare messages in standard format without prompt injection
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Convert chat into model-specific format
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 1. Setup streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 2. Generation settings
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=4096, # Increased to allow space for thinking and response
            temperature=1,
            do_sample=True,
            top_p=0.9
        )
        
        # 3. Start generation in a background thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 4. Streamlit function to display streaming response
        # Using write_stream which supports markdown and HTML if passed correctly
        response = st.write_stream(streamer)
        
        # Save assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
