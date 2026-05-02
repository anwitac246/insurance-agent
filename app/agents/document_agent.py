import os
import base64
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from app.core.config import settings
from app.models.state import ClaimState
import pytesseract
from PIL import Image

# Initialize Groq LLMs
vision_llm = ChatGroq(model="llama-3.2-11b-vision-preview", api_key=settings.GROQ_API_KEY)
text_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_documents(state: ClaimState) -> ClaimState:
    documents = state.get("local_files", {}).get("documents", [])
    images = state.get("local_files", {}).get("images", [])
    
    extracted_text = ""
    # Process text docs via OCR
    for doc in documents:
        if doc.endswith('.txt'):
            with open(doc, 'r') as f:
                extracted_text += f.read() + "\n"
        elif doc.endswith(('.png', '.jpg', '.jpeg')):
             try:
                 extracted_text += pytesseract.image_to_string(Image.open(doc)) + "\n"
             except Exception:
                 pass
                 
    # Process images for damage via Vision LLM
    damage_summary = ""
    if images:
        image_messages = []
        for img in images:
            b64_img = encode_image(img)
            image_messages.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            )
        
        if image_messages:
            msg = HumanMessage(content=[{"type": "text", "text": "Describe the damage to the vehicle in these images in detail."}] + image_messages)
            try:
                vision_res = vision_llm.invoke([msg])
                damage_summary = vision_res.content
            except Exception as e:
                damage_summary = f"Error analyzing images: {str(e)}"
    
    # Synthesize with Text LLM
    prompt = f"""
    Analyze the following extracted claim text and damage summary.
    Extracted Text: {extracted_text}
    Damage Summary: {damage_summary}
    
    Extract user details, vehicle details, incident summary.
    Identify if FIR, RC, License, and Repair Estimates are present.
    Format your response as structured data.
    """
    
    try:
        extraction_res = text_llm.invoke(prompt)
        text_content = extraction_res.content
    except Exception:
        text_content = "Extraction failed."
        
    state["extracted_data"] = {
        "raw_extraction": text_content,
        "damage_summary": damage_summary
    }
    
    # Simple check for missing docs (mock logic)
    missing = []
    if "FIR" not in extracted_text.upper():
        missing.append("FIR")
        
    state["missing_docs"] = missing
    
    if missing:
         state["status"] = "pending_docs"
    else:
         state["status"] = "processing"
         
    return state
