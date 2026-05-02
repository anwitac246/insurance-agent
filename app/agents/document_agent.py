import os
import base64
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from app.core.config import settings
from app.models.state import ClaimState, MissingData
from app.models.schemas import ExtractedClaimData
import pytesseract
from PIL import Image

# Initialize Groq LLMs
vision_llm = ChatGroq(model="llama-3.2-11b-vision-preview", api_key=settings.GROQ_API_KEY)
text_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)

# Use structured output
structured_llm = text_llm.with_structured_output(ExtractedClaimData)

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def validate_extracted_data(data: ExtractedClaimData) -> MissingData:
    missing_docs = []
    missing_fields = []

    # 1. Check Document Presence
    if not data.insurance_policy: missing_docs.append("Insurance Policy")
    if not data.claim_form: missing_docs.append("Claim Form")
    if not data.vehicle_rc: missing_docs.append("Vehicle RC")
    if not data.driving_license: missing_docs.append("Driving License")
    if not data.repair_estimate: missing_docs.append("Repair Estimate")
    if not data.identity_proof: missing_docs.append("Identity Proof")
    
    # 2. Check Required Fields within present documents
    if data.insurance_policy:
        if not data.insurance_policy.policy_number: missing_fields.append("policy.policy_number")
        if not data.insurance_policy.policyholder_name: missing_fields.append("policy.policyholder_name")
        if not data.insurance_policy.vehicle_number: missing_fields.append("policy.vehicle_number")
        if not data.insurance_policy.policy_start_date: missing_fields.append("policy.policy_start_date")
        if not data.insurance_policy.policy_end_date: missing_fields.append("policy.policy_end_date")
        
    if data.claim_form:
        if not data.claim_form.policy_number: missing_fields.append("claim_form.policy_number")
        if data.claim_form.signature_present is None: missing_fields.append("claim_form.signature_present")
        if not data.claim_form.incident_date: missing_fields.append("claim_form.incident_date")

    if data.vehicle_rc:
        if not data.vehicle_rc.vehicle_number: missing_fields.append("rc.vehicle_number")
        if not data.vehicle_rc.chassis_number: missing_fields.append("rc.chassis_number")
        
    if data.identity_proof:
        if not data.identity_proof.name: missing_fields.append("id.name")

    # 3. Cross-Document Validation
    if data.insurance_policy and data.claim_form:
        p_num = (data.insurance_policy.policy_number or "").strip()
        c_num = (data.claim_form.policy_number or "").strip()
        if p_num and c_num and p_num != c_num:
            missing_fields.append(f"cross_check: policy_number mismatch ({p_num} != {c_num})")

    if data.insurance_policy and data.vehicle_rc:
        p_veh = (data.insurance_policy.vehicle_number or "").strip()
        rc_veh = (data.vehicle_rc.vehicle_number or "").strip()
        if p_veh and rc_veh and p_veh != rc_veh:
            missing_fields.append(f"cross_check: vehicle_number mismatch between Policy and RC ({p_veh} != {rc_veh})")
            
    if data.insurance_policy and data.identity_proof:
        p_name = (data.insurance_policy.policyholder_name or "").lower().strip()
        id_name = (data.identity_proof.name or "").lower().strip()
        if p_name and id_name and p_name != id_name:
            missing_fields.append(f"cross_check: name mismatch between Policy and Identity Proof ({p_name} != {id_name})")

    return MissingData(missing_documents=missing_docs, missing_fields=missing_fields)

def process_documents(state: ClaimState) -> ClaimState:
    documents = state.get("local_files", {}).get("documents", [])
    images = state.get("local_files", {}).get("images", [])
    
    extracted_text = ""
    # Process text docs via OCR
    for doc in documents:
        if doc.endswith('.txt'):
            with open(doc, 'r', encoding='utf-8') as f:
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
    
    # Synthesize with Structured LLM
    prompt = f"""
    Analyze the following extracted claim text and damage summary.
    Extracted Text: {extracted_text}
    Damage Summary: {damage_summary}
    
    Carefully extract ALL required fields for each document type if they are present in the text.
    Do NOT hallucinate fields. If a field is not explicitly present, omit it or set to null.
    """
    
    try:
        extracted_data_obj = structured_llm.invoke(prompt)
    except Exception:
        extracted_data_obj = None
        
    state["extracted_data"] = extracted_data_obj.model_dump() if extracted_data_obj else {}
    
    # Run rigid validation
    if extracted_data_obj:
        missing_data = validate_extracted_data(extracted_data_obj)
    else:
        missing_data = MissingData(missing_documents=["All Required Documents"], missing_fields=["Extraction Failed"])
        
    state["missing_data"] = missing_data
    
    # Determine Status
    if missing_data["missing_documents"] or missing_data["missing_fields"]:
         state["status"] = "pending_docs"
    else:
         state["status"] = "processing"
         
    return state
