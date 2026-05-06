def calculate_performance_drop(clean_accuracy: float, noisy_accuracy: float) -> float:
    """Calculates absolute drop in performance due to robustness failures."""
    return clean_accuracy - noisy_accuracy

def add_ocr_noise(text: str) -> str:
    """Simulates OCR blur and noise on the input text."""
    if not text:
        return text
    # Common OCR mistakes
    noisy_text = text.replace("0", "O").replace("1", "I").replace("5", "S").replace("-", "~")
    # Simulate cropped document
    lines = noisy_text.split('\n')
    if len(lines) > 2:
        noisy_text = '\n'.join(lines[:-1]) # crop last line
    return noisy_text + "\n[UNREADABLE_BLOCK]"
