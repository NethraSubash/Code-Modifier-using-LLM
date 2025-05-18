# Code-Modifier-using-LLM

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Model and tokenizer setup
MODEL_NAME = "microsoft/phi-1_5"

# Load tokenizer and set pad token
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

# Load model with optimized device handling
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    # sets the data type for model weights
    device_map="auto"
)

# LoRA Configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

# Apply LoRA optimization
model = get_peft_model(base_model, lora_config).to(device)

# Function to modify Python code based on instructions
def modify_code(original_code, instruction):
    prompt = f"""You are an AI assistant that rewrites Python code.
Modify only the necessary parts based on the instruction below.

### Original Code:
{original_code}

### Instruction:
{instruction}

### Rewritten Code:
ONLY return properly formatted Python code.
"""

    try:
        # Tokenization with structured padding strategy
        inputs = tokenizer(prompt, return_tensors="pt", padding="longest", truncation=True).to(device)

        # Generate output using deterministic decoding
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Allow longer responses
            do_sample=False,  # Prevent randomness
            temperature=0.1,  # Reduce creative variation
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode and extract the modified code
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        modified_code = decoded.split("### Rewritten Code:")[-1].strip()

        # Validate output before returning
        if not any(keyword in modified_code for keyword in ["def ", "import ", "for ", "if ", "print"]):
            modified_code = "Error: Model output does not contain valid Python code."

        explanation = f"The code was modified based on your instruction: '{instruction}'"
        return explanation, modified_code

    except Exception as e:
        return "An error occurred during processing.", f"Error: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🔥 Python Code Modifier (Powered by Phi-1.5 + LoRA)")

    with gr.Row():
        original_code = gr.Code(label="Original Python Code", language="python")
        instruction = gr.Textbox(label="Modification Instruction")

    with gr.Row():
        explanation = gr.Textbox(label="Modification Explanation")
        modified_code = gr.Code(label="Modified Python Code", language="python")

    integrate_button = gr.Button("Integrate Code")  # Updated button label
    integrate_button.click(fn=modify_code, inputs=[original_code, instruction], outputs=[explanation, modified_code])

demo.launch()
