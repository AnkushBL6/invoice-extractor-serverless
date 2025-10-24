import runpod
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from PIL import Image
import torch
import json
import re
import base64
import io

print("🔥 Loading model...")

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="Ankushbl6/qwen2_5_vl_7b_invoice_lora_final",
    load_in_4bit=True,
    max_seq_length=2048,
)

FastVisionModel.for_inference(model)
data_collator = UnslothVisionDataCollator(model, tokenizer)

print("✅ Model ready")

def handler(job):
    try:
        image_b64 = job['input']['image']
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # YOUR ORIGINAL FULL PROMPT
        instruction = """Please carefully examine this invoice image and extract all the information into the following structured JSON format. Pay close attention to details and ensure accuracy in number formatting and text extraction.

Extract the data into this exact JSON structure:

{
  "header": {
    "invoice_no": "Invoice number or reference ID",
    "invoice_date": "Date the invoice was issued (maintain original format)",
    "due_date": "Payment due date if specified",
    "sender_name": "Name of the company/person issuing the invoice",
    "sender_addr": "Complete address of the sender/issuer",
    "rcpt_name": "Name of the recipient/customer",
    "rcpt_addr": "Address of the recipient/customer",
    "bank_iban": "International Bank Account Number",
    "bank_name": "Name of the bank",
    "bank_acc_no": "Bank account number",
    "bank_routing": "Bank routing number",
    "bank_swift": "SWIFT/BIC code",
    "bank_acc_name": "Account holder name",
    "bank_branch": "Bank branch information"
  },
  "items": [
    {
      "descriptions": "Detailed description of the item/service",
      "SKU": "Stock Keeping Unit or item code",
      "quantity": "Quantity of items",
      "unit_price": "Price per unit",
      "amount": "Total amount for this line item",
      "tax": "Tax amount for this item",
      "Line_total": "Total amount including tax for this line"
    }
  ],
  "summary": {
    "subtotal": "Subtotal amount before tax",
    "tax_rate": "Tax rate percentage or description",
    "tax_amount": "Total tax amount",
    "total_amount": "Final total amount to be paid",
    "currency": "Currency code (USD, EUR, etc.)"
  }
}

Important guidelines:
- Preserve original number formatting (including commas, decimals)
- If multiple line items exist, include all of them in the items array
- Use empty string "" for any field that is not present or cannot be clearly identified
- Maintain accuracy in financial figures - double-check all numbers
- Extract text exactly as it appears, including special characters and formatting
- For dates, preserve the original format shown in the invoice
- For due dates calculate the date based on payment terms if present
- Also line item wise tax calculation has to be done properly basis the given tax rate
- If currency symbols are present, note them appropriately
- If discounts are given, then tax calculation should be done after calculating (unit price * quantity - discount), then calculate tax for that amount
- For due date, if payment terms is upon receipt, upon publication, then it is the invoice date itself.

Return only the JSON object with the extracted information."""
        
        training_sample = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ]
            }]
        }
        
        batch = data_collator([training_sample])
        batch = {k: v.to(model.device) if v is not None else None for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch['image_grid_thw'],
                max_new_tokens=1536,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][batch['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        # Extract JSON from response
        cleaned_response = response.replace("assistant", "").strip()
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        
        if json_match:
            extracted_data = json.loads(json_match.group())
            return {"result": extracted_data, "status": "success"}
        else:
            return {"result": {}, "raw_response": response[:500], "status": "no_json"}
            
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "status": "failed"}

runpod.serverless.start({"handler": handler})
