import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import random
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class QwenEvaluator:
    def __init__(self, base_model_name="Qwen/Qwen2.5-1.5B-Instruct", adapter_path="qwen_simple_output"):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
        )

        print("Loading fine-tuned model...")
        self.finetuned_model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.finetuned_model.eval()
        print("Models loaded!")

    def classify_sentence(self, sentence, use_finetuned=True):
        prompt = f"Is this medical statement compliant or non-compliant? Answer only 'compliant' or 'non-compliant'.\nStatement: {sentence}\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        model = self.finetuned_model if use_finetuned else self.base_model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=len(inputs['input_ids'][0]) + 20,
                temperature=0.3, do_sample=True, pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip().lower()

        return "compliant" if "compliant" in generated and "non-compliant" not in generated else "non-compliant"

with open("test.json", 'r') as f:
    test_data = json.load(f)

print(f"Testing all {len(test_data)} samples...")

evaluator = QwenEvaluator()

true_labels = []
base_predictions = []
finetuned_predictions = []

for i, item in enumerate(test_data, 1):
    true_labels.append(item['class'])
    
    base_pred = evaluator.classify_sentence(item['sentence'], use_finetuned=False)
    base_predictions.append(base_pred)
    
    finetuned_pred = evaluator.classify_sentence(item['sentence'], use_finetuned=True)
    finetuned_predictions.append(finetuned_pred)
    
    print(f"{i}/{len(test_data)}")

cm_base = confusion_matrix(true_labels, base_predictions, labels=['compliant', 'non-compliant'])
cm_finetuned = confusion_matrix(true_labels, finetuned_predictions, labels=['compliant', 'non-compliant'])

base_accuracy = sum(1 for t, p in zip(true_labels, base_predictions) if t == p) / len(true_labels)
finetuned_accuracy = sum(1 for t, p in zip(true_labels, finetuned_predictions) if t == p) / len(true_labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.heatmap(cm_base, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['compliant', 'non-compliant'],
            yticklabels=['compliant', 'non-compliant'], ax=ax1)
ax1.set_title(f'Base Model\nAccuracy: {base_accuracy:.2%}')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

sns.heatmap(cm_finetuned, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['compliant', 'non-compliant'],
            yticklabels=['compliant', 'non-compliant'], ax=ax2)
ax2.set_title(f'Fine-tuned Model\nAccuracy: {finetuned_accuracy:.2%}')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nBase Model Accuracy: {base_accuracy:.2%}")
print(f"Fine-tuned Model Accuracy: {finetuned_accuracy:.2%}")
print(f"Improvement: {(finetuned_accuracy - base_accuracy):.2%}")