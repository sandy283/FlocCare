## For Testing and running the Main Chat Agent (connected with RAG tool)
1. create an environment: python -m venv floccare
2. activate and install the required libraries: source floccare/bin/activate
3. Install requirements: pip install -r requirements.txt
4. Datasets folder has the dataset which was use for training purpose and the scripts are Qwen_Lora_Finetune
5. For running the main chatbot, go to CustomAgent: cd CustomAgent
6. Run the chat agent using streamlit run script.py

## Fine tuning approach and configs
1. Activate the environment and install requirements (same as above)
2. Navigate to the fine-tuning directory: cd Qwen_Lora_Finetune
3. (Optional) Deduplicate training data: python deduplicate.py
4. Start fine-tuning the Qwen model: python finetune.py
5. Evaluate the fine-tuned model: python evaluate.py
6. Fine-tuned model checkpoints are saved in qwen_simple_output/
7. Model uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
8. Training data: medical_compliance_deduplicated.csv (51MB)
9. Base model: Qwen/Qwen2.5-1.5B-Instruct with custom LoRA configuration