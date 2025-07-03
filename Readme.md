## For Testing and running the Main Chat Agent (connected with RAG tool)
1. create an environment: python -m venv floccare
2. activate and install the required libraries: source floccare/bin/activate
3. Install requirements: pip install -r requirements.txt
4. Datasets folder has the dataset which was use for training purpose and the scripts are Qwen_Lora_Finetune
5. For running the main chatbot, go to CustomAgent: cd CustomAgent
6. Run the chat agent using streamlit run script.py
7. change the LLM using at the backened (for now you won't be able to use the ollama because it requires the GPU, which is not there on the streamlit. So instead use the API key for gemini to proceed)

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

## Test Cases ##
- Compliant Questions (should show COMPLIANT for most regulations)
1.	“Consuming at least 2.5 cups of fruits and vegetables each day as part of a healthy diet may help reduce the risk of some chronic diseases.”

2.	“This supplement is not intended to diagnose, treat, cure, or prevent any disease.”

3.	 “Clinical studies suggest that our device may improve sleep quality in adults with mild insomnia, but results may vary.”

4.	 “Our sunscreen helps protect against sunburn when used as directed with other sun protection measures.”

5.	 “In clinical trials, our medication demonstrated similar efficacy to the leading brand in reducing blood pressure.”

- Non-Compliant Questions (should show NON-COMPLIANT for ALL regulations - FDA, EMA, and HSA)
1.	 “This herbal remedy cures all types of cancer within three months.”

2.	 “Guaranteed results: You will lose 10 pounds in one week or your money back.”

3.	 “Our medical device is the safest and most effective option available, with no side effects.”

4.	 “Our supplement works better than any prescription drug on the market.”

5.	 “This surgery has a 100% success rate and is completely risk-free.”

**Expected behavior for non-compliant claims:**
- Initial assessment will show NON-COMPLIANT for your selected regulation
- When you click "Check Other Regulations", it will show NON-COMPLIANT for the other 2 regulations  
- Multi-regulation summary should show "3 regulation(s) found non-compliant"
- Clicking "Provide Explanation" should show detailed explanations for all 3 regulations (FDA, EMA, HSA)

streamlit hosted demo:- https://floccare.streamlit.app/
