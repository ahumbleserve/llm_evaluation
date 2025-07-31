import logging
import pandas as pd
from transformers import pipeline
from bert_score import score
import json
import os

# Create output directory if it doesn't exist
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Configure logging to capture success/failure signals and edge cases
logging.basicConfig(
    filename=os.path.join(output_dir, 'llm_evaluation.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_dataset(file_path):
    """Load a CSV dataset with prompts and reference answers."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise

def evaluate_llm(prompts, reference_answers, model_name="t5-small"):
    """Evaluate an LLM with a scoring rubric (BERTScore) and capture logs."""
    try:
        # Initialize the model
        nlp = pipeline("text2text-generation", model=model_name)
        
        results = []
        for prompt, reference in zip(prompts, reference_answers):
            # Generate response from LLM
            response = nlp(prompt, max_length=50, num_beams=5)[0]
            generated_text = response['generated_text']
            
            # Calculate coherence metric (BERTScore)
            P, R, F1 = score([generated_text], [reference], lang="en", verbose=False)
            
            result = {
                "prompt": prompt,
                "generated": generated_text,
                "reference": reference,
                "bertscore_f1": F1.tolist()[0],
                "success": F1.tolist()[0] > 0.8  # Threshold for success
            }
            
            # Log edge cases
            if F1.tolist()[0] < 0.5:
                logging.warning(f"Edge case detected: Low BERTScore {F1.tolist()[0]} for prompt: {prompt}")
            
            results.append(result)
        
        logging.info(f"Evaluation completed for {len(prompts)} prompts")
        return pd.DataFrame(results)
    
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise

def save_results(results_df, output_path=os.path.join(output_dir, 'evaluation_results.json')):
    """Save evaluation results to a JSON file."""
    try:
        results_df.to_json(output_path, orient="records", indent=2)
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")
        raise

def main():
    # Example dataset
    dataset = pd.DataFrame({
        "prompt": ["What is the capital of France?", "What is 2+2?"],
        "reference": ["The capital is Paris.", "The answer is 4."]
    })
    
    # Save temporary dataset to simulate loading
    dataset_path = os.path.join(output_dir, 'sample_dataset.csv')
    dataset.to_csv(dataset_path, index=False)
    
    # Run pipeline
    df = load_dataset(dataset_path)
    results = evaluate_llm(df["prompt"], df["reference"])
    save_results(results)

if __name__ == "__main__":
    main()