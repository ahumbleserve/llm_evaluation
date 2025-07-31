import logging
import pandas as pd
from transformers import pipeline
from bert_score import score
from datetime import datetime
import json
import os

# Criar diretório de saída, se não existir
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Configuração de logging para capturar sinais de sucesso/falha e casos de borda
logging.basicConfig(
    filename=os.path.join(output_dir, 'llm_evaluation.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_dataset(file_path):
    """Carrega um dataset CSV com prompts e respostas de referência."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise

def evaluate_llm(prompts, reference_answers, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Avalia um LLM com uma rubrica de pontuação (BERTScore) e captura logs."""
    try:
        # Inicializa o modelo
        nlp = pipeline("text-classification", model=model_name)
        
        results = []
        for prompt, reference in zip(prompts, reference_answers):
            # Gera resposta do LLM
            response = nlp(prompt)[0]
            generated_text = response['label']
            
            # Calcula métrica de coerência (BERTScore)
            P, R, F1 = score([generated_text], [reference], lang="en", verbose=False)
            
            result = {
                "prompt": prompt,
                "generated": generated_text,
                "reference": reference,
                "bertscore_f1": F1.tolist()[0],
                "success": F1.tolist()[0] > 0.8  # Threshold para sucesso
            }
            
            # Log de casos de borda
            if F1.tolist()[0] < 0.5:
                logging.warning(f"Edge case detected: Low BERTScore {F1.tolist()[0]} for prompt: {prompt}")
            
            results.append(result)
        
        logging.info(f"Evaluation completed for {len(prompts)} prompts")
        return pd.DataFrame(results)
    
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise

def save_results(results_df, output_path=os.path.join(output_dir, 'evaluation_results.json')):
    """Salva os resultados da avaliação em um arquivo JSON."""
    try:
        results_df.to_json(output_path, orient="records", indent=2)
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")
        raise

def main():
    # Exemplo de dataset
    dataset = pd.DataFrame({
        "prompt": ["What is the capital of France?", "What is 2+2?"],
        "reference": ["The capital is Paris.", "The answer is 4."]
    })
    
    # Salva dataset temporário para simular carregamento
    dataset_path = os.path.join(output_dir, 'sample_dataset.csv')
    dataset.to_csv(dataset_path, index=False)
    dataset_path = os.path.join(output_dir, 'custom_dataset.csv')
    
    # Executa pipeline
    df = load_dataset(dataset_path)
    #results = evaluate_llm(df["prompt"], df["reference"])
    results = evaluate_llm(df["prompt"], df["reference"], model_name="bert-base-uncased")
    save_results(results)

if __name__ == "__main__":
    main()
