from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import nltk
from detoxify import Detoxify
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = FastAPI(title="LLM Evaluation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or better: ["https://llm-frontend-...a.run.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class EvaluationItem(BaseModel):
    user_query: str
    expected_response: str
    actual_response: str

class LLMEvaluator:
    def __init__(self, sbert_model_name: str = 'all-MiniLM-L6-v2', detoxify_model_name: str = 'original'):
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.detoxify_model = Detoxify(detoxify_model_name)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1

    def compute_toxicity(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        scores = self.detoxify_model.predict(df[column].tolist())
        return pd.concat([df, pd.DataFrame(scores, index=df.index)], axis=1)

    def compute_semantic_similarity(self, df: pd.DataFrame, ref_col: str, cand_col: str) -> pd.DataFrame:
        embeddings_ref = self.sbert_model.encode(df[ref_col].tolist(), convert_to_tensor=True)
        embeddings_cand = self.sbert_model.encode(df[cand_col].tolist(), convert_to_tensor=True)
        cos_scores = util.cos_sim(embeddings_ref, embeddings_cand)
        df['similarity_score'] = [cos_scores[i][i].item() for i in range(len(cos_scores))]
        return df

    def _calculate_bleu_row(self, row, ref_col, cand_col):
        reference = [word_tokenize(row[ref_col].lower())]
        candidate = word_tokenize(row[cand_col].lower())
        try:
            return sentence_bleu(reference, candidate, smoothing_function=self.smoothing_function)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def compute_bleu(self, df: pd.DataFrame, ref_col: str, cand_col: str) -> pd.DataFrame:
        df['BLEU_score'] = df.apply(lambda row: self._calculate_bleu_row(row, ref_col, cand_col), axis=1)
        return df

    def _calculate_rouge_row(self, row, ref_col, cand_col):
        scores = self.rouge_scorer.score(row[ref_col], row[cand_col])
        return pd.Series([scores['rouge1'].fmeasure, scores['rougeL'].fmeasure])

    def compute_rouge(self, df: pd.DataFrame, ref_col: str, cand_col: str) -> pd.DataFrame:
        rouge_scores = df.apply(lambda row: self._calculate_rouge_row(row, ref_col, cand_col), axis=1)
        rouge_scores.columns = ['ROUGE1_F1', 'ROUGEL_F1']
        return pd.concat([df, rouge_scores], axis=1)

    def compute_bertscore(self, df: pd.DataFrame, ref_col: str, cand_col: str) -> pd.DataFrame:
        _, _, f1 = score(df[cand_col].tolist(), df[ref_col].tolist(), lang="en", verbose=False)
        df['BERTScore_F1'] = f1.numpy()
        return df

    def run_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_toxicity(df, 'actual_response')
        df = self.compute_semantic_similarity(df, 'expected_response', 'actual_response')
        df = self.compute_bleu(df, 'expected_response', 'actual_response')
        df = self.compute_rouge(df, 'expected_response', 'actual_response')
        df = self.compute_bertscore(df, 'expected_response', 'actual_response')
        return df

# Instantiate evaluator once (at app startup)
evaluator = LLMEvaluator()

@app.post("/evaluate")
def evaluate_responses(items: List[EvaluationItem]):
    if not items:
        raise HTTPException(status_code=400, detail="Empty payload.")
    
    df = pd.DataFrame([{
        "user_query": item.user_query,
        "expected_response": item.expected_response,
        "actual_response": item.actual_response
    } for item in items])

    try:
        evaluated_df = evaluator.run_all(df)
        evaluated_df = evaluated_df.round(2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return only evaluation columns + input
    return evaluated_df.to_dict(orient="records")
