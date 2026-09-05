import pandas as pd
from pathlib import Path

df = pd.read_csv(r'D:\LawRAG\src\evaluation\rag_eval\evals\experiments\mystifying_dorsey.csv')

print(f"| Faithfulness | {df['faithfulness'].mean():.2f} |")
print(f"| Answer Relevancy | {df['answer_relevancy'].mean():.2f} |")
print(f"| Context Precision | {df['context_precision'].mean():.2f} |")
print(f"| Context Recall | {df['context_recall'].mean():.2f} |")
print(f"| Correctness (pass rate) | {(df['correctness'] == 'pass').mean() * 100:.1f}% |")

Path("eval").mkdir(exist_ok=True)
df.to_csv("eval/results.csv", index=False)