import os, sys, json, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import demo_analysis, ai_analyse

df=pd.read_csv(os.path.join(os.path.dirname(__file__),"..","data","eval_1000.csv")).fillna("")
rows=[]
for i,r in df.iterrows():
    x=r.to_dict(); x["id"]=str(i+1); rows.append(x)
gt=json.load(open(os.path.join(os.path.dirname(__file__),"..","data","eval_ground_truth.json")))

def report(name, pred):
    def acc(k): return sum(bool(pred[i].get(k))==bool(gt[i][k]) for i in range(len(gt)))/len(gt)
    print(name, {k:round(acc(k),3) for k in ["churn_signal","feature_request"]})

# Baseline/fallback sanity check
a=demo_analysis(rows)
print("dataset rows:",len(rows))
print("top problems:",[(x["name"],x["mentions"]) for x in a["problems"][:5]])
print("churn candidates:",len(a["churn_signals"]),"feature candidates:",len(a["feature_requests"]))

# Only run expensive model eval when explicitly requested.
if os.getenv("RUN_AI_EVAL")=="1":
    analysed=ai_analyse(rows)
    pred=[r["ai"] for r in rows]
    report("OpenAI classifier",pred)
    print("AI evaluation completed")
else:
    print("Set RUN_AI_EVAL=1 and OPENAI_API_KEY to run the model evaluation.")
