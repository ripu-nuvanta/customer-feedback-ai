import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import demo_analysis, make_answer
rows=[
 {"id":"1","customer_name":"A","message":"We may cancel because onboarding is confusing.","channel":"email","date":"2026-08-01","plan":"Pro","revenue":"1000"},
 {"id":"2","customer_name":"B","message":"Please add a Salesforce integration.","channel":"email","date":"2026-08-02","plan":"Pro","revenue":"2000"},
 {"id":"3","customer_name":"C","message":"Reports are slow.","channel":"chat","date":"2026-08-03","plan":"Enterprise","revenue":"5000"},
]
a=demo_analysis(rows)
assert a["total"]==3
assert len(a["churn_signals"])==1
assert len(a["feature_requests"])==1
print("basic analysis tests passed")
