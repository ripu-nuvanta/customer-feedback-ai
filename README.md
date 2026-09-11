# Customer Feedback AI

Customer Feedback AI is an AI-powered customer feedback analysis platform that helps teams turn customer conversations into actionable insights.

Instead of manually reviewing hundreds of customer conversations, teams can use AI to identify recurring problems, sentiment, unresolved issues, churn signals, feature requests, and other important customer insights.

## Features

- Customer feedback overview
- Sentiment analysis
- Recurring problem identification
- Unresolved issue tracking
- Churn and retention signals
- Customer conversation analysis
- Feature request identification
- High-value customer feedback
- AI-powered "Ask Your Data"
- Evidence-based answers from customer conversations

### Preserving Analyzed Data

After each dataset is analyzed, the analyzed customer feedback and generated insights are also preserved in CSV format.

Each analysis run is stored separately so that previous analyzed datasets are not overwritten or deleted.

The CSV data can be used later for:

- Reviewing previous analysis results
- Further data analysis
- Reporting
- Historical comparison
- Reusing analyzed customer feedback

This allows the system to preserve the analyzed data even after a new dataset is uploaded and analyzed.

## Project Structure

```text
customer-feedback-ai/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── public/
│   ├── package.json
│   └── .env.example
│
├── .gitignore
├── README.md
└── LICENSE