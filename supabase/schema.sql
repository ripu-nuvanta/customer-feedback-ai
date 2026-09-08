create extension if not exists vector;

create table if not exists feedback_conversations (
  id uuid primary key default gen_random_uuid(),
  customer_id text not null,
  customer_name text not null,
  conversation_date timestamptz,
  channel text not null,
  message text not null,
  customer_type text,
  plan text,
  revenue numeric,
  category text,
  sentiment text,
  churn_signal boolean default false,
  feature_request boolean default false,
  competitor_mention boolean default false,
  topic text,
  reason text,
  embedding vector(1536),
  created_at timestamptz default now()
);

create index if not exists feedback_conversations_embedding_idx
on feedback_conversations using hnsw (embedding vector_cosine_ops);

create index if not exists feedback_conversations_topic_idx on feedback_conversations(topic);
create index if not exists feedback_conversations_customer_idx on feedback_conversations(customer_id);
create index if not exists feedback_conversations_date_idx on feedback_conversations(conversation_date);

create or replace function match_feedback(
  query_embedding vector(1536),
  match_count int default 10
)
returns table (
  id uuid, customer_name text, channel text, conversation_date timestamptz,
  message text, customer_type text, plan text, revenue numeric, similarity float
)
language sql stable
as $$
  select id, customer_name, channel, conversation_date, message,
         customer_type, plan, revenue,
         1 - (embedding <=> query_embedding) as similarity
  from feedback_conversations
  where embedding is not null
  order by embedding <=> query_embedding
  limit match_count;
$$;
