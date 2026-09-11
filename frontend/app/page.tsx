"use client";

import { useEffect, useMemo, useState } from "react";

type Conversation = {
  id: string;
  customer_id?: string;
  customer_name?: string;
  customer?: string;
  date?: string;
  channel?: string;
  message?: string;
  original_message?: string;
  feedback_summary?: string;
  plan?: string;
  revenue?: number | string;
  customer_type?: string;
  ai_generated_summary?: boolean;
  sentiment?: string;
  feature_request?: boolean;
  churn_signal?: boolean;
  unresolved_issue?: boolean;
  topic?: string;
  problem?: string;
};

type Problem = {
  name: string;
  mentions: number;
  percentage?: number;
  summary?: string;
  source_ids?: string[];
};

type FeatureRequest = {
  id: string;
  customer: string;
  message: string;
  summary?: string;
  reason?: string;
  source_ids?: string[];
  channel?: string;
  date?: string;
  sentiment?: string;
  topic?: string;
};

type ChurnSignal = {
  id: string;
  customer: string;
  message: string;
  evidence?: string;
  reason?: string;
  severity?: string;
  plan?: string;
  source_ids?: string[];
  channel?: string;
  date?: string;
  sentiment?: string;
  topic?: string;
  unresolved_issue?: boolean;
};

type HighValue = {
  id: string;
  customer: string;
  revenue?: number | string;
  message: string;
  evidence?: string;
  summary?: string;
};

type Segment = {
  segment: string;
  count: number;
  churn: number;
  negative: number;
  summary?: string;
};

type Analysis = {
  total?: number;
  unique_customers?: number;
  ai_analyzed_evidence_rows?: number;
  overview?: {
    headline?: string;
    summary?: string;
    priority?: string;
    source_ids?: string[];
    ai_generated?: boolean;
  };
  problems?: Problem[];
  feature_requests?: FeatureRequest[];
  churn_signals?: ChurnSignal[];
  sentiment?: { positive?: number; negative?: number; neutral?: number };
  high_value?: HighValue[];
  segments?: Segment[];
  source_counts?: { email?: number; call?: number; chat?: number };
  suggested_questions?: string[];
  analysis_method?: string;
};

type AskResponse = {
  answer?: string;
  key_findings?: string[];
  recommendation?: string;
  confidence?: string;
  source_ids?: string[];
  sources?: Conversation[];
  suggested_questions?: string[];
  ai_used?: boolean;
};

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

function text(v: unknown) {
  return v === null || v === undefined ? "" : String(v);
}

function customerName(item: Conversation | any) {
  return item?.customer_name || item?.customer || "Unknown customer";
}

function iconForChannel(channel?: string) {
  const value = (channel || "").toLowerCase();
  if (value.includes("email")) return "✉";
  if (value.includes("call")) return "☎";
  if (value.includes("chat")) return "●";
  return "▣";
}

function money(value: unknown) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? `$${n.toLocaleString()}` : "";
}


export default function Home() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [conversationResults, setConversationResults] = useState<Conversation[]>([]);
  const [answerSources, setAnswerSources] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [error, setError] = useState("");
  const [activeSection, setActiveSection] = useState("Overview");
  const [showAllFeatures, setShowAllFeatures] = useState(false);
  const [showAllChurn, setShowAllChurn] = useState(false);
  const [sentimentFilter, setSentimentFilter] = useState("all");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [channelFilter, setChannelFilter] = useState("all");
  const [insightFilter, setInsightFilter] = useState("all");

  const problems = analysis?.problems || [];
  const features = analysis?.feature_requests || [];
  const churn = analysis?.churn_signals || [];
  const highValue = analysis?.high_value || [];
  const segments = analysis?.segments || [];
  const sentiment = analysis?.sentiment || {};
  const total = analysis?.total || 0;

  const sourceConversations = conversationResults;
  const filteredAnswerSources = useMemo(() => {
    return answerSources.filter((item) => {
      const sentimentOk = sentimentFilter === "all" || (item.sentiment || "neutral").toLowerCase() === sentimentFilter;
      const channelOk = channelFilter === "all" || (item.channel || "").toLowerCase() === channelFilter;
      const categoryOk =
        categoryFilter === "all" ||
        (categoryFilter === "churn" && !!item.churn_signal) ||
        (categoryFilter === "feature" && !!item.feature_request) ||
        (categoryFilter === "unresolved" && !!item.unresolved_issue) ||
        (categoryFilter === "general" && !item.churn_signal && !item.feature_request && !item.unresolved_issue);
      return sentimentOk && channelOk && categoryOk;
    });
  }, [answerSources, sentimentFilter, categoryFilter, channelFilter]);

  const filteredConversations = useMemo(() => {
    return sourceConversations.filter((item) => {
      const sentimentOk = sentimentFilter === "all" || (item.sentiment || "neutral").toLowerCase() === sentimentFilter;
      const channelOk = channelFilter === "all" || (item.channel || "").toLowerCase() === channelFilter;
      const categoryOk =
        categoryFilter === "all" ||
        (categoryFilter === "churn" && !!item.churn_signal) ||
        (categoryFilter === "feature" && !!item.feature_request) ||
        (categoryFilter === "unresolved" && !!item.unresolved_issue) ||
        (categoryFilter === "general" && !item.churn_signal && !item.feature_request && !item.unresolved_issue);
      return sentimentOk && channelOk && categoryOk;
    });
  }, [sourceConversations, sentimentFilter, categoryFilter, channelFilter]);

  const filteredFeatures = useMemo(() => {
    if (insightFilter === "churn") return [];
    return features;
  }, [features, insightFilter]);

  const filteredChurn = useMemo(() => {
    if (insightFilter === "feature") return [];
    return churn;
  }, [churn, insightFilter]);

  const quickQuestions = useMemo(() => {
    return analysis?.suggested_questions || [];
  }, [analysis]);

  async function waitForAnalysis() {
    /*
     * IMPORTANT:
     * The backend can now process the full dataset through many small
     * AI batches. That can legitimately take several minutes.
     *
     * Previously this stopped after ~95 seconds, which caused the frontend
     * to report an error while the backend was still working.
     *
     * We now allow up to 15 minutes for the backend analysis to finish.
     * The frontend still polls every second and immediately exits when
     * the backend reports "complete" or "error".
     */
    const MAX_WAIT_SECONDS = 15 * 60;

    for (let i = 0; i < MAX_WAIT_SECONDS; i++) {
      try {
        const response = await fetch(`${API}/api/analysis/status`, {
          cache: "no-store",
        });

        if (!response.ok) {
          throw new Error(
            `Analysis status request failed with HTTP ${response.status}.`
          );
        }

        const status = await response.json();

        if (status.status === "complete" && status.analysis) {
          setAnalysis(status.analysis as Analysis);
          setLoading(false);
          setLoadingMessage("");
          return;
        }

        if (status.status === "error") {
          throw new Error(status.error || "AI analysis failed.");
        }

        setLoadingMessage(
          status.message ||
            "Customer Feedback AI is analyzing the customer evidence…"
        );
      } catch (err) {
        /*
         * Preserve backend errors exactly as before.
         * Network/status failures should still surface to the user
         * rather than silently continuing forever.
         */
        throw err;
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    throw new Error(
      "AI analysis is taking longer than expected. The backend is still processing the data. Please check the backend terminal for the current OpenAI request status."
    );
  }

  async function startAnalysis(endpoint: string, options?: RequestInit) {
    setLoading(true);
    setSentimentFilter("all");
    setCategoryFilter("all");
    setChannelFilter("all");
    setError("");
    setAnswer(null);
    setAnswerSources([]);
    setConversationResults([]);
    setAnalysis(null);
    setLoadingMessage("Loading customer evidence…");
    try {
      const response = await fetch(`${API}${endpoint}`, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data?.detail || "Unable to load customer data.");
      setLoadingMessage(data.message || "Customer Feedback AI analysis is starting…");
      await waitForAnalysis();
      setActiveSection("Overview");
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Could not connect to the Customer Feedback AI backend.");
      setLoading(false);
    }
  }

  async function uploadCSV(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    await startAnalysis("/api/upload", { method: "POST", body: formData });
    event.target.value = "";
  }

  async function askQuestion(customQuestion?: string) {
    const q = (customQuestion ?? question).trim();
    if (!q || loading || !analysis) return;
    setQuestion(q);
    setLoading(true);
    setLoadingMessage("Customer Feedback AI is retrieving evidence and writing an answer…");
    setError("");
    setAnswer(null);
    try {
      const response = await fetch(`${API}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data?.detail || "Unable to answer the question.");
      setAnswer(data as AskResponse);
      setAnswerSources(data.sources || []);
      setActiveSection("Ask your data");
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Something went wrong while asking Customer Feedback AI.");
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  }

  useEffect(() => {
    if (!analysis || conversationResults.length) return;
    fetch(`${API}/api/conversations`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => setConversationResults(Array.isArray(rows) ? rows : []))
      .catch(() => undefined);
  }, [analysis, conversationResults.length]);

  return (
    <main className="min-h-screen bg-[#f7f8fa] text-[#172033]">
      <header className="sticky top-0 z-40 border-b border-[#e7e9ee] bg-white/95 backdrop-blur">
        <div className="mx-auto flex h-[72px] max-w-[1500px] items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#172033] text-sm font-bold text-white">C</div>
            <div>
              <div className="text-xl font-extrabold tracking-tight text-[#172033]">Customer Feedback AI</div>
              <h1 className="mt-0.5 text-sm font-bold tracking-tight text-[#172033]">by Nuvanta AI</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <label className="cursor-pointer rounded-lg border border-[#dfe3ea] bg-white px-4 py-2 text-sm font-semibold hover:bg-[#f5f6f8]">
              ↑ Upload CSV
              <input type="file" accept=".csv" className="hidden" onChange={uploadCSV} />
            </label>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1500px]">
        <aside className="hidden min-h-[calc(100vh-72px)] w-[240px] shrink-0 border-r border-[#e7e9ee] bg-white px-4 py-6 lg:block">
          <div className="mb-3 px-3 text-[11px] font-semibold uppercase tracking-[0.16em] text-[#9aa1ad]">Workspace</div>
          <nav className="space-y-1">
            {[["⌂", "Overview"], ["⌕", "Ask your data"], ["◌", "Conversations"], ["↗", "Insights"]].map(([icon, label]) => (
              <button key={label} onClick={() => setActiveSection(label)} className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm font-medium ${activeSection === label ? "bg-[#172033] text-white" : "text-[#697386] hover:bg-[#f7f8fa]"}`}>
                <span className="w-5 text-center">{icon}</span>{label}
              </button>
            ))}
          </nav>
          <div className="mb-3 mt-9 px-3 text-[11px] font-semibold uppercase tracking-[0.16em] text-[#9aa1ad]">Data sources</div>
          <div className="space-y-1">
            {[["✉", "Email"], ["◉", "WhatsApp"], ["▣", "Support tickets"], ["☎", "Call transcripts"], ["↑", "CSV uploads"]].map(([icon, label]) => (
              <div key={label} className="flex items-center justify-between rounded-lg px-3 py-2.5 text-sm text-[#697386]">
                <span className="flex items-center gap-3"><span className="w-5 text-center">{icon}</span>{label}</span>
                <span className="text-[10px] font-semibold uppercase text-emerald-600">Ready</span>
              </div>
            ))}
          </div>
          <div className="mt-10 rounded-xl bg-[#172033] p-4 text-white">
            <div className="mb-2 text-xs font-semibold">Evidence workspace</div>
            <p className="text-xs leading-5 text-[#c8cfda]">Customer Feedback AI reads the supplied conversations and keeps original source text separate as evidence.</p>
          </div>
        </aside>

        <section className="min-w-0 flex-1 px-5 py-8 sm:px-8">
          {analysis && activeSection === "Conversations" && (
            <FilterBar
              sentiment={sentimentFilter}
              category={categoryFilter}
              channel={channelFilter}
              onSentiment={setSentimentFilter}
              onCategory={setCategoryFilter}
              onChannel={setChannelFilter}
            />
          )}

          {analysis && activeSection === "Insights" && (
            <InsightsFilter
              value={insightFilter}
              onChange={setInsightFilter}
            />
          )}

          {error && <div className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm leading-6 text-red-700">{error}</div>}

          {!analysis && loading && (
            <div className="flex min-h-[calc(100vh-160px)] items-center justify-center">
              <div className="w-full max-w-xl rounded-2xl border border-[#e7e9ee] bg-white p-10 text-center shadow-sm">
                <div className="mx-auto mb-5 h-10 w-10 animate-spin rounded-full border-2 border-[#dfe3ea] border-t-[#172033]" />
                <div className="text-lg font-semibold">Analysing customer feedback…</div>
                <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-[#7b8495]">{loadingMessage || "Customer Feedback AI is reading the supplied customer evidence."}</p>
                <div className="mt-6 rounded-lg bg-[#f6f7f9] px-4 py-3 text-left text-xs leading-5 text-[#667085]">The dashboard waits for the complete analysis to finish before showing the results.</div>
              </div>
            </div>
          )}

          {!analysis && !loading && (
            <div className="flex min-h-[calc(100vh-160px)] items-center justify-center">
              <div className="w-full max-w-2xl rounded-2xl border border-dashed border-[#d9dde5] bg-white p-12 text-center">
                <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-[#172033] text-xl font-bold text-white">AI</div>
                <h2 className="text-2xl font-semibold">Load customer feedback</h2>
                <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-[#7b8495]">Upload your customer-feedback CSV. Customer Feedback AI will analyze the complete evidence and generate the overview, insights, questions and answers.</p>
                <div className="mt-6 flex justify-center">
                  <label className="cursor-pointer rounded-lg border border-[#dfe3ea] bg-white px-5 py-2.5 text-sm font-semibold">Upload CSV<input type="file" accept=".csv" className="hidden" onChange={uploadCSV} /></label>
                </div>
              </div>
            </div>
          )}

          {analysis && (
            <>
              {activeSection === "Overview" && (
                <Overview analysis={analysis} problems={problems} sentiment={sentiment} total={total} onAsk={(q) => askQuestion(q)} />
              )}
              {activeSection === "Ask your data" && (
                <AskPage question={question} setQuestion={setQuestion} loading={loading} quickQuestions={quickQuestions} answer={answer} onAsk={askQuestion} conversations={filteredAnswerSources} />
              )}
              {activeSection === "Conversations" && (
                <ConversationsPage conversations={filteredConversations} />
              )}
              {activeSection === "Insights" && (
                <InsightsPage analysis={analysis} features={filteredFeatures} churn={filteredChurn} highValue={highValue} showAllFeatures={showAllFeatures} showAllChurn={showAllChurn} setShowAllFeatures={setShowAllFeatures} setShowAllChurn={setShowAllChurn} />
              )}
            </>
          )}
        </section>
      </div>
    </main>
  );
}

function Overview({ analysis, problems, sentiment, total, onAsk }: { analysis: Analysis; problems: Problem[]; sentiment: any; total: number; onAsk: (q: string) => void }) {
  const overview = analysis.overview;
  return (
    <div className="mx-auto max-w-[1100px]">
      <div className="mb-8">
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[#e1e5eb] bg-white px-3 py-1.5 text-xs font-semibold text-[#667085]"><span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />Customer overview</div>
        <h2 className="max-w-4xl text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">What customers are telling your team.</h2>
        <p className="mt-3 max-w-3xl text-[15px] leading-7 text-[#697386]">A decision-ready summary grounded in the supplied customer evidence, with source conversations available behind each signal.</p>
      </div>

      <section className="mb-6 rounded-2xl bg-[#172033] p-7 text-white sm:p-8">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#aeb8c8]">Strongest customer signal</div>
        <h3 className="mt-3 text-2xl font-semibold sm:text-3xl">{overview?.headline || "The strongest recurring customer signal."}</h3>
        <p className="mt-5 max-w-4xl whitespace-pre-line text-[15px] leading-8 text-[#d5dbe5]">{overview?.summary || "No overview summary is available."}</p>
        <div className="mt-6 border-t border-white/10 pt-5"><div className="text-[11px] font-semibold uppercase tracking-[0.15em] text-[#aeb8c8]">Recommended priority</div><p className="mt-2 text-sm leading-7 text-white">{overview?.priority || "No priority was returned."}</p></div>
      </section>

      <div className="mb-6 grid grid-cols-2 gap-4 xl:grid-cols-4">
        <Metric label="Conversations" value={total} description="Source records loaded" />
        <Metric label="Churn signals" value={analysis.churn_signals?.length || 0} description="Retention risk signals" />
        <Metric label="Feature requests" value={analysis.feature_requests?.length || 0} description="Capability demand" />
        <Metric label="Customers" value={analysis.unique_customers || 0} description="Distinct customer identities" />
      </div>

      <section className="mb-6 rounded-2xl border border-[#e5e8ed] bg-white p-6">
        <SectionHeader eyebrow="Recurring problems" title="What is creating the most friction?" description="Related problems are grouped by the evidence in the dataset." />
        <div className="mt-6 space-y-4">
          {problems.slice(0, 6).map((p, i) => (
            <div key={`${p.name}-${i}`} className="rounded-xl border border-[#edf0f3] p-5">
              <div className="flex items-start justify-between gap-4"><div className="flex items-center gap-3"><span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#f1f3f6] text-xs font-bold">{i + 1}</span><div className="text-sm font-semibold">{p.name}</div></div><span className="text-xs font-semibold text-[#7b8495]">{p.mentions} evidence records</span></div>
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[#eef0f3]"><div className="h-full rounded-full bg-[#172033]" style={{ width: `${Math.min(100, Math.max(4, p.percentage || 0))}%` }} /></div>
              <p className="mt-4 text-sm leading-7 text-[#596477]">{p.summary || "No problem summary is available."}</p>
              {p.source_ids?.length ? <div className="mt-3 text-[11px] font-semibold uppercase tracking-[0.12em] text-[#98a0ad]">Sources: {p.source_ids.slice(0, 6).map((id) => `[${id}]`).join(" ")}</div> : null}
            </div>
          ))}
        </div>
      </section>

      <section className="mb-6 rounded-2xl border border-[#e5e8ed] bg-white p-6">
        <SectionHeader title="Sentiment" description="Sentiment distribution across the analyzed evidence." />
        <div className="mt-5 grid gap-4 md:grid-cols-3"><SentimentCard label="Positive" value={sentiment.positive || 0} /><SentimentCard label="Negative" value={sentiment.negative || 0} /><SentimentCard label="Neutral" value={sentiment.neutral || 0} /></div>
        <div className="mt-5 rounded-xl bg-[#f7f8fa] px-4 py-3 text-xs leading-5 text-[#7b8495]">All {total.toLocaleString()} uploaded records are analyzed. Ask your data retrieves supporting evidence from the same dataset.</div>
      </section>
    </div>
  );
}

function AskPage({ question, setQuestion, loading, quickQuestions, answer, onAsk, conversations }: { question: string; setQuestion: (v: string) => void; loading: boolean; quickQuestions: string[]; answer: AskResponse | null; onAsk: (q?: string) => void; conversations: Conversation[] }) {
  return (
    <div className="mx-auto max-w-[1100px]">
      <div className="mb-8"><div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#89919f]">Customer intelligence</div><h2 className="text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">Ask your customer data.</h2><p className="mt-3 max-w-3xl text-[15px] leading-7 text-[#697386]">Ask in plain language. Customer Feedback AI retrieves the most relevant evidence from the analyzed dataset and builds the answer around what customers actually said.</p></div>
      <section className="rounded-2xl border border-[#e5e8ed] bg-white p-6">
        <div className="flex flex-col gap-3 sm:flex-row"><input value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") onAsk(question); }} placeholder="e.g. Which unresolved problems put renewals at risk?" className="min-w-0 flex-1 rounded-xl border border-[#dfe3ea] bg-white px-4 py-3 text-sm outline-none focus:border-[#172033]" /><button disabled={!question.trim() || loading} onClick={() => onAsk(question)} className="rounded-xl bg-[#172033] px-6 py-3 text-sm font-semibold text-white disabled:opacity-40">{loading ? "Thinking…" : "Ask"}</button></div>
        <div className="mt-5 text-[10px] font-semibold uppercase tracking-[0.15em] text-[#9aa1ad]">Recommended questions</div>
        <div className="mt-3 grid gap-2 md:grid-cols-2">
          {quickQuestions.map((q) => <button key={q} onClick={() => setQuestion(q)} title={q} className="rounded-xl border border-[#e1e5ea] bg-white px-4 py-3 text-left text-sm font-medium leading-6 text-[#596477] break-words whitespace-normal hover:border-[#cbd1da] hover:bg-[#f7f8fa]">{q}</button>)}
        </div>
        <p className="mt-4 text-[11px] text-[#98a0ad]">Selecting a recommendation only fills the question. The answer is generated when you press Ask.</p>
      </section>

      {answer && <section className="mt-6 rounded-2xl border border-[#dfe4eb] bg-white p-6">
        <div className="flex items-center justify-between gap-4"><div className="flex items-center gap-3"><div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#172033] text-xs font-bold text-white">C</div><div><div className="text-sm font-semibold">Answer from customer data</div><div className="text-[11px] text-[#89919f]">{answer.confidence || "medium"} confidence · {answer.source_ids?.length || 0} cited sources</div></div></div><span className="rounded-full bg-[#eef8f1] px-3 py-1 text-[10px] font-semibold text-emerald-700">Data grounded</span></div>
        <div className="mt-6 whitespace-pre-line text-[15px] leading-8 text-[#394355]">{answer.answer}</div>
        {answer.key_findings?.length ? <div className="mt-6 border-t border-[#e7e9ee] pt-5"><div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#89919f]">Key findings</div><div className="mt-3 space-y-3">{answer.key_findings.map((f, i) => <div key={i} className="rounded-xl bg-[#f7f8fa] px-4 py-3 text-sm leading-7 text-[#596477]">{f}</div>)}</div></div> : null}
        {answer.recommendation ? <div className="mt-5 rounded-xl bg-[#172033] p-5 text-sm leading-7 text-white"><div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[#aeb8c8]">Recommended action</div>{answer.recommendation}</div> : null}
        {conversations.length ? <div className="mt-6 border-t border-[#e7e9ee] pt-5"><div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#89919f]">Supporting conversations</div><div className="space-y-3">{conversations.map((c) => <ConversationCard key={c.id} conversation={c} />)}</div></div> : null}
      </section>}
    </div>
  );
}

function ConversationsPage({ conversations }: { conversations: Conversation[] }) {
  return <div className="mx-auto max-w-[1100px]"><div className="mb-8"><div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#89919f]">Source evidence</div><h2 className="text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">Conversations behind the insight.</h2><p className="mt-3 max-w-3xl text-[15px] leading-7 text-[#697386]">Original customer source text is preserved exactly as supplied. Use the filters above to focus on the conversations you want to review.</p></div><section className="rounded-2xl border border-[#e5e8ed] bg-white p-6"><div className="mb-5 text-xs font-semibold text-[#596477]">{conversations.length} source conversations shown</div><div className="grid gap-3 md:grid-cols-2">{conversations.map((c) => <ConversationCard key={c.id} conversation={c} />)}</div></section></div>;
}

function InsightsPage({ analysis, features, churn, highValue, showAllFeatures, showAllChurn, setShowAllFeatures, setShowAllChurn }: { analysis: Analysis; features: FeatureRequest[]; churn: ChurnSignal[]; highValue: HighValue[]; showAllFeatures: boolean; showAllChurn: boolean; setShowAllFeatures: (v: boolean) => void; setShowAllChurn: (v: boolean) => void }) {
  const visibleFeatures = showAllFeatures ? features : features.slice(0, 2);
  const visibleChurn = showAllChurn ? churn : churn.slice(0, 2);
  return <div className="mx-auto max-w-[1100px]"><div className="mb-8"><div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#89919f]">Customer intelligence</div><h2 className="text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">Signals worth acting on.</h2><p className="mt-3 max-w-3xl text-[15px] leading-7 text-[#697386]">The patterns below connect individual customer conversations to broader product demand, unresolved friction and retention risk.</p></div>
    <section className="mb-6 rounded-2xl border border-[#e5e8ed] bg-white p-6"><SectionHeader title="Feature requests" description="Includes direct requests and subtle capability demand, such as asking whether an integration, action or workflow is supported." /><div className="mt-6 grid gap-3 md:grid-cols-2">{visibleFeatures.map((f, i) => <div key={f.id || i} className="rounded-xl border border-[#edf0f3] p-5"><div className="flex items-start justify-between gap-3"><div><div className="text-sm font-semibold">{f.customer}</div><div className="mt-1 text-[11px] text-[#9299a5]">{f.channel || "source"}{f.date ? ` · ${f.date}` : ""}</div></div><span className="rounded-full bg-[#f4f5f7] px-2 py-1 text-[10px] font-semibold text-[#7a8390]">Feature request</span></div><div className="mt-4 text-sm font-medium leading-6 text-[#394355]">{f.message}</div></div>)}</div>{!visibleFeatures.length && <EmptyFilterState />} {features.length > 2 && <button onClick={() => setShowAllFeatures(!showAllFeatures)} className="mt-5 text-sm font-semibold underline underline-offset-4">{showAllFeatures ? "Show less" : "See all feature requests"}</button>}</section>
    <section className="mb-6 rounded-2xl border border-[#e5e8ed] bg-white p-6"><SectionHeader title="Silent churn & retention risk" description="Risk can appear through recent unresolved problems, repeated friction, blocked workflows, renewal pressure, alternatives or declining confidence — not just an explicit cancellation statement." /><div className="mt-6 space-y-3">{visibleChurn.map((c, i) => <div key={c.id || i} className="rounded-xl border border-[#f0e1e1] bg-[#fffafa] p-5"><div className="flex items-start gap-4"><div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#f8eaea] text-sm">!</div><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><span className="text-sm font-semibold">{c.customer}</span>{c.plan ? <span className="rounded-full bg-white px-2 py-1 text-[10px] font-semibold text-[#8b929e]">{c.plan}</span> : null}{c.severity ? <span className="rounded-full bg-[#f8eaea] px-2 py-1 text-[10px] font-semibold text-[#a34b4b]">{c.severity}</span> : null}</div><p className="mt-3 text-sm leading-7 text-[#596477]">{c.message}</p></div></div></div>)}</div>{!visibleChurn.length && <EmptyFilterState />} {churn.length > 2 && <button onClick={() => setShowAllChurn(!showAllChurn)} className="mt-5 text-sm font-semibold underline underline-offset-4">{showAllChurn ? "Show less" : "See all churn"}</button>}</section>
    <section className="rounded-2xl border border-[#e5e8ed] bg-white p-6"><SectionHeader title="High-value customer feedback" description="Customer evidence is shown directly, so business impact can be considered alongside what was actually said." /><div className="mt-6 grid gap-3 md:grid-cols-2">{highValue.slice(0, 8).map((h, i) => <div key={h.id || i} className="rounded-xl border border-[#edf0f3] p-5"><div className="flex items-center justify-between"><div className="text-sm font-semibold">{h.customer}</div><span className="rounded-full bg-[#f4f5f7] px-2 py-1 text-[10px] font-semibold">{money(h.revenue)}</span></div><p className="mt-4 text-sm leading-7 text-[#596477]">{h.message}</p></div>)}</div></section>
  </div>;
}

function InsightsFilter({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <section className="mb-6 rounded-2xl border border-[#e5e8ed] bg-white p-4 sm:p-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="text-sm font-semibold">Filter insights</div>
          <div className="mt-1 text-xs text-[#89919f]">Choose the signal you want to review.</div>
        </div>
        <label className="flex items-center gap-2 rounded-lg border border-[#dfe3ea] bg-white px-3 py-2 text-xs text-[#596477]">
          <span className="font-semibold text-[#89919f]">Signal</span>
          <select value={value} onChange={(e) => onChange(e.target.value)} className="bg-transparent font-semibold text-[#394355] outline-none">
            <option value="all">All insights</option>
            <option value="feature">Feature requests</option>
            <option value="churn">Churn risk</option>
          </select>
        </label>
      </div>
    </section>
  );
}

function EmptyFilterState() { return <div className="mt-6 rounded-xl bg-[#f7f8fa] px-4 py-4 text-sm text-[#7b8495]">No records match the current filters.</div>; }

function FilterBar({ sentiment, category, channel, onSentiment, onCategory, onChannel }: { sentiment: string; category: string; channel: string; onSentiment: (v: string) => void; onCategory: (v: string) => void; onChannel: (v: string) => void }) {
  return <section className="mb-6 rounded-2xl border border-[#e5e8ed] bg-white p-4 sm:p-5"><div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between"><div><div className="text-sm font-semibold">Filter customer evidence</div><div className="mt-1 text-xs text-[#89919f]">Use the same filters across Ask your data, Conversations and Insights.</div></div><div className="flex flex-wrap gap-2"><FilterSelect label="Sentiment" value={sentiment} onChange={onSentiment} options={[["all","All sentiments"],["positive","Positive"],["neutral","Neutral"],["negative","Negative"]]} /><FilterSelect label="Category" value={category} onChange={onCategory} options={[["all","All categories"],["churn","Churn signal"],["feature","Feature request"],["unresolved","Unresolved issue"],["general","General feedback"]]} /><FilterSelect label="Channel" value={channel} onChange={onChannel} options={[["all","All channels"],["email","Email"],["chat","Chat"],["call","Call"]]} /></div></div></section>;
}

function FilterSelect({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: string[][] }) { return <label className="flex items-center gap-2 rounded-lg border border-[#dfe3ea] bg-white px-3 py-2 text-xs text-[#596477]"><span className="font-semibold text-[#89919f]">{label}</span><select value={value} onChange={(e) => onChange(e.target.value)} className="bg-transparent font-semibold text-[#394355] outline-none">{options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>; }

function ConversationCard({ conversation }: { conversation: Conversation }) {
  return <div className="rounded-xl border border-[#edf0f3] bg-white p-5"><div className="flex items-start justify-between gap-4"><div className="flex min-w-0 items-center gap-3"><div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#f1f3f6] text-xs">{iconForChannel(conversation.channel)}</div><div className="min-w-0"><div className="truncate text-sm font-semibold">{customerName(conversation)}</div><div className="mt-0.5 text-[11px] capitalize text-[#9299a5]">{conversation.channel || "customer feedback"}{conversation.date ? ` · ${conversation.date}` : ""}</div></div></div><div className="flex flex-wrap justify-end gap-1.5">{conversation.sentiment ? <span className="rounded-full bg-[#f4f5f7] px-2 py-1 text-[10px] font-semibold capitalize text-[#697386]">{conversation.sentiment}</span> : null}{conversation.churn_signal ? <span className="rounded-full bg-[#fff1f1] px-2 py-1 text-[10px] font-semibold text-[#a34b4b]">Churn signal</span> : null}{conversation.feature_request ? <span className="rounded-full bg-[#f4f5f7] px-2 py-1 text-[10px] font-semibold text-[#697386]">Feature request</span> : null}{conversation.unresolved_issue ? <span className="rounded-full bg-[#fff7ed] px-2 py-1 text-[10px] font-semibold text-[#9a6a2f]">Unresolved</span> : null}</div></div><div className="mt-4 whitespace-pre-line text-sm leading-7 text-[#596477]">{conversation.message || conversation.original_message || "No source text."}</div></div>;
}

function SectionHeader({ eyebrow, title, description }: { eyebrow?: string; title: string; description: string }) { return <div>{eyebrow ? <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-[#9aa1ad]">{eyebrow}</div> : null}<h3 className="text-xl font-semibold tracking-tight">{title}</h3><p className="mt-1 text-sm leading-6 text-[#7b8495]">{description}</p></div>; }

function Metric({ label, value, description }: { label: string; value: number; description: string }) { return <div className="rounded-xl border border-[#e5e8ed] bg-white p-5"><div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#89919f]">{label}</div><div className="mt-2 text-2xl font-semibold">{value.toLocaleString()}</div><div className="mt-1 text-xs text-[#9299a5]">{description}</div></div>; }

function SentimentCard({ label, value }: { label: string; value: number }) { return <div className="rounded-xl border border-[#edf0f3] p-5"><div className="text-xs font-semibold text-[#89919f]">{label}</div><div className="mt-2 text-2xl font-semibold">{value.toLocaleString()}</div></div>; }