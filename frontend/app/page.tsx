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

function shortQuestion(value: string) {
  const cleaned = value.replace(/\s+/g, " ").trim();
  return cleaned.length > 90 ? `${cleaned.slice(0, 87).trimEnd()}…` : cleaned;
}

export default function Home() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [conversationResults, setConversationResults] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [error, setError] = useState("");
  const [activeSection, setActiveSection] = useState("Overview");
  const [showAllFeatures, setShowAllFeatures] = useState(false);
  const [showAllChurn, setShowAllChurn] = useState(false);

  const problems = analysis?.problems || [];
  const features = analysis?.feature_requests || [];
  const churn = analysis?.churn_signals || [];
  const highValue = analysis?.high_value || [];
  const segments = analysis?.segments || [];
  const sentiment = analysis?.sentiment || {};
  const total = analysis?.total || 0;

  const quickQuestions = useMemo(() => {
    return (analysis?.suggested_questions || []).slice(0, 5);
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
            "AI is analyzing the customer evidence…"
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
    setError("");
    setAnswer(null);
    setConversationResults([]);
    setAnalysis(null);
    setLoadingMessage("Loading customer evidence…");
    try {
      const response = await fetch(`${API}${endpoint}`, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data?.detail || "Unable to load customer data.");
      setLoadingMessage(data.message || "AI analysis is starting…");
      await waitForAnalysis();
      setActiveSection("Overview");
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Could not connect to the backend.");
      setLoading(false);
    }
  }

  async function loadDemo() {
    await startAnalysis("/api/demo", { method: "POST" });
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
    setLoadingMessage("AI is retrieving evidence and writing an answer…");
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
      setConversationResults(data.sources || []);
      setActiveSection("Ask your data");
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Something went wrong while asking the AI.");
    } finally {
      setLoading(false);
      setLoadingMessage("");
    }
  }

  useEffect(() => {
    if (!analysis || conversationResults.length) return;
    fetch(`${API}/api/conversations`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => setConversationResults(Array.isArray(rows) ? rows.slice(0, 12) : []))
      .catch(() => undefined);
  }, [analysis, conversationResults.length]);

  return (
    <main className="min-h-screen bg-[#f7f8fa] text-[#172033]">
      <header className="sticky top-0 z-40 border-b border-[#e7e9ee] bg-white/95 backdrop-blur">
        <div className="mx-auto flex h-[72px] max-w-[1500px] items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#172033] text-sm font-bold text-white">AI</div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#7b8495]">Customer Intelligence</div>
              <h1 className="text-lg font-semibold tracking-tight">Customer Feedback AI</h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <label className="cursor-pointer rounded-lg border border-[#dfe3ea] bg-white px-4 py-2 text-sm font-semibold hover:bg-[#f5f6f8]">
              ↑ Upload CSV
              <input type="file" accept=".csv" className="hidden" onChange={uploadCSV} />
            </label>
            <button onClick={loadDemo} disabled={loading} className="rounded-lg bg-[#172033] px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
              {loading && !analysis ? "Analysing…" : "Load demo"}
            </button>
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
            <div className="mb-2 text-xs font-semibold">AI evidence workspace</div>
            <p className="text-xs leading-5 text-[#c8cfda]">AI interprets the supplied conversations; original source text stays separate as evidence.</p>
          </div>
        </aside>

        <section className="min-w-0 flex-1 px-5 py-8 sm:px-8">
          {error && <div className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm leading-6 text-red-700">{error}</div>}

          {!analysis && loading && (
            <div className="flex min-h-[calc(100vh-160px)] items-center justify-center">
              <div className="w-full max-w-xl rounded-2xl border border-[#e7e9ee] bg-white p-10 text-center shadow-sm">
                <div className="mx-auto mb-5 h-10 w-10 animate-spin rounded-full border-2 border-[#dfe3ea] border-t-[#172033]" />
                <div className="text-lg font-semibold">Analysing customer feedback…</div>
                <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-[#7b8495]">{loadingMessage || "AI is reading the supplied customer evidence."}</p>
                <div className="mt-6 rounded-lg bg-[#f6f7f9] px-4 py-3 text-left text-xs leading-5 text-[#667085]">No scripted business answer is being used. The dashboard waits for the real AI analysis to finish.</div>
              </div>
            </div>
          )}

          {!analysis && !loading && (
            <div className="flex min-h-[calc(100vh-160px)] items-center justify-center">
              <div className="w-full max-w-2xl rounded-2xl border border-dashed border-[#d9dde5] bg-white p-12 text-center">
                <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-[#172033] text-xl font-bold text-white">AI</div>
                <h2 className="text-2xl font-semibold">Load customer feedback</h2>
                <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-[#7b8495]">Upload your CSV or load the demo. The AI will analyze the evidence and generate the overview, insights, questions and answers.</p>
                <div className="mt-6 flex justify-center gap-3">
                  <button onClick={loadDemo} className="rounded-lg bg-[#172033] px-5 py-2.5 text-sm font-semibold text-white">Load demo data</button>
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
                <AskPage question={question} setQuestion={setQuestion} loading={loading} quickQuestions={quickQuestions} answer={answer} onAsk={askQuestion} />
              )}
              {activeSection === "Conversations" && (
                <ConversationsPage conversations={conversationResults} />
              )}
              {activeSection === "Insights" && (
                <InsightsPage analysis={analysis} features={features} churn={churn} highValue={highValue} showAllFeatures={showAllFeatures} showAllChurn={showAllChurn} setShowAllFeatures={setShowAllFeatures} setShowAllChurn={setShowAllChurn} />
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
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[#e1e5eb] bg-white px-3 py-1.5 text-xs font-semibold text-[#667085]"><span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />AI-generated overview</div>
        <h2 className="max-w-4xl text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">What customers are telling your team.</h2>
        <p className="mt-3 max-w-3xl text-[15px] leading-7 text-[#697386]">A decision-ready summary grounded in the supplied customer evidence, with source conversations available behind each signal.</p>
      </div>

      <section className="mb-6 rounded-2xl bg-[#172033] p-7 text-white sm:p-8">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#aeb8c8]">Strongest customer signal</div>
        <h3 className="mt-3 text-2xl font-semibold sm:text-3xl">{overview?.headline || "AI is identifying the strongest recurring customer signal."}</h3>
        <p className="mt-5 max-w-4xl whitespace-pre-line text-[15px] leading-8 text-[#d5dbe5]">{overview?.summary || "The AI did not return an overview summary."}</p>
        <div className="mt-6 border-t border-white/10 pt-5"><div className="text-[11px] font-semibold uppercase tracking-[0.15em] text-[#aeb8c8]">Recommended priority</div><p className="mt-2 text-sm leading-7 text-white">{overview?.priority || "No priority was returned."}</p></div>
      </section>

      <div className="mb-6 grid grid-cols-2 gap-4 xl:grid-cols-4">
        <Metric label="Conversations" value={total} description="Source records loaded" />
        <Metric label="Churn signals" value={analysis.churn_signals?.length || 0} description="AI-detected retention risk" />
        <Metric label="Feature requests" value={analysis.feature_requests?.length || 0} description="AI-detected capability demand" />
        <Metric label="Customers" value={analysis.unique_customers || 0} description="Distinct customer identities" />
      </div>

      <section className="mb-6 rounded-2xl border border-[#e5e8ed] bg-white p-6">
        <SectionHeader eyebrow="Recurring problems" title="What is creating the most friction?" description="AI groups semantically related problems and explains why they matter." />
        <div className="mt-6 space-y-4">
          {problems.slice(0, 6).map((p, i) => (
            <div key={`${p.name}-${i}`} className="rounded-xl border border-[#edf0f3] p-5">
              <div className="flex items-start justify-between gap-4"><div className="flex items-center gap-3"><span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#f1f3f6] text-xs font-bold">{i + 1}</span><div className="text-sm font-semibold">{p.name}</div></div><span className="text-xs font-semibold text-[#7b8495]">{p.mentions} evidence records</span></div>
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[#eef0f3]"><div className="h-full rounded-full bg-[#172033]" style={{ width: `${Math.min(100, Math.max(4, p.percentage || 0))}%` }} /></div>
              <p className="mt-4 text-sm leading-7 text-[#596477]">{p.summary || "AI did not provide a problem summary."}</p>
              {p.source_ids?.length ? <div className="mt-3 text-[11px] font-semibold uppercase tracking-[0.12em] text-[#98a0ad]">Sources: {p.source_ids.slice(0, 6).map((id) => `[${id}]`).join(" ")}</div> : null}
            </div>
          ))}
        </div>
      </section>

      <section className="mb-6 rounded-2xl border border-[#e5e8ed] bg-white p-6">
        <SectionHeader title="Sentiment" description="AI classification across the evidence set used for the analysis." />
        <div className="mt-5 grid gap-4 md:grid-cols-3"><SentimentCard label="Positive" value={sentiment.positive || 0} /><SentimentCard label="Negative" value={sentiment.negative || 0} /><SentimentCard label="Neutral" value={sentiment.neutral || 0} /></div>
        <div className="mt-5 rounded-xl bg-[#f7f8fa] px-4 py-3 text-xs leading-5 text-[#7b8495]">AI evidence coverage: {analysis.ai_analyzed_evidence_rows || 0} representative source records were semantically analyzed from {total.toLocaleString()} loaded records. Ask Your Data retrieves from the full loaded source set.</div>
      </section>
    </div>
  );
}

function AskPage({ question, setQuestion, loading, quickQuestions, answer, onAsk }: { question: string; setQuestion: (v: string) => void; loading: boolean; quickQuestions: string[]; answer: AskResponse | null; onAsk: (q?: string) => void }) {
  return (
    <div className="mx-auto max-w-[1100px]">
      <div className="mb-8"><div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#89919f]">AI search</div><h2 className="text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">Ask your customer data.</h2><p className="mt-3 max-w-3xl text-[15px] leading-7 text-[#697386]">Ask in plain language. The AI retrieves relevant source conversations and writes a new answer for your exact question.</p></div>
      <section className="rounded-2xl border border-[#e5e8ed] bg-white p-6">
        <div className="flex flex-col gap-3 sm:flex-row"><input value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") onAsk(question); }} placeholder="e.g. Which unresolved problems put renewals at risk?" className="min-w-0 flex-1 rounded-xl border border-[#dfe3ea] bg-white px-4 py-3 text-sm outline-none focus:border-[#172033]" /><button disabled={!question.trim() || loading} onClick={() => onAsk(question)} className="rounded-xl bg-[#172033] px-6 py-3 text-sm font-semibold text-white disabled:opacity-40">{loading ? "Thinking…" : "Ask"}</button></div>
        <div className="mt-5 text-[10px] font-semibold uppercase tracking-[0.15em] text-[#9aa1ad]">Recommended questions</div>
        <div className="mt-3 grid gap-2 md:grid-cols-2">
          {quickQuestions.map((q) => <button key={q} onClick={() => setQuestion(q)} title={q} className="rounded-xl border border-[#e1e5ea] bg-white px-4 py-3 text-left text-xs font-medium leading-5 text-[#596477] hover:border-[#cbd1da] hover:bg-[#f7f8fa]"><span className="line-clamp-2">{shortQuestion(q)}</span></button>)}
        </div>
        <p className="mt-4 text-[11px] text-[#98a0ad]">Selecting a recommendation only fills the question. The AI runs when you press Ask.</p>
      </section>

      {answer && <section className="mt-6 rounded-2xl border border-[#dfe4eb] bg-white p-6">
        <div className="flex items-center justify-between gap-4"><div className="flex items-center gap-3"><div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#172033] text-xs font-bold text-white">AI</div><div><div className="text-sm font-semibold">AI-generated answer</div><div className="text-[11px] text-[#89919f]">{answer.confidence || "medium"} confidence · {answer.source_ids?.length || 0} cited sources</div></div></div><span className="rounded-full bg-[#eef8f1] px-3 py-1 text-[10px] font-semibold text-emerald-700">AI · evidence grounded</span></div>
        <div className="mt-6 whitespace-pre-line text-[15px] leading-8 text-[#394355]">{answer.answer}</div>
        {answer.key_findings?.length ? <div className="mt-6 border-t border-[#e7e9ee] pt-5"><div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#89919f]">Key findings</div><div className="mt-3 space-y-3">{answer.key_findings.map((f, i) => <div key={i} className="rounded-xl bg-[#f7f8fa] px-4 py-3 text-sm leading-7 text-[#596477]">{f}</div>)}</div></div> : null}
        {answer.recommendation ? <div className="mt-5 rounded-xl bg-[#172033] p-5 text-sm leading-7 text-white"><div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[#aeb8c8]">Recommended action</div>{answer.recommendation}</div> : null}
        {answer.sources?.length ? <div className="mt-6 border-t border-[#e7e9ee] pt-5"><div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#89919f]">Supporting source conversations</div><div className="space-y-3">{answer.sources.map((c) => <ConversationCard key={c.id} conversation={c} />)}</div></div> : null}
      </section>}
    </div>
  );
}

function ConversationsPage({ conversations }: { conversations: Conversation[] }) {
  return <div className="mx-auto max-w-[1100px]"><div className="mb-8"><div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#89919f]">Source evidence</div><h2 className="text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">Conversations behind the insight.</h2><p className="mt-3 max-w-3xl text-[15px] leading-7 text-[#697386]">Original source text is preserved. Longer customer threads are assembled only from records that actually belong to the same customer; AI interpretation is shown separately and is never presented as an original customer quote.</p></div><section className="rounded-2xl border border-[#e5e8ed] bg-white p-6"><div className="mb-5 text-xs font-semibold text-[#596477]">{conversations.length} source conversations shown</div><div className="grid gap-3 md:grid-cols-2">{conversations.map((c) => <ConversationCard key={c.id} conversation={c} />)}</div></section></div>;
}

function InsightsPage({ analysis, features, churn, highValue, showAllFeatures, showAllChurn, setShowAllFeatures, setShowAllChurn }: { analysis: Analysis; features: FeatureRequest[]; churn: ChurnSignal[]; highValue: HighValue[]; showAllFeatures: boolean; showAllChurn: boolean; setShowAllFeatures: (v: boolean) => void; setShowAllChurn: (v: boolean) => void }) {
  const visibleFeatures = showAllFeatures ? features : features.slice(0, 8);
  const visibleChurn = showAllChurn ? churn : churn.slice(0, 8);
  return <div className="mx-auto max-w-[1100px]"><div className="mb-8"><div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#89919f]">AI insights</div><h2 className="text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">Signals worth acting on.</h2><p className="mt-3 max-w-3xl text-[15px] leading-7 text-[#697386]">These are semantic AI interpretations of customer evidence — including subtle capability demand and silent retention risk.</p></div>
    <section className="mb-6 rounded-2xl border border-[#e5e8ed] bg-white p-6"><SectionHeader title="Feature requests" description="Explicit requests plus subtle capability demand such as unsupported integrations, APIs, exports and automation." /><div className="mt-6 grid gap-3 md:grid-cols-2">{visibleFeatures.map((f, i) => <div key={f.id || i} className="rounded-xl border border-[#edf0f3] p-5"><div className="flex items-start justify-between gap-3"><div><div className="text-sm font-semibold">{f.customer}</div><div className="mt-1 text-[11px] text-[#9299a5]">{f.channel || "source"}{f.date ? ` · ${f.date}` : ""}</div></div><span className="rounded-full bg-[#f4f5f7] px-2 py-1 text-[10px] font-semibold text-[#7a8390]">Feature</span></div><div className="mt-4 text-sm font-medium leading-6 text-[#394355]">{f.message}</div>{f.summary ? <p className="mt-3 text-sm leading-7 text-[#596477]">{f.summary}</p> : null}{f.reason ? <div className="mt-3 rounded-lg bg-[#f7f8fa] px-3 py-2 text-xs leading-5 text-[#697386]"><b>Why AI classified it:</b> {f.reason}</div> : null}</div>)}</div>{features.length > 8 && <button onClick={() => setShowAllFeatures(!showAllFeatures)} className="mt-5 text-sm font-semibold underline underline-offset-4">{showAllFeatures ? "Show less" : `Show all ${features.length} requests`}</button>}</section>
    <section className="mb-6 rounded-2xl border border-[#e5e8ed] bg-white p-6"><SectionHeader title="Silent churn & retention risk" description="AI looks beyond explicit cancellation language: unresolved issues, repeated friction, blocked workflows, alternatives, renewal pressure and declining confidence." /><div className="mt-6 space-y-3">{visibleChurn.map((c, i) => <div key={c.id || i} className="rounded-xl border border-[#f0e1e1] bg-[#fffafa] p-5"><div className="flex items-start gap-4"><div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#f8eaea] text-sm">!</div><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><span className="text-sm font-semibold">{c.customer}</span>{c.plan ? <span className="rounded-full bg-white px-2 py-1 text-[10px] font-semibold text-[#8b929e]">{c.plan}</span> : null}{c.severity ? <span className="rounded-full bg-[#f8eaea] px-2 py-1 text-[10px] font-semibold text-[#a34b4b]">{c.severity}</span> : null}</div><p className="mt-3 text-sm leading-7 text-[#596477]">{c.message}</p>{c.evidence ? <div className="mt-4 rounded-lg border border-[#f1dddd] bg-white px-4 py-3 text-xs leading-6 text-[#697386]"><div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#9a7b7b]">Original evidence</div>{c.evidence}</div> : null}{c.reason ? <div className="mt-3 text-xs leading-6 text-[#697386]"><b>Why this matters:</b> {c.reason}</div> : null}</div></div></div>)}</div>{churn.length > 8 && <button onClick={() => setShowAllChurn(!showAllChurn)} className="mt-5 text-sm font-semibold underline underline-offset-4">{showAllChurn ? "Show less" : `Show all ${churn.length} signals`}</button>}</section>
    <section className="rounded-2xl border border-[#e5e8ed] bg-white p-6"><SectionHeader title="High-value customer feedback" description="AI summaries are shown separately from the original evidence." /><div className="mt-6 grid gap-3 md:grid-cols-2">{highValue.slice(0, 8).map((h, i) => <div key={h.id || i} className="rounded-xl border border-[#edf0f3] p-5"><div className="flex items-center justify-between"><div className="text-sm font-semibold">{h.customer}</div><span className="rounded-full bg-[#f4f5f7] px-2 py-1 text-[10px] font-semibold">{money(h.revenue)}</span></div><p className="mt-4 text-sm leading-7 text-[#596477]">{h.message}</p>{h.summary ? <p className="mt-3 text-sm leading-7 text-[#596477]">{h.summary}</p> : null}{h.evidence ? <div className="mt-3 rounded-lg bg-[#f7f8fa] px-3 py-2 text-xs leading-5 text-[#697386]">Original evidence: {h.evidence}</div> : null}</div>)}</div></section>
  </div>;
}

function ConversationCard({ conversation }: { conversation: Conversation }) {
  return <div className="rounded-xl border border-[#edf0f3] bg-white p-5"><div className="flex items-start justify-between gap-4"><div className="flex min-w-0 items-center gap-3"><div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#f1f3f6] text-xs">{iconForChannel(conversation.channel)}</div><div className="min-w-0"><div className="truncate text-sm font-semibold">{customerName(conversation)}</div><div className="mt-0.5 text-[11px] capitalize text-[#9299a5]">{conversation.channel || "customer feedback"}{conversation.date ? ` · ${conversation.date}` : ""}</div></div></div></div><div className="mt-4 whitespace-pre-line text-sm leading-7 text-[#596477]">{conversation.message || conversation.original_message || "No source text."}</div>{conversation.feedback_summary ? <div className="mt-4 rounded-xl bg-[#f7f8fa] p-4"><div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#89919f]">AI interpretation</div><p className="text-sm leading-7 text-[#596477]">{conversation.feedback_summary}</p></div> : null}</div>;
}

function SectionHeader({ eyebrow, title, description }: { eyebrow?: string; title: string; description: string }) { return <div>{eyebrow ? <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-[#9aa1ad]">{eyebrow}</div> : null}<h3 className="text-xl font-semibold tracking-tight">{title}</h3><p className="mt-1 text-sm leading-6 text-[#7b8495]">{description}</p></div>; }

function Metric({ label, value, description }: { label: string; value: number; description: string }) { return <div className="rounded-xl border border-[#e5e8ed] bg-white p-5"><div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[#89919f]">{label}</div><div className="mt-2 text-2xl font-semibold">{value.toLocaleString()}</div><div className="mt-1 text-xs text-[#9299a5]">{description}</div></div>; }

function SentimentCard({ label, value }: { label: string; value: number }) { return <div className="rounded-xl border border-[#edf0f3] p-5"><div className="text-xs font-semibold text-[#89919f]">{label}</div><div className="mt-2 text-2xl font-semibold">{value.toLocaleString()}</div></div>; }