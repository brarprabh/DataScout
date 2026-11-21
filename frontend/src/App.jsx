import React, { useState, useEffect } from 'react';
import {
  FileText,
  Upload,
  Search,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  Layers,
  RefreshCw,
  BarChart3,
  Terminal,
  Clock,
  Zap,
  Globe
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell
} from 'recharts';

const BACKEND_URL = 'http://127.0.0.1:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('workbench'); // 'workbench' | 'evaluation'
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  // Search & Query state
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [queryResult, setQueryResult] = useState(null);

  // Analytics & Benchmark state
  const [analytics, setAnalytics] = useState(null);
  const [loadingAnalytics, setLoadingAnalytics] = useState(false);

  useEffect(() => {
    fetchDocuments();
    fetchAnalytics();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (err) {
      console.error('Error fetching documents:', err);
    }
  };

  const fetchAnalytics = async () => {
    setLoadingAnalytics(true);
    try {
      const res = await fetch(`${BACKEND_URL}/analytics`);
      if (res.ok) {
        const data = await res.json();
        setAnalytics(data);
      }
    } catch (err) {
      console.error('Error fetching analytics:', err);
    } finally {
      setLoadingAnalytics(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadStatus({ type: 'error', message: 'Only PDF files are supported' });
      return;
    }

    setUploading(true);
    setUploadStatus(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${BACKEND_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setUploadStatus({
          type: 'success',
          message: `Indexed ${data.chunks_indexed} chunks from ${data.filename}`,
        });
        fetchDocuments();
        fetchAnalytics();
      } else {
        const err = await res.json();
        setUploadStatus({ type: 'error', message: err.detail || 'Upload failed' });
      }
    } catch (err) {
      setUploadStatus({ type: 'error', message: 'Failed to connect to backend server' });
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), top_k: 5 }),
      });

      if (res.ok) {
        const data = await res.json();
        setQueryResult(data);
        fetchAnalytics();
      }
    } catch (err) {
      console.error('Query execution error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F6F5F0] text-[#111111]">
      {/* Top Header Grid */}
      <header className="border-b border-[#D4D2C8] bg-[#F6F5F0] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-5 h-5 bg-[#FF4F00] flex items-center justify-center text-white text-xs font-bold font-mono">
            D
          </div>
          <span className="font-mono text-sm tracking-widest font-bold uppercase">
            DATASCOUT // AUDITABLE HYBRID RAG
          </span>
          <span className="text-xs bg-[#E5E3D8] border border-[#D4D2C8] px-2 py-0.5 font-mono text-[#555555]">
            RRF-K60
          </span>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center border border-[#D4D2C8] bg-[#EBE9DE]">
          <button
            onClick={() => setActiveTab('workbench')}
            className={`px-4 py-1.5 text-xs font-mono font-medium flex items-center space-x-2 transition-colors ${
              activeTab === 'workbench'
                ? 'bg-[#111111] text-[#F6F5F0]'
                : 'text-[#555555] hover:text-[#111111]'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>RETRIEVAL WORKBENCH</span>
          </button>
          <button
            onClick={() => {
              setActiveTab('evaluation');
              fetchAnalytics();
            }}
            className={`px-4 py-1.5 text-xs font-mono font-medium flex items-center space-x-2 transition-colors ${
              activeTab === 'evaluation'
                ? 'bg-[#111111] text-[#F6F5F0]'
                : 'text-[#555555] hover:text-[#111111]'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>EVALUATION BENCHMARK</span>
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      {activeTab === 'workbench' ? (
        <main className="flex-1 grid grid-cols-12 min-h-0 divide-x divide-[#D4D2C8]">
          {/* LEFT PANE: Document Ingestion & KB Status */}
          <aside className="col-span-3 p-5 flex flex-col space-y-6">
            <div className="space-y-1">
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-[#555555]">
                01 // KNOWLEDGE BASE INGESTION
              </h2>
              <p className="text-xs text-[#777777]">
                PyMuPDF 500-char sliding window with 50-char overlap.
              </p>
            </div>

            {/* Drag & Drop Upload Card */}
            <div className="border border-dashed border-[#B5B3A6] bg-[#EFECE1] p-5 text-center relative hover:bg-[#EAE6D8] transition-colors">
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                disabled={uploading}
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
              <Upload className="w-6 h-6 mx-auto mb-2 text-[#777777]" />
              <p className="text-xs font-mono font-semibold uppercase">
                {uploading ? 'INGESTING & EMBEDDING...' : 'DROP PDF TO INDEX'}
              </p>
              <p className="text-[11px] text-[#777777] mt-1">
                Generates Gemini embeddings & BM25 tokens
              </p>
            </div>

            {uploadStatus && (
              <div
                className={`p-3 border text-xs font-mono flex items-start space-x-2 ${
                  uploadStatus.type === 'success'
                    ? 'bg-[#EBF7EE] border-[#72C185] text-[#1B642C]'
                    : 'bg-[#FBEBEB] border-[#E88E8E] text-[#9A2222]'
                }`}
              >
                {uploadStatus.type === 'success' ? (
                  <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                ) : (
                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                )}
                <span>{uploadStatus.message}</span>
              </div>
            )}

            {/* Indexed Document List */}
            <div className="flex-1 flex flex-col space-y-2 overflow-hidden">
              <div className="flex items-center justify-between border-b border-[#D4D2C8] pb-1">
                <span className="text-[11px] font-mono uppercase text-[#777777]">
                  INDEXED SOURCES ({documents.length})
                </span>
                <button
                  onClick={fetchDocuments}
                  className="p-1 hover:text-[#FF4F00] text-[#777777]"
                >
                  <RefreshCw className="w-3 h-3" />
                </button>
              </div>

              <div className="space-y-2 overflow-y-auto pr-1">
                {documents.length === 0 ? (
                  <div className="text-xs text-[#888888] font-mono p-4 border border-[#D4D2C8] bg-[#EAE8DE] text-center">
                    No documents indexed yet. Upload a PDF or run benchmarks.
                  </div>
                ) : (
                  documents.map((doc, idx) => (
                    <div
                      key={idx}
                      className="border border-[#D4D2C8] bg-[#FAF9F5] p-3 flex flex-col space-y-1"
                    >
                      <div className="flex items-center space-x-2">
                        <FileText className="w-3.5 h-3.5 text-[#FF4F00] shrink-0" />
                        <span className="text-xs font-mono font-medium truncate">
                          {doc.filename}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[11px] font-mono text-[#666666]">
                        <span>CHUNKS:</span>
                        <span className="bg-[#EAE7DC] px-1.5 py-0.2 border border-[#D4D2C8] text-[#111111]">
                          {doc.chunk_count}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </aside>

          {/* CENTER PANE: Query Input & Grounded Response */}
          <section className="col-span-5 p-5 flex flex-col space-y-5">
            <div className="space-y-1">
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-[#555555]">
                02 // RETRIEVAL & ANSWER GENERATION
              </h2>
              <p className="text-xs text-[#777777]">
                Self-correcting LangGraph loop with hybrid RRF and Tavily web fallback.
              </p>
            </div>

            {/* Query Form */}
            <form onSubmit={handleQuery} className="flex border border-[#111111]">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about system architecture, ERR codes, or web fallback..."
                className="flex-1 bg-[#FAF9F5] px-3 py-2 text-xs font-mono outline-none text-[#111111] placeholder-[#888888]"
              />
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="bg-[#111111] text-[#F6F5F0] px-4 py-2 text-xs font-mono font-bold uppercase hover:bg-[#FF4F00] transition-colors flex items-center space-x-1 disabled:opacity-50"
              >
                <Search className="w-3.5 h-3.5" />
                <span>{loading ? 'RETRIEVING...' : 'EXECUTE'}</span>
              </button>
            </form>

            {/* Telemetry Bar */}
            {queryResult && (
              <div className="border border-[#D4D2C8] bg-[#FAF9F5] p-3 flex items-center justify-between font-mono text-xs">
                <div className="flex items-center space-x-2">
                  <span className="text-[#666666]">SOURCE:</span>
                  <span
                    className={`px-2 py-0.5 text-[11px] font-bold border ${
                      queryResult.source_mode === 'kb'
                        ? 'bg-[#EBF7EE] border-[#72C185] text-[#1B642C]'
                        : queryResult.source_mode === 'web_fallback'
                        ? 'bg-[#FFF3E8] border-[#FFB67D] text-[#D84300]'
                        : 'bg-[#EFECE1] border-[#D4D2C8] text-[#555555]'
                    }`}
                  >
                    {queryResult.source_mode === 'kb'
                      ? 'INTERNAL KB'
                      : queryResult.source_mode === 'web_fallback'
                      ? 'TAVILY WEB FALLBACK'
                      : 'NO MATCH'}
                  </span>
                </div>

                <div className="flex items-center space-x-4 text-[#555555]">
                  <span className="flex items-center space-x-1">
                    <Clock className="w-3 h-3" />
                    <span>{queryResult.latency_ms} ms</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <Zap className="w-3 h-3 text-[#FF4F00]" />
                    <span>{queryResult.cache_hit ? 'CACHE HIT' : 'COLD RUN'}</span>
                  </span>
                </div>
              </div>
            )}

            {/* LangGraph Trace Steps */}
            {queryResult && queryResult.trace_steps && (
              <div className="border border-[#D4D2C8] bg-[#FAF9F5] p-3 space-y-2">
                <div className="flex items-center space-x-2 border-b border-[#D4D2C8] pb-1">
                  <Terminal className="w-3.5 h-3.5 text-[#555555]" />
                  <span className="text-[11px] font-mono font-bold uppercase text-[#555555]">
                    LANGGRAPH CORRECTIVE TRACE
                  </span>
                </div>
                <div className="space-y-1 font-mono text-[11px]">
                  {queryResult.trace_steps.map((step, idx) => (
                    <div key={idx} className="flex items-start space-x-2">
                      <span className="text-[#FF4F00] font-bold">[{idx + 1}]</span>
                      <span className="text-[#333333]">{step}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Grounded Answer Card */}
            <div className="flex-1 border border-[#D4D2C8] bg-[#FAF9F5] p-4 flex flex-col space-y-2 overflow-y-auto">
              <div className="flex items-center justify-between border-b border-[#D4D2C8] pb-1">
                <span className="text-[11px] font-mono uppercase text-[#777777]">
                  GROUNDED SYNTHESIS
                </span>
                <span className="text-[10px] font-mono text-[#888888]">
                  gemini-3.1-flash-lite
                </span>
              </div>
              {queryResult ? (
                <div className="text-xs leading-relaxed font-sans text-[#111111] whitespace-pre-wrap">
                  {queryResult.answer}
                </div>
              ) : (
                <div className="text-xs text-[#888888] font-mono text-center my-auto">
                  Execute a query to inspect the grounded answer and LangGraph execution trace.
                </div>
              )}
            </div>
          </section>

          {/* RIGHT PANE: X-Ray Inspector */}
          <aside className="col-span-4 p-5 flex flex-col space-y-4 overflow-y-auto">
            <div className="space-y-1">
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-[#555555]">
                03 // X-RAY INSPECTOR
              </h2>
              <p className="text-xs text-[#777777]">
                Per-chunk RRF scores, BM25 ranks, dense vector ranks, and relevance audit.
              </p>
            </div>

            {/* Web Sources Cards (if Tavily Fallback triggered) */}
            {queryResult && queryResult.web_sources && queryResult.web_sources.length > 0 && (
              <div className="space-y-2 border border-[#FF4F00] p-3 bg-[#FFFBF7]">
                <div className="flex items-center space-x-2 text-[#D84300] font-mono text-xs font-bold border-b border-[#FFE0CC] pb-1">
                  <Globe className="w-3.5 h-3.5" />
                  <span>TAVILY WEB SOURCES ({queryResult.web_sources.length})</span>
                </div>
                {queryResult.web_sources.map((ws, idx) => (
                  <div key={idx} className="border border-[#FFE0CC] p-2 bg-white space-y-1">
                    <a
                      href={ws.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xs font-mono font-bold text-[#D84300] hover:underline flex items-center space-x-1"
                    >
                      <span className="truncate">{ws.title}</span>
                      <ExternalLink className="w-3 h-3 shrink-0" />
                    </a>
                    <p className="text-[11px] text-[#555555] line-clamp-2">
                      {ws.snippet}
                    </p>
                  </div>
                ))}
              </div>
            )}

            {/* Retrieved Chunks Audit Cards */}
            <div className="space-y-3">
              <span className="text-[11px] font-mono uppercase text-[#777777] block border-b border-[#D4D2C8] pb-1">
                RETRIEVED CHUNKS ({queryResult?.chunks?.length || 0})
              </span>

              {!queryResult || queryResult.chunks.length === 0 ? (
                <div className="text-xs text-[#888888] font-mono p-4 border border-[#D4D2C8] bg-[#EAE8DE] text-center">
                  No retrieval audit data to display.
                </div>
              ) : (
                queryResult.chunks.map((c, idx) => (
                  <div
                    key={idx}
                    className="border border-[#D4D2C8] bg-[#FAF9F5] p-3 flex flex-col space-y-2"
                  >
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="font-bold text-[#111111] truncate">{c.chunk_id}</span>
                      <span
                        className={`text-[10px] px-1.5 py-0.5 border ${
                          c.is_relevant === true
                            ? 'bg-[#EBF7EE] border-[#72C185] text-[#1B642C]'
                            : c.is_relevant === false
                            ? 'bg-[#FBEBEB] border-[#E88E8E] text-[#9A2222]'
                            : 'bg-[#EAE7DC] border-[#D4D2C8] text-[#555555]'
                        }`}
                      >
                        {c.is_relevant === true
                          ? 'RELEVANT'
                          : c.is_relevant === false
                          ? 'IRRELEVANT'
                          : 'UNGRADED'}
                      </span>
                    </div>

                    {/* Metrics Row */}
                    <div className="grid grid-cols-3 gap-1 bg-[#EFECE1] border border-[#D4D2C8] p-1.5 text-center font-mono text-[11px]">
                      <div>
                        <div className="text-[9px] text-[#777777]">RRF SCORE</div>
                        <div className="font-bold text-[#FF4F00]">{c.rrf_score}</div>
                      </div>
                      <div>
                        <div className="text-[9px] text-[#777777]">BM25 RANK</div>
                        <div className="font-bold">
                          {c.bm25_rank !== null ? `#${c.bm25_rank}` : '—'}
                        </div>
                      </div>
                      <div>
                        <div className="text-[9px] text-[#777777]">VECTOR RANK</div>
                        <div className="font-bold">
                          {c.vector_rank !== null ? `#${c.vector_rank}` : '—'}
                        </div>
                      </div>
                    </div>

                    {/* Chunk Snippet */}
                    <p className="text-[11px] font-sans text-[#444444] bg-[#FAF9F5] border border-[#E5E3D8] p-2 leading-relaxed">
                      {c.text}
                    </p>

                    <div className="flex items-center justify-between text-[10px] font-mono text-[#888888]">
                      <span>SOURCE: {c.source}</span>
                      <span>PAGE: {c.page}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </aside>
        </main>
      ) : (
        /* EVALUATION BENCHMARK TAB */
        <main className="flex-1 p-8 max-w-6xl mx-auto w-full space-y-8">
          <div className="flex items-center justify-between border-b border-[#D4D2C8] pb-4">
            <div>
              <h1 className="font-mono text-base font-bold uppercase tracking-wider">
                RETRIEVAL BENCHMARKS & EVALUATION AUDIT
              </h1>
              <p className="text-xs text-[#666666] mt-0.5 font-mono">
                Real measured metrics comparing Vector-only vs BM25-only vs Hybrid RRF
              </p>
            </div>
            <button
              onClick={fetchAnalytics}
              disabled={loadingAnalytics}
              className="px-3 py-1.5 border border-[#111111] bg-[#FAF9F5] text-xs font-mono font-bold flex items-center space-x-1.5 hover:bg-[#FF4F00] hover:text-white transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingAnalytics ? 'animate-spin' : ''}`} />
              <span>REFRESH METRICS</span>
            </button>
          </div>

          {/* Live Metrics Summary Cards */}
          {analytics?.live_metrics && (
            <div className="grid grid-cols-4 gap-4 font-mono">
              <div className="border border-[#D4D2C8] bg-[#FAF9F5] p-4">
                <div className="text-[10px] uppercase text-[#777777]">TOTAL DOCUMENTS</div>
                <div className="text-2xl font-bold mt-1">
                  {analytics.live_metrics.total_documents}
                </div>
              </div>
              <div className="border border-[#D4D2C8] bg-[#FAF9F5] p-4">
                <div className="text-[10px] uppercase text-[#777777]">TOTAL CHUNKS</div>
                <div className="text-2xl font-bold text-[#FF4F00] mt-1">
                  {analytics.live_metrics.total_chunks}
                </div>
              </div>
              <div className="border border-[#D4D2C8] bg-[#FAF9F5] p-4">
                <div className="text-[10px] uppercase text-[#777777]">CACHE ENTRIES</div>
                <div className="text-2xl font-bold mt-1">
                  {analytics.live_metrics.cache_entries}
                </div>
              </div>
              <div className="border border-[#D4D2C8] bg-[#FAF9F5] p-4">
                <div className="text-[10px] uppercase text-[#777777]">CACHE HIT RATE</div>
                <div className="text-2xl font-bold text-[#1B642C] mt-1">
                  {(analytics.live_metrics.cache_hit_rate * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          )}

          {/* Comparison Table */}
          {analytics?.benchmarks?.metrics && (
            <div className="space-y-4">
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-[#555555]">
                12-QUESTION BENCHMARK RESULTS (K=5)
              </h2>

              <div className="border border-[#D4D2C8] bg-[#FAF9F5] overflow-hidden">
                <table className="w-full text-left font-mono text-xs divide-y divide-[#D4D2C8]">
                  <thead className="bg-[#EFECE1] text-[#555555]">
                    <tr>
                      <th className="py-2.5 px-4 font-bold">RETRIEVAL MODE</th>
                      <th className="py-2.5 px-4 font-bold">RECALL@5</th>
                      <th className="py-2.5 px-4 font-bold">PRECISION@5</th>
                      <th className="py-2.5 px-4 font-bold">MRR (MEAN RECIPROCAL RANK)</th>
                      <th className="py-2.5 px-4 font-bold">P95 LATENCY (MS)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#D4D2C8]">
                    {Object.entries(analytics.benchmarks.metrics).map(([mode, m]) => (
                      <tr
                        key={mode}
                        className={mode === 'hybrid_rrf' ? 'bg-[#FFF8F2] font-semibold' : ''}
                      >
                        <td className="py-3 px-4 flex items-center space-x-2">
                          <span className={mode === 'hybrid_rrf' ? 'text-[#FF4F00]' : ''}>
                            {mode.toUpperCase()}
                          </span>
                          {mode === 'hybrid_rrf' && (
                            <span className="text-[9px] bg-[#FF4F00] text-white px-1.5 py-0.2">
                              OPTIMAL
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-4">{m.recall_at_5.toFixed(4)}</td>
                        <td className="py-3 px-4">{m.precision_at_5.toFixed(4)}</td>
                        <td className="py-3 px-4 text-[#FF4F00] font-bold">
                          {m.mrr.toFixed(4)}
                        </td>
                        <td className="py-3 px-4">{m.p95_latency_ms} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Horizontal Bar Charts for MRR & Latency */}
              <div className="grid grid-cols-2 gap-6 pt-4">
                <div className="border border-[#D4D2C8] bg-[#FAF9F5] p-5">
                  <div className="text-xs font-mono font-bold uppercase text-[#555555] mb-4">
                    MRR (HIGHER IS BETTER)
                  </div>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        layout="vertical"
                        data={[
                          { name: 'Vector', mrr: analytics.benchmarks.metrics.vector_only?.mrr || 0 },
                          { name: 'BM25', mrr: analytics.benchmarks.metrics.bm25_only?.mrr || 0 },
                          { name: 'Hybrid RRF', mrr: analytics.benchmarks.metrics.hybrid_rrf?.mrr || 0 },
                        ]}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#D4D2C8" />
                        <XAxis type="number" domain={[0.8, 1.0]} stroke="#777777" tick={{ fontSize: 10 }} />
                        <YAxis dataKey="name" type="category" stroke="#111111" tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Bar dataKey="mrr" fill="#FF4F00">
                          <Cell fill="#888888" />
                          <Cell fill="#555555" />
                          <Cell fill="#FF4F00" />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="border border-[#D4D2C8] bg-[#FAF9F5] p-5">
                  <div className="text-xs font-mono font-bold uppercase text-[#555555] mb-4">
                    P95 LATENCY MS (LOWER IS BETTER)
                  </div>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        layout="vertical"
                        data={[
                          { name: 'Vector', latency: analytics.benchmarks.metrics.vector_only?.p95_latency_ms || 0 },
                          { name: 'BM25', latency: analytics.benchmarks.metrics.bm25_only?.p95_latency_ms || 0 },
                          { name: 'Hybrid RRF', latency: analytics.benchmarks.metrics.hybrid_rrf?.p95_latency_ms || 0 },
                        ]}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#D4D2C8" />
                        <XAxis type="number" stroke="#777777" tick={{ fontSize: 10 }} />
                        <YAxis dataKey="name" type="category" stroke="#111111" tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Bar dataKey="latency" fill="#111111">
                          <Cell fill="#888888" />
                          <Cell fill="#1B642C" />
                          <Cell fill="#FF4F00" />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      )}
    </div>
  );
}
