const { useState, useEffect } = React;
const API = "http://localhost:8000/api/v1";

function useApi(endpoint) {
  const [data, setData] = useState(null);
  useEffect(() => {
    fetch(`${API}${endpoint}`).then(r => r.json()).then(setData).catch(() => {});
  }, [endpoint]);
  return data;
}

const getRiskColor = l => ({ CRITICAL:"#dc2626",HIGH:"#ea580c",MEDIUM:"#d97706",LOW:"#16a34a" }[l]||"#6b7785");
const getRiskPill = l => ({ CRITICAL:"pill-red",HIGH:"pill-red",MEDIUM:"pill-orange",LOW:"pill-green" }[l]||"pill-gray");
const getRiskFill = l => ({ CRITICAL:"linear-gradient(90deg,#fca5a5,#ef4444)",HIGH:"linear-gradient(90deg,#fdba74,#f97316)",MEDIUM:"linear-gradient(90deg,#fcd34d,#f59e0b)",LOW:"linear-gradient(90deg,#86efac,#22c55e)" }[l]||"#e4e8ed");

function Sidebar({ active, setActive }) {
  const mods = [
    { id:"dashboard", label:"Overview", icon:"▦", live:true },
    { id:"fraud", label:"Fraud Detection", icon:"🛡", live:true },
    { id:"loan", label:"Loan Default", icon:"🏦", live:true },
    { id:"health", label:"Health Risk", icon:"❤", live:true },
    { id:"equipment", label:"Equipment Failure", icon:"⚙", live:true },
    { id:"network", label:"Network Anomaly", icon:"🔒", live:true },
    { id:"churn", label:"Customer Churn", icon:"📉", live:true },
  ];
  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <h1>SENTINEL-ML</h1>
        <p>Risk Intelligence Platform</p>
      </div>
      <div className="sidebar-section">Modules</div>
      {mods.map(m => (
        <button key={m.id} className={`nav-item ${active===m.id?"active":""}`}
          onClick={() => m.live && setActive(m.id)}
          style={{ opacity:m.live?1:0.5, cursor:m.live?"pointer":"not-allowed" }}>
          <span className="icon">{m.icon}</span>
          {m.label}
          <span className={`nav-badge ${m.live?"badge-live":"badge-soon"}`}>{m.live?"Live":"Soon"}</span>
        </button>
      ))}
      <div className="sidebar-footer">
        <div><span className="status-dot" />All systems operational</div>
        <div style={{ marginTop:5, fontSize:10, fontFamily:"'DM Mono',monospace" }}>v1.0.0 · 6 modules active</div>
      </div>
    </div>
  );
}

function StatCard({ label, value, sub, color, tag }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color }}>{value}</div>
      <div className="stat-sub">{sub}</div>
      {tag && <div className="stat-change">▲ {tag}</div>}
    </div>
  );
}

function MetricRow({ name, value, pct }) {
  return (
    <div className="metric-row">
      <span className="metric-name">{name}</span>
      <div style={{ display:"flex", alignItems:"center", gap:10 }}>
        <span className="metric-value">{value}</span>
        <div className="progress-bar"><div className="progress-fill" style={{ width:`${Math.min((pct||0)*100,100)}%` }} /></div>
      </div>
    </div>
  );
}

function ResultDetail({ label, value }) {
  return (
    <div className="result-detail-item">
      <div className="result-detail-label">{label}</div>
      <div className="result-detail-value">{value}</div>
    </div>
  );
}

function Dashboard() {
  const report = useApi("/model/report");
  const health = useApi("/health");
  const mods = [
    { id:"fraud", icon:"🛡", name:"Fraud Detection", desc:"Real-time transaction fraud scoring with XGBoost, LightGBM and Isolation Forest ensemble", company:"JPMorgan · Goldman Sachs", cls:"fraud", live:true },
    { id:"loan", icon:"🏦", name:"Loan Default", desc:"Credit risk scoring and default probability prediction trained on 2.26M real loans", company:"Morgan Stanley · Citi", cls:"loan", live:true },
    { id:"health", icon:"❤", name:"Health Risk", desc:"Combined heart disease and diabetes prediction using SVM and Gradient Boosting", company:"Google Health · DeepMind", cls:"health", live:true },
    { id:"equipment", icon:"⚙", name:"Equipment Failure", desc:"Predictive maintenance and machine failure detection using NASA sensor data", company:"Tesla · SpaceX", cls:"equipment", live:true },
    { id:"network", icon:"🔒", name:"Network Anomaly", desc:"Real-time cyber threat detection and intrusion prevention system", company:"Oracle · Microsoft", cls:"network", live:true },
    { id:"churn", icon:"📉", name:"Customer Churn", desc:"Churn probability and retention risk scoring trained on telecom customer data", company:"Google · Salesforce", cls:"churn", live:true },
  ];
  return (
    <div>
      <div className="section-title">Platform Overview</div>
      <div className="section-sub">SENTINEL-ML — Unified Multi-Domain Intelligent Risk Detection System</div>
      <div className="stats-grid">
        <StatCard label="Active Modules" value="6 / 6" sub="All modules live" color="#2563eb" tag="Platform complete" />
        <StatCard label="Fraud AUC-ROC" value="0.9845" sub="XGBoost · 56,746 test records" color="#16a34a" tag="Production grade" />
        <StatCard label="Loan Business Value" value="$315M+" sub="LightGBM · 2.26M loans trained" color="#d97706" />
        <StatCard label="Network Business Value" value="$602M+" sub="RandomForest · AUC 0.9644" color="#7c3aed" />
      </div>
      <div className="grid-2" style={{ marginBottom:20 }}>
        <div className="card">
          <div className="card-header">
            <div><div className="card-title">Fraud Detection Model Performance</div><div className="card-subtitle">XGBoost · Temporal split · 56,746 test transactions</div></div>
            <span className="pill pill-green">Live</span>
          </div>
          <div className="card-body">
            {report && ["auc_roc","avg_precision","f1","recall","precision","mcc"].map(k => (
              <MetricRow key={k} name={k.replace(/_/g," ").toUpperCase()} value={report[k]?.toFixed(4)} pct={Math.abs(report[k])} />
            ))}
          </div>
        </div>
        <div className="card">
          <div className="card-header"><div><div className="card-title">Fraud Confusion Matrix</div><div className="card-subtitle">At optimal business cost threshold</div></div></div>
          <div className="card-body">
            {report && (
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
                {[
                  { l:"True Positives", v:report.fraud_caught, s:"Fraud caught", c:"#16a34a" },
                  { l:"False Negatives", v:report.fraud_missed, s:"Fraud missed", c:"#dc2626" },
                  { l:"False Positives", v:report.false_alarms, s:"False alarms", c:"#d97706" },
                  { l:"True Negatives", v:(report.test_transactions-report.fraud_caught-report.fraud_missed-report.false_alarms), s:"Legit approved", c:"#2563eb" },
                ].map(i => (
                  <div key={i.l} style={{ background:"var(--gray-50)", borderRadius:8, padding:14, border:"1px solid var(--border)" }}>
                    <div style={{ fontSize:10, color:"var(--gray-400)", marginBottom:6, fontWeight:700, textTransform:"uppercase", letterSpacing:1 }}>{i.l}</div>
                    <div style={{ fontSize:28, fontWeight:700, color:i.c, fontFamily:"'DM Mono',monospace" }}>{i.v?.toLocaleString()}</div>
                    <div style={{ fontSize:11, color:"var(--gray-500)", marginTop:4 }}>{i.s}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
      <div style={{ fontSize:12, fontWeight:700, marginBottom:12, color:"var(--gray-500)", textTransform:"uppercase", letterSpacing:1.5 }}>Risk Detection Modules</div>
      <div className="modules-grid">
        {mods.map(m => (
          <div key={m.id} className={`module-card ${m.cls}`}>
            <div className="module-icon">{m.icon}</div>
            <div className="module-name">{m.name}</div>
            <div className="module-desc">{m.desc}</div>
            <div className="module-company">Used by: <span>{m.company}</span></div>
          </div>
        ))}
      </div>
      {health && (
        <div className="card">
          <div className="card-header"><div className="card-title">System Health</div><span className="pill pill-green">Operational</span></div>
          <div className="card-body">
            <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:12 }}>
              {Object.entries(health.modules_loaded).map(([mod,loaded]) => (
                <div key={mod} style={{ display:"flex", alignItems:"center", gap:8, fontSize:12 }}>
                  <div style={{ width:8, height:8, borderRadius:"50%", background:loaded?"#22c55e":"#cdd3db", flexShrink:0 }} />
                  <span style={{ color:loaded?"#2d3748":"#9aa5b4", textTransform:"capitalize", fontWeight:loaded?600:400 }}>{mod.replace(/_/g," ")}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function FraudModule() {
  const legit = { V1:-1.359807,V2:-0.072781,V3:2.536347,V4:1.378155,V5:-0.338321,V6:0.462388,V7:0.239599,V8:0.098698,V9:0.363787,V10:0.090794,V11:-0.5516,V12:-0.617801,V13:-0.99139,V14:-0.311169,V15:1.468177,V16:-0.470401,V17:0.207971,V18:0.025791,V19:0.403993,V20:0.251412,V21:-0.018307,V22:0.277838,V23:-0.110474,V24:0.066928,V25:0.128539,V26:-0.189115,V27:0.133558,V28:-0.021053,Amount:149.62,Time:0 };
  const fraud = { V1:-2.312227,V2:1.951992,V3:-1.609851,V4:3.997906,V5:-0.522188,V6:-1.426545,V7:-2.537387,V8:1.391657,V9:-2.770089,V10:-2.772272,V11:3.202033,V12:-2.899907,V13:-0.595222,V14:-4.289254,V15:0.389724,V16:-1.140747,V17:-2.830056,V18:-0.016822,V19:0.416956,V20:0.126911,V21:0.517232,V22:-0.035049,V23:-0.465211,V24:0.320198,V25:0.044519,V26:0.17784,V27:0.261145,V28:-0.143276,Amount:1,Time:406 };
  const [form, setForm] = useState(legit);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const report = useApi("/model/report");
  const submit = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/predict/fraud`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(form) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail||"Failed");
      setResult(data);
      setHistory(h => [{ id:Date.now(), amt:form.Amount, time:new Date().toLocaleTimeString(), score:data.fraud_score, action:data.action, risk:data.risk_level }, ...h.slice(0,9)]);
    } catch(e) { setError(e.message); } finally { setLoading(false); }
  };
  const vFeats = Array.from({ length:28 }, (_,i) => `V${i+1}`);
  return (
    <div>
      <div className="section-title">Fraud Detection</div>
      <div className="section-sub">Real-time transaction risk · XGBoost + LightGBM + Isolation Forest · AUC {report?.auc_roc?.toFixed(4)}</div>
      <div className="grid-2" style={{ alignItems:"start" }}>
        <div className="card">
          <div className="card-header">
            <div><div className="card-title">Transaction Input</div><div className="card-subtitle">PCA features V1-V28</div></div>
            <div style={{ display:"flex", gap:6 }}>
              <button className="btn btn-secondary" style={{ fontSize:11 }} onClick={() => { setForm(legit); setResult(null); }}>Legit</button>
              <button className="btn btn-danger" style={{ fontSize:11 }} onClick={() => { setForm(fraud); setResult(null); }}>Fraud</button>
            </div>
          </div>
          <div className="card-body">
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:14 }}>
              {["Amount","Time"].map(f => (
                <div className="form-group" key={f}>
                  <label className="form-label">{f} {f==="Amount"?"(USD)":"(s)"}</label>
                  <input className="form-input" type="number" value={form[f]} onChange={e => setForm(p => ({...p,[f]:parseFloat(e.target.value)||0}))} />
                </div>
              ))}
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:7 }}>
              {vFeats.map(f => (
                <div className="form-group" key={f}>
                  <label className="form-label">{f}</label>
                  <input className="form-input" type="number" step="0.000001" value={form[f]} onChange={e => setForm(p => ({...p,[f]:parseFloat(e.target.value)||0}))} />
                </div>
              ))}
            </div>
            <button className="btn btn-primary" style={{ width:"100%", padding:"11px 0", fontSize:13, marginTop:14 }} onClick={submit} disabled={loading}>
              {loading?"Analyzing...":"Run Fraud Analysis"}
            </button>
            {error && <div className="error-box">{error}</div>}
          </div>
        </div>
        <div>
          <div className="card" style={{ marginBottom:14 }}>
            <div className="card-header"><div className="card-title">Prediction Result</div></div>
            <div className="card-body">
              {!result && !loading && <div className="loading" style={{ flexDirection:"column", gap:8 }}><div style={{ fontSize:32 }}>🛡</div><div>Submit a transaction to analyze</div></div>}
              {loading && <div className="loading"><div className="spinner" />Analyzing...</div>}
              {result && !loading && (
                <div className="result-panel">
                  <div className="result-score" style={{ color:getRiskColor(result.risk_level) }}>{(result.fraud_score*100).toFixed(2)}%</div>
                  <div className="result-label">{result.risk_level} Risk</div>
                  <div className="risk-meter"><div className="risk-fill" style={{ width:`${result.fraud_score*100}%`, background:getRiskFill(result.risk_level) }} /></div>
                  <div style={{ fontSize:11, color:"var(--gray-400)", marginBottom:12, fontFamily:"'DM Mono',monospace" }}>threshold {result.threshold_used} · score {result.fraud_score.toFixed(6)}</div>
                  <div className={`action-badge ${result.action==="APPROVE"?"action-approve":"action-block"}`}>{result.action}</div>
                  <div className="result-detail-grid">
                    <ResultDetail label="Fraud Score" value={result.fraud_score.toFixed(6)} />
                    <ResultDetail label="Decision" value={result.action} />
                    <ResultDetail label="Risk Level" value={result.risk_level} />
                    <ResultDetail label="Is Fraud" value={result.is_fraud?"YES":"NO"} />
                  </div>
                </div>
              )}
            </div>
          </div>
          <div className="card">
            <div className="card-header"><div className="card-title">Model Metrics</div><span className="pill pill-green">XGBoost</span></div>
            <div className="card-body">
              {report && [["AUC-ROC",report.auc_roc],["Avg Precision",report.avg_precision],["F1 Score",report.f1],["Recall",report.recall]].map(([n,v]) => (
                <MetricRow key={n} name={n} value={v?.toFixed(4)} pct={v} />
              ))}
            </div>
          </div>
        </div>
      </div>
      {history.length > 0 && (
        <div className="card" style={{ marginTop:16 }}>
          <div className="card-header"><div className="card-title">Session History</div></div>
          <div className="table-container">
            <table>
              <thead><tr><th>Time</th><th>Amount</th><th>Score</th><th>Risk</th><th>Action</th></tr></thead>
              <tbody>{history.map(h => (
                <tr key={h.id}>
                  <td style={{ color:"var(--gray-400)", fontFamily:"'DM Mono',monospace", fontSize:12 }}>{h.time}</td>
                  <td style={{ fontFamily:"'DM Mono',monospace" }}>${h.amt.toFixed(2)}</td>
                  <td style={{ fontFamily:"'DM Mono',monospace" }}>{(h.score*100).toFixed(3)}%</td>
                  <td><span className={`pill ${getRiskPill(h.risk)}`}>{h.risk}</span></td>
                  <td><span className={`pill ${h.action==="APPROVE"?"pill-green":"pill-red"}`}>{h.action}</span></td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function LoanModule() {
  const good = { loan_amnt:10000,funded_amnt:10000,int_rate:8.5,installment:180,annual_inc:85000,dti:12,delinq_2yrs:0,fico_range_low:740,fico_range_high:760,open_acc:8,pub_rec:0,revol_bal:3000,revol_util:15,total_acc:18,mort_acc:1,pub_rec_bankruptcies:0,emp_length:8,grade:"A",home_ownership:"MORTGAGE",purpose:"debt_consolidation" };
  const risky = { loan_amnt:25000,funded_amnt:25000,int_rate:24.5,installment:750,annual_inc:32000,dti:38,delinq_2yrs:3,fico_range_low:610,fico_range_high:630,open_acc:12,pub_rec:2,revol_bal:18000,revol_util:88,total_acc:22,mort_acc:0,pub_rec_bankruptcies:1,emp_length:1,grade:"F",home_ownership:"RENT",purpose:"small_business" };
  const [form, setForm] = useState(good);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const setF = (k,v) => setForm(p => ({...p,[k]:isNaN(parseFloat(v))?v:parseFloat(v)}));
  const submit = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/predict/loan-default`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(form) });
      const data = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(data.detail)||"Failed");
      setResult(data);
      setHistory(h => [{ id:Date.now(), amt:form.loan_amnt, grade:form.grade, time:new Date().toLocaleTimeString(), score:data.default_score, decision:data.decision, risk:data.risk_level }, ...h.slice(0,9)]);
    } catch(e) { setError(e.message); } finally { setLoading(false); }
  };
  const numF = [["loan_amnt","Loan Amount ($)"],["funded_amnt","Funded ($)"],["int_rate","Interest Rate (%)"],["installment","Installment ($)"],["annual_inc","Annual Income ($)"],["dti","DTI"],["delinq_2yrs","Delinquencies"],["fico_range_low","FICO Low"],["fico_range_high","FICO High"],["open_acc","Open Accounts"],["pub_rec","Public Records"],["revol_bal","Revolving Bal ($)"],["revol_util","Revolving Util (%)"],["total_acc","Total Accounts"],["mort_acc","Mortgage Accounts"],["pub_rec_bankruptcies","Bankruptcies"],["emp_length","Employment (yrs)"]];
  return (
    <div>
      <div className="section-title">Loan Default Prediction</div>
      <div className="section-sub">Credit risk · LightGBM + XGBoost · 2.26M loans · AUC 0.6864</div>
      <div className="grid-2" style={{ alignItems:"start" }}>
        <div className="card">
          <div className="card-header">
            <div><div className="card-title">Loan Application</div></div>
            <div style={{ display:"flex", gap:6 }}>
              <button className="btn btn-success" style={{ fontSize:11 }} onClick={() => { setForm(good); setResult(null); }}>Good Loan</button>
              <button className="btn btn-danger" style={{ fontSize:11 }} onClick={() => { setForm(risky); setResult(null); }}>Risky Loan</button>
            </div>
          </div>
          <div className="card-body">
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:10, marginBottom:12 }}>
              {[["grade","Grade",["A","B","C","D","E","F","G"]],["home_ownership","Ownership",["MORTGAGE","RENT","OWN","ANY"]],["purpose","Purpose",["debt_consolidation","credit_card","home_improvement","other","small_business","medical","car"]]].map(([k,l,opts]) => (
                <div className="form-group" key={k}>
                  <label className="form-label">{l}</label>
                  <select className="form-input" value={form[k]} onChange={e => setF(k,e.target.value)}>
                    {opts.map(o => <option key={o} value={o}>{o.replace(/_/g," ")}</option>)}
                  </select>
                </div>
              ))}
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
              {numF.map(([k,l]) => (<div className="form-group" key={k}><label className="form-label">{l}</label><input className="form-input" type="number" value={form[k]} onChange={e => setF(k,e.target.value)} /></div>))}
            </div>
            <button className="btn btn-primary" style={{ width:"100%", padding:"11px 0", fontSize:13, marginTop:14 }} onClick={submit} disabled={loading}>
              {loading?"Analyzing...":"Run Credit Risk Analysis"}
            </button>
            {error && <div className="error-box">{error}</div>}
          </div>
        </div>
        <div>
          <div className="card" style={{ marginBottom:14 }}>
            <div className="card-header"><div className="card-title">Credit Risk Result</div></div>
            <div className="card-body">
              {!result && !loading && <div className="loading" style={{ flexDirection:"column", gap:8 }}><div style={{ fontSize:32 }}>🏦</div><div>Submit application to analyze</div></div>}
              {loading && <div className="loading"><div className="spinner" />Analyzing...</div>}
              {result && !loading && (
                <div className="result-panel">
                  <div className="result-score" style={{ color:getRiskColor(result.risk_level) }}>{(result.default_score*100).toFixed(2)}%</div>
                  <div className="result-label">Default Probability · {result.risk_level}</div>
                  <div className="risk-meter"><div className="risk-fill" style={{ width:`${result.default_score*100}%`, background:getRiskFill(result.risk_level) }} /></div>
                  <div className={`action-badge ${result.decision==="APPROVE"?"action-approve":"action-block"}`}>{result.decision}</div>
                  <div className="result-detail-grid">
                    <ResultDetail label="Default Score" value={result.default_score.toFixed(6)} />
                    <ResultDetail label="Decision" value={result.decision} />
                    <ResultDetail label="Risk Level" value={result.risk_level} />
                    <ResultDetail label="Composite Risk" value={result.composite_risk_score.toFixed(4)} />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function HealthModule() {
  const healthy = { age:35,sex:0,cp:0,trestbps:115,chol:200,fbs:0,restecg:0,thalach:175,exang:0,oldpeak:0.2,slope:2,ca:0,thal:2,Pregnancies:1,Glucose:90,BloodPressure:70,SkinThickness:18,Insulin:60,BMI:22.5,DiabetesPedigreeFunction:0.2,Age:35 };
  const atRisk = { age:62,sex:1,cp:2,trestbps:160,chol:320,fbs:1,restecg:2,thalach:108,exang:1,oldpeak:3.5,slope:0,ca:3,thal:3,Pregnancies:6,Glucose:175,BloodPressure:92,SkinThickness:35,Insulin:200,BMI:38.5,DiabetesPedigreeFunction:1.2,Age:62 };
  const [form, setForm] = useState(healthy);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const setF = (k,v) => setForm(p => ({...p,[k]:parseFloat(v)||0}));
  const submit = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/predict/health-risk`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(form) });
      const data = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(data.detail)||"Failed");
      setResult(data);
    } catch(e) { setError(e.message); } finally { setLoading(false); }
  };
  const hF = [["age","Age"],["sex","Sex (1=M)"],["cp","Chest Pain 0-3"],["trestbps","Resting BP"],["chol","Cholesterol"],["fbs","Fast BS>120"],["restecg","ECG 0-2"],["thalach","Max HR"],["exang","Ex Angina"],["oldpeak","ST Dep"],["slope","Slope 0-2"],["ca","Vessels 0-4"],["thal","Thal 0-3"]];
  const dF = [["Pregnancies","Pregnancies"],["Glucose","Glucose"],["BloodPressure","Blood Pressure"],["SkinThickness","Skin Thickness"],["Insulin","Insulin"],["BMI","BMI"],["DiabetesPedigreeFunction","DPF"],["Age","Age"]];
  return (
    <div>
      <div className="section-title">Health Risk Assessment</div>
      <div className="section-sub">Heart disease and diabetes prediction · SVM + Gradient Boosting</div>
      <div className="grid-2" style={{ alignItems:"start" }}>
        <div>
          <div className="card" style={{ marginBottom:14 }}>
            <div className="card-header">
              <div><div className="card-title">Cardiovascular Indicators</div></div>
              <div style={{ display:"flex", gap:6 }}>
                <button className="btn btn-success" style={{ fontSize:11 }} onClick={() => { setForm(healthy); setResult(null); }}>Healthy</button>
                <button className="btn btn-danger" style={{ fontSize:11 }} onClick={() => { setForm(atRisk); setResult(null); }}>At Risk</button>
              </div>
            </div>
            <div className="card-body">
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                {hF.map(([k,l]) => (<div className="form-group" key={k}><label className="form-label">{l}</label><input className="form-input" type="number" step="any" value={form[k]} onChange={e => setF(k,e.target.value)} /></div>))}
              </div>
            </div>
          </div>
          <div className="card">
            <div className="card-header"><div className="card-title">Metabolic Indicators</div></div>
            <div className="card-body">
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
                {dF.map(([k,l]) => (<div className="form-group" key={k}><label className="form-label">{l}</label><input className="form-input" type="number" step="any" value={form[k]} onChange={e => setF(k,e.target.value)} /></div>))}
              </div>
              <button className="btn btn-primary" style={{ width:"100%", padding:"11px 0", fontSize:13, marginTop:14 }} onClick={submit} disabled={loading}>
                {loading?"Analyzing...":"Run Health Risk Assessment"}
              </button>
              {error && <div className="error-box">{error}</div>}
            </div>
          </div>
        </div>
        <div className="card">
          <div className="card-header"><div className="card-title">Health Risk Result</div></div>
          <div className="card-body">
            {!result && !loading && <div className="loading" style={{ flexDirection:"column", gap:8 }}><div style={{ fontSize:32 }}>❤</div><div>Submit patient data to assess</div></div>}
            {loading && <div className="loading"><div className="spinner" />Running assessment...</div>}
            {result && !loading && (
              <div>
                <div className="health-score-row">
                  {[{label:"Heart Disease",value:result.heart_disease_score,risk:result.heart_risk_level},{label:"Diabetes",value:result.diabetes_score,risk:result.diabetes_risk_level},{label:"Combined",value:result.combined_health_score,risk:result.overall_risk_level}].map(s => (
                    <div key={s.label} className="health-score-circle">
                      <div className="health-score-value" style={{ color:getRiskColor(s.risk) }}>{(s.value*100).toFixed(0)}%</div>
                      <div className="health-score-label">{s.label}</div>
                      <div style={{ marginTop:6 }}><span className={`pill ${getRiskPill(s.risk)}`}>{s.risk}</span></div>
                    </div>
                  ))}
                </div>
                <div className="result-panel">
                  <div className="result-score" style={{ color:getRiskColor(result.overall_risk_level), fontSize:42 }}>{result.overall_risk_level}</div>
                  <div className="risk-meter"><div className="risk-fill" style={{ width:`${result.combined_health_score*100}%`, background:getRiskFill(result.overall_risk_level) }} /></div>
                  <div style={{ marginTop:14, padding:"12px 16px", background:"var(--white)", borderRadius:8, border:"1px solid var(--border)", fontSize:12.5, color:getRiskColor(result.overall_risk_level), fontWeight:600 }}>{result.recommendation}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function EquipmentModule() {
  const healthyR = Array.from({length:30},(_,i) => ({ cycle:i+1,s2:641.82+i*0.01,s3:1589.70+i*0.05,s4:1400.60+i*0.02,s7:554.36,s8:2388.02,s9:9046.19-i*0.1,s11:47.47+i*0.01,s12:521.66,s13:2388.02,s14:8138.62-i*0.5,s15:8.4195,s17:392,s20:39.06,s21:23.419 }));
  const criticalR = Array.from({length:30},(_,i) => ({ cycle:i+1,s2:642.50+i*0.3,s3:1595.00+i*0.8,s4:1410.00+i*0.5,s7:550.00,s8:2390.00,s9:9020.00-i*2,s11:47.80+i*0.1,s12:525.00,s13:2390.00,s14:8050.00-i*8,s15:8.50+i*0.05,s17:392,s20:38.00,s21:23.50+i*0.05 }));
  const [readings, setReadings] = useState(healthyR);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [nc, setNc] = useState(30);
  const submit = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/predict/equipment-failure`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ sensor_readings:readings.slice(0,nc) }) });
      const data = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(data.detail)||"Failed");
      setResult(data);
    } catch(e) { setError(e.message); } finally { setLoading(false); }
  };
  return (
    <div>
      <div className="section-title">Equipment Failure Prediction</div>
      <div className="section-sub">Remaining Useful Life · XGBoost · NASA CMAPSS · Tesla/SpaceX methodology</div>
      <div className="grid-2" style={{ alignItems:"start" }}>
        <div className="card">
          <div className="card-header">
            <div><div className="card-title">Engine Sensor Input</div></div>
            <div style={{ display:"flex", gap:6 }}>
              <button className="btn btn-success" style={{ fontSize:11 }} onClick={() => { setReadings(healthyR); setResult(null); }}>Healthy Engine</button>
              <button className="btn btn-danger" style={{ fontSize:11 }} onClick={() => { setReadings(criticalR); setResult(null); }}>Degraded Engine</button>
            </div>
          </div>
          <div className="card-body">
            <div className="form-group" style={{ marginBottom:14 }}>
              <label className="form-label">Cycles to use: {nc}</label>
              <input className="form-input" type="range" min={5} max={30} value={nc} onChange={e => setNc(parseInt(e.target.value))} style={{ padding:0 }} />
            </div>
            <div style={{ background:"var(--gray-50)", border:"1px solid var(--border)", borderRadius:8, padding:14, marginBottom:14 }}>
              <div style={{ fontSize:12, fontWeight:700, color:"var(--gray-600)", marginBottom:8 }}>Sensor Summary</div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:6 }}>
                {["s4","s11","s14","s15","s20","s21"].map(s => (
                  <div key={s} style={{ fontSize:11, color:"var(--gray-500)" }}><span style={{ fontWeight:700, color:"var(--gray-700)" }}>{s}: </span>{readings[0]?.[s]?.toFixed(2)||"N/A"}</div>
                ))}
              </div>
            </div>
            <button className="btn btn-primary" style={{ width:"100%", padding:"11px 0", fontSize:13 }} onClick={submit} disabled={loading}>
              {loading?"Predicting...":"Predict Remaining Useful Life"}
            </button>
            {error && <div className="error-box">{error}</div>}
          </div>
        </div>
        <div className="card">
          <div className="card-header"><div className="card-title">Prediction Result</div></div>
          <div className="card-body">
            {!result && !loading && <div className="loading" style={{ flexDirection:"column", gap:8 }}><div style={{ fontSize:32 }}>⚙</div><div>Load sensor data to predict RUL</div></div>}
            {loading && <div className="loading"><div className="spinner" />Analyzing sensors...</div>}
            {result && !loading && (
              <div className="result-panel">
                <div className="result-score" style={{ color:getRiskColor(result.urgency_level) }}>{Math.round(result.predicted_rul)}</div>
                <div className="result-label">Cycles Remaining · {result.urgency_level}</div>
                <div className="risk-meter"><div className="risk-fill" style={{ width:`${Math.min((result.predicted_rul/125)*100,100)}%`, background:result.predicted_rul>60?"linear-gradient(90deg,#86efac,#22c55e)":result.predicted_rul>30?"linear-gradient(90deg,#fcd34d,#f59e0b)":"linear-gradient(90deg,#fca5a5,#ef4444)" }} /></div>
                <div style={{ margin:"14px 0", padding:"12px 16px", background:"var(--white)", borderRadius:8, border:"1px solid var(--border)", fontSize:12.5, fontWeight:600, color:getRiskColor(result.urgency_level) }}>{result.recommendation}</div>
                <div className="result-detail-grid">
                  <ResultDetail label="Predicted RUL" value={`${result.predicted_rul.toFixed(1)} cycles`} />
                  <ResultDetail label="Alert Threshold" value={`${result.alert_threshold} cycles`} />
                  <ResultDetail label="Health Index" value={result.health_index.toFixed(4)} />
                  <ResultDetail label="Failure Risk" value={result.failure_risk_score.toFixed(4)} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function NetworkModule() {
  const normal = { duration:0,protocol_type:"tcp",service:"http",flag:"SF",src_bytes:181,dst_bytes:5450,land:0,wrong_fragment:0,urgent:0,hot:0,num_failed_logins:0,logged_in:1,num_compromised:0,root_shell:0,su_attempted:0,num_root:0,num_file_creations:0,num_shells:0,num_access_files:0,is_host_login:0,is_guest_login:0,count:8,srv_count:8,serror_rate:0,srv_serror_rate:0,rerror_rate:0,srv_rerror_rate:0,same_srv_rate:1,diff_srv_rate:0,srv_diff_host_rate:0,dst_host_count:9,dst_host_srv_count:9,dst_host_same_srv_rate:1,dst_host_diff_srv_rate:0,dst_host_same_src_port_rate:0.11,dst_host_srv_diff_host_rate:0,dst_host_serror_rate:0,dst_host_srv_serror_rate:0,dst_host_rerror_rate:0,dst_host_srv_rerror_rate:0,service_attack_rate:0.1,service_frequency:5000 };
  const attack = { duration:0,protocol_type:"icmp",service:"ecr_i",flag:"SF",src_bytes:1032,dst_bytes:0,land:0,wrong_fragment:0,urgent:0,hot:0,num_failed_logins:0,logged_in:0,num_compromised:0,root_shell:0,su_attempted:0,num_root:0,num_file_creations:0,num_shells:0,num_access_files:0,is_host_login:0,is_guest_login:0,count:511,srv_count:511,serror_rate:0,srv_serror_rate:0,rerror_rate:0,srv_rerror_rate:0,same_srv_rate:1,diff_srv_rate:0,srv_diff_host_rate:0,dst_host_count:255,dst_host_srv_count:255,dst_host_same_srv_rate:1,dst_host_diff_srv_rate:0,dst_host_same_src_port_rate:1,dst_host_srv_diff_host_rate:0,dst_host_serror_rate:0,dst_host_srv_serror_rate:0,dst_host_rerror_rate:0,dst_host_srv_rerror_rate:0,service_attack_rate:0.98,service_frequency:200 };
  const [form, setForm] = useState(normal);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const setF = (k,v) => setForm(p => ({...p,[k]:isNaN(parseFloat(v))?v:parseFloat(v)}));
  const submit = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/predict/network-anomaly`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(form) });
      const data = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(data.detail)||"Failed");
      setResult(data);
    } catch(e) { setError(e.message); } finally { setLoading(false); }
  };
  const nF = [["duration","Duration"],["src_bytes","Src Bytes"],["dst_bytes","Dst Bytes"],["count","Count"],["srv_count","Srv Count"],["serror_rate","SYN Err Rate"],["rerror_rate","REJ Err Rate"],["same_srv_rate","Same Svc Rate"],["dst_host_count","Dst Host Count"],["service_attack_rate","Svc Attack Rate"]];
  return (
    <div>
      <div className="section-title">Network Anomaly Detection</div>
      <div className="section-sub">Cyber threat detection · RandomForest · NSL-KDD · AUC 0.9644 · $602M business value</div>
      <div className="grid-2" style={{ alignItems:"start" }}>
        <div className="card">
          <div className="card-header">
            <div><div className="card-title">Network Traffic Input</div></div>
            <div style={{ display:"flex", gap:6 }}>
              <button className="btn btn-success" style={{ fontSize:11 }} onClick={() => { setForm(normal); setResult(null); }}>Normal Traffic</button>
              <button className="btn btn-danger" style={{ fontSize:11 }} onClick={() => { setForm(attack); setResult(null); }}>Attack Traffic</button>
            </div>
          </div>
          <div className="card-body">
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:10, marginBottom:12 }}>
              {[["protocol_type","Protocol",["tcp","udp","icmp"]],["flag","Flag",["SF","S0","REJ","RSTR"]]].map(([k,l,opts]) => (
                <div className="form-group" key={k}><label className="form-label">{l}</label>
                  <select className="form-input" value={form[k]} onChange={e => setF(k,e.target.value)}>{opts.map(o => <option key={o} value={o}>{o}</option>)}</select>
                </div>
              ))}
              <div className="form-group"><label className="form-label">Logged In</label><select className="form-input" value={form.logged_in} onChange={e => setF("logged_in",e.target.value)}><option value={0}>No</option><option value={1}>Yes</option></select></div>
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
              {nF.map(([k,l]) => (<div className="form-group" key={k}><label className="form-label">{l}</label><input className="form-input" type="number" step="any" value={form[k]||0} onChange={e => setF(k,e.target.value)} /></div>))}
            </div>
            <button className="btn btn-primary" style={{ width:"100%", padding:"11px 0", fontSize:13, marginTop:14 }} onClick={submit} disabled={loading}>
              {loading?"Analyzing...":"Run Threat Analysis"}
            </button>
            {error && <div className="error-box">{error}</div>}
          </div>
        </div>
        <div className="card">
          <div className="card-header"><div className="card-title">Threat Detection Result</div></div>
          <div className="card-body">
            {!result && !loading && <div className="loading" style={{ flexDirection:"column", gap:8 }}><div style={{ fontSize:32 }}>🔒</div><div>Submit traffic to analyze</div></div>}
            {loading && <div className="loading"><div className="spinner" />Analyzing traffic...</div>}
            {result && !loading && (
              <div className="result-panel">
                <div className="result-score" style={{ color:getRiskColor(result.threat_level) }}>{(result.attack_score*100).toFixed(1)}%</div>
                <div className="result-label">Attack Probability · {result.threat_level}</div>
                <div className="risk-meter"><div className="risk-fill" style={{ width:`${result.attack_score*100}%`, background:getRiskFill(result.threat_level) }} /></div>
                <div className={`action-badge ${result.action==="ALLOW"?"action-approve":"action-block"}`}>{result.action}</div>
                <div style={{ marginTop:14, padding:"12px 16px", background:"var(--white)", borderRadius:8, border:"1px solid var(--border)", fontSize:12.5, fontWeight:600, color:getRiskColor(result.threat_level) }}>{result.recommendation}</div>
                <div className="result-detail-grid">
                  <ResultDetail label="Attack Score" value={result.attack_score.toFixed(6)} />
                  <ResultDetail label="Action" value={result.action} />
                  <ResultDetail label="Threat Level" value={result.threat_level} />
                  <ResultDetail label="Network Risk" value={result.network_risk_score.toFixed(4)} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ChurnModule() {
  const loyal = { gender:"Male",SeniorCitizen:0,Partner:"Yes",Dependents:"Yes",tenure:60,PhoneService:"Yes",MultipleLines:"Yes",InternetService:"DSL",OnlineSecurity:"Yes",OnlineBackup:"Yes",DeviceProtection:"Yes",TechSupport:"Yes",StreamingTV:"No",StreamingMovies:"No",Contract:"Two year",PaperlessBilling:"No",PaymentMethod:"Bank transfer (automatic)",MonthlyCharges:55.0,TotalCharges:3300.0 };
  const atrisk = { gender:"Female",SeniorCitizen:0,Partner:"No",Dependents:"No",tenure:2,PhoneService:"Yes",MultipleLines:"No",InternetService:"Fiber optic",OnlineSecurity:"No",OnlineBackup:"No",DeviceProtection:"No",TechSupport:"No",StreamingTV:"Yes",StreamingMovies:"Yes",Contract:"Month-to-month",PaperlessBilling:"Yes",PaymentMethod:"Electronic check",MonthlyCharges:95.0,TotalCharges:190.0 };
  const [form, setForm] = useState(loyal);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const strFields = ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"];
  const setF = (k,v) => setForm(p => ({...p,[k]:strFields.includes(k)?v:parseFloat(v)||0}));
  const submit = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API}/predict/customer-churn`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(form) });
      const data = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(data.detail)||"Failed");
      setResult(data);
      setHistory(h => [{ id:Date.now(), time:new Date().toLocaleTimeString(), score:data.churn_score, risk:data.risk_level, action:data.action, tenure:form.tenure, contract:form.Contract }, ...h.slice(0,9)]);
    } catch(e) { setError(e.message); } finally { setLoading(false); }
  };
  const selects = [["gender","Gender",["Male","Female"]],["Partner","Partner",["Yes","No"]],["Dependents","Dependents",["Yes","No"]],["PhoneService","Phone Service",["Yes","No"]],["MultipleLines","Multiple Lines",["Yes","No","No phone service"]],["InternetService","Internet",["DSL","Fiber optic","No"]],["OnlineSecurity","Online Security",["Yes","No","No internet service"]],["OnlineBackup","Online Backup",["Yes","No","No internet service"]],["DeviceProtection","Device Protect",["Yes","No","No internet service"]],["TechSupport","Tech Support",["Yes","No","No internet service"]],["StreamingTV","Streaming TV",["Yes","No","No internet service"]],["StreamingMovies","Streaming Movies",["Yes","No","No internet service"]],["Contract","Contract",["Month-to-month","One year","Two year"]],["PaperlessBilling","Paperless Bill",["Yes","No"]],["PaymentMethod","Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]]];
  return (
    <div>
      <div className="section-title">Customer Churn Prediction</div>
      <div className="section-sub">Retention risk scoring · XGBoost · Telco Dataset · AUC 0.8363</div>
      <div className="grid-2" style={{ alignItems:"start" }}>
        <div className="card">
          <div className="card-header">
            <div><div className="card-title">Customer Profile</div></div>
            <div style={{ display:"flex", gap:6 }}>
              <button className="btn btn-success" style={{ fontSize:11 }} onClick={() => { setForm(loyal); setResult(null); }}>Loyal Customer</button>
              <button className="btn btn-danger" style={{ fontSize:11 }} onClick={() => { setForm(atrisk); setResult(null); }}>At-Risk Customer</button>
            </div>
          </div>
          <div className="card-body">
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:12 }}>
              <div className="form-group"><label className="form-label">Tenure (months)</label><input className="form-input" type="number" value={form.tenure} onChange={e => setF("tenure",e.target.value)} /></div>
              <div className="form-group"><label className="form-label">Senior Citizen</label><select className="form-input" value={form.SeniorCitizen} onChange={e => setF("SeniorCitizen",e.target.value)}><option value={0}>No</option><option value={1}>Yes</option></select></div>
              <div className="form-group"><label className="form-label">Monthly Charges ($)</label><input className="form-input" type="number" value={form.MonthlyCharges} onChange={e => setF("MonthlyCharges",e.target.value)} /></div>
              <div className="form-group"><label className="form-label">Total Charges ($)</label><input className="form-input" type="number" value={form.TotalCharges} onChange={e => setF("TotalCharges",e.target.value)} /></div>
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8 }}>
              {selects.map(([k,l,opts]) => (
                <div className="form-group" key={k}><label className="form-label">{l}</label>
                  <select className="form-input" value={form[k]} onChange={e => setF(k,e.target.value)}>{opts.map(o => <option key={o} value={o}>{o}</option>)}</select>
                </div>
              ))}
            </div>
            <button className="btn btn-primary" style={{ width:"100%", padding:"11px 0", fontSize:13, marginTop:14 }} onClick={submit} disabled={loading}>
              {loading?"Analyzing...":"Predict Churn Risk"}
            </button>
            {error && <div className="error-box">{error}</div>}
          </div>
        </div>
        <div>
          <div className="card" style={{ marginBottom:14 }}>
            <div className="card-header"><div className="card-title">Churn Prediction Result</div></div>
            <div className="card-body">
              {!result && !loading && <div className="loading" style={{ flexDirection:"column", gap:8 }}><div style={{ fontSize:32 }}>📉</div><div>Submit a customer profile to analyze</div></div>}
              {loading && <div className="loading"><div className="spinner" />Analyzing churn risk...</div>}
              {result && !loading && (
                <div className="result-panel">
                  <div className="result-score" style={{ color:getRiskColor(result.risk_level) }}>{(result.churn_score*100).toFixed(1)}%</div>
                  <div className="result-label">Churn Probability · {result.risk_level}</div>
                  <div className="risk-meter"><div className="risk-fill" style={{ width:`${result.churn_score*100}%`, background:getRiskFill(result.risk_level) }} /></div>
                  <div style={{ fontSize:11, color:"var(--gray-400)", marginBottom:12, fontFamily:"'DM Mono',monospace" }}>threshold {result.threshold_used} · score {result.churn_score.toFixed(6)}</div>
                  <div className={`action-badge ${result.will_churn?"action-block":"action-approve"}`}>{result.action}</div>
                  <div style={{ marginTop:14, padding:"12px 16px", background:"var(--white)", borderRadius:8, border:"1px solid var(--border)", fontSize:12.5, fontWeight:600, color:getRiskColor(result.risk_level) }}>{result.recommendation}</div>
                  <div className="result-detail-grid">
                    <ResultDetail label="Churn Score" value={result.churn_score.toFixed(6)} />
                    <ResultDetail label="Action" value={result.action} />
                    <ResultDetail label="Risk Level" value={result.risk_level} />
                    <ResultDetail label="Churn Risk Score" value={result.churn_risk_score.toFixed(4)} />
                  </div>
                </div>
              )}
            </div>
          </div>
          <div className="card">
            <div className="card-header"><div className="card-title">Model Information</div><span className="pill pill-teal">XGBoost</span></div>
            <div className="card-body">
              {[["Dataset","7,043 customers","Telco Customer Churn"],["AUC-ROC","0.8363","1,409 test customers"],["Recall","98.4%","368/374 churners caught"],["Business Value","$71,400","At optimal threshold"],["Top Signal","Contract type","Month-to-month = 42.7% churn"],["Revenue at Risk","$139,131/mo","Full customer base"]].map(([n,v,s]) => (
                <div key={n} className="metric-row">
                  <div><div style={{ fontSize:12.5, fontWeight:600, color:"var(--gray-700)" }}>{n}</div><div style={{ fontSize:11, color:"var(--gray-400)" }}>{s}</div></div>
                  <span className="pill pill-teal">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      {history.length > 0 && (
        <div className="card" style={{ marginTop:16 }}>
          <div className="card-header"><div className="card-title">Session History</div><span className="pill pill-blue">{history.length} predictions</span></div>
          <div className="table-container">
            <table>
              <thead><tr><th>Time</th><th>Tenure</th><th>Contract</th><th>Churn Score</th><th>Risk</th><th>Action</th></tr></thead>
              <tbody>{history.map(h => (
                <tr key={h.id}>
                  <td style={{ color:"var(--gray-400)", fontFamily:"'DM Mono',monospace", fontSize:12 }}>{h.time}</td>
                  <td>{h.tenure} mo</td>
                  <td><span className="pill pill-blue">{h.contract}</span></td>
                  <td style={{ fontFamily:"'DM Mono',monospace" }}>{(h.score*100).toFixed(2)}%</td>
                  <td><span className={`pill ${getRiskPill(h.risk)}`}>{h.risk}</span></td>
                  <td><span className={`pill ${h.action.includes("MONITOR")?"pill-green":"pill-red"}`}>{h.action}</span></td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function App() {
  const [active, setActive] = useState("dashboard");
  const pages = { dashboard:<Dashboard />, fraud:<FraudModule />, loan:<LoanModule />, health:<HealthModule />, equipment:<EquipmentModule />, network:<NetworkModule />, churn:<ChurnModule /> };
  const titles = {
    dashboard:["Platform Overview","SENTINEL-ML · Unified Risk Detection"],
    fraud:["Fraud Detection","JPMorgan · Goldman Sachs · Real-time transaction scoring"],
    loan:["Loan Default Prediction","Morgan Stanley · Citi · Credit risk assessment"],
    health:["Health Risk Assessment","Google Health · DeepMind · Combined disease prediction"],
    equipment:["Equipment Failure Prediction","Tesla · SpaceX · Remaining Useful Life"],
    network:["Network Anomaly Detection","Oracle · Microsoft · Cyber threat detection"],
    churn:["Customer Churn Prediction","Google · Salesforce · Retention risk scoring"],
  };
  const [title, sub] = titles[active]||titles.dashboard;
  return (
    <div className="platform">
      <Sidebar active={active} setActive={setActive} />
      <div className="main">
        <div className="topbar">
          <div><div className="topbar-title">{title}</div><div className="topbar-subtitle">{sub}</div></div>
          <div className="topbar-actions">
            <span style={{ fontSize:11, color:"var(--gray-400)", fontFamily:"'DM Mono',monospace" }}>api · localhost:8000</span>
            <span className="pill pill-green">● Live</span>
          </div>
        </div>
        <div className="content">{pages[active]||<Dashboard />}</div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
