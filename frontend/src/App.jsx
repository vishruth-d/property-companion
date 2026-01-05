import React, { useState, useEffect, useCallback } from 'react';
import { 
  Home, Upload, FileSpreadsheet, BarChart3, TrendingUp, TrendingDown, Calculator,
  CheckCircle, XCircle, Loader2, AlertTriangle, ChevronDown, ChevronUp,
  ArrowLeft, Target, MapPin, Building, Activity, PieChart, Download, RefreshCw,
  Minus, Shield, PoundSterling, Award, Bookmark, BookmarkPlus, Trash2,
  Columns, Eye, X, ExternalLink, Search, Filter, Save, Ruler, Clock,
  AlertCircle, Info, DollarSign, Percent, Users, Zap
} from 'lucide-react';

// ============================================
// CONFIG
// ============================================
const API_BASE = 'http://localhost:8000';
const MAX_COMPARE = 5;
const MAX_SAVED = 50;

// ============================================
// API
// ============================================
const API = {
  getVersions: () => fetch(`${API_BASE}/api/versions`).then(r => r.json()),
  health: () => fetch(`${API_BASE}/api/health`).then(r => r.json()),
  predict: (prop, v) => fetch(`${API_BASE}/api/predict?version=${v}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(prop)
  }).then(r => r.json()),
  uploadExcel: async (file, v) => {
    const fd = new FormData(); fd.append('file', file);
    return fetch(`${API_BASE}/api/upload-excel?version=${v}`, {method: 'POST', body: fd}).then(r => r.json());
  },
  getComparables: (prop, v) => fetch(`${API_BASE}/api/comparables?version=${v}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(prop)
  }).then(r => r.json()),
  getStats: v => fetch(`${API_BASE}/api/stats/${v}`).then(r => r.json()),
  getStatsSummary: () => fetch(`${API_BASE}/api/stats-summary`).then(r => r.json()),
  saveAnalysis: d => fetch(`${API_BASE}/api/analyses/save`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(d)
  }).then(r => r.json()),
  getAnalyses: () => fetch(`${API_BASE}/api/analyses`).then(r => r.json()),
  deleteAnalysis: id => fetch(`${API_BASE}/api/analyses/${id}`, {method: 'DELETE'}).then(r => r.json()),
};

// ============================================
// CONSTANTS & HELPERS
// ============================================
const TENURE = [{v:'F',l:'Freehold'},{v:'L',l:'Leasehold'}];
const AGE = [{v:'N',l:'New Build'},{v:'Y',l:'Existing'}];
const TYPES = ['Flat','Maisonette','House','Bungalow'];
const FORMS = ['Detached','Semi-Detached','Terraced','End-Terrace'];
const LAS = ['Barnet','Brent','Ealing','Enfield','Harrow','Hillingdon','Hounslow','Haringey','Hertsmere','Richmond upon Thames','Three Rivers','Watford','Buckinghamshire','Slough','Dacorum'];
const ENERGY = ['A','B','C','D','E','F','G'];

const fmt = n => n?.toLocaleString('en-GB') ?? 'N/A';
const fmtDate = d => d ? new Date(d).toLocaleDateString('en-GB',{month:'short',year:'numeric'}) : 'N/A';
const pct = (v, d=1) => v?.toFixed(d) + '%' ?? 'N/A';

const REC = {
  STRONG_BUY: {bg:'bg-green-500/20',border:'border-green-500',text:'text-green-400',label:'STRONG BUY',icon:TrendingUp,desc:'Excellent opportunity'},
  BUY: {bg:'bg-emerald-500/20',border:'border-emerald-500',text:'text-emerald-400',label:'BUY',icon:TrendingUp,desc:'Good value'},
  NEUTRAL: {bg:'bg-blue-500/20',border:'border-blue-500',text:'text-blue-400',label:'FAIR',icon:Minus,desc:'Market rate'},
  NEGOTIATE: {bg:'bg-amber-500/20',border:'border-amber-500',text:'text-amber-400',label:'NEGOTIATE',icon:TrendingDown,desc:'Push for discount'},
  AVOID: {bg:'bg-red-500/20',border:'border-red-500',text:'text-red-400',label:'AVOID',icon:XCircle,desc:'Overpriced'},
};

const VERDICT = {
  STRONG_UNDERVALUED: {bg:'bg-green-600',label:'STRONG UNDER',color:'green'},
  LIKELY_UNDERVALUED: {bg:'bg-emerald-600',label:'UNDERVALUED',color:'emerald'},
  FAIR_VALUE: {bg:'bg-blue-600',label:'FAIR VALUE',color:'blue'},
  LIKELY_OVERVALUED: {bg:'bg-amber-600',label:'OVERPRICED',color:'amber'},
  STRONG_OVERVALUED: {bg:'bg-red-600',label:'STRONG OVER',color:'red'},
};

const CONFIDENCE = {
  HIGH: {bg:'bg-green-500/20',border:'border-green-500',text:'text-green-400',icon:'🟢'},
  MEDIUM: {bg:'bg-amber-500/20',border:'border-amber-500',text:'text-amber-400',icon:'🟡'},
  LOW: {bg:'bg-red-500/20',border:'border-red-500',text:'text-red-400',icon:'🔴'},
};

// ============================================
// UI COMPONENTS
// ============================================
const Card = ({children, className='', highlight=false}) => (
  <div className={`bg-gray-800/80 backdrop-blur rounded-2xl p-5 ${highlight ? 'ring-2 ring-purple-500/50' : ''} ${className}`}>
    {children}
  </div>
);

const Badge = ({type='default', children, className=''}) => {
  const styles = {
    default: 'bg-gray-700 text-gray-300',
    success: 'bg-green-600/30 text-green-400 border border-green-500/50',
    warning: 'bg-amber-600/30 text-amber-400 border border-amber-500/50',
    danger: 'bg-red-600/30 text-red-400 border border-red-500/50',
    info: 'bg-blue-600/30 text-blue-400 border border-blue-500/50',
    purple: 'bg-purple-600/30 text-purple-400 border border-purple-500/50',
  };
  return <span className={`px-2.5 py-1 rounded-lg text-xs font-semibold ${styles[type]} ${className}`}>{children}</span>;
};

const VersionBadge = ({v, active=false, onClick, stats=null}) => {
  const colors = {v1:'from-blue-600 to-blue-700',v2:'from-purple-600 to-purple-700',v3:'from-green-600 to-green-700'};
  const labels = {v1:'V1 Baseline',v2:'V2 Enhanced',v3:'V3 Production'};
  return (
    <button onClick={onClick} className={`px-4 py-2.5 rounded-xl font-medium transition-all flex flex-col items-center ${active ? `bg-gradient-to-r ${colors[v]} shadow-lg scale-105` : 'bg-gray-700/50 hover:bg-gray-700'}`}>
      <span>{labels[v]}{v==='v3' && <span className="ml-1.5 text-yellow-400">★</span>}</span>
      {stats && <span className="text-xs opacity-70 mt-0.5">MAPE: {stats.average_mape?.toFixed(1)}%</span>}
    </button>
  );
};

const RecBadge = ({rec, size='md'}) => {
  const r = REC[rec] || REC.NEUTRAL;
  const Icon = r.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-bold ${r.bg} ${r.text} border ${r.border} ${size==='sm'?'text-xs':'text-sm'}`}>
      <Icon className="w-4 h-4"/>{r.label}
    </span>
  );
};

const VerdictBadge = ({verdict}) => {
  const v = VERDICT[verdict] || {bg:'bg-gray-600',label:verdict};
  return <span className={`px-2.5 py-1 rounded-lg text-xs font-bold ${v.bg}`}>{v.label}</span>;
};

const ConfidenceBadge = ({confidence}) => {
  const c = CONFIDENCE[confidence] || CONFIDENCE.MEDIUM;
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium ${c.bg} ${c.text} border ${c.border}`}>
      <span>{c.icon}</span>{confidence}
    </span>
  );
};

const StatCard = ({label, value, sub, highlight=false, icon:Icon, trend=null}) => (
  <div className={`p-4 rounded-xl ${highlight ? 'bg-gradient-to-br from-purple-900/40 to-purple-800/20 border border-purple-500/30' : 'bg-gray-700/40'}`}>
    <div className="flex items-center justify-between mb-1">
      <span className="text-xs text-gray-400 uppercase tracking-wide">{label}</span>
      {Icon && <Icon className="w-4 h-4 text-gray-500"/>}
    </div>
    <div className="flex items-baseline gap-2">
      <p className={`text-xl font-bold ${highlight ? 'text-purple-300' : ''}`}>{value}</p>
      {trend !== null && (
        <span className={`text-xs ${trend > 0 ? 'text-red-400' : 'text-green-400'}`}>
          {trend > 0 ? '↑' : '↓'}{Math.abs(trend).toFixed(1)}%
        </span>
      )}
    </div>
    {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
  </div>
);

const ProgressBar = ({value, max=100, color='purple'}) => (
  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
    <div className={`h-full bg-${color}-500 transition-all`} style={{width:`${Math.min(100, (value/max)*100)}%`}}/>
  </div>
);

const Modal = ({open, onClose, title, children, wide=false}) => {
  if (!open) return null;
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className={`bg-gray-800 rounded-2xl ${wide ? 'max-w-5xl' : 'max-w-2xl'} w-full max-h-[90vh] overflow-hidden`} onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold">{title}</h2>
          <button onClick={onClose} className="p-2 hover:bg-gray-700 rounded-lg"><X className="w-5 h-5"/></button>
        </div>
        <div className="p-4 overflow-y-auto max-h-[calc(90vh-4rem)]">{children}</div>
      </div>
    </div>
  );
};

const Tooltip = ({children, text}) => (
  <div className="group relative inline-block">
    {children}
    <div className="invisible group-hover:visible absolute z-10 bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-xs text-gray-200 rounded-lg whitespace-nowrap">
      {text}
      <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900"/>
    </div>
  </div>
);

// ============================================
// HOME SCREEN
// ============================================
const HomeScreen = ({onNavigate}) => {
  const [saved, setSaved] = useState([]);
  const [health, setHealth] = useState(null);
  
  useEffect(() => { 
    API.getAnalyses().then(d => setSaved(d.analyses || [])).catch(()=>{});
    API.health().then(setHealth).catch(()=>{});
  }, []);
  
  return (
    <div className="max-w-4xl mx-auto py-8">
      <div className="text-center mb-12">
        <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl shadow-purple-500/20">
          <Home className="w-10 h-10"/>
        </div>
        <h1 className="text-4xl font-bold mb-2">PropertyCompanion</h1>
        <p className="text-gray-400">Production-grade automated valuation model for UK properties</p>
        <p className="text-gray-500 text-sm mt-1">£150k - £3M | NW London & neighbouring boroughs</p>
        
        {health && (
          <div className="mt-4 inline-flex items-center gap-2 px-3 py-1.5 bg-green-500/10 border border-green-500/30 rounded-full text-sm text-green-400">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"/>
            API Connected • {health.versions_loaded?.length || 0} models loaded
          </div>
        )}
      </div>
      
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <button onClick={() => onNavigate('valuate')} className="group bg-gradient-to-br from-purple-900/50 to-gray-800/50 rounded-2xl p-8 border border-purple-500/20 hover:border-purple-500/50 transition-all text-left">
          <div className="w-14 h-14 bg-purple-500/20 rounded-xl flex items-center justify-center mb-4 group-hover:bg-purple-500/30 transition">
            <Calculator className="w-7 h-7 text-purple-400"/>
          </div>
          <h2 className="text-2xl font-bold mb-2">Valuate Property</h2>
          <p className="text-gray-400 text-sm mb-4">ML predictions with conformal intervals</p>
          <div className="flex flex-wrap gap-2">
            <Badge>Manual/Excel</Badge><Badge>Comparables</Badge><Badge>Mortgage</Badge>
          </div>
        </button>
        
        <button onClick={() => onNavigate('stats')} className="group bg-gradient-to-br from-blue-900/50 to-gray-800/50 rounded-2xl p-8 border border-blue-500/20 hover:border-blue-500/50 transition-all text-left">
          <div className="w-14 h-14 bg-blue-500/20 rounded-xl flex items-center justify-center mb-4 group-hover:bg-blue-500/30 transition">
            <BarChart3 className="w-7 h-7 text-blue-400"/>
          </div>
          <h2 className="text-2xl font-bold mb-2">Model Statistics</h2>
          <p className="text-gray-400 text-sm mb-4">Performance metrics & validation</p>
          <div className="flex flex-wrap gap-2">
            <Badge>Cross-version</Badge><Badge>MAPE</Badge><Badge>Segments</Badge>
          </div>
        </button>
      </div>
      
      {/* Quick Stats */}
      <Card className="mb-8">
        <h3 className="font-semibold mb-4 flex items-center gap-2"><Zap className="w-5 h-5 text-yellow-400"/>Quick Stats</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Models" value="13" sub="Tuned per version"/>
          <StatCard label="Segments" value="5" sub="Price brackets"/>
          <StatCard label="Coverage" value="80%" sub="Conformal interval"/>
          <StatCard label="Regions" value="15" sub="Local authorities"/>
        </div>
      </Card>
      
      {saved.length > 0 && (
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold flex items-center gap-2"><Bookmark className="w-5 h-5 text-purple-400"/>Saved Analyses ({saved.length})</h3>
            <button onClick={() => onNavigate('saved')} className="text-sm text-purple-400 hover:text-purple-300">View all →</button>
          </div>
          <div className="space-y-2">
            {saved.slice(0,3).map(a => (
              <div key={a.id} className="flex items-center justify-between p-3 bg-gray-700/30 rounded-xl">
                <div>
                  <p className="font-medium">{a.name}</p>
                  <p className="text-xs text-gray-400">{a.property_input?.postcode_sector} • {fmtDate(a.timestamp)}</p>
                </div>
                <RecBadge rec={a.analysis?.recommendation} size="sm"/>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

// ============================================
// VALUATION SCREEN
// ============================================
const ValuationScreen = ({onBack}) => {
  const [version, setVersion] = useState('v3');
  const [versionStats, setVersionStats] = useState({});
  const [mode, setMode] = useState('manual');
  const [form, setForm] = useState({
    postcode_sector:'',total_floor_area:'',number_bedrooms:'',number_bathrooms:'',listing_price:'',
    property_type:'House',built_form:'Semi-Detached',tenure_type:'F',old_new:'Y',
    local_authority_label:'Harrow',current_energy_rating:'D'
  });
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [errors, setErrors] = useState({});
  const [saved, setSaved] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [showDetails, setShowDetails] = useState(false);
  
  useEffect(() => {
    // Load stats for all versions
    Promise.all(['v1','v2','v3'].map(v => API.getStats(v).then(s => ({v, s})).catch(() => ({v, s: null}))))
      .then(results => {
        const stats = {};
        results.forEach(({v, s}) => { if(s) stats[v] = s; });
        setVersionStats(stats);
      });
  }, []);
  
  const change = (f,v) => { setForm(p=>({...p,[f]:v})); if(errors[f]) setErrors(p=>({...p,[f]:null})); };
  
  const validate = () => {
    const e = {};
    if(!form.postcode_sector) e.postcode_sector = 'Required';
    if(!form.total_floor_area || +form.total_floor_area <= 0) e.total_floor_area = 'Required';
    if(!form.number_bedrooms || +form.number_bedrooms <= 0) e.number_bedrooms = 'Required';
    if(!form.number_bathrooms || +form.number_bathrooms <= 0) e.number_bathrooms = 'Required';
    if(!form.listing_price || +form.listing_price <= 0) e.listing_price = 'Required';
    setErrors(e);
    return !Object.keys(e).length;
  };
  
  const runManual = async () => {
    if(!validate()) return;
    setLoading(true); setResult(null); setSaved(false);
    try {
      const prop = {
        postcode_sector: form.postcode_sector.toUpperCase(),
        old_new: form.old_new,
        tenure_type: form.tenure_type,
        property_type: form.property_type,
        built_form: form.built_form,
        total_floor_area: +form.total_floor_area,
        local_authority_label: form.local_authority_label,
        number_bedrooms: +form.number_bedrooms,
        number_bathrooms: +form.number_bathrooms,
        current_energy_rating: form.current_energy_rating,
        listing_price: +form.listing_price,
      };
      const res = await API.predict(prop, version);
      setResult({property: prop, ...res});
    } catch(e) {
      setErrors({general:'Prediction failed. Check API connection.'});
    }
    setLoading(false);
  };
  
  const runBatch = async () => {
    if(!file) return;
    setLoading(true); setBatchResults(null);
    try {
      const res = await API.uploadExcel(file, version);
      if(res.error) setErrors({general: res.error});
      else setBatchResults(res);
    } catch(e) {
      setErrors({general:'Batch processing failed.'});
    }
    setLoading(false);
  };
  
  const handleSave = async () => {
    if(!saveName.trim() || !result) return;
    try {
      await API.saveAnalysis({
        name: saveName.trim(),
        analysis: result,
        property_input: result.property,
        version,
        tags: [result.recommendation, result.verdict].filter(Boolean)
      });
      setSaved(true);
      setShowSaveModal(false);
      setSaveName('');
    } catch(e) {}
  };
  
  return (
    <div className="max-w-5xl mx-auto py-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <button onClick={onBack} className="flex items-center gap-2 text-gray-400 hover:text-white transition">
          <ArrowLeft className="w-5 h-5"/>Back
        </button>
        <h1 className="text-xl font-bold">Property Valuation</h1>
        <div className="w-20"/>
      </div>
      
      {/* Version Selection */}
      <div className="flex justify-center gap-3 mb-6">
        {['v1','v2','v3'].map(v => (
          <VersionBadge key={v} v={v} active={version===v} onClick={()=>setVersion(v)} stats={versionStats[v]}/>
        ))}
      </div>
      
      {/* Mode Toggle & Input */}
      <Card className="mb-6">
        <div className="flex gap-2 mb-6">
          <button onClick={()=>setMode('manual')} className={`flex-1 py-3 rounded-xl font-medium transition flex items-center justify-center gap-2 ${mode==='manual'?'bg-purple-600':'bg-gray-700 hover:bg-gray-600'}`}>
            <Calculator className="w-5 h-5"/>Manual Input
          </button>
          <button onClick={()=>setMode('excel')} className={`flex-1 py-3 rounded-xl font-medium transition flex items-center justify-center gap-2 ${mode==='excel'?'bg-purple-600':'bg-gray-700 hover:bg-gray-600'}`}>
            <FileSpreadsheet className="w-5 h-5"/>Excel Upload
          </button>
        </div>
        
        {mode === 'manual' ? (
          <div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
              {[
                {f:'listing_price',l:'Listing Price (£)',t:'number',r:true,icon:PoundSterling},
                {f:'postcode_sector',l:'Postcode Sector',t:'text',r:true,ph:'HA3 8',icon:MapPin},
                {f:'total_floor_area',l:'Floor Area (m²)',t:'number',r:true,icon:Ruler},
                {f:'number_bedrooms',l:'Bedrooms',t:'number',r:true},
                {f:'number_bathrooms',l:'Bathrooms',t:'number',r:true},
              ].map(({f,l,t,r,ph,icon:Icon})=>(
                <div key={f}>
                  <label className="flex items-center gap-1.5 text-sm text-gray-400 mb-1.5">
                    {Icon && <Icon className="w-3.5 h-3.5"/>}{l}{r&&<span className="text-red-400">*</span>}
                  </label>
                  <input type={t} value={form[f]} onChange={e=>change(f,e.target.value)} placeholder={ph||''} 
                    className={`w-full bg-gray-700/50 border ${errors[f]?'border-red-500':'border-gray-600'} rounded-xl px-4 py-2.5 focus:border-purple-500 focus:outline-none transition`}/>
                  {errors[f] && <p className="text-red-400 text-xs mt-1">{errors[f]}</p>}
                </div>
              ))}
              {[
                {f:'property_type',l:'Type',o:TYPES},
                {f:'built_form',l:'Built Form',o:FORMS},
                {f:'tenure_type',l:'Tenure',o:TENURE},
                {f:'old_new',l:'Age',o:AGE},
                {f:'local_authority_label',l:'Local Authority',o:LAS},
                {f:'current_energy_rating',l:'EPC',o:ENERGY.map(e=>({v:e,l:e}))},
              ].map(({f,l,o})=>(
                <div key={f}>
                  <label className="block text-sm text-gray-400 mb-1.5">{l}</label>
                  <select value={form[f]} onChange={e=>change(f,e.target.value)} 
                    className="w-full bg-gray-700/50 border border-gray-600 rounded-xl px-4 py-2.5 focus:border-purple-500 focus:outline-none transition">
                    {o.map(x=>typeof x==='string'?<option key={x} value={x}>{x}</option>:<option key={x.v} value={x.v}>{x.l}</option>)}
                  </select>
                </div>
              ))}
            </div>
            {errors.general && <div className="bg-red-500/10 border border-red-500/50 rounded-xl p-3 mb-4 text-red-400 text-sm flex items-center gap-2"><AlertTriangle className="w-5 h-5"/>{errors.general}</div>}
            <button onClick={runManual} disabled={loading} className="w-full bg-gradient-to-r from-purple-600 to-pink-600 py-3.5 rounded-xl font-semibold flex items-center justify-center gap-2 hover:opacity-90 transition disabled:opacity-50">
              {loading ? <Loader2 className="w-5 h-5 animate-spin"/> : <Calculator className="w-5 h-5"/>}
              {loading ? 'Analyzing...' : 'Get Valuation'}
            </button>
          </div>
        ) : (
          <div>
            <div className="border-2 border-dashed border-gray-600 rounded-xl p-10 text-center mb-4 hover:border-purple-500/50 transition">
              <Upload className="w-12 h-12 text-gray-500 mx-auto mb-4"/>
              <p className="text-gray-300 mb-2">Upload Excel/CSV with listings</p>
              <p className="text-xs text-gray-500 mb-4">Required: POSTCODE_SECTOR, PROPERTY_TYPE, BUILT_FORM, TOTAL_FLOOR_AREA,<br/>NUMBER_BEDROOMS, NUMBER_BATHROOMS, LISTING_PRICE</p>
              <input type="file" accept=".xlsx,.xls,.csv" onChange={e=>setFile(e.target.files[0])} className="hidden" id="file"/>
              <label htmlFor="file" className="px-6 py-2.5 bg-purple-600 hover:bg-purple-500 rounded-xl cursor-pointer inline-block transition">Select File</label>
              {file && <p className="mt-4 text-green-400 text-sm flex items-center justify-center gap-2"><CheckCircle className="w-4 h-4"/>{file.name}</p>}
            </div>
            <button onClick={runBatch} disabled={loading||!file} className="w-full bg-gradient-to-r from-purple-600 to-pink-600 py-3.5 rounded-xl font-semibold flex items-center justify-center gap-2 hover:opacity-90 transition disabled:opacity-50">
              {loading ? <Loader2 className="w-5 h-5 animate-spin"/> : <FileSpreadsheet className="w-5 h-5"/>}
              {loading ? 'Processing...' : 'Analyze All'}
            </button>
          </div>
        )}
      </Card>
      
      {/* SINGLE RESULT */}
      {result && mode==='manual' && (
        <div className="space-y-4">
          {/* Header Card */}
          <Card className="border border-gray-700/50">
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <MapPin className="w-5 h-5 text-purple-400"/>
                  <h2 className="text-xl font-bold">{result.property.postcode_sector}</h2>
                  <Badge type="purple">{version.toUpperCase()}</Badge>
                </div>
                <p className="text-gray-400 text-sm">
                  {result.property.number_bedrooms} bed • {result.property.number_bathrooms} bath • {result.property.total_floor_area}m² • {result.property.built_form} {result.property.property_type}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={()=>setShowSaveModal(true)} disabled={saved} className={`p-2 rounded-lg transition ${saved?'bg-green-600/20 text-green-400':'bg-gray-700 hover:bg-gray-600'}`}>
                  {saved ? <Bookmark className="w-5 h-5"/> : <BookmarkPlus className="w-5 h-5"/>}
                </button>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <VerdictBadge verdict={result.verdict}/>
              <RecBadge rec={result.recommendation}/>
              <ConfidenceBadge confidence={result.confidence}/>
              <Badge type="info">{result.segment}</Badge>
            </div>
          </Card>
          
          {/* Main Valuation */}
          <Card className={`border-2 ${result.recommendation==='STRONG_BUY'||result.recommendation==='BUY'?'border-green-500/30 bg-green-500/5':result.recommendation==='AVOID'?'border-red-500/30 bg-red-500/5':result.recommendation==='NEGOTIATE'?'border-amber-500/30 bg-amber-500/5':'border-blue-500/30 bg-blue-500/5'}`}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <StatCard label="Listing Price" value={`£${fmt(result.property.listing_price)}`}/>
              <StatCard label="ML Prediction" value={`£${fmt(Math.round(result.point_estimate))}`} highlight icon={Target}/>
              <StatCard label="Difference" value={`${result.diff_pct>0?'+':''}${result.diff_pct?.toFixed(1)}%`} sub={`£${fmt(Math.abs(Math.round(result.diff||0)))}`} trend={result.diff_pct}/>
              <StatCard label="Price/m²" value={`£${fmt(Math.round(result.property.listing_price/result.property.total_floor_area))}`}/>
            </div>
            
            {/* Confidence Interval */}
            <div className="bg-gray-900/50 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Shield className="w-4 h-4 text-blue-400"/>
                  <span className="text-sm font-medium">{((result.conformal_interval?.coverage||0.8)*100).toFixed(0)}% Confidence Interval</span>
                </div>
                <span className={`text-xs px-2 py-1 rounded ${result.within_ci ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                  {result.within_ci ? '✓ Within CI' : '⚠ Outside CI'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div><p className="text-xs text-gray-400">Lower</p><p className="font-bold">£{fmt(Math.round(result.conformal_interval?.lower||0))}</p></div>
                <div className="bg-purple-500/10 rounded-lg py-2"><p className="text-xs text-gray-400">Estimate</p><p className="font-bold text-purple-400">£{fmt(Math.round(result.point_estimate))}</p></div>
                <div><p className="text-xs text-gray-400">Upper</p><p className="font-bold">£{fmt(Math.round(result.conformal_interval?.upper||0))}</p></div>
              </div>
              <div className="mt-3">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>£{fmt(Math.round(result.conformal_interval?.lower||0))}</span>
                  <span>£{fmt(Math.round(result.conformal_interval?.upper||0))}</span>
                </div>
                <div className="h-2 bg-gray-700 rounded-full relative">
                  <div className="absolute h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full" style={{left:'10%',right:'10%'}}/>
                  <div className="absolute w-1 h-4 bg-white rounded -top-1" style={{left:`${Math.min(90,Math.max(10,50 + (result.diff_pct||0)))}%`}}/>
                </div>
                <p className="text-xs text-center text-gray-500 mt-1">Listing position</p>
              </div>
            </div>
          </Card>
          
          {/* Confidence Reasons */}
          {result.confidence_reasons?.length > 0 && (
            <Card>
              <h3 className="font-semibold mb-3 flex items-center gap-2"><Info className="w-5 h-5 text-blue-400"/>Confidence Analysis</h3>
              <div className="space-y-2">
                {result.confidence_reasons.map((r,i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0"/>
                    <span className="text-gray-300">{r}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
          
          {/* All Model Predictions */}
          <Card>
            <button onClick={()=>setShowDetails(!showDetails)} className="w-full flex items-center justify-between">
              <h3 className="font-semibold flex items-center gap-2"><Activity className="w-5 h-5 text-purple-400"/>All Model Predictions</h3>
              {showDetails ? <ChevronUp className="w-5 h-5"/> : <ChevronDown className="w-5 h-5"/>}
            </button>
            {showDetails && result.predictions && (
              <div className="mt-4 space-y-2">
                {Object.entries(result.predictions).map(([name, pred]) => (
                  <div key={name} className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                    <span className="text-sm">{name}</span>
                    <span className="font-bold">£{fmt(Math.round(pred))}</span>
                  </div>
                ))}
                <div className="pt-2 border-t border-gray-700">
                  <div className="flex items-center justify-between text-sm text-gray-400">
                    <span>Model Agreement (CV)</span>
                    <span>{result.cv?.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            )}
          </Card>
          
          {/* Mortgage */}
          {result.mortgage && (
            <Card>
              <h3 className="font-semibold mb-4 flex items-center gap-2"><PoundSterling className="w-5 h-5 text-amber-400"/>Mortgage ({result.mortgage.ltv_pct}% LTV @ {result.mortgage.rate_pct}%)</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <StatCard label="Deposit" value={`£${fmt(result.mortgage.deposit)}`}/>
                <StatCard label="Loan" value={`£${fmt(result.mortgage.loan_amount)}`}/>
                <StatCard label="Monthly" value={`£${fmt(result.mortgage.monthly_payment)}`} highlight/>
                <StatCard label="Total Interest" value={`£${fmt(result.mortgage.total_interest)}`} sub={`${result.mortgage.term_years} years`}/>
              </div>
            </Card>
          )}
          
          {/* Comparables */}
          {result.comparables?.length > 0 && (
            <Card>
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold flex items-center gap-2"><Building className="w-5 h-5 text-purple-400"/>Comparables</h3>
                {result.comparables_valuation?.has_valuation && (
                  <span className="text-sm"><span className="text-gray-400">Comp avg:</span> <span className="font-bold text-purple-400">£{fmt(Math.round(result.comparables_valuation.weighted_avg))}</span></span>
                )}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-gray-700 text-gray-400 text-xs">
                    <th className="text-left py-2 px-2">#</th><th className="text-left py-2 px-2">Match</th><th className="text-left py-2 px-2">Postcode</th>
                    <th className="text-left py-2 px-2">Date</th><th className="text-right py-2 px-2">Sold</th><th className="text-right py-2 px-2">Adj.</th>
                    <th className="text-right py-2 px-2">£/m²</th><th className="text-center py-2 px-2">Bed</th><th className="text-right py-2 px-2">Area</th>
                  </tr></thead>
                  <tbody>
                    {result.comparables.slice(0,10).map((c,i)=>(
                      <tr key={i} className={`border-b border-gray-700/30 ${c.used?'bg-purple-500/5':''}`}>
                        <td className="py-2.5 px-2">{c.used?'★':''}{i+1}</td>
                        <td className="py-2.5 px-2"><span className={`px-2 py-0.5 rounded text-xs ${c.similarity>=85?'bg-green-500/20 text-green-400':c.similarity>=75?'bg-amber-500/20 text-amber-400':'bg-gray-700 text-gray-400'}`}>{c.similarity?.toFixed(0)}%</span></td>
                        <td className="py-2.5 px-2">{c.postcode}</td>
                        <td className="py-2.5 px-2 text-gray-400">{c.date ? fmtDate(c.date) : `${c.months_ago}mo ago`}</td>
                        <td className="py-2.5 px-2 text-right">£{fmt(c.original_price)}</td>
                        <td className="py-2.5 px-2 text-right text-purple-400">£{fmt(c.adjusted_price)}</td>
                        <td className="py-2.5 px-2 text-right">£{fmt(Math.round(c.adjusted_psm))}</td>
                        <td className="py-2.5 px-2 text-center">{c.bedrooms||'?'}</td>
                        <td className="py-2.5 px-2 text-right">{c.floor_area}m²</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
          
          {/* Next Steps & Negotiation */}
          <Card className={`border-2 ${REC[result.recommendation]?.border||'border-gray-600'}`}>
            <div className="flex items-center gap-3 mb-4">
              <RecBadge rec={result.recommendation}/>
              <span className="text-gray-300">{result.recommendation_detail}</span>
            </div>
            
            {result.negotiation && (
              <div className="bg-gray-900/50 rounded-xl p-4 mb-4">
                <h4 className="font-medium mb-3 flex items-center gap-2">💰 Negotiation Strategy</h4>
                <div className="grid grid-cols-4 gap-3 text-center">
                  <div><p className="text-xs text-gray-400">Opening</p><p className="text-green-400 font-bold">£{fmt(result.negotiation.opening)}</p></div>
                  <div><p className="text-xs text-gray-400">Target</p><p className="text-amber-400 font-bold">£{fmt(result.negotiation.target)}</p></div>
                  <div><p className="text-xs text-gray-400">Maximum</p><p className="text-orange-400 font-bold">£{fmt(result.negotiation.maximum)}</p></div>
                  <div><p className="text-xs text-gray-400">Walk Away</p><p className="text-red-400 font-bold">£{fmt(result.negotiation.walkaway)}</p></div>
                </div>
              </div>
            )}
            
            {result.next_steps?.length > 0 && (
              <div>
                <h4 className="font-medium mb-3">📋 Next Steps</h4>
                <ul className="space-y-2">
                  {result.next_steps.map((s,i)=><li key={i} className="flex items-start gap-2 text-gray-300 text-sm"><span className="text-purple-400 mt-0.5">•</span>{s}</li>)}
                </ul>
              </div>
            )}
          </Card>
        </div>
      )}
      
      {/* BATCH RESULTS */}
      {batchResults && mode==='excel' && (
        <div className="space-y-4">
          {/* Analysis Summary */}
          {batchResults.analysis && (
            <Card>
              <h3 className="font-semibold mb-4 flex items-center gap-2"><PieChart className="w-5 h-5 text-blue-400"/>Analysis Summary</h3>
              <div className="grid grid-cols-5 gap-3 mb-4">
                {['STRONG_BUY','BUY','NEUTRAL','NEGOTIATE','AVOID'].map(r=>(
                  <div key={r} className={`p-3 rounded-xl text-center ${REC[r]?.bg} border ${REC[r]?.border}`}>
                    <p className="text-2xl font-bold">{batchResults.analysis.recommendations?.[r]||0}</p>
                    <p className="text-xs text-gray-400">{r.replace('_',' ')}</p>
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-700/30 rounded-xl p-3">
                  <p className="text-sm text-gray-400 mb-1">High Confidence Opportunities</p>
                  <p className="text-lg"><span className="text-green-400 font-bold">{batchResults.analysis.opportunities?.undervalued_high_conf||0}</span> undervalued • <span className="text-red-400 font-bold">{batchResults.analysis.opportunities?.overvalued_high_conf||0}</span> overvalued</p>
                </div>
                <div className="bg-gray-700/30 rounded-xl p-3">
                  <p className="text-sm text-gray-400 mb-1">Bias</p>
                  <p className="text-lg">Mean: <span className="font-bold">{batchResults.analysis.bias?.mean_diff_pct?.toFixed(1)}%</span> • Median: <span className="font-bold">{batchResults.analysis.bias?.median_diff_pct?.toFixed(1)}%</span></p>
                </div>
              </div>
            </Card>
          )}
          
          {/* Top Opportunities */}
          {batchResults.analysis?.opportunities?.undervalued_listings?.length > 0 && (
            <Card>
              <h3 className="font-semibold mb-4 flex items-center gap-2"><TrendingUp className="w-5 h-5 text-green-400"/>🟢 High Confidence Undervalued</h3>
              <div className="space-y-2">
                {batchResults.analysis.opportunities.undervalued_listings.slice(0,5).map((o,i)=>(
                  <div key={i} className="flex items-center justify-between p-3 bg-green-500/10 rounded-xl border border-green-500/20">
                    <div><span className="font-medium">{o.postcode}</span><span className="text-gray-400 ml-3">£{fmt(o.listing)}</span></div>
                    <div className="flex items-center gap-3">
                      <span className="text-purple-400">→ £{fmt(Math.round(o.predicted))}</span>
                      <Badge type="success">{o.diff_pct?.toFixed(1)}%</Badge>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
          
          {/* Results Table */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold">All Results ({batchResults.successful}/{batchResults.total})</h3>
              <button className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm transition">
                <Download className="w-4 h-4"/>Export
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="border-b border-gray-700 text-gray-400 text-xs">
                  <th className="text-left py-2 px-2">#</th><th className="text-left py-2 px-2">Postcode</th><th className="text-left py-2 px-2">Type</th>
                  <th className="text-right py-2 px-2">Listing</th><th className="text-right py-2 px-2">Predicted</th><th className="text-right py-2 px-2">Diff</th>
                  <th className="text-center py-2 px-2">Verdict</th><th className="text-center py-2 px-2">Action</th>
                </tr></thead>
                <tbody>
                  {batchResults.results?.map((r,i)=>(
                    <tr key={i} className="border-b border-gray-700/30">
                      <td className="py-2.5 px-2">{r.row+1}</td>
                      <td className="py-2.5 px-2">{r.postcode||'Error'}</td>
                      <td className="py-2.5 px-2 text-gray-400">{r.property_type}</td>
                      <td className="py-2.5 px-2 text-right">{r.success?`£${fmt(r.listing_price)}`:'-'}</td>
                      <td className="py-2.5 px-2 text-right text-purple-400">{r.success?`£${fmt(Math.round(r.predicted))}`:'-'}</td>
                      <td className={`py-2.5 px-2 text-right ${r.diff_pct>0?'text-red-400':'text-green-400'}`}>{r.success?`${r.diff_pct>0?'+':''}${r.diff_pct?.toFixed(1)}%`:'-'}</td>
                      <td className="py-2.5 px-2 text-center">{r.success?<VerdictBadge verdict={r.verdict}/>:<span className="text-red-400 text-xs">Error</span>}</td>
                      <td className="py-2.5 px-2 text-center">{r.success?<RecBadge rec={r.recommendation} size="sm"/>:'-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
      
      {/* Save Modal */}
      <Modal open={showSaveModal} onClose={()=>setShowSaveModal(false)} title="Save Analysis">
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1.5">Name</label>
            <input type="text" value={saveName} onChange={e=>setSaveName(e.target.value)} placeholder={`${result?.property?.postcode_sector || 'Property'} Analysis`}
              className="w-full bg-gray-700 border border-gray-600 rounded-xl px-4 py-2.5 focus:border-purple-500 focus:outline-none"/>
          </div>
          <button onClick={handleSave} className="w-full bg-purple-600 hover:bg-purple-500 py-3 rounded-xl font-semibold transition">
            Save Analysis
          </button>
        </div>
      </Modal>
    </div>
  );
};

// ============================================
// STATS DASHBOARD
// ============================================
const StatsDashboard = ({onBack}) => {
  const [version, setVersion] = useState('v3');
  const [stats, setStats] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    setLoading(true);
    Promise.all([API.getStats(version), API.getStatsSummary()])
      .then(([s, sum]) => { setStats(s); setSummary(sum); })
      .finally(() => setLoading(false));
  }, [version]);
  
  if(loading) return <div className="flex items-center justify-center h-96"><Loader2 className="w-8 h-8 animate-spin text-purple-400"/></div>;
  
  return (
    <div className="max-w-5xl mx-auto py-6">
      <div className="flex items-center justify-between mb-6">
        <button onClick={onBack} className="flex items-center gap-2 text-gray-400 hover:text-white transition"><ArrowLeft className="w-5 h-5"/>Back</button>
        <h1 className="text-xl font-bold">Model Statistics</h1>
        <div className="w-20"/>
      </div>
      
      <div className="flex justify-center gap-3 mb-6">
        {['v1','v2','v3'].map(v => <VersionBadge key={v} v={v} active={version===v} onClick={()=>setVersion(v)}/>)}
      </div>
      
      {stats && (
        <>
          {/* Overview */}
          <Card className="mb-6">
            <h3 className="font-semibold mb-4">Overview</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Models Loaded" value={stats.n_models} icon={Activity}/>
              <StatCard label="Average MAPE" value={stats.average_mape ? `${stats.average_mape.toFixed(2)}%` : 'N/A'} highlight icon={Target}/>
              <StatCard label="Quantile Models" value={stats.has_quantile ? 'Yes' : 'No'} icon={BarChart3}/>
              <StatCard label="Conformal" value={stats.has_conformal ? 'Yes' : 'No'} icon={Shield}/>
            </div>
          </Card>
          
          {/* Segment MAPEs */}
          {stats.segment_mapes && (
            <Card className="mb-6">
              <h3 className="font-semibold mb-4">MAPE by Segment</h3>
              <div className="space-y-4">
                {Object.entries(stats.segment_mapes).map(([opt, mapes]) => (
                  <div key={opt} className="p-4 bg-gray-700/30 rounded-xl">
                    <h4 className="font-medium mb-3">Option {opt}</h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                      {Object.entries(mapes).map(([seg, mape]) => (
                        <div key={seg} className="bg-gray-800/50 rounded-lg p-3">
                          <p className="text-xs text-gray-400 truncate">{seg}</p>
                          <p className={`font-bold ${mape < 8 ? 'text-green-400' : mape < 12 ? 'text-amber-400' : 'text-red-400'}`}>
                            {typeof mape === 'number' ? mape.toFixed(2) : mape}%
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
          
          {/* Cross-Version Comparison */}
          {summary && (
            <Card>
              <h3 className="font-semibold mb-4">Cross-Version Comparison</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-3">Version</th>
                      <th className="text-center py-3 px-3">Models</th>
                      <th className="text-center py-3 px-3">Avg MAPE</th>
                      <th className="text-center py-3 px-3">Quantile</th>
                      <th className="text-center py-3 px-3">Conformal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(summary).map(([v, s]) => (
                      <tr key={v} className={`border-b border-gray-700/30 ${v === version ? 'bg-purple-500/10' : ''}`}>
                        <td className="py-3 px-3 font-medium">{v.toUpperCase()}</td>
                        <td className="py-3 px-3 text-center">{s.n_models || '-'}</td>
                        <td className="py-3 px-3 text-center">
                          {s.average_mape ? (
                            <span className={s.average_mape < 8 ? 'text-green-400' : s.average_mape < 12 ? 'text-amber-400' : 'text-red-400'}>
                              {s.average_mape.toFixed(2)}%
                            </span>
                          ) : '-'}
                        </td>
                        <td className="py-3 px-3 text-center">{s.has_quantile ? <CheckCircle className="w-5 h-5 text-green-400 mx-auto"/> : <XCircle className="w-5 h-5 text-gray-500 mx-auto"/>}</td>
                        <td className="py-3 px-3 text-center">{s.has_conformal ? <CheckCircle className="w-5 h-5 text-green-400 mx-auto"/> : <XCircle className="w-5 h-5 text-gray-500 mx-auto"/>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
};

// ============================================
// SAVED SCREEN
// ============================================
const SavedScreen = ({onBack}) => {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    API.getAnalyses().then(d => setAnalyses(d.analyses || [])).finally(() => setLoading(false));
  }, []);
  
  const handleDelete = async id => {
    await API.deleteAnalysis(id);
    setAnalyses(p => p.filter(a => a.id !== id));
  };
  
  if(loading) return <div className="flex items-center justify-center h-96"><Loader2 className="w-8 h-8 animate-spin text-purple-400"/></div>;
  
  return (
    <div className="max-w-5xl mx-auto py-6">
      <div className="flex items-center justify-between mb-6">
        <button onClick={onBack} className="flex items-center gap-2 text-gray-400 hover:text-white transition"><ArrowLeft className="w-5 h-5"/>Back</button>
        <h1 className="text-xl font-bold">Saved Analyses</h1>
        <div className="w-20"/>
      </div>
      
      {analyses.length === 0 ? (
        <Card className="text-center py-12">
          <Bookmark className="w-12 h-12 text-gray-600 mx-auto mb-4"/>
          <p className="text-gray-400">No saved analyses yet</p>
        </Card>
      ) : (
        <div className="space-y-3">
          {analyses.map(a => (
            <div key={a.id} className="flex items-center gap-4 p-4 bg-gray-800/80 rounded-xl border border-transparent hover:border-gray-700 transition">
              <div className="flex-1">
                <p className="font-medium">{a.name}</p>
                <p className="text-sm text-gray-400">{a.property_input?.postcode_sector} • {a.property_input?.property_type} • £{fmt(a.property_input?.listing_price)}</p>
                <p className="text-xs text-gray-500">{fmtDate(a.timestamp)} • {a.version}</p>
              </div>
              <RecBadge rec={a.analysis?.recommendation} size="sm"/>
              <button onClick={()=>handleDelete(a.id)} className="p-2 text-gray-400 hover:text-red-400 transition"><Trash2 className="w-5 h-5"/></button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ============================================
// MAIN APP
// ============================================
export default function PropertyCompanion() {
  const [screen, setScreen] = useState('home');
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-900 to-gray-800 text-white">
      <div className="p-4 md:p-6">
        {screen === 'home' && <HomeScreen onNavigate={setScreen}/>}
        {screen === 'valuate' && <ValuationScreen onBack={()=>setScreen('home')}/>}
        {screen === 'stats' && <StatsDashboard onBack={()=>setScreen('home')}/>}
        {screen === 'saved' && <SavedScreen onBack={()=>setScreen('home')}/>}
      </div>
    </div>
  );
}
