#!/usr/bin/env python3
"""
JPROP Video Analyzer — powered by Google Gemini (free tier)
Analyzes the FULL video including audio/speech against Meta Andromeda criteria.
No frame extraction needed — Gemini reads the entire video natively.
"""
import os, json, tempfile, logging, re, time
from flask import Flask, request, jsonify, render_template_string
import google.generativeai as genai

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MIME_TYPES = {
    ".mp4": "video/mp4", ".mov": "video/quicktime",
    ".avi": "video/avi", ".mkv": "video/x-matroska",
    ".webm": "video/webm", ".3gp": "video/3gpp",
}

# ── Analysis prompt ─────────────────────────────────────────────────

ANALYSIS_PROMPT = """You are an expert Meta ads creative analyst specialising in Andromeda's AI retrieval system for real estate advertising in Malaysia.

Watch this video in full — including visuals, text overlays, and spoken audio.

Andromeda is Meta's deep neural network retrieval engine. It reads the SEMANTIC meaning of your ad creative (what you show, what you say, what emotion you trigger) and uses it to match your ad to the right users BEFORE the auction. A weak semantic signal = wrong audience = high CPM = high CPL.

ANALYZE AND SCORE:

1. HOOK SCORE (first 3 seconds) — Does it stop the scroll?
   10: Strong pattern interrupt (bold statement, shocking visual, direct question)
   7-9: Clear, engaging opening with curiosity or emotion
   5-6: Average, not immediately compelling
   3-4: Slow start, generic opening
   1-2: No hook at all

2. ANDROMEDA SIGNAL SCORE — How clearly does Andromeda know who to show this to?
   10: Crystal clear — product, audience, location, price point all signalled
   7-9: Clear product/service and audience signal
   5-6: Somewhat clear, some mixed signals
   3-4: Confusing offer or audience
   1-2: No clear semantic signal

3. AUDIO/SPEECH HOOK (first 3 seconds spoken words) — Rate the opening line quality
   What was said? Was it compelling? Does it qualify or disqualify viewers?

4. CPM PREDICTION (how expensive will delivery be?)
   Low RM20-50: Very clear signal, Andromeda finds large cheap audience
   Medium RM50-100: Decent signal, manageable cost
   High RM100-180: Weak or mixed signal
   Very High RM180+: Poor signal, Andromeda struggles to match

5. CPL PREDICTION (cost per lead)
   Low RM30-80: Strong hook + clear offer + obvious CTA
   Medium RM80-150: Some friction
   High RM150-250: Weak hook or unclear offer
   Very High RM250+: Multiple issues

6. BUYER PERSONA PREDICTION — Based purely on what this video shows and says, who is the human being most likely to stop scrolling, watch, and submit a lead form?

   This is what Andromeda reads from your creative to find its audience. Be specific:
   - Age range and gender skew
   - Monthly household income (RM)
   - Life stage (e.g. fresh graduate, young couple, growing family, investor)
   - Location and commute mindset
   - Core pain point (what problem are they trying to solve RIGHT NOW)
   - Primary motivation (investment return? family space? first home? status?)
   - Property type they're mentally shopping for
   - Budget range implied by the creative signals
   - Risk: Does the creative accidentally attract the WRONG buyer? (e.g. lifestyle video attracts dreamers not buyers)

7. META INTEREST TARGETING MATCH — Based on the video's content, list the Meta interest/behaviour categories that align with this buyer. These are searchable in Meta Ads Manager Detailed Targeting.
   Format: category name + why this video signals it
   Include 5-8 categories across: interests, behaviours, demographics

8. TOP 3 SPECIFIC IMPROVEMENTS — be precise, actionable, reference actual moments in the video

9. VERDICT:
   "Ready to publish" — Strong across all metrics
   "Test first (ABO RM30/day)" — Promising but needs validation
   "Needs rework" — Specific fixable issues
   "Do not run" — Fundamental creative problems

Return ONLY valid JSON, no markdown, no extra text:
{
    "hook_score": 7,
    "hook_feedback": "What works and what doesn't in the first 3 seconds visually",
    "hook_tip": "One specific change to make the hook stronger",
    "audio_hook": "Exact quote or description of opening spoken words",
    "audio_hook_score": 6,
    "audio_hook_feedback": "Was the opening line compelling? Does it qualify viewers?",
    "andromeda_score": 6,
    "andromeda_feedback": "What semantic signal this sends to Andromeda",
    "andromeda_audience": "One-line summary of who Andromeda will match this ad to",
    "cpm_prediction": "Medium (RM60-90)",
    "cpm_reason": "Why this CPM range — reference specific creative elements",
    "cpl_prediction": "High (RM130-180)",
    "cpl_reason": "Why this CPL range — reference specific creative elements",
    "buyer_persona": {
        "age_range": "30-42",
        "gender_skew": "Mixed, slight male skew",
        "income_monthly_rm": "RM6,000 - RM12,000",
        "life_stage": "Young family, first or second property",
        "location_mindset": "Based in KL/PJ, willing to move to Klang Valley suburbs",
        "pain_point": "Paying rent but no equity, needs more space for growing family",
        "motivation": "Long-term asset, family upgrade",
        "property_mindset": "3-bedroom landed or large condo, below RM600k",
        "implied_budget": "RM400k - RM650k",
        "persona_mismatch_risk": "Lifestyle visuals may attract aspirational viewers who cannot afford — recommend adding price anchor or monthly instalment in video"
    },
    "meta_targeting": [
        {"category": "Property investment", "type": "Interest", "reason": "Video mentions ROI and rental yield"},
        {"category": "First home buyer Malaysia", "type": "Interest", "reason": "Dialogue references stamp duty exemption"},
        {"category": "Likely to move", "type": "Behaviour", "reason": "Creative addresses people considering relocation"},
        {"category": "Parents of young children", "type": "Demographic", "reason": "Family lifestyle footage in video"},
        {"category": "Home ownership", "type": "Interest", "reason": "Core theme of the creative"}
    ],
    "strengths": [
        "Specific strength 1",
        "Specific strength 2"
    ],
    "improvements": [
        "Specific improvement 1 with timestamp or reference",
        "Specific improvement 2 with timestamp or reference",
        "Specific improvement 3 with timestamp or reference"
    ],
    "verdict": "Test first (ABO RM30/day)",
    "verdict_reason": "One clear sentence why"
}"""


# ── Gemini analysis ─────────────────────────────────────────────────

def analyze_with_gemini(video_path: str, mime_type: str) -> dict:
    genai.configure(api_key=GEMINI_API_KEY)

    log.info("Uploading video to Gemini...")
    video_file = genai.upload_file(video_path, mime_type=mime_type)

    # Wait for Gemini to process the video
    wait = 0
    while video_file.state.name == "PROCESSING":
        time.sleep(3)
        wait += 3
        video_file = genai.get_file(video_file.name)
        log.info(f"Processing... {wait}s elapsed")
        if wait > 180:
            raise TimeoutError("Video processing timed out after 3 minutes")

    if video_file.state.name == "FAILED":
        raise ValueError("Gemini failed to process the video. Check format.")

    log.info("Video ready. Running analysis...")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        [video_file, ANALYSIS_PROMPT],
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=3500,
        )
    )

    # Clean up uploaded file from Gemini
    try:
        genai.delete_file(video_file.name)
    except Exception:
        pass

    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not GEMINI_API_KEY:
        return jsonify({"error": "GEMINI_API_KEY not configured on server"}), 500

    ext = os.path.splitext(f.filename)[1].lower()
    mime_type = MIME_TYPES.get(ext, "video/mp4")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        video_path = tmp.name
        f.save(video_path)

    file_size_mb = os.path.getsize(video_path) / 1024 / 1024
    log.info(f"Analyzing: {f.filename} ({file_size_mb:.1f}MB)")

    try:
        result = analyze_with_gemini(video_path, mime_type)
        result["filename"] = f.filename
        result["file_size_mb"] = round(file_size_mb, 1)
        return jsonify(result)
    except json.JSONDecodeError:
        return jsonify({"error": "Analysis returned invalid format. Try again."}), 500
    except Exception as e:
        log.error(f"Analysis error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/health")
def health():
    return "OK"


# ── UI ───────────────────────────────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JPROP Video Analyzer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e4e6ea; min-height: 100vh; }
  .container { max-width: 800px; margin: 0 auto; padding: 40px 20px; }
  h1 { font-size: 26px; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: #8a8d94; font-size: 14px; margin-bottom: 36px; }
  .badge { display:inline-block; background:#0d2818; color:#34c759; border:1px solid #34c759;
           border-radius:6px; padding:2px 8px; font-size:11px; margin-left:8px; vertical-align:middle; }

  .upload-box { border: 2px dashed #2e3140; border-radius: 16px; padding: 48px 32px;
                text-align: center; cursor: pointer; transition: all 0.2s; background: #161820; }
  .upload-box:hover, .upload-box.drag { border-color: #4f6ef7; background: #1a1d2e; }
  .upload-box .icon { font-size: 48px; margin-bottom: 12px; }
  .upload-box h2 { font-size: 18px; margin-bottom: 8px; }
  .upload-box p { color: #8a8d94; font-size: 13px; }
  #fileInput { display: none; }

  .btn { background: #4f6ef7; color: white; border: none; border-radius: 10px;
         padding: 14px 32px; font-size: 15px; font-weight: 600; cursor: pointer;
         margin-top: 20px; transition: background 0.2s; width: 100%; }
  .btn:hover { background: #3d5ae0; }
  .btn:disabled { background: #2e3140; cursor: not-allowed; }

  .progress { display: none; margin-top: 28px; text-align: center; }
  .progress-bar-bg { background: #2e3140; border-radius: 99px; height: 6px; margin: 12px 0; }
  .progress-bar { background: #4f6ef7; height: 6px; border-radius: 99px; width: 0%; transition: width 1s ease; }
  .progress-text { color: #8a8d94; font-size: 13px; }
  .spinner { display:inline-block; width:20px; height:20px; border:2px solid #2e3140;
             border-top-color:#4f6ef7; border-radius:50%; animation:spin 0.8s linear infinite; margin-bottom:12px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .results { display: none; margin-top: 36px; }
  .result-meta { color: #8a8d94; font-size: 13px; margin-bottom: 20px; }
  .result-meta strong { color: #e4e6ea; }

  .scores-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px; }
  @media(max-width:500px) { .scores-grid { grid-template-columns: 1fr; } }

  .score-card { background: #161820; border-radius: 12px; padding: 18px; border: 1px solid #2e3140; }
  .score-card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #8a8d94; margin-bottom: 8px; }
  .score-card .value { font-size: 30px; font-weight: 700; }
  .score-card .sub-value { font-size:17px; font-weight:700; margin-top:4px; }
  .score-card .sub { font-size: 12px; color: #8a8d94; margin-top: 6px; line-height:1.5; }
  .score-bar { background: #2e3140; border-radius: 99px; height: 4px; margin: 8px 0 4px; }
  .score-bar-fill { height: 4px; border-radius: 99px; }

  .green { color: #34c759; } .yellow { color: #ffd60a; } .orange { color: #ff9500; } .red { color: #ff3b30; }
  .bg-green { background: #34c759; } .bg-yellow { background: #ffd60a; }
  .bg-orange { background: #ff9500; } .bg-red { background: #ff3b30; }

  .section { background: #161820; border-radius: 12px; padding: 20px; border: 1px solid #2e3140; margin-bottom: 14px; }
  .section h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #8a8d94; margin-bottom: 12px; }
  .section p { font-size: 14px; line-height: 1.6; }
  .highlight { background: #1e2236; border-radius: 8px; padding: 10px 14px; margin-top: 10px;
               font-size: 13px; color: #a0c4ff; font-style: italic; }

  .list-items { list-style: none; }
  .list-items li { padding: 9px 0; border-bottom: 1px solid #2e3140; font-size: 14px; line-height: 1.5; }
  .list-items li:last-child { border-bottom: none; }
  .improvement li::before { content: "→ "; color: #4f6ef7; font-weight:700; }
  .strength li::before { content: "✓ "; color: #34c759; font-weight:700; }

  .verdict-box { border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  .verdict-ready  { background:#0d2818; border:1px solid #34c759; }
  .verdict-test   { background:#1c1a0a; border:1px solid #ffd60a; }
  .verdict-rework { background:#1c100a; border:1px solid #ff9500; }
  .verdict-no     { background:#1c0a0a; border:1px solid #ff3b30; }
  .verdict-box h3 { font-size:17px; font-weight:700; margin-bottom:6px; }
  .verdict-box p  { font-size:14px; opacity:0.85; line-height:1.5; }

  .error-box { background:#1c0a0a; border:1px solid #ff3b30; border-radius:12px;
               padding:20px; margin-top:20px; display:none; }
  .error-box p { color:#ff6b6b; font-size:14px; }
  .try-again { background:#2e3140; color:#e4e6ea; border:none; border-radius:10px;
               padding:12px 24px; font-size:14px; cursor:pointer; margin-top:20px; }
  .try-again:hover { background:#3a3f52; }
  .divider { border: none; border-top: 1px solid #2e3140; margin: 16px 0; }
</style>
</head>
<body>
<div class="container">
  <h1>🎬 Video Analyzer <span class="badge">FREE</span></h1>
  <p class="subtitle">Full video analysis — visuals + audio · Scored against Meta Andromeda criteria</p>

  <div class="upload-box" id="uploadBox" onclick="document.getElementById('fileInput').click()">
    <div class="icon">📹</div>
    <h2 id="uploadTitle">Drop your video here</h2>
    <p id="uploadSub">MP4, MOV, AVI, MKV · Max 200MB</p>
    <input type="file" id="fileInput" accept="video/*">
  </div>

  <button class="btn" id="analyzeBtn" disabled onclick="analyzeVideo()">Choose a video to analyze</button>

  <div class="progress" id="progress">
    <div class="spinner"></div>
    <div class="progress-bar-bg"><div class="progress-bar" id="progressBar"></div></div>
    <p class="progress-text" id="progressText">Uploading video...</p>
  </div>

  <div class="error-box" id="errorBox"><p id="errorMsg"></p></div>
  <div class="results" id="results"></div>
</div>

<script>
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadBox = document.getElementById('uploadBox');
let selectedFile = null;

fileInput.addEventListener('change', e => {
  selectedFile = e.target.files[0];
  if (selectedFile) {
    document.getElementById('uploadTitle').textContent = selectedFile.name;
    document.getElementById('uploadSub').textContent = (selectedFile.size/1024/1024).toFixed(1) + ' MB';
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = '🔍 Analyze This Creative';
  }
});
uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.classList.add('drag'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('drag'));
uploadBox.addEventListener('drop', e => {
  e.preventDefault(); uploadBox.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) {
    selectedFile = file;
    document.getElementById('uploadTitle').textContent = file.name;
    document.getElementById('uploadSub').textContent = (file.size/1024/1024).toFixed(1) + ' MB';
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = '🔍 Analyze This Creative';
  }
});

function scoreColor(s) { return s>=8?'green':s>=6?'yellow':s>=4?'orange':'red'; }
function scoreBg(s)    { return s>=8?'bg-green':s>=6?'bg-yellow':s>=4?'bg-orange':'bg-red'; }
function verdictCls(v) {
  const l = (v||'').toLowerCase();
  return l.includes('publish')?'verdict-ready':l.includes('test')?'verdict-test':l.includes('rework')?'verdict-rework':'verdict-no';
}
function verdictIcon(v) {
  const l = (v||'').toLowerCase();
  return l.includes('publish')?'✅':l.includes('test')?'🧪':l.includes('rework')?'⚠️':'🚫';
}

function setProgress(pct, text) {
  document.getElementById('progressBar').style.width = pct + '%';
  document.getElementById('progressText').textContent = text;
}

function renderPersona(p) {
  if (!p) return '';
  const rows = [
    ['Age Range', p.age_range],
    ['Gender Skew', p.gender_skew],
    ['Monthly Income', p.income_monthly_rm],
    ['Life Stage', p.life_stage],
    ['Location Mindset', p.location_mindset],
    ['Core Pain Point', p.pain_point],
    ['Primary Motivation', p.motivation],
    ['Property Mindset', p.property_mindset],
    ['Implied Budget', p.implied_budget],
  ].filter(r=>r[1]).map(r=>`
    <div style="display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #2e3140;">
      <div style="min-width:140px;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;color:#8a8d94;padding-top:2px">${r[0]}</div>
      <div style="font-size:14px;line-height:1.5">${r[1]}</div>
    </div>`).join('');
  const mismatch = p.persona_mismatch_risk
    ? `<div style="background:#1c100a;border:1px solid #ff9500;border-radius:8px;padding:12px 14px;margin-top:14px;font-size:13px;color:#ff9500">⚠️ <strong>Mismatch Risk:</strong> ${p.persona_mismatch_risk}</div>`
    : '';
  return `<div class="section" style="margin-bottom:14px">
    <h3>👤 Predicted Buyer Persona</h3>
    <p style="font-size:12px;color:#8a8d94;margin-bottom:12px">Based on what Andromeda reads from your video — visuals, language, and audio signals</p>
    ${rows}
    ${mismatch}
  </div>`;
}

function renderTargeting(arr) {
  if (!arr || !arr.length) return '';
  const typeColor = t => t==='Interest'?'#4f6ef7':t==='Behaviour'?'#34c759':t==='Demographic'?'#ff9500':'#8a8d94';
  const items = arr.map(t=>`
    <div style="padding:10px 0;border-bottom:1px solid #2e3140;display:flex;gap:12px;align-items:flex-start;">
      <span style="background:${typeColor(t.type)}22;color:${typeColor(t.type)};border:1px solid ${typeColor(t.type)}44;
            border-radius:4px;padding:1px 7px;font-size:10px;white-space:nowrap;margin-top:2px">${t.type||''}</span>
      <div>
        <div style="font-size:14px;font-weight:600">${t.category||''}</div>
        <div style="font-size:12px;color:#8a8d94;margin-top:3px">${t.reason||''}</div>
      </div>
    </div>`).join('');
  return `<div class="section" style="margin-bottom:14px">
    <h3>🎯 Meta Interest & Behaviour Match</h3>
    <p style="font-size:12px;color:#8a8d94;margin-bottom:4px">Search these in Meta Ads Manager → Detailed Targeting to find their IDs</p>
    ${items}
  </div>`;
}

function renderResults(d) {
  const hs = d.hook_score||0, as = d.andromeda_score||0, ahs = d.audio_hook_score||0;
  const imps = (d.improvements||[]).map(i=>`<li>${i}</li>`).join('');
  const strs = (d.strengths||[]).map(s=>`<li>${s}</li>`).join('');
  return `
    <p class="result-meta"><strong>${d.filename}</strong> · ${d.file_size_mb}MB · Full video + audio analyzed</p>

    <div class="verdict-box ${verdictCls(d.verdict)}">
      <h3>${verdictIcon(d.verdict)} ${d.verdict||'Unknown'}</h3>
      <p>${d.verdict_reason||''}</p>
    </div>

    <div class="scores-grid">
      <div class="score-card">
        <div class="label">Hook Score (first 3s)</div>
        <div class="value ${scoreColor(hs)}">${hs}<span style="font-size:15px;color:#8a8d94">/10</span></div>
        <div class="score-bar"><div class="score-bar-fill ${scoreBg(hs)}" style="width:${hs*10}%"></div></div>
        <div class="sub">${d.hook_feedback||''}</div>
        ${d.hook_tip?`<div class="sub" style="color:#ffd60a;margin-top:6px">💡 ${d.hook_tip}</div>`:''}
      </div>
      <div class="score-card">
        <div class="label">Audio Hook (spoken words)</div>
        <div class="value ${scoreColor(ahs)}">${ahs}<span style="font-size:15px;color:#8a8d94">/10</span></div>
        <div class="score-bar"><div class="score-bar-fill ${scoreBg(ahs)}" style="width:${ahs*10}%"></div></div>
        ${d.audio_hook?`<div class="highlight">"${d.audio_hook}"</div>`:''}
        <div class="sub">${d.audio_hook_feedback||''}</div>
      </div>
      <div class="score-card">
        <div class="label">Andromeda Signal</div>
        <div class="value ${scoreColor(as)}">${as}<span style="font-size:15px;color:#8a8d94">/10</span></div>
        <div class="score-bar"><div class="score-bar-fill ${scoreBg(as)}" style="width:${as*10}%"></div></div>
        <div class="sub">${d.andromeda_feedback||''}</div>
      </div>
      <div class="score-card">
        <div class="label">Andromeda Audience</div>
        <div class="sub-value" style="color:#a0c4ff;font-size:14px;line-height:1.5">${d.andromeda_audience||'—'}</div>
        <hr class="divider">
        <div class="label" style="margin-top:0">CPM · CPL Prediction</div>
        <div class="sub">${d.cpm_prediction||'—'} · ${d.cpl_prediction||'—'}</div>
      </div>
    </div>

    <div class="scores-grid">
      <div class="section">
        <h3>📊 CPM Prediction</h3>
        <p style="font-size:16px;font-weight:600">${d.cpm_prediction||'—'}</p>
        <p style="margin-top:6px">${d.cpm_reason||''}</p>
      </div>
      <div class="section">
        <h3>💰 CPL Prediction</h3>
        <p style="font-size:16px;font-weight:600">${d.cpl_prediction||'—'}</p>
        <p style="margin-top:6px">${d.cpl_reason||''}</p>
      </div>
    </div>

    ${renderPersona(d.buyer_persona)}
    ${renderTargeting(d.meta_targeting)}

    ${strs?`<div class="section"><h3>✅ Strengths</h3><ul class="list-items strength">${strs}</ul></div>`:''}
    <div class="section"><h3>🔧 Top Improvements</h3><ul class="list-items improvement">${imps}</ul></div>

    <button class="try-again" onclick="resetForm()">← Analyze another video</button>
  `;
}

async function analyzeVideo() {
  if (!selectedFile) return;
  document.getElementById('errorBox').style.display = 'none';
  document.getElementById('results').style.display = 'none';
  document.getElementById('progress').style.display = 'block';
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  setProgress(5, 'Uploading video to Gemini...');

  const steps = [
    [20,'Processing video frames...'],
    [38,'Transcribing audio and speech...'],
    [52,'Analyzing hook (first 3 seconds)...'],
    [65,'Scoring Andromeda signal...'],
    [78,'Predicting buyer persona...'],
    [88,'Matching Meta interest categories...'],
    [88,'Predicting CPM and CPL...'],
    [95,'Writing improvement suggestions...'],
  ];
  let i=0;
  const timer = setInterval(()=>{ if(i<steps.length){setProgress(steps[i][0],steps[i][1]);i++;} },4000);

  const formData = new FormData();
  formData.append('video', selectedFile);

  try {
    const res = await fetch('/analyze', {method:'POST', body:formData});
    clearInterval(timer);
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error||'Analysis failed');
    setProgress(100, 'Done!');
    setTimeout(()=>{
      document.getElementById('progress').style.display='none';
      const el = document.getElementById('results');
      el.innerHTML = renderResults(data);
      el.style.display='block';
      analyzeBtn.style.display='none';
      uploadBox.style.display='none';
    },400);
  } catch(err) {
    clearInterval(timer);
    document.getElementById('progress').style.display='none';
    document.getElementById('errorMsg').textContent='❌ '+err.message;
    document.getElementById('errorBox').style.display='block';
    analyzeBtn.disabled=false;
    analyzeBtn.textContent='Try Again';
  }
}

function resetForm() {
  selectedFile=null; fileInput.value='';
  document.getElementById('uploadTitle').textContent='Drop your video here';
  document.getElementById('uploadSub').textContent='MP4, MOV, AVI, MKV · Max 200MB';
  uploadBox.style.display='block';
  analyzeBtn.style.display='block';
  analyzeBtn.disabled=true;
  analyzeBtn.textContent='Choose a video to analyze';
  document.getElementById('results').style.display='none';
  document.getElementById('progress').style.display='none';
  document.getElementById('errorBox').style.display='none';
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
