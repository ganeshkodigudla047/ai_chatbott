import ssl
import json
import random
from datetime import datetime
from difflib import get_close_matches
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import markdown as md_lib

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="Avanthi College Assistant", page_icon="🎓", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stAppViewContainer"] { background: #f0f4f8 !important; }
[data-testid="stMain"] { padding-top: 0 !important; }
[data-testid="block-container"] {
    padding-top: 0 !important; padding-bottom: 8px !important; max-width: 100% !important;
}
textarea {
    border-radius: 12px !important; border: 2px solid #d0d7e8 !important;
    padding: 8px 14px !important; font-size: 14px !important; background: #f8faff !important;
    resize: none !important; line-height: 1.4 !important;
}
textarea:focus { border-color: #1e2a4a !important; background: #fff !important; outline: none !important; }
.stFormSubmitButton > button { border-radius: 20px !important; font-size: 13px !important; height: 42px !important; }
.stButton > button { border-radius: 20px !important; font-size: 13px !important; height: 42px !important; }
.stFormSubmitButton > button:disabled { visibility: hidden !important; }
.stForm { border: none !important; padding: 0 !important; box-shadow: none !important; }
section[data-testid="stMain"] > div { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(_version="v11"):
    import os, pickle
    from sentence_transformers import SentenceTransformer
    cache_path = f"model/embeddings_{_version}.pkl"
    with open("data/intents.json", encoding="utf-8") as f:
        intents = json.load(f)["intents"]
    tags, patterns = [], []
    for intent in intents:
        for pattern in intent["patterns"]:
            tags.append(intent["tag"])
            patterns.append(pattern.lower())
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = model.encode(patterns, convert_to_numpy=True)
        try:
            os.makedirs("model", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings, f)
        except Exception:
            pass  # read-only filesystem on cloud — skip caching
    model.encode(["warmup"], convert_to_numpy=True)
    return intents, model, embeddings, tags

intents, embedder, pattern_embeddings, pattern_tags = load_model("v15")

@st.cache_resource(show_spinner=False)
def build_vocab(_version="v15"):
    vocab = set()
    for intent in intents:
        for pattern in intent["patterns"]:
            for word in pattern.lower().split():
                if len(word) > 2:
                    vocab.add(word)
    return list(vocab)

VOCAB = build_vocab("v15")

# ── TTS ────────────────────────────────────────────────────────────────────────
_tts_proc = None

def speak_text(text):
    import re
    clean = re.sub(r'\*+|#+|`|_|\|', '', text)
    clean = re.sub(r'\-{2,}', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    st.session_state["_tts_text"] = clean

def recognize_speech():
    import sys
    if sys.platform != "win32":
        return ""  # Microphone not supported on cloud
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=8, phrase_time_limit=10)
        return r.recognize_google(audio)
    except Exception:
        return ""

# ── Slang / gibberish / typo ───────────────────────────────────────────────────
SLANG = {
    "clg":"college","col":"college","u":"you","r":"are","ur":"your",
    "hw":"how","wt":"what","wat":"what","whr":"where","whn":"when",
    "abt":"about","dept":"department","sem":"semester","lib":"library",
    "tpo":"placement officer","crt":"campus recruitment training","hos":"hostel",
    "xam":"exam","xams":"exams","hlp":"help","plz":"please","pls":"please",
    "thx":"thanks","ty":"thank you","gud":"good","idk":"i don't know"
}

def normalize(text):
    return " ".join(SLANG.get(t, t) for t in text.lower().strip().split())

def is_gibberish(text):
    import re
    words = text.strip().split()
    if not words: return True
    real = sum(1 for w in words if re.match(r'^[a-zA-Z]{2,}$', w))
    if real / len(words) < 0.5: return True
    if len(words) == 1 and len(words[0]) > 3:
        if sum(1 for c in words[0].lower() if c in "aeiou") == 0:
            return True
    return False

def correct_typos(text):
    corrected = []
    for word in text.split():
        matches = get_close_matches(word.lower(), VOCAB, n=1, cutoff=0.75)
        corrected.append(matches[0] if matches else word)
    return " ".join(corrected)

# ── Events ─────────────────────────────────────────────────────────────────────
MONTHLY_EVENTS = {
    1:"🎉 Sankranthi Event is happening this month (January)!",
    2:"🔧 Synergy Technical Fest is happening this month (February)!",
    3:"🎓 Farewell ceremony is this month (March). Final year students, get ready!",
    4:"🏆 Graduation Day is this month (April). Congratulations to all graduates!",
    8:"🏅 Sports Day is happening this month (August). Time to show your athletic skills!",
    9:"💡 Diploma Tech Fest is this month (September)!",
    11:"🤝 Induction Meet Event is this month (November). Welcome to new students!"
}

def get_event_response(tag, user_input):
    today_kw = ["today","now","this week","happening","any thing","anything","tdy"]
    if any(k in user_input.lower() for k in today_kw):
        month = datetime.now().month
        if month in MONTHLY_EVENTS:
            return MONTHLY_EVENTS[month]
        return (f"No major events this month ({datetime.now().strftime('%B')}). "
                "Upcoming: Sankranthi (Jan), Synergy (Feb), Farewell (Mar), "
                "Graduation (Apr), Sports (Aug), Diploma Fest (Sep), Induction (Nov).")
    return None

# ── Context / dept tracking ────────────────────────────────────────────────────
DEPT_TAGS = {
    "cse": ["cse_faculty","cse_syllabus","cse_course_info","cse_flat_faculty","cse_cd_faculty",
            "cse_networks_faculty","cse_programming_faculty"],
    "ece": ["ece_faculty","ece_syllabus","ece_course_info","ece_vlsi_faculty","ece_communication_faculty"],
    "eee": ["eee_faculty","eee_syllabus","eee_course_info","eee_power_faculty","eee_control_faculty"],
    "mech": ["mech_faculty","mech_syllabus","mech_course_info","mech_thermo_faculty","mech_design_faculty"],
    "csd": ["csd_faculty","csd_syllabus","csd_course_info","csd_faculty_individual"],
    "csm": ["csm_faculty","csm_syllabus","csm_course_info","csm_faculty_individual"],
}
TAG_TO_DEPT = {tag: dept for dept, tags in DEPT_TAGS.items() for tag in tags}

FOLLOWUP_TOPIC = {
    "subjects":"subjects","syllabus":"syllabus","curriculum":"syllabus",
    "faculty":"faculty","teachers":"faculty","staff":"faculty",
    "hod":"hod","incharge":"faculty",
    "course":"course details","details":"course details",
    "info":"course details","about":"course details",
}
TOPIC_PHRASE = {"syllabus":"subjects","faculty":"faculty","course_info":"course details"}
DEPT_TO_COURSE_TAG = {
    "cse":"cse_course_info","ece":"ece_course_info","eee":"eee_course_info",
    "mech":"mech_course_info","csd":"csd_course_info","csm":"csm_course_info",
}
DEPT_TOPIC_TO_TAG = {
    ("cse","syllabus"):"cse_syllabus",   ("cse","subjects"):"cse_syllabus",
    ("cse","faculty"):"cse_faculty",     ("cse","hod"):"cse_faculty",
    ("cse","course details"):"cse_course_info",
    ("ece","syllabus"):"ece_syllabus",   ("ece","subjects"):"ece_syllabus",
    ("ece","faculty"):"ece_faculty",     ("ece","hod"):"ece_faculty",
    ("ece","course details"):"ece_course_info",
    ("eee","syllabus"):"eee_syllabus",   ("eee","subjects"):"eee_syllabus",
    ("eee","faculty"):"eee_faculty",     ("eee","hod"):"eee_faculty",
    ("eee","course details"):"eee_course_info",
    ("mech","syllabus"):"mech_syllabus", ("mech","subjects"):"mech_syllabus",
    ("mech","faculty"):"mech_faculty",   ("mech","hod"):"mech_faculty",
    ("mech","course details"):"mech_course_info",
    ("csd","syllabus"):"csd_syllabus",   ("csd","subjects"):"csd_syllabus",
    ("csd","faculty"):"csd_faculty",     ("csd","hod"):"csd_faculty",
    ("csd","course details"):"csd_course_info",
    ("csm","syllabus"):"csm_syllabus",   ("csm","subjects"):"csm_syllabus",
    ("csm","faculty"):"csm_faculty",     ("csm","hod"):"csm_faculty",
    ("csm","course details"):"csm_course_info",
}
LAB_DIRECT_TAG = {
    "vlsi":"vlsi_lab","vlsi lab":"vlsi_lab",
    "jkc":"jkc_lab","jkc lab":"jkc_lab",
    "cew":"cew_lab","cew lab":"cew_lab",
    "digital electronics lab":"digital_electronics_lab","digital lab":"digital_electronics_lab",
    "embedded lab":"embedded_lab","embedded systems lab":"embedded_lab",
    "microprocessor lab":"microprocessor_lab","micro lab":"microprocessor_lab",
    "power lab":"power_lab","power systems lab":"power_lab",
    "control lab":"control_lab","control systems lab":"control_lab",
    "workshop lab":"workshop_lab","workshop":"workshop_lab",
    "fluid mechanics lab":"fluid_mechanics_lab","fluid lab":"fluid_mechanics_lab",
    "chemistry lab":"chemistry_lab","physics lab":"physics_lab",
    "ece lab":"ece_lab","electronics lab":"ece_lab",
    "eee lab":"eee_lab","electrical lab":"eee_lab",
    "mech lab":"mech_lab","mechanical lab":"mech_lab",
    "computer lab":"computer_lab",
}

def resolve_context(user_input, last_dept):
    tokens = user_input.lower().strip().split()
    inp = user_input.lower().strip()
    if inp in LAB_DIRECT_TAG:
        return user_input, LAB_DIRECT_TAG[inp]
    dept_keywords = [
        ("computer science","cse"),("data science","csd"),("machine learning","csm"),
        ("electronics","ece"),("electrical","eee"),("mechanical","mech"),
        ("cse","cse"),("csd","csd"),("csm","csm"),("ece","ece"),("eee","eee"),("mech","mech"),
    ]
    detected_dept = None
    for kw, dept in dept_keywords:
        if kw in inp:
            detected_dept = dept; break
    active_dept = detected_dept or last_dept
    topic_phrase = None
    for token in tokens:
        if token in FOLLOWUP_TOPIC:
            topic_phrase = FOLLOWUP_TOPIC[token]; break
    is_short = len(tokens) <= 3
    if active_dept and topic_phrase and is_short:
        tag = DEPT_TOPIC_TO_TAG.get((active_dept, topic_phrase))
        if tag: return user_input, tag
    if is_short and detected_dept and not topic_phrase:
        if last_dept:
            phrase = TOPIC_PHRASE.get(st.session_state.get("last_topic","course_info"), "course details")
            tag = DEPT_TOPIC_TO_TAG.get((detected_dept, phrase))
            if tag: return user_input, tag
        tag = DEPT_TO_COURSE_TAG.get(detected_dept)
        if tag: return user_input, tag
    if is_short and topic_phrase and not detected_dept and last_dept:
        tag = DEPT_TOPIC_TO_TAG.get((last_dept, topic_phrase))
        if tag: return user_input, tag
    return user_input, None

# List-format responses (used when user asks for list)
LIST_RESPONSES = {
    "computer_lab": (
        "**Computer Labs:**\n\n"
        "**Locations:**\n"
        "- JKC Lab — 3rd floor, Main Block (CSE dept)\n"
        "- Ground Floor Lab — Main Block\n"
        "- New Lab-1 & Lab-2 — New Building, Ground Floor\n\n"
        "**Facilities:**\n"
        "- 200+ systems with Windows/Linux\n"
        "- High-speed internet\n"
        "- Software: VS Code, Python, Java, MATLAB\n\n"
        "**Timings:** 9 AM to 5 PM"
    ),
}

def chatbot_reply(user_input):
    text = normalize(user_input)
    stripped = text.strip()
    if len(stripped) < 2:
        return random.choice([
            "Could you say a bit more? I want to make sure I help you correctly!",
            "That's a bit short for me to understand. What would you like to know?",
        ])
    if is_gibberish(stripped):
        return random.choice([
            "That doesn't look like a valid question. Try asking about fees, faculty, labs, bus routes, or events!",
            "I didn't quite catch that. Could you rephrase your question?",
        ])
    last_dept = st.session_state.get("last_dept", None)
    expanded, direct_tag = resolve_context(stripped, last_dept)
    if direct_tag:
        tag = direct_tag
    else:
        corrected = correct_typos(expanded)
        user_emb = embedder.encode([corrected], convert_to_numpy=True)
        sims = cosine_similarity(user_emb, pattern_embeddings)[0]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] < 0.40:
            return "I'm not sure about that. Could you rephrase? You can ask about labs, fees, faculty, bus routes, events, and more."
        tag = pattern_tags[best_idx]
    if tag in TAG_TO_DEPT:
        st.session_state["last_dept"] = TAG_TO_DEPT[tag]
    if "syllabus" in tag: st.session_state["last_topic"] = "syllabus"
    elif "faculty" in tag: st.session_state["last_topic"] = "faculty"
    elif "course_info" in tag: st.session_state["last_topic"] = "course_info"
    if tag == "events":
        ev = get_event_response(tag, user_input)
        if ev: return ev
    list_kw = ["list", "in list", "as list", "show list", "list format", "in a list"]
    if any(kw in user_input.lower() for kw in list_kw) and tag in LIST_RESPONSES:
        return LIST_RESPONSES[tag]
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I don't have information on that. Please contact the college office."

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"bot","text":"👋 Hello! I'm the Avanthi College Assistant. How can I help you today?"}]
if "voice_output" not in st.session_state: st.session_state.voice_output = False
if "last_dept" not in st.session_state: st.session_state.last_dept = None
if "last_topic" not in st.session_state: st.session_state.last_topic = "course_info"
if "_toggle_voice" not in st.session_state: st.session_state["_toggle_voice"] = False
if "_tts_text" not in st.session_state: st.session_state["_tts_text"] = ""

def process(user_text):
    response = chatbot_reply(user_text)
    st.session_state.messages.append({"role":"user","text":user_text})
    st.session_state.messages.append({"role":"bot","text":response})
    if st.session_state.voice_output:
        speak_text(response)


def clear_chat():
    global _tts_proc
    if _tts_proc is not None:
        try: _tts_proc.kill()
        except Exception: pass
        _tts_proc = None
    pid = st.session_state.get("_tts_pid")
    if pid:
        try:
            import os, signal; os.kill(pid, signal.SIGTERM)
        except Exception: pass
        st.session_state["_tts_pid"] = None
    st.session_state.messages = [{"role":"bot","text":"👋 Hello! I'm the Avanthi College Assistant. How can I help you today?"}]
    st.session_state.last_dept = None
    st.session_state.last_topic = "course_info"

# ── Build chat HTML ────────────────────────────────────────────────────────────
chat_html = ""
for msg in st.session_state.messages:
    if msg["role"] == "user":
        safe = msg["text"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        chat_html += (
            '<div style="display:flex;align-items:flex-end;justify-content:flex-end;margin:5px 0">'
            f'<div style="max-width:68%;padding:10px 15px;font-size:14px;line-height:1.55;'
            f'word-wrap:break-word;border-radius:18px;border-bottom-right-radius:4px;'
            f'background:#1e2a4a;color:#fff">{safe}</div>'
            '<div style="width:30px;height:30px;border-radius:50%;background:#1e2a4a;'
            'display:flex;align-items:center;justify-content:center;font-size:15px;'
            'flex-shrink:0;margin-left:7px">🧑</div></div>\n'
        )
    else:
        rendered = md_lib.markdown(msg["text"], extensions=["tables"])
        chat_html += (
            '<div style="display:flex;align-items:flex-end;justify-content:flex-start;margin:5px 0">'
            '<div style="width:30px;height:30px;border-radius:50%;background:#e8edf8;'
            'display:flex;align-items:center;justify-content:center;font-size:15px;'
            'flex-shrink:0;margin-right:7px">🎓</div>'
            f'<div style="max-width:68%;padding:10px 15px;font-size:14px;line-height:1.55;'
            f'word-wrap:break-word;border-radius:18px;border-bottom-left-radius:4px;'
            f'background:#fff;color:#1a1a2e;box-shadow:0 2px 8px rgba(0,0,0,0.07)">{rendered}</div>'
            '</div>\n'
        )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#1e2a4a;padding:10px 20px;display:flex;align-items:center;
            gap:12px;box-shadow:0 2px 8px rgba(0,0,0,.2);margin-bottom:4px">
  <span style="font-size:22px">🎓</span>
  <div>
    <div style="color:#fff;font-size:16px;font-weight:700">Avanthi College Assistant</div>
    <div style="color:#a0b0d0;font-size:11px">Ask about faculty, fees, labs, bus routes, events &amp; more</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Chat window ────────────────────────────────────────────────────────────────
components.html(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
* {{ box-sizing:border-box; margin:0; padding:0; font-family:'Segoe UI',sans-serif; }}
html,body {{ height:100%; background:#f0f4f8; overflow:hidden; }}
#chat {{ height:100%; overflow-y:auto; padding:10px 16px 4px; }}
table {{ border-collapse:collapse; width:100%; margin-top:6px; font-size:13px; }}
th,td {{ border:1px solid #dde3f0; padding:5px 10px; text-align:left; }}
th {{ background:#f0f4ff; font-weight:600; }}
ul,ol {{ padding-left:18px; margin-top:4px; }}
p {{ margin:2px 0; }}
strong {{ font-weight:700; }}
</style></head><body>
<div id="chat">{chat_html}</div>
<script>
(function(){{
  var c=document.getElementById("chat");
  function s(){{c.scrollTop=c.scrollHeight;}}
  s(); requestAnimationFrame(function(){{s();requestAnimationFrame(s);}});
  setTimeout(s,150); setTimeout(s,400);
  var us=false;
  c.addEventListener("scroll",function(){{us=(c.scrollHeight-c.scrollTop-c.clientHeight)>80;}});
  new MutationObserver(function(){{if(!us)s();}}).observe(c,{{childList:true,subtree:true}});
}})();
</script></body></html>""", height=430, scrolling=False)

# Browser TTS via separate component (avoids f-string escaping issues)
_tts_out = (st.session_state.get("_tts_text","") if st.session_state.get("voice_output") else "")
st.session_state["_tts_text"] = ""  # clear after use
if _tts_out:
    import json as _json
    _tts_safe = _json.dumps(_tts_out)  # properly escaped JS string
    components.html(f"""<script>
(function(){{
  var txt = {_tts_safe};
  if(txt && window.speechSynthesis){{
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(txt);
    u.rate = 1.0; u.pitch = 1.0; u.volume = 1.0;
    window.speechSynthesis.speak(u);
  }}
}})();
</script>""", height=0, scrolling=False)

# ── Input row ──────────────────────────────────────────────────────────────────
st.markdown('<div style="background:#f0f4f8;border-top:1px solid #dde3f0;padding-top:4px">', unsafe_allow_html=True)

if st.session_state.get("_toggle_voice"):
    st.session_state["_toggle_voice"] = False
    st.session_state.voice_output = not st.session_state.voice_output

_voice_on = st.session_state.get("voice_output", False)
_voice_label = "🔊 ON" if _voice_on else "🔇 OFF"
_voice_color = "#dcfce7" if _voice_on else "#fff"
_prefill = st.session_state.pop("_mic_prefill", "") if "_mic_prefill" in st.session_state else ""

# Streamlit form — hidden visually, submitted by JS
with st.form("chat_form", clear_on_submit=True):
    ci, c_send, c_voice, c_clear = st.columns([5,1,1,1])
    user_input = ci.text_input("msg", label_visibility="collapsed",
                               placeholder="msg", value=_prefill, key="user_msg_form")
    send  = c_send.form_submit_button("Send")
    voice = c_voice.form_submit_button(_voice_label)
    clear = c_clear.form_submit_button("Clear")
    if voice: st.session_state["_toggle_voice"] = True; st.rerun()
    if send and user_input.strip(): process(user_input.strip()); st.rerun()
    if clear: clear_chat(); st.rerun()

# Hide the Streamlit form and overlay our custom HTML input row
st.markdown(f"""<style>
/* hide native form */
div[data-testid="stForm"] {{ display:none !important; }}
</style>
<div style="display:flex;gap:6px;align-items:center;padding:2px 0 4px;">
  <div style="flex:1;position:relative;display:flex;align-items:center;
              background:#f8faff;border:2px solid #d0d7e8;border-radius:24px;
              padding:0 40px 0 14px;height:42px;" id="inputWrap">
    <input id="msgInput" type="text" placeholder="💬 Type your message..."
           value="{_prefill}"
           style="flex:1;border:none;outline:none;background:transparent;
                  font-size:14px;color:#1a1a2e;"
           onkeydown="if(event.key==='Enter')sendMsg()"/>
    <button onclick="startMic()" title="Speak"
            id="micBtn"
            style="position:absolute;right:10px;background:none;border:none;
                   cursor:pointer;font-size:17px;padding:0;opacity:0.5;">🎤</button>
  </div>
  <button onclick="sendMsg()"
          style="background:#fff;border:1.5px solid #d0d7e8;border-radius:20px;
                 padding:0 16px;height:42px;font-size:13px;color:#1e2a4a;cursor:pointer;">Send</button>
  <button onclick="toggleVoice()"
          style="background:{_voice_color};border:1.5px solid #d0d7e8;border-radius:20px;
                 padding:0 16px;height:42px;font-size:13px;color:#1e2a4a;cursor:pointer;">{_voice_label}</button>
  <button onclick="clearChat()"
          style="background:#fff;border:1.5px solid #d0d7e8;border-radius:20px;
                 padding:0 16px;height:42px;font-size:13px;color:#1e2a4a;cursor:pointer;">Clear</button>
</div>
<script>
function fillAndSubmit(txt, btnId) {{
  // Find the hidden Streamlit input and submit button, fill and click
  var inputs = window.parent.document.querySelectorAll('input[data-testid="stTextInput"]');
  var btns = window.parent.document.querySelectorAll('button[kind="formSubmit"], button[data-testid="baseButton-secondaryFormSubmit"]');
  if(inputs.length > 0) {{
    var nativeInput = inputs[0];
    var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    nativeSetter.call(nativeInput, txt);
    nativeInput.dispatchEvent(new Event('input', {{bubbles:true}}));
  }}
  // Click the right submit button
  var allBtns = window.parent.document.querySelectorAll('button');
  for(var i=0;i<allBtns.length;i++) {{
    if(allBtns[i].innerText.trim() === btnId) {{ allBtns[i].click(); break; }}
  }}
}}
function sendMsg() {{
  var txt = document.getElementById("msgInput").value.trim();
  if(txt) fillAndSubmit(txt, "Send");
}}
function toggleVoice() {{ fillAndSubmit("", "{_voice_label}"); }}
function clearChat() {{ fillAndSubmit("", "Clear"); }}
function startMic() {{
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){{ alert("Use Chrome or Edge for mic."); return; }}
  var btn = document.getElementById("micBtn");
  var inp = document.getElementById("msgInput");
  var r = new SR();
  r.lang = "en-IN"; r.interimResults = false; r.maxAlternatives = 1;
  btn.style.opacity="1"; btn.innerText="🔴";
  r.start();
  r.onresult = function(e) {{
    inp.value = e.results[0][0].transcript;
    btn.style.opacity="0.5"; btn.innerText="🎤";
  }};
  r.onerror = function() {{ btn.style.opacity="0.5"; btn.innerText="🎤"; }};
  r.onend = function() {{ if(btn.innerText==="🔴"){{ btn.style.opacity="0.5"; btn.innerText="🎤"; }} }};
}}
document.getElementById("msgInput").focus();
</script>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)