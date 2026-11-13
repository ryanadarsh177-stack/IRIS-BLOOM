
from __future__ import annotations

import base64
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "iris.csv"
ART_PATH = BASE_DIR / "assets" / "iris_luminous.svg"
FEATURES = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
DISPLAY = {
    "sepal_length_cm": "Sepal length",
    "sepal_width_cm": "Sepal width",
    "petal_length_cm": "Petal length",
    "petal_width_cm": "Petal width",
}
SPECIES = ["setosa", "versicolor", "virginica"]
PALETTE = {"setosa": "#ff9dbd", "versicolor": "#c2a4ff", "virginica": "#91dfcf"}
SPECIES_INFO = {
    "setosa": "A petite bloom with short petals and a distinctive compact profile.",
    "versicolor": "The in-between beauty, with measurements that bridge the other two species.",
    "virginica": "The largest Iris species in this classic dataset, often with longer petals.",
}
PRESETS = {
    "🌸 Setosa": [5.0, 3.4, 1.5, 0.2],
    "🪻 Versicolor": [6.0, 2.8, 4.5, 1.3],
    "🌺 Virginica": [6.5, 3.0, 5.5, 2.0],
    "✨ My own flower": [5.8, 3.0, 4.3, 1.3],
}

st.set_page_config(page_title="Iris Bloom | Interactive Flower Lab", page_icon="🌸", layout="wide", initial_sidebar_state="expanded")

def styles():
    st.markdown(r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
    :root{--cream:#fff7f8;--ink:#3c263b;--muted:#856b7e;--pink:#f58eb5;--rose:#d96f9e;--lilac:#b59ae8;--mint:#8bd7c4}
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
    .stApp{color:var(--ink);background:
      radial-gradient(ellipse at 10% 5%,rgba(255,191,211,.58),transparent 34%),
      radial-gradient(ellipse at 90% 24%,rgba(201,183,245,.42),transparent 32%),
      radial-gradient(ellipse at 70% 95%,rgba(163,231,211,.38),transparent 35%),
      linear-gradient(135deg,#fff8f5 0%,#fff2f7 46%,#f5f0ff 100%);background-attachment:fixed}
    [data-testid="stHeader"]{background:rgba(255,247,250,.58);backdrop-filter:blur(18px)}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(255,239,246,.92),rgba(246,239,255,.94));border-right:1px solid rgba(190,128,164,.18)}
    [data-testid="stSidebar"] *{color:#563b53}
    .block-container{max-width:1400px;padding-top:1.6rem;padding-bottom:4rem}
    h1,h2,h3{font-family:'Playfair Display',serif!important;color:#55334e!important;letter-spacing:-.025em}
    p,label,.stCaption{color:#80677b}
    .brand{display:flex;align-items:center;gap:12px;margin:0 0 25px}
    .brand-icon{height:44px;width:44px;border-radius:16px;display:grid;place-items:center;font-size:24px;background:linear-gradient(145deg,#ffd7e6,#e6d9ff);box-shadow:0 8px 25px #d99bbd66;animation:breathe 4s ease-in-out infinite}
    .brand-name{font:800 18px Manrope;color:#573850;letter-spacing:.04em}
    .brand-sub{font-size:10px;text-transform:uppercase;letter-spacing:.22em;color:#a17b98}
    .hero{position:relative;overflow:hidden;min-height:345px;border-radius:32px;padding:38px 42px;background:
      linear-gradient(115deg,rgba(255,255,255,.86),rgba(255,236,245,.74) 48%,rgba(237,228,255,.74));
      border:1px solid rgba(255,255,255,.92);box-shadow:0 24px 65px rgba(159,105,143,.16),inset 0 1px 0 white;backdrop-filter:blur(24px)}
    .hero:before{content:"";position:absolute;inset:-50%;background:conic-gradient(from 0deg,transparent 0 72%,rgba(255,255,255,.48) 78%,transparent 84%);animation:shine 18s linear infinite;pointer-events:none}
    .hero-copy{position:relative;z-index:3;max-width:700px}
    .eyebrow{font-size:10px;letter-spacing:.25em;text-transform:uppercase;color:#b06e95;font-weight:800}
    .hero-title{font:800 clamp(38px,5vw,66px)/1.02 'Playfair Display',serif;letter-spacing:-.05em;color:#59354f;margin:17px 0}
    .hero-title span{background:linear-gradient(90deg,#d76f9d,#9c79d1,#65bca9);-webkit-background-clip:text;background-clip:text;color:transparent}
    .hero-text{font-size:15px;line-height:1.8;max-width:620px;color:#826b7d}
    .hero-art{position:absolute;right:0;top:-8px;width:min(39%,430px);height:360px;object-fit:contain;filter:drop-shadow(0 20px 28px #b77eae55);animation:float 6s ease-in-out infinite;z-index:2}
    .petal{position:absolute;z-index:1;font-size:25px;opacity:.32;animation:drift 13s ease-in-out infinite}
    .p1{right:37%;top:12%;animation-delay:-3s}.p2{right:8%;bottom:12%;font-size:34px;animation-delay:-7s}.p3{left:47%;bottom:7%;font-size:18px;animation-delay:-1s}
    @keyframes float{0%,100%{transform:translateY(0) rotate(-2deg)}50%{transform:translateY(-13px) rotate(2deg)}}
    @keyframes drift{0%,100%{transform:translate(0,0) rotate(0)}50%{transform:translate(16px,-22px) rotate(35deg)}}
    @keyframes shine{to{transform:rotate(360deg)}}
    @keyframes breathe{0%,100%{transform:scale(1)}50%{transform:scale(1.07)}}
    .glass{border:1px solid rgba(255,255,255,.9);border-radius:23px;padding:21px 23px;background:linear-gradient(145deg,rgba(255,255,255,.84),rgba(255,247,251,.56));box-shadow:0 12px 36px rgba(164,112,148,.12),inset 0 1px 0 white;backdrop-filter:blur(18px);transition:transform .25s,box-shadow .25s}
    .glass:hover{transform:translateY(-4px);box-shadow:0 18px 42px rgba(164,112,148,.19)}
    .metric-label{font-size:10px;letter-spacing:.17em;text-transform:uppercase;color:#a17b97;font-weight:800}
    .metric-value{font:800 30px Manrope;color:#633e5a;margin-top:5px}
    .metric-hint{font-size:12px;color:#987e91;margin-top:4px}
    .section-kicker{font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:#b8759a;font-weight:800;margin-bottom:7px}
    .section-title{font:700 28px 'Playfair Display';color:#5b3853;margin-bottom:5px}
    .section-note{font-size:13px;color:#927b8d;margin-bottom:17px}
    .stButton>button{border:1px solid #f0c4d8;border-radius:15px;padding:.65rem 1rem;color:#713f61;background:linear-gradient(110deg,#ffe0eb,#e9ddff 60%,#d9f4ec);box-shadow:0 7px 22px #c58aa633;font-weight:700;transition:all .22s}
    .stButton>button:hover{transform:translateY(-2px) scale(1.015);border-color:#d99abc;box-shadow:0 12px 28px #c58aa655;color:#57334f}
    .stSlider [data-baseweb="slider"] div[role="slider"]{background:#d978a5;border-color:white;box-shadow:0 0 0 5px #e9a9c344}
    [data-testid="stMetric"]{background:#ffffffa8;border:1px solid white;border-radius:17px;padding:13px}
    [data-testid="stMetricLabel"],[data-testid="stMetricValue"]{color:#633e5a}
    .stTabs [data-baseweb="tab-list"]{gap:8px;background:#ffffff83;padding:7px;border:1px solid white;border-radius:17px}
    .stTabs [data-baseweb="tab"]{border-radius:12px;color:#936d88;padding:9px 16px}
    .stTabs [aria-selected="true"]{background:linear-gradient(110deg,#ffe0eb,#e9ddff);color:#613c59}
    [data-testid="stDataFrame"]{border:1px solid white;border-radius:16px;overflow:hidden}
    [data-testid="stAlert"]{border-radius:15px}
    .flower-divider{font-size:19px;letter-spacing:12px;text-align:center;color:#d48cac;margin:15px 0}
    .footer{text-align:center;color:#b18da5;font-size:11px;letter-spacing:.12em;padding:30px 0 5px}
    @media(max-width:850px){.hero{padding:27px 24px}.hero-art{right:-65px;width:58%;opacity:.27}.hero-copy{max-width:100%}.hero-title{font-size:42px}}
    @media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[FEATURES + ["species"]].copy()
    for col in FEATURES: df[col] = pd.to_numeric(df[col], errors="coerce")
    df["species"] = df["species"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=FEATURES + ["species"])
    return df[df.species.isin(SPECIES)].reset_index(drop=True)

@st.cache_resource(show_spinner="Growing the models…")
def train_models():
    df = load_data()
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURES], df.species, test_size=.2, random_state=42, stratify=df.species)
    models = {
        "K-Nearest Neighbors": Pipeline([("scale", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results[name] = {"accuracy": accuracy_score(y_test, pred), "matrix": confusion_matrix(y_test, pred, labels=SPECIES),
                         "report": classification_report(y_test, pred, labels=SPECIES, output_dict=True, zero_division=0)}
    return df, models, results, len(x_train), len(x_test)

def card(label, value, hint):
    st.markdown(f'<div class="glass"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-hint">{hint}</div></div>', unsafe_allow_html=True)

def predict_for(values, model):
    row = pd.DataFrame([dict(zip(FEATURES, values))])
    pred = str(model.predict(row)[0])
    probs = model.predict_proba(row)[0]
    classes = model.named_steps["classifier"].classes_
    return pred, {str(c): float(p) for c,p in zip(classes,probs)}

styles()
if "measurements" not in st.session_state: st.session_state.measurements = PRESETS["✨ My own flower"].copy()
if "feedback_sent" not in st.session_state: st.session_state.feedback_sent = False

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">🌷</div><div><div class="brand-name">IRIS BLOOM</div><div class="brand-sub">The little flower lab</div></div></div>', unsafe_allow_html=True)
    st.markdown("### Design your flower")
    model_name = st.selectbox("Choose a classifier", ["K-Nearest Neighbors", "Logistic Regression"])
    preset = st.selectbox("Start with a flower preset", list(PRESETS.keys()), index=3)
    if st.button("↻ Load this flower", use_container_width=True):
        st.session_state.measurements = PRESETS[preset].copy()
        st.rerun()
    vals = st.session_state.measurements
    st.markdown("**Sepal dimensions**")
    vals[0] = st.slider("Sepal length · cm", 4.0, 8.0, float(vals[0]), .1, key="sl")
    vals[1] = st.slider("Sepal width · cm", 2.0, 4.5, float(vals[1]), .1, key="sw")
    st.markdown("**Petal dimensions**")
    vals[2] = st.slider("Petal length · cm", 1.0, 7.0, float(vals[2]), .1, key="pl")
    vals[3] = st.slider("Petal width · cm", .1, 2.5, float(vals[3]), .1, key="pw")
    st.session_state.measurements = vals
    st.markdown("---")
    st.caption("Move any slider to see the prediction and flower profile update instantly.")

try:
    data, models, evaluation, train_count, test_count = train_models()
except Exception as exc:
    st.error(f"Could not prepare the flower lab: {exc}")
    st.stop()

predicted, probability_map = predict_for(st.session_state.measurements, models[model_name])
top_probability = probability_map[predicted]
art_uri = "data:image/svg+xml;base64," + base64.b64encode(ART_PATH.read_bytes()).decode() if ART_PATH.exists() else ""

st.markdown(f"""
<section class="hero">
 <div class="petal p1">✿</div><div class="petal p2">❀</div><div class="petal p3">✧</div>
 <div class="hero-copy">
  <div class="eyebrow">A tiny botanical machine-learning playground</div>
  <div class="hero-title">Let curiosity<br><span>blossom.</span></div>
  <div class="hero-text">Meet the Iris flower through a living, interactive classification studio. Play with petal and sepal measurements, watch the prediction bloom, explore the data, and tell us what you think.</div>
  <div style="margin-top:20px"><span class="eyebrow">150 specimens&nbsp; · &nbsp;4 dimensions&nbsp; · &nbsp;3 species</span></div>
 </div>
 {f'<img class="hero-art" src="{art_uri}" alt="Illustrated iris flower" />' if art_uri else ''}
</section>
""", unsafe_allow_html=True)

st.write("")
metrics = st.columns(4)
for col, args in zip(metrics, [("THE COLLECTION",str(len(data)),"flower observations"),("DIMENSIONS","04","sepal + petal features"),("SPECIES","03","unique Iris varieties"),("MODEL ACCURACY",f"{evaluation[model_name]['accuracy']:.1%}","held-out test data")]):
    with col: card(*args)

st.write("")
tab_play, tab_garden, tab_lab, tab_feedback = st.tabs(["🌸  Play with a flower", "🌿  Explore the garden", "🧪  Model greenhouse", "💌  Your feedback"])

with tab_play:
    st.markdown('<div class="section-kicker">01 / Live flower identification</div><div class="section-title">Your flower is taking shape.</div><div class="section-note">Change the measurements in the sidebar. Your result and probabilities recalculate automatically.</div>', unsafe_allow_html=True)
    a,b = st.columns([1,1], gap="large")
    with a:
        st.markdown(f'<div class="glass"><div class="metric-label">THE MODEL THINKS IT IS</div><div class="metric-value" style="font-family:Playfair Display;font-size:42px">{predicted.title()} 🌸</div><p>{SPECIES_INFO[predicted]}</p><div class="metric-hint">Top predicted probability · {top_probability:.1%}</div></div>', unsafe_allow_html=True)
        st.progress(top_probability)
        if top_probability >= .8: st.success("The model has a strong preference for this species.")
        elif top_probability >= .55: st.info("The model leans toward this species, but there is some uncertainty.")
        else: st.warning("This flower is a tricky one. The model is not very certain.")
        st.markdown("**Your current measurements**")
        pretty = pd.DataFrame({"Measurement":[DISPLAY[x] for x in FEATURES],"Centimetres":[f"{x:.1f} cm" for x in st.session_state.measurements]})
        st.dataframe(pretty, use_container_width=True, hide_index=True)
    with b:
        st.markdown("**How the model sees the possibilities**")
        pframe = pd.DataFrame({"Species":[s.title() for s in SPECIES],"Probability":[probability_map[s] for s in SPECIES],"Color":[PALETTE[s] for s in SPECIES]})
        fig = px.bar(pframe, x="Probability", y="Species", orientation="h", text=pframe["Probability"].map(lambda x:f"{x:.0%}"), color="Species",
                     color_discrete_map={s.title():PALETTE[s] for s in SPECIES}, range_x=[0,1])
        fig.update_layout(height=280, margin=dict(l=5,r=15,t=10,b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.25)",
                          font_color="#74546c",xaxis=dict(tickformat=".0%",gridcolor="#ead8e3"),yaxis=dict(title=None))
        fig.update_traces(textposition="outside",cliponaxis=False)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        st.caption("Probabilities are model estimates, not a guarantee of botanical identity.")
    st.markdown('<div class="flower-divider">✿ ❀ ✿ ❀ ✿</div>', unsafe_allow_html=True)
    st.markdown("**Try a little experiment**")
    quick = st.columns(3)
    for col, (name, vals) in zip(quick, list(PRESETS.items())[:3]):
        with col:
            if st.button(f"Try {name}", use_container_width=True, key="quick_"+name):
                for key, val in zip(["sl","sw","pl","pw"], vals): st.session_state[key] = float(val)
                st.session_state.measurements = vals.copy()
                st.rerun()

with tab_garden:
    st.markdown('<div class="section-kicker">02 / The flower collection</div><div class="section-title">Explore the Iris garden.</div><div class="section-note">Filter the collection and explore how petal and sepal dimensions separate the three species.</div>', unsafe_allow_html=True)
    chosen = st.multiselect("Show species", SPECIES, default=SPECIES, format_func=str.title)
    filtered = data[data.species.isin(chosen)]
    x_axis = st.selectbox("Horizontal axis", FEATURES, index=2, format_func=lambda x:DISPLAY[x])
    y_axis = st.selectbox("Vertical axis", FEATURES, index=3, format_func=lambda x:DISPLAY[x])
    scatter = px.scatter(filtered, x=x_axis, y=y_axis, color="species", color_discrete_map=PALETTE,
                         hover_data=FEATURES, labels={x_axis:DISPLAY[x_axis]+" (cm)",y_axis:DISPLAY[y_axis]+" (cm)"},
                         category_orders={"species":SPECIES})
    scatter.update_traces(marker=dict(size=11,opacity=.82,line=dict(width=1,color="white")))
    scatter.update_layout(height=470,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.35)",
                          font_color="#76586f",legend_title_text="Species",xaxis=dict(gridcolor="#eadde6"),yaxis=dict(gridcolor="#eadde6"))
    st.plotly_chart(scatter,use_container_width=True)
    c1,c2,c3 = st.columns(3)
    for col,s in zip([c1,c2,c3],SPECIES):
        with col:
            count=int((data.species==s).sum())
            card(s.upper(),str(count),"flowers in the collection")
    with st.expander("Open the raw dataset"):
        st.dataframe(filtered.rename(columns=DISPLAY).rename(columns={"species":"Species"}),use_container_width=True,hide_index=True)
        st.download_button("⬇ Download filtered CSV",filtered.to_csv(index=False).encode("utf-8"),"iris_filtered.csv","text/csv")

with tab_lab:
    st.markdown('<div class="section-kicker">03 / Model greenhouse</div><div class="section-title">Behind the bloom.</div><div class="section-note">Compare two supervised learning models trained and evaluated on the same stratified split.</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    for col,(name,res) in zip([c1,c2],evaluation.items()):
        with col: card(name.upper(),f"{res['accuracy']:.1%}","test-set accuracy")
    selected=evaluation[model_name]
    left,right=st.columns([1,1],gap="large")
    with left:
        st.markdown("**Confusion matrix**")
        fig,ax=plt.subplots(figsize=(5,3.8),dpi=140)
        fig.patch.set_alpha(0); ax.set_facecolor("#fff8fb")
        im=ax.imshow(selected["matrix"],cmap="PuRd")
        ax.set_xticks(range(3),[s.title() for s in SPECIES],rotation=15);ax.set_yticks(range(3),[s.title() for s in SPECIES])
        ax.set_xlabel("Predicted");ax.set_ylabel("Actual")
        for i in range(3):
            for j in range(3): ax.text(j,i,str(selected["matrix"][i,j]),ha="center",va="center",color="#56384e",fontweight="bold")
        fig.tight_layout();st.pyplot(fig,use_container_width=True);plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report=pd.DataFrame(selected["report"]).T.loc[[*SPECIES,"macro avg","weighted avg"],["precision","recall","f1-score","support"]]
        st.dataframe(report.round(3),use_container_width=True)
        st.caption(f"{train_count} training observations · {test_count} test observations")
    comp=pd.DataFrame([{"Model":n,"Accuracy":r["accuracy"]} for n,r in evaluation.items()])
    st.markdown("**Model comparison**")
    st.bar_chart(comp.set_index("Model"),y="Accuracy")

with tab_feedback:
    st.markdown('<div class="section-kicker">04 / A note for the gardener</div><div class="section-title">Help this little project grow.</div><div class="section-note">Your feedback helps us make the experience more useful, intuitive, and delightful.</div>', unsafe_allow_html=True)
    if st.session_state.feedback_sent:
        st.success("Thank you! Your feedback has been received in this session. 🌷")
        if st.button("Leave another note"): st.session_state.feedback_sent=False;st.rerun()
    else:
        with st.form("feedback_form"):
            rating=st.feedback("stars",key="experience_rating")
            sentiment=st.radio("How did the experience feel?",["Delightful","Interesting","A little confusing","Needs more features"],horizontal=True)
            comments=st.text_area("What should we improve or add?",placeholder="Tell us what you liked, what felt confusing, or what you'd love to explore…",height=120)
            submitted=st.form_submit_button("Send my feedback ✿",use_container_width=True)
            if submitted:
                if rating is None: st.warning("Please leave a star rating before sending.")
                else:
                    st.session_state.feedback_sent=True
                    st.session_state.feedback_payload={"rating":rating+1,"sentiment":sentiment,"comments":comments}
                    st.rerun()
    st.info("Feedback is held in the current app session. For permanent collection, connect a database or form endpoint before publishing.")

st.divider()
with st.expander("🌱 How does the flower identification work?"):
    st.write("K-Nearest Neighbors compares a new flower with nearby training examples and predicts the most common species among its five nearest neighbors. Logistic Regression learns relationships between the four measurements and estimates a probability for each species. Both models use StandardScaler and the same stratified 80:20 train/test split.")
st.markdown('<div class="footer">IRIS BLOOM · SECTION 8A · GROWN WITH CURIOSITY ✿</div>',unsafe_allow_html=True)
