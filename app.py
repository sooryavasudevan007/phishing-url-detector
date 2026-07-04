import re
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)

@st.cache_resource
def load_model_assets():
    model = joblib.load("lightweight_url_model.pkl")
    feature_columns = joblib.load("lightweight_feature_columns.pkl")
    return model, feature_columns

model, FEATURE_COLUMNS = load_model_assets()

SENSITIVE_WORDS = [
    "secure", "account", "update", "login", "verify", "signin", "banking",
    "confirm", "password", "alert", "billing", "suspend", "limited",
]

KNOWN_BRANDS = [
    "paypal", "apple", "amazon", "microsoft", "google", "facebook", "netflix",
    "bankofamerica", "wellsfargo", "chase", "ebay", "instagram", "dropbox",
]

def extract_lexical_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = url
    hostname_parts = hostname.split(".")
    subdomain_level = max(len(hostname_parts) - 2, 0)
    path_level = len([p for p in path.split("/") if p])
    features = {
        "NumDots": full_url.count("."),
        "SubdomainLevel": subdomain_level,
        "PathLevel": path_level,
        "UrlLength": len(full_url),
        "NumDash": full_url.count("-"),
        "NumDashInHostname": hostname.count("-"),
        "AtSymbol": int("@" in full_url),
        "TildeSymbol": int("~" in full_url),
        "NumUnderscore": full_url.count("_"),
        "NumPercent": full_url.count("%"),
        "NumQueryComponents": len(query.split("&")) if query else 0,
        "NumAmpersand": full_url.count("&"),
        "NumHash": full_url.count("#"),
        "NumNumericChars": sum(c.isdigit() for c in full_url),
        "NoHttps": int(parsed.scheme != "https"),
        "RandomString": int(
            bool(re.search(r"[a-z0-9]{10,}", hostname.lower()))
            and sum(ch in "aeiou" for ch in hostname.lower()) / max(len(hostname), 1) < 0.2
        ),
        "IpAddress": int(bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname.split(":")[0]))),
        "DomainInSubdomains": int(any(b in ".".join(hostname_parts[:-2]) for b in KNOWN_BRANDS)),
        "DomainInPaths": int(any(b in path.lower() for b in KNOWN_BRANDS)),
        "HttpsInHostname": int("https" in hostname.lower()),
        "HostnameLength": len(hostname),
        "PathLength": len(path),
        "QueryLength": len(query),
        "DoubleSlashInPath": int("//" in path),
        "NumSensitiveWords": sum(word in full_url.lower() for word in SENSITIVE_WORDS),
        "EmbeddedBrandName": int(any(b in (path + query).lower() for b in KNOWN_BRANDS)),
    }
    return features

def predict_url(url: str) -> dict:
    feats = extract_lexical_features(url)
    row = pd.DataFrame([[feats[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    proba_phishing = float(model.predict_proba(row)[0, 1])
    label = "Phishing" if proba_phishing >= 0.5 else "Legitimate"
    return {"prediction": label, "phishing_probability": proba_phishing, "features": feats}

st.title("🛡️ AI-Based Phishing URL Detector")
st.markdown(
    "Paste a URL below and the model will estimate whether it's **phishing** or "
    "**legitimate**, based only on the structure of the URL itself — no page is "
    "visited or downloaded."
)

url_input = st.text_input("URL to check", placeholder="e.g. https://www.example.com/login")
check_clicked = st.button("Check URL", type="primary")

if check_clicked and url_input.strip():
    with st.spinner("Analysing URL..."):
        result = predict_url(url_input.strip())
    proba = result["phishing_probability"]
    label = result["prediction"]
    st.divider()
    if label == "Phishing":
        st.error(f"⚠️ **{label}** — estimated phishing probability: {proba:.1%}")
    else:
        st.success(f"✅ **{label}** — estimated phishing probability: {proba:.1%}")
    st.progress(proba)
    with st.expander("See extracted URL features"):
        feat_df = pd.DataFrame(result["features"].items(), columns=["Feature", "Value"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
    st.caption(
        "Note: this lightweight model only looks at the URL's text structure — "
        "it does not fetch or render the actual webpage."
    )
elif check_clicked:
    st.warning("Please enter a URL first.")

st.divider()
with st.expander("ℹ️ About this project"):
    st.markdown("""
        This demo accompanies the **AI-Based Phishing URL Detection Using Machine Learning**
        project (ECE 569A, Team 16 — Secure Mind AI).

        - **Full model** used 10,000 webpages and 48 features, comparing Logistic Regression,
          Random Forest, SVM, and XGBoost.
        - **This demo** uses a lightweight XGBoost model trained on 26 URL-only features
          for instant, safe screening without visiting the site.
    """)
