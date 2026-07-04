import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

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


def is_valid_url(url: str) -> bool:
    url = url.strip()
    if " " in url:
        return False
    test_url = url if "://" in url else "http://" + url
    try:
        parsed = urlparse(test_url)
        hostname = parsed.netloc or ""
        hostname = hostname.split(":")[0]
        if not hostname or "." not in hostname:
            return False
        tld = hostname.rsplit(".", 1)[-1]
        if not re.match(r"^[a-zA-Z]{2,6}$", tld):
            if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
                return False
        domain_body = hostname.rsplit(".", 1)[0]
        if len(domain_body) < 1:
            return False
        return True
    except Exception:
        return False


def extract_lexical_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = url
    hostname_parts = hostname.split(".")
    subdomain_level = max(len(hostname_parts) - 2, 0)
    path_level = len([p for p in path.split("/") if p])
    return {
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


def fetch_html_features(url: str) -> dict:
    """Fetch the actual page and extract extra HTML-based features."""
    full_url = url if "://" in url else "http://" + url
    try:
        resp = requests.get(full_url, timeout=8, allow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")

        all_links = soup.find_all("a", href=True)
        parsed_base = urlparse(full_url)
        base_domain = parsed_base.netloc

        external_links = [
            a for a in all_links
            if urlparse(a["href"]).netloc and urlparse(a["href"]).netloc != base_domain
        ]
        pct_external = len(external_links) / max(len(all_links), 1)

        forms = soup.find_all("form")
        has_login_form = any(
            inp.get("type", "").lower() in ["password", "email"]
            for form in forms for inp in form.find_all("input")
        )

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        suspicious_title = any(w in title.lower() for w in SENSITIVE_WORDS)

        iframes = len(soup.find_all("iframe"))
        missing_title = int(len(title) == 0)

        return {
            "fetched": True,
            "pct_external_links": round(pct_external, 3),
            "has_login_form": int(has_login_form),
            "suspicious_title": int(suspicious_title),
            "num_iframes": iframes,
            "missing_title": missing_title,
            "page_title": title or "(no title)",
            "status_code": resp.status_code,
            "redirected": resp.url != full_url,
            "final_url": resp.url,
        }
    except requests.exceptions.Timeout:
        return {"fetched": False, "error": "Page took too long to respond (timeout)."}
    except requests.exceptions.ConnectionError:
        return {"fetched": False, "error": "Could not connect to the page."}
    except Exception as e:
        return {"fetched": False, "error": str(e)}


def compute_deep_score(url_proba: float, html: dict) -> float:
    """
    Combine URL-model probability with HTML signals into a final score.
    Weights: URL model 60%, HTML signals 40%.
    """
    if not html.get("fetched"):
        return url_proba

    html_score = 0.0
    html_score += html["pct_external_links"] * 0.30
    html_score += html["has_login_form"] * 0.25
    html_score += html["suspicious_title"] * 0.20
    html_score += min(html["num_iframes"] / 5, 1.0) * 0.15
    html_score += html["missing_title"] * 0.10

    combined = 0.60 * url_proba + 0.40 * html_score
    return round(min(combined, 1.0), 4)


def predict_url(url: str, deep: bool = False) -> dict:
    feats = extract_lexical_features(url)
    row = pd.DataFrame([[feats[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    url_proba = float(model.predict_proba(row)[0, 1])

    html_info = {}
    final_proba = url_proba

    if deep:
        html_info = fetch_html_features(url)
        final_proba = compute_deep_score(url_proba, html_info)

    label = "Phishing" if final_proba >= 0.5 else "Legitimate"
    return {
        "prediction": label,
        "phishing_probability": final_proba,
        "url_only_probability": url_proba,
        "features": feats,
        "html_info": html_info,
    }


# ── UI ───────────────────────────────────────────────────────────────────────

st.title("🛡️ AI-Based Phishing URL Detector")
st.markdown(
    "Paste a URL below and the model will estimate whether it's **phishing** or "
    "**legitimate**."
)

url_input = st.text_input("URL to check", placeholder="e.g. https://www.example.com/login")

st.markdown("**Analysis Mode**")
mode = st.radio(
    "mode",
    ["🔵 Quick — URL structure only (instant)", "🟢 Deep — Fetch page HTML (~5-10 sec, more accurate)"],
    label_visibility="collapsed",
)
deep_mode = mode.startswith("🟢")

if deep_mode:
    st.info(
        "⚠️ **Deep mode will actually visit the URL** to fetch its HTML. "
        "Only use this on URLs you are reasonably confident are safe to visit, "
        "or that you own. Do not use on highly suspicious links."
    )

check_clicked = st.button("Check URL", type="primary")

if check_clicked:
    url = url_input.strip()

    if not url:
        st.warning("⚠️ Please enter a URL first.")

    elif not is_valid_url(url):
        st.error(
            "❌ **Invalid URL** — please enter a real website address.\n\n"
            "**Valid examples:**\n"
            "- `https://www.google.com`\n"
            "- `http://paypal-secure-login.verify-account.tk/signin`\n\n"
            "Random text like `jhghjhgjh` is not a URL and cannot be analysed."
        )

    else:
        with st.spinner("Analysing URL..." if not deep_mode else "Fetching page and analysing... this may take a few seconds."):
            result = predict_url(url, deep=deep_mode)

        proba = result["phishing_probability"]
        label = result["prediction"]

        st.divider()

        if label == "Phishing":
            st.error(f"⚠️ **{label}** — estimated phishing probability: {proba:.1%}")
        else:
            st.success(f"✅ **{label}** — estimated phishing probability: {proba:.1%}")

        st.progress(proba)

        if deep_mode:
            html = result.get("html_info", {})
            if html.get("fetched"):
                col1, col2, col3 = st.columns(3)
                col1.metric("URL-only score", f"{result['url_only_probability']:.1%}")
                col2.metric("External links", f"{html['pct_external_links']:.1%}")
                col3.metric("Login form found", "Yes" if html["has_login_form"] else "No")

                with st.expander("🌐 Page details"):
                    st.write(f"**Page title:** {html['page_title']}")
                    st.write(f"**Status code:** {html['status_code']}")
                    st.write(f"**Redirected:** {'Yes → ' + html['final_url'] if html['redirected'] else 'No'}")
                    st.write(f"**Iframes found:** {html['num_iframes']}")
                    st.write(f"**Suspicious title keywords:** {'Yes' if html['suspicious_title'] else 'No'}")
            else:
                st.warning(f"⚠️ Could not fetch page: {html.get('error', 'Unknown error')}. Showing URL-only result.")

        with st.expander("See extracted URL features"):
            feat_df = pd.DataFrame(result["features"].items(), columns=["Feature", "Value"])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        if not deep_mode:
            st.caption(
                "Note: Quick mode only looks at URL structure — no page is visited. "
                "Switch to Deep mode for a more thorough analysis."
            )

st.divider()
with st.expander("ℹ️ About this project"):
    st.markdown("""
        This demo accompanies the **AI-Based Phishing URL Detection Using Machine Learning**
        project (ECE 569A, Team 16 — Secure Mind AI).

        - **Full model** used 10,000 webpages and 48 features, comparing Logistic Regression,
          Random Forest, SVM, and XGBoost.
        - **Quick mode** uses a lightweight XGBoost model on 26 URL-only features — instant, safe.
        - **Deep mode** additionally fetches the page HTML to extract login forms, external links,
          iframes, and title keywords for a more accurate combined score.
    """)
