import re
import requests
from html.parser import HTMLParser
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

class PageParser(HTMLParser):
    def __init__(self, base_domain):
        super().__init__()
        self.base_domain = base_domain
        self.title = ""
        self._in_title = False
        self.links = []
        self.input_types = []
        self.forms = 0
        self.iframes = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        elif tag == "input":
            self.input_types.append(attrs.get("type", "").lower())
        elif tag == "form":
            self.forms += 1
        elif tag == "iframe":
            self.iframes += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data

    def pct_external_links(self):
        if not self.links:
            return 0.0
        external = [l for l in self.links if urlparse(l).netloc and urlparse(l).netloc != self.base_domain]
        return len(external) / len(self.links)

    def has_login_form(self):
        return any(t in ["password", "email"] for t in self.input_types)


def is_valid_url(url):
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
        return len(hostname.rsplit(".", 1)[0]) >= 1
    except Exception:
        return False


def extract_lexical_features(url):
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = url
    hostname_parts = hostname.split(".")
    return {
        "NumDots": full_url.count("."),
        "SubdomainLevel": max(len(hostname_parts) - 2, 0),
        "PathLevel": len([p for p in path.split("/") if p]),
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
        "RandomString": int(bool(re.search(r"[a-z0-9]{10,}", hostname.lower())) and sum(ch in "aeiou" for ch in hostname.lower()) / max(len(hostname), 1) < 0.2),
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


def fetch_html_features(url):
    full_url = url if "://" in url else "http://" + url
    try:
        resp = requests.get(full_url, timeout=8, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        base_domain = urlparse(full_url).netloc
        parser = PageParser(base_domain)
        parser.feed(resp.text)
        title = parser.title.strip()
        return {
            "fetched": True,
            "unreachable": False,
            "pct_external_links": round(parser.pct_external_links(), 3),
            "has_login_form": int(parser.has_login_form()),
            "suspicious_title": int(any(w in title.lower() for w in SENSITIVE_WORDS)),
            "num_iframes": parser.iframes,
            "missing_title": int(len(title) == 0),
            "page_title": title or "(no title)",
            "status_code": resp.status_code,
            "redirected": resp.url != full_url,
            "final_url": resp.url,
        }
    except requests.exceptions.Timeout:
        return {"fetched": False, "unreachable": True, "error": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"fetched": False, "unreachable": True, "error": "connection_error"}
    except Exception as e:
        return {"fetched": False, "unreachable": False, "error": str(e)}


def compute_deep_score(url_proba, html):
    """
    Combine URL model score with HTML signals.
    If the page is unreachable (timeout or connection error),
    treat that as a strong suspicious signal — a legitimate site
    that someone is sharing a link to should normally be reachable.
    """
    if not html.get("fetched"):
        if html.get("unreachable"):
            # Page doesn't exist or can't be reached —
            # boost score significantly toward phishing
            boosted = min(url_proba + 0.45, 0.95)
            return round(boosted, 4)
        # Other errors — return URL-only score unchanged
        return url_proba

    html_score = (
        html["pct_external_links"] * 0.30 +
        html["has_login_form"] * 0.25 +
        html["suspicious_title"] * 0.20 +
        min(html["num_iframes"] / 5, 1.0) * 0.15 +
        html["missing_title"] * 0.10
    )
    return round(min(0.60 * url_proba + 0.40 * html_score, 1.0), 4)


def predict_url(url, deep=False):
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
    "Paste a URL below and the model will estimate whether it's **phishing** or **legitimate**."
)

url_input = st.text_input("URL to check", placeholder="e.g. https://www.example.com/login")

st.markdown("**Analysis Mode**")
mode = st.radio(
    "mode",
    [
        "🔵 Quick — URL structure only (instant)",
        "🟢 Deep — Fetch page HTML (~5-10 sec, more accurate)",
    ],
    label_visibility="collapsed",
)
deep_mode = mode.startswith("🟢")

if deep_mode:
    st.info(
        "⚠️ **Deep mode will actually visit the URL** to fetch its HTML. "
        "Only use on URLs you are reasonably confident are safe."
    )

check_clicked = st.button("Check URL", type="primary")

if check_clicked:
    url = url_input.strip()

    if not url:
        st.warning("⚠️ Please enter a URL first.")

    elif not is_valid_url(url):
        st.error(
            "❌ **Invalid URL** — please enter a real website address.\n\n"
            "Examples:\n"
            "- `https://www.google.com`\n"
            "- `http://paypal-secure-login.verify-account.tk/signin`\n\n"
            "Random text like `jhghjhgjh` cannot be analysed."
        )

    else:
        with st.spinner("Analysing URL..." if not deep_mode else "Fetching page and analysing..."):
            result = predict_url(url, deep=deep_mode)

        proba = result["phishing_probability"]
        label = result["prediction"]
        html = result.get("html_info", {})

        st.divider()

        if label == "Phishing":
            st.error(f"⚠️ **{label}** — estimated phishing probability: {proba:.1%}")
        else:
            st.success(f"✅ **{label}** — estimated phishing probability: {proba:.1%}")

        st.progress(proba)

        # Deep mode result details
        if deep_mode:
            if html.get("fetched"):
                col1, col2, col3 = st.columns(3)
                col1.metric("URL-only score", f"{result['url_only_probability']:.1%}")
                col2.metric("External links", f"{html['pct_external_links']:.1%}")
                col3.metric("Login form", "Yes" if html["has_login_form"] else "No")
                with st.expander("🌐 Page details"):
                    st.write(f"**Page title:** {html['page_title']}")
                    st.write(f"**Status code:** {html['status_code']}")
                    st.write(f"**Redirected:** {'Yes → ' + html['final_url'] if html['redirected'] else 'No'}")
                    st.write(f"**Iframes found:** {html['num_iframes']}")
                    st.write(f"**Suspicious title:** {'Yes' if html['suspicious_title'] else 'No'}")

            elif html.get("unreachable"):
                # Page couldn't be reached — explain why score was boosted
                error_type = html.get("error", "")
                if error_type == "timeout":
                    reason = "the page took too long to respond"
                else:
                    reason = "the page does not exist or could not be reached"

                st.warning(
                    f"⚠️ **Page unreachable** — {reason}.\n\n"
                    "A legitimate website that someone shares a link to should normally "
                    "be reachable. An unreachable URL is itself a suspicious signal, "
                    "so the phishing probability has been increased accordingly.\n\n"
                    "**Do not attempt to visit this URL.**"
                )
            else:
                st.warning(f"⚠️ Could not analyse page: {html.get('error', 'Unknown error')}. Showing URL-only result.")

        with st.expander("📊 See extracted URL features"):
            feat_df = pd.DataFrame(result["features"].items(), columns=["Feature", "Value"])
            st.dataframe(feat_df, width="stretch")

        if not deep_mode:
            st.caption("Quick mode only looks at URL structure — no page is visited.")

st.divider()
with st.expander("ℹ️ About this project"):
    st.markdown("""
        **AI-Based Phishing URL Detection** — ECE 569A, Team 16 — Secure Mind AI.

        - **Quick mode:** lightweight XGBoost on 26 URL-only features — instant and safe.
        - **Deep mode:** fetches page HTML for extra signals. If the page is unreachable,
          that itself is treated as a suspicious signal and the phishing score is boosted.
    """)
