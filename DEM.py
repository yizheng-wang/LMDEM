# dem_streamlit_gmsh_app.py
# ------------------------------------------------------------
# Streamlit DEM app (Gmsh route B):
# - Upload a .msh (Gmsh) with Physical Groups:
#     2D:  Omega
#     1D:  Gamma_u   (Dirichlet boundary)
#     1D:  Gamma_t   (Neumann / traction boundary)
#
# - Domain integration: triangle Gauss (1-pt or 3-pt) on Omega triangles
# - Boundary integration:
#     Gamma_u: line Gauss (2-pt/3-pt) for penalty BC (and for hard BC distance)
#     Gamma_t: line Gauss (2-pt/3-pt) for traction work
#
# Problems:
# - Poisson (scalar):     Pi = ∫ 0.5|∇u|^2 dΩ - ∫ f u dΩ + Wbc
# - Linear elasticity:    Pi = ∫ psi(eps) dΩ - ∫ b·u dΩ - ∫ t·u dΓ + Wbc
# - Neo-Hookean (2D):     Pi = ∫ psi(F) dΩ - ∫ b·u dΩ - ∫ t·u dΓ + Wbc
#
# Dirichlet handling:
# - Penalty:   Wbc = λ ∫_{Gamma_u} ||u - ubar||^2 dΓ
# - Hard:      u(x)=ubar(x) + d(x,Gamma_u)*NN(x)
#   where d(x,Gamma_u) is the point-to-segment distance to Gamma_u.
#   (Works for arbitrary geometry; exact on Gamma_u because d=0 on the boundary.)
# - You can enable Hard, Penalty, or both.
#
# Performance improvements applied (your “5 points”):
# 1) Cache heavy mesh parsing + quadrature build via st.cache_data
# 2) Faster mesh preview using triplot (no per-triangle Python loops)
# 3) Optional “Show mesh preview” checkbox to avoid re-drawing on every rerun
# 4) Lighter mesh signature (avoid unstable hash() for bytes; use stable fingerprint)
# 5) Spinner feedback during load/build phases
# ------------------------------------------------------------
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import json
import re

import io
import math
import time
import zlib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from soap import SOAP

import torch
import torch.nn as nn
from torch.autograd import grad

# mesh reader
import meshio
import io
import os
import tempfile
import subprocess
import shlex
import shutil

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =============================
# LLM API integration (multi-provider support)
# =============================
import os
from typing import Literal
from openai import OpenAI

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import requests
except ImportError:
    requests = None

def get_client_ip() -> str:
    """
    Get client IP address from Streamlit context or external API.
    Returns IP address string or 'unknown' if unavailable.
    """
    detected_ip = None
    
    # Try Streamlit's built-in context (Streamlit 1.28+)
    try:
        if hasattr(st, 'context') and hasattr(st.context, 'ip_address'):
            ip = st.context.ip_address
            if ip:
                detected_ip = str(ip)
                # If it's localhost, we'll try external API as fallback
                if detected_ip and not detected_ip.startswith('127.') and detected_ip != 'localhost':
                    return detected_ip
    except Exception:
        pass
    
    # Fallback: try headers from context
    try:
        if hasattr(st, 'context') and hasattr(st.context, 'headers'):
            headers = st.context.headers
            # Check common IP headers (in order of preference)
            for header in ['x-forwarded-for', 'x-real-ip', 'x-client-ip', 'cf-connecting-ip']:
                if header in headers:
                    ip = headers[header]
                    # x-forwarded-for can contain multiple IPs, take the first one
                    if ',' in ip:
                        ip = ip.split(',')[0].strip()
                    if ip and not ip.startswith('127.') and ip != 'localhost':
                        return ip
    except Exception:
        pass
    
    # Fallback: try st.client (older Streamlit versions)
    try:
        if hasattr(st, 'client') and hasattr(st.client, 'request'):
            headers = st.client.request.headers
            for header in ['x-forwarded-for', 'x-real-ip', 'x-client-ip', 'cf-connecting-ip']:
                if header in headers:
                    ip = headers[header]
                    if ',' in ip:
                        ip = ip.split(',')[0].strip()
                    if ip and not ip.startswith('127.') and ip != 'localhost':
                        return ip
            if hasattr(st.client.request, 'remote_addr'):
                ip = st.client.request.remote_addr
                if ip and not ip.startswith('127.') and ip != 'localhost':
                    return ip
    except Exception:
        pass
    
    # Final fallback: use external API to get real public IP (if localhost detected or no IP found)
    if not detected_ip or detected_ip.startswith('127.') or detected_ip == 'localhost':
        try:
            if requests:
                # Try multiple free IP detection APIs
                api_urls = [
                    'https://api.ipify.org?format=text',
                    'https://ifconfig.me/ip',
                    'https://icanhazip.com',
                    'https://checkip.amazonaws.com',
                ]
                for url in api_urls:
                    try:
                        response = requests.get(url, timeout=3)
                        if response.status_code == 200:
                            ip = response.text.strip()
                            # Validate IP format (basic check)
                            if ip and '.' in ip and not ip.startswith('127.'):
                                return ip
                    except Exception:
                        continue
        except Exception:
            pass
    
    return detected_ip if detected_ip else 'unknown'


def get_ip_location(ip: str) -> dict | None:
    """
    Get geographic location from IP address using free API.
    Returns dict with: lat, lon, city, country, country_code, region, timezone
    Falls back to None if API fails or IP is invalid.
    """
    if not ip or ip == 'unknown':
        return None
    
    # Skip localhost IPs (they can't be geolocated)
    if ip.startswith('127.') or ip.startswith('localhost') or ip.startswith('::1'):
        return None
    
    try:
        # Use ip-api.com (free, no key required, 45 requests/minute)
        # Format: http://ip-api.com/json/{ip}?fields=status,message,country,countryCode,region,regionName,city,lat,lon,timezone
        if requests is None:
            return None
        
        url = f"http://ip-api.com/json/{ip}?fields=status,message,country,countryCode,region,regionName,city,lat,lon,timezone"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get('status') == 'success':
            lat = float(data.get('lat', 0))
            lon = float(data.get('lon', 0))
            # Validate coordinates
            if lat == 0 and lon == 0:
                return None
            return {
                'lat': lat,
                'lon': lon,
                'city': str(data.get('city', '')),
                'country': str(data.get('country', '')),
                'country_code': str(data.get('countryCode', '')),
                'region': str(data.get('regionName', '')),
                'timezone': str(data.get('timezone', '')),
            }
        else:
            # API returned error
            return None
    except Exception as e:
        # API failed, return None
        return None
    
    return None


def get_and_increment_visit_count() -> tuple[int, dict]:
    """
    Get and increment the total visit count, and record visitor location.
    Stores the count in 'visit_count.json' and visitor records in 'visitor_locations.json'.
    Returns (count, location_info_dict).
    """
    from pathlib import Path
    from datetime import datetime
    
    script_dir = Path(__file__).parent
    count_file = script_dir / "visit_count.json"
    locations_file = script_dir / "visitor_locations.json"
    
    # Get client IP and location
    ip = get_client_ip()
    location = get_ip_location(ip) if ip != 'unknown' else None
    
    # Read existing count or initialize to 0
    try:
        if count_file.exists():
            with open(count_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = data.get('total_visits', 0)
        else:
            count = 0
    except Exception:
        count = 0
    
    # Increment count
    count += 1
    
    # Save updated count
    try:
        with open(count_file, 'w', encoding='utf-8') as f:
            json.dump({'total_visits': count}, f, indent=2)
    except Exception:
        pass
    
    # Record visitor location
    if location:
        try:
            # Read existing locations
            if locations_file.exists():
                with open(locations_file, 'r', encoding='utf-8') as f:
                    locations_data = json.load(f)
                    visitors = locations_data.get('visitors', [])
            else:
                visitors = []
            
            # Add new visitor record
            visitor_record = {
                'ip': ip,
                'timestamp': datetime.now().isoformat(),
                'lat': location['lat'],
                'lon': location['lon'],
                'city': location['city'],
                'country': location['country'],
                'country_code': location['country_code'],
                'region': location['region'],
                'timezone': location['timezone'],
            }
            visitors.append(visitor_record)
            
            # Keep only last 1000 records to avoid file bloat
            if len(visitors) > 1000:
                visitors = visitors[-1000:]
            
            # Save updated locations
            with open(locations_file, 'w', encoding='utf-8') as f:
                json.dump({'visitors': visitors}, f, indent=2)
        except Exception:
            pass
    
    location_info = location if location else {'ip': ip, 'error': 'Location unavailable'}
    return count, location_info


def load_visitor_locations() -> list[dict]:
    """
    Load all visitor location records from JSON file.
    Returns list of visitor records with lat, lon, city, country, etc.
    """
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    locations_file = script_dir / "visitor_locations.json"
    
    try:
        if locations_file.exists():
            with open(locations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('visitors', [])
    except Exception:
        pass
    
    return []


def get_visitor_map_data() -> tuple[list[dict], int]:
    """
    Get visitor locations formatted for Streamlit map visualization.
    Returns (map_dataframe_dict_list, unique_visitor_count).
    Each dict has: lat, lon.
    Each visitor gets a red dot on the map (no grouping - show all visitors).
    """
    visitors = load_visitor_locations()
    
    if not visitors:
        return [], 0
    
    # Count unique IPs
    unique_ips = set(v.get('ip', '') for v in visitors if v.get('ip'))
    unique_count = len(unique_ips)
    
    # Show each visitor as a separate point (no grouping - each visitor gets a red dot)
    map_data = []
    for v in visitors:
        if 'lat' in v and 'lon' in v and v.get('lat') != 0 and v.get('lon') != 0:
            # Each visitor gets their own point
            map_data.append({
                'lat': v['lat'],
                'lon': v['lon'],
            })
    
    return map_data, unique_count


# LLM Provider types
LLMProvider = Literal["openai", "anthropic", "google", "ollama"]

def _get_llm_api_key(provider: LLMProvider) -> str:
    """
    Get LLM API key for a specific provider from:
      1) per-session UI input (st.session_state)
      2) environment variable
      3) Streamlit secrets
    """
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "ollama": "OLLAMA_BASE_URL",  # Ollama uses URL, not key
    }
    
    secret_key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "ollama": "OLLAMA_BASE_URL",
    }
    
    ui_key_map = {
        "openai": "openai_api_key_ui",
        "anthropic": "anthropic_api_key_ui",
        "google": "google_api_key_ui",
        "ollama": "ollama_base_url_ui",
    }
    
    # 1) session-only UI override
    try:
        ui_key_name = ui_key_map.get(provider, "")
        if ui_key_name:
            ui_key = str(st.session_state.get(ui_key_name, "") or "").strip()
            if ui_key:
                return ui_key
    except Exception:
        pass
    
    # 2) environment variable
    env_var = env_var_map.get(provider, "")
    if env_var:
        key = os.getenv(env_var, "").strip()
        if key:
            return key
    
    # 3) Streamlit secrets
    try:
        secret_key = secret_key_map.get(provider, "")
        if secret_key:
            key2 = str(st.secrets.get(secret_key, "")).strip()
            if key2:
                return key2
    except Exception:
        pass
    
    return ""

def _get_openai_client():
    """Get OpenAI client (backward compatibility)."""
    key = _get_llm_api_key("openai")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it via environment variable "
            "OPENAI_API_KEY or Streamlit Secrets (OPENAI_API_KEY)."
        )
    return OpenAI(api_key=key)

def _call_llm(
    provider: LLMProvider,
    model: str,
    system: str,
    user: str,
) -> str:
    """
    Universal LLM call interface supporting multiple providers.
    Returns the response text.
    """
    if provider == "openai":
        key = _get_llm_api_key("openai")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
    
    elif provider == "anthropic":
        if Anthropic is None:
            raise RuntimeError("Anthropic SDK not installed. Run: pip install anthropic")
        key = _get_llm_api_key("anthropic")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        client = Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    
    elif provider == "google":
        if genai is None:
            raise RuntimeError("Google Generative AI SDK not installed. Run: pip install google-generativeai")
        key = _get_llm_api_key("google")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        genai.configure(api_key=key)
        model_obj = genai.GenerativeModel(model_name=model)
        # Google's API combines system and user into a single prompt
        full_prompt = f"{system}\n\n{user}"
        resp = model_obj.generate_content(full_prompt)
        return resp.text
    
    elif provider == "ollama":
        if requests is None:
            raise RuntimeError("requests library not installed. Run: pip install requests")
        base_url = _get_llm_api_key("ollama") or "http://localhost:11434"
        url = f"{base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def _parse_llm_geo_json(text: str) -> tuple[str, str]:
    """
    Parse LLM response as JSON: {"chat": "...", "geo": "..."}.
    Return (chat, geo). Robust fallbacks:
      1) strict json.loads
      2) extract first {...} block and json.loads
      3) fallback to fenced ```geo``` extraction
      4) final fallback: whole text as chat, empty geo
    """
    if text is None:
        return "", ""

    raw = text.strip()

    # ---- 1) strict JSON ----
    try:
        obj = json.loads(raw)
        chat = str(obj.get("chat", "")).strip()
        geo = str(obj.get("geo", "")).strip()
        return chat, (geo + "\n" if geo else "")
    except Exception:
        pass

    # ---- 2) try find a JSON object substring ----
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            chat = str(obj.get("chat", "")).strip()
            geo = str(obj.get("geo", "")).strip()
            return chat, (geo + "\n" if geo else "")
        except Exception:
            pass

    # ---- 3) fallback: fenced geo block ----
    m = re.search(r"```geo\s*(.*?)```", raw, flags=re.S | re.I)
    if m:
        geo = m.group(1).strip() + "\n"
        chat = (raw[:m.start()]).strip()
        return chat, geo

    # ---- 4) nothing worked ----
    return raw, ""


def _extract_geo_only(text: str) -> str:
    """
    Robustly extract .geo from LLM output.
    Accept either:
      - plain geo text
      - ```geo ... ```
      - ```text ... ```
    """
    if text is None:
        return ""
    # fenced block
    m = re.search(r"```(?:geo|text)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"



# -----------------------------
# Matplotlib
# -----------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110

# -----------------------------
# Streamlit
# -----------------------------
st.set_page_config(page_title="Deep Energy Method", page_icon="⚡", layout="wide")

# Track visit count (only increment once per session)
if 'visit_count_initialized' not in st.session_state:
    total_visits, location_info = get_and_increment_visit_count()
    st.session_state['visit_count_initialized'] = True
    st.session_state['total_visits'] = total_visits
    st.session_state['visitor_location'] = location_info
else:
    total_visits = st.session_state.get('total_visits', 0)
    location_info = st.session_state.get('visitor_location', {})

st.title("🤩 Deep Energy Method based on LLM")
st.caption("The author is Yizheng Wang, email: wang-yz19@tsinghua.org.cn")
st.caption(f"📊 Total visits: {total_visits:,}")

# Visitor location map
with st.expander("🌍 Visitor Map (click to view)", expanded=False):
    import pandas as pd
    try:
        import pydeck as pdk
        use_pydeck = True
    except ImportError:
        use_pydeck = False
        st.warning("⚠️ pydeck not installed. Install with `pip install pydeck` for red dot markers. Using default st.map() for now.")
    
    map_data, unique_visitors = get_visitor_map_data()
    
    # Collect all map points (all visitors, each gets a red dot)
    all_map_points = []
    if map_data:
        all_map_points.extend(map_data)
    
    # Add current visitor location if available (will be saved on next page load)
    if location_info and 'lat' in location_info and 'lon' in location_info:
        current_lat = location_info.get('lat')
        current_lon = location_info.get('lon')
        # Validate coordinates
        if current_lat and current_lon and current_lat != 0 and current_lon != 0:
            # Always add current location to show on map (even if not saved yet)
            all_map_points.append({
                'lat': current_lat,
                'lon': current_lon,
            })
    
    # Debug and test section
    with st.expander("🔧 Debug & Test", expanded=False):
        current_ip = get_client_ip()
        st.write(f"**Current IP:** `{current_ip}`")
        
        if st.button("Test IP Location API", key="test_ip_location"):
            if current_ip and current_ip != 'unknown' and not current_ip.startswith('127.'):
                with st.spinner("Fetching location..."):
                    test_location = get_ip_location(current_ip)
                    if test_location:
                        st.success("✅ Location fetched successfully!")
                        st.json(test_location)
                        # Show on map
                        df_test = pd.DataFrame([{
                            'lat': test_location['lat'],
                            'lon': test_location['lon']
                        }])
                        st.map(df_test, zoom=5)
                    else:
                        st.error("❌ Failed to get location. The API may be unavailable or rate-limited.")
            else:
                st.warning("⚠️ Cannot test: IP is localhost or unknown. Deploy the app to test with a public IP.")
        
        if st.checkbox("Show debug info", value=False, key="visitor_map_debug"):
            st.code(f"Current IP: {current_ip}")
            st.code(f"Location info: {location_info}")
            visitors_raw = load_visitor_locations()
            st.code(f"Total visitor records: {len(visitors_raw)}")
            st.code(f"Map data points: {len(all_map_points)}")
            if visitors_raw:
                st.json(visitors_raw[:3])  # Show first 3 records
    
    if all_map_points:
        # Create DataFrame for map
        df_map = pd.DataFrame([
            {'lat': d['lat'], 'lon': d['lon']} for d in all_map_points
        ])
        
        st.markdown(f"**Unique visitors:** {unique_visitors} | **Locations:** {len(all_map_points)}")
        
        # Red dot size control
        dot_size = st.slider(
            "🔴 Red dot size",
            min_value=1000,
            max_value=200000,
            value=50000,
            step=5000,
            help="Adjust the size of red dots on the map (in meters)"
        )
        
        # Debug: show data
        if st.checkbox("Show map data", value=False, key="show_map_data_debug"):
            st.dataframe(df_map)
        
        if use_pydeck:
            try:
                # Use pydeck for red dot markers - each visitor gets a red dot
                # Simple configuration for global view
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position=["lon", "lat"],
                    get_color=[255, 0, 0, 255],  # Bright red (RGBA)
                    get_radius=dot_size,  # Adjustable radius in meters
                    pickable=False,  # Disable picking for better performance
                )
                
                # Global view - always start with world map centered
                center_lat = 20  # Fixed center for global view
                center_lon = 0   # Fixed center for global view
                
                # Always use zoom level 1 for global/world view
                zoom_level = 1
                
                view_state = pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=zoom_level,
                    pitch=0,
                    bearing=0,
                )
                
                # Use simple map style (no Mapbox token required)
                r = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style='light',  # Simple light style
                )
                st.pydeck_chart(r)
            except Exception as e:
                st.warning(f"⚠️ pydeck map failed: {str(e)}. Falling back to st.map()")
                # Fallback to st.map() (blue markers, but it works reliably) - always global view
                st.map(df_map, zoom=1)
        else:
            # Use st.map() (blue markers, but it works reliably) - always global view
            st.map(df_map, zoom=1)
        
        # Show location details
        st.markdown("**Recent visitor locations:**")
        visitors = load_visitor_locations()
        recent_visitors = sorted(visitors, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        
        for v in recent_visitors:
            if 'city' in v and 'country' in v:
                city = v.get('city', 'Unknown')
                country = v.get('country', 'Unknown')
                timestamp = v.get('timestamp', '')[:19] if v.get('timestamp') else 'Unknown time'
                st.caption(f"📍 {city}, {country} — {timestamp}")
    else:
        # No data available
        current_ip = get_client_ip()
        if current_ip == 'unknown':
            st.warning("⚠️ Could not detect your IP address. This may be because you're accessing the app locally (127.0.0.1) or through a proxy.")
        elif current_ip.startswith('127.') or current_ip.startswith('localhost'):
            st.info("ℹ️ You're accessing the app locally. IP geolocation only works for public IP addresses. Deploy the app to see visitor locations.")
        elif location_info and 'error' in location_info:
            st.warning(f"⚠️ Could not get location for IP {current_ip}. The geolocation API may be temporarily unavailable.")
        else:
            st.info("No visitor location data available yet. Locations will appear as visitors access the app.")

# -----------------------------
# UI polish (chat look & feel)
# -----------------------------
st.markdown(
    """
<style>
/* Subtle app background */
section.main {
  background:
    radial-gradient(1200px 600px at 15% 5%, rgba(124, 58, 237, .10), transparent 55%),
    radial-gradient(900px 500px at 85% 10%, rgba(6, 182, 212, .10), transparent 55%);
}

/* Chat message cards */
div[data-testid="stChatMessage"] {
  border-radius: 16px;
  border: 1px solid rgba(120, 120, 120, .18);
  box-shadow: 0 8px 22px rgba(0,0,0,.06);
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  font-size: 0.98rem;
  line-height: 1.45;
}

/* Make assistant/user bubbles feel different when Streamlit provides role classes */
div.stChatMessage.stChatMessage--assistant,
div.stChatMessage.stChatMessage--assistant div[data-testid="stChatMessage"] {
  background: linear-gradient(135deg, rgba(124, 58, 237, .07), rgba(6, 182, 212, .06));
}
div.stChatMessage.stChatMessage--user,
div.stChatMessage.stChatMessage--user div[data-testid="stChatMessage"] {
  background: rgba(255,255,255,.02);
}

/* Chat input */
div[data-testid="stChatInput"] textarea {
  border-radius: 14px !important;
  border: 1px solid rgba(120, 120, 120, .25) !important;
  background: rgba(20, 20, 25, .20) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Utilities
# ============================================================



def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def safe_rel_err(pred, ref, eps=1e-12):
    denom = np.maximum(np.abs(ref), eps)
    return np.abs(pred - ref) / denom


def rel_l2_error(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    """Global relative L2 error: ||pred-ref||_2 / max(||ref||_2, eps)."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    r = np.asarray(ref, dtype=np.float64).reshape(-1)
    num = float(np.linalg.norm(p - r))
    den = float(np.linalg.norm(r))
    return num / max(den, float(eps))


def rel_l2_error_per_node_vec(pred_xy: np.ndarray, ref_xy: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-node relative vector L2: ||p-r||_2 / max(||r||_2, eps). Returns (N,)."""
    p = np.asarray(pred_xy, dtype=np.float64)
    r = np.asarray(ref_xy, dtype=np.float64)
    num = np.linalg.norm(p - r, axis=1)
    den = np.linalg.norm(r, axis=1)
    den = np.maximum(den, float(eps))
    return num / den


def eval_expr_torch(expr: str, *, x=None, y=None, z=None, t=None, pi=None):
    """
    torch-safe eval for expressions like:
      sin(pi*x)*sin(pi*y)
      0
      x*(1-x)
    Always returns a torch.Tensor with shape like x.
    """
    allowed = {
        "x": x,
        "y": y,
        "z": z,
        "t": t,
        "pi": torch.pi if pi is None else pi,
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
        "abs": torch.abs,
        "tanh": torch.tanh,
        "sinh": torch.sinh,
        "cosh": torch.cosh,
        "where": torch.where,
        "maximum": torch.maximum,
        "minimum": torch.minimum,
    }
    out = eval(expr, {"__builtins__": {}}, allowed)

    # ---- force tensor output ----
    if torch.is_tensor(out):
        return out

    # scalar number -> tensor broadcast
    if x is None:
        return torch.tensor(float(out), device=y.device if y is not None else None)

    return torch.full_like(x, float(out))


def eval_expr_torch_ext(expr: str, *, x=None, y=None, z=None, t=None, pi=None, **vars):
    """
    Extended torch-safe eval:
    - Same safe function set as eval_expr_torch
    - Plus user-provided tensors/scalars via **vars (e.g., u, ux, eps_xx, lam, mu)
    """
    allowed = {
        "x": x,
        "y": y,
        "z": z,
        "t": t,
        "pi": torch.pi if pi is None else pi,
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
        "abs": torch.abs,
        "tanh": torch.tanh,
        "sinh": torch.sinh,
        "cosh": torch.cosh,
        "where": torch.where,
        "maximum": torch.maximum,
        "minimum": torch.minimum,
    }
    for k, v in (vars or {}).items():
        if v is not None:
            allowed[str(k)] = v

    out = eval(str(expr), {"__builtins__": {}}, allowed)
    if torch.is_tensor(out):
        return out
    # scalar number -> tensor broadcast
    ref = x if x is not None else (y if y is not None else (z if z is not None else None))
    if ref is None:
        return torch.tensor(float(out))
    return torch.full_like(ref, float(out))


def to_torch(x: np.ndarray, device, requires_grad=False):
    t = torch.tensor(x, device=device, dtype=torch.float32)
    t.requires_grad_(requires_grad)
    return t


def stable_fingerprint(b: bytes, head_n: int = 20000) -> int:
    """
    Stable quick fingerprint for bytes: crc32 on head+tail chunks + length.
    (Fast; avoids Python's randomized hash()).
    """
    n = len(b)
    head = b[: min(head_n, n)]
    tail = b[max(0, n - head_n) : n]
    c1 = zlib.crc32(head)
    c2 = zlib.crc32(tail)
    return (n << 32) ^ (c1 << 16) ^ c2


def _sf_conn_is_valid(sf: dict | None, n_nodes: int) -> tuple[bool, str]:
    """
    Validate that sf['conn'] indices are within [0, n_nodes-1].
    Returns (ok, message).
    """
    if sf is None:
        return True, "sf is None"
    if "conn" not in sf:
        return False, "missing 'conn'"
    conn = np.asarray(sf.get("conn"))
    if conn.size == 0:
        return True, "empty conn"
    if conn.ndim != 2:
        return False, f"conn must be 2D, got shape={conn.shape}"
    if not np.issubdtype(conn.dtype, np.integer):
        return False, f"conn dtype must be integer, got {conn.dtype}"
    cmin = int(conn.min())
    cmax = int(conn.max())
    if cmin < 0 or cmax >= int(n_nodes):
        return False, f"conn index out of range: min={cmin}, max={cmax}, n_nodes={int(n_nodes)}"
    return True, "ok"


def _meshio_vtu_bytes(mesh: meshio.Mesh) -> bytes:
    """
    Write a meshio.Mesh to VTU bytes (ParaView-friendly).
    Uses an in-memory buffer when possible; falls back to a temp file.
    """
    try:
        buf = io.BytesIO()
        meshio.write(buf, mesh, file_format="vtu")
        return buf.getvalue()
    except Exception:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vtu") as f:
            tmp = f.name
        try:
            meshio.write(tmp, mesh, file_format="vtu")
            with open(tmp, "rb") as rf:
                return rf.read()
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass


def build_paraview_mesh_2d(
    pts: np.ndarray,
    tris_plot: np.ndarray,
    *,
    quads: np.ndarray | None,
    omega_triangles_n: int,
):
    """
    Build a meshio.Mesh for 2D ParaView export.
    - If quads are available, exports true quad cells + original triangles.
    - Otherwise exports triangulated surface (tris_plot).
    """
    P = np.asarray(pts, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] < 2:
        raise ValueError("pts must be (N,2+) for 2D export")
    # ParaView/VTU prefers 3D points; pad z=0
    if P.shape[1] == 2:
        P3 = np.hstack([P[:, :2], np.zeros((P.shape[0], 1), dtype=np.float64)])
    else:
        P3 = np.hstack([P[:, :2], np.zeros((P.shape[0], 1), dtype=np.float64)])

    tri_plot = np.asarray(tris_plot) if tris_plot is not None else np.zeros((0, 3), dtype=int)
    tri_plot = tri_plot[:, :3].astype(int) if tri_plot.size else np.zeros((0, 3), dtype=int)
    qu = np.asarray(quads)[:, :4].astype(int) if (quads is not None and np.asarray(quads).size > 0) else np.zeros((0, 4), dtype=int)

    cells = []
    if qu.size > 0:
        ntri = int(max(0, min(int(omega_triangles_n), int(tri_plot.shape[0]))))
        tri_orig = tri_plot[:ntri, :] if ntri > 0 else np.zeros((0, 3), dtype=int)
        if tri_orig.size > 0:
            cells.append(("triangle", tri_orig))
        cells.append(("quad", qu))
    else:
        if tri_plot.size > 0:
            cells.append(("triangle", tri_plot))

    return meshio.Mesh(points=P3.astype(np.float64), cells=cells)


def build_paraview_mesh_3d(pts: np.ndarray, tets_conn: np.ndarray) -> meshio.Mesh:
    """
    Build a meshio.Mesh for 3D ParaView export (Tet4).
    """
    P = np.asarray(pts, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("pts must be (N,3) for 3D export")
    tets = np.asarray(tets_conn, dtype=int)
    if tets.ndim != 2 or tets.shape[1] < 4:
        raise ValueError("tets_conn must be (Nt,4+) tetra connectivity")
    tets = tets[:, :4].astype(int)
    return meshio.Mesh(points=P.astype(np.float64), cells=[("tetra", tets)])


def _as_vec3_point_data(U: np.ndarray) -> np.ndarray:
    """
    Convert an (N,1)/(N,2)/(N,3) field into an (N,3) float32 vector (ParaView-friendly).
    """
    Uarr = np.asarray(U)
    if Uarr.ndim != 2 or Uarr.shape[0] <= 0:
        raise ValueError("U must be a 2D array (N,dim)")
    u3 = np.zeros((Uarr.shape[0], 3), dtype=np.float32)
    k = int(min(3, int(Uarr.shape[1])))
    if k > 0:
        u3[:, :k] = Uarr[:, :k].astype(np.float32)
    return u3


def add_paraview_point_fields(
    mesh: meshio.Mesh,
    *,
    U: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    mises: np.ndarray | None = None,
    displacement_name: str = "displacement",
    stress_name: str = "Stress",
    mises_name: str = "vonMises",
):
    """
    Attach common point-data fields to a meshio mesh for ParaView.
    - U: displacement/solution field. Written only as `displacement_name` (no legacy duplicate).
    - sigma: stress components (2D: [sxx,syy,sxy], 3D: [sxx,syy,szz,sxy,syz,szx])
    - mises: von Mises scalar (N,) or (N,1)
    """
    if U is not None:
        Uarr = np.asarray(U)
        if Uarr.ndim != 2 or int(Uarr.shape[0]) != int(mesh.points.shape[0]):
            raise ValueError("U must be (N,dim) with N == number of mesh points")
        if int(Uarr.shape[1]) == 1:
            mesh.point_data[str(displacement_name)] = Uarr[:, 0].astype(np.float32)
        else:
            u3 = _as_vec3_point_data(Uarr)
            mesh.point_data[str(displacement_name)] = u3

    if sigma is not None:
        s = np.asarray(sigma)
        if s.ndim != 2 or int(s.shape[0]) != int(mesh.points.shape[0]):
            raise ValueError("sigma must be (N,ncomp) with N == number of mesh points")
        mesh.point_data[stress_name] = s.astype(np.float32)

    if mises is not None:
        m = np.asarray(mises)
        if m.ndim == 2 and m.shape[1] == 1:
            m = m[:, 0]
        if m.ndim != 1 or int(m.shape[0]) != int(mesh.points.shape[0]):
            raise ValueError("mises must be (N,) or (N,1) with N == number of mesh points")
        mesh.point_data[mises_name] = m.astype(np.float32)


def _split_cmdline(cmdline: str) -> list[str]:
    """
    Split a user-provided cmdline into argv, in a Windows-friendly way.
    Accepts either:
      - 'gmsh'
      - 'C:\\Program Files\\Gmsh\\gmsh.exe'
      - 'C:\\...\\gmsh.exe -v 2'
    """
    cmdline = (cmdline or "").strip()
    if not cmdline:
        return []
    # If user pasted a path (possibly containing spaces) without quotes,
    # treat it as a single argv item when it exists as file/dir.
    cmdline_unquoted = cmdline.strip("\"'")
    if os.path.exists(cmdline_unquoted):
        return [cmdline_unquoted]
    try:
        return shlex.split(cmdline, posix=False)
    except Exception:
        # fall back to a simple split
        return cmdline.split()


def gmsh_geo_to_msh_bytes(
    geo_text: str,
    *,
    gmsh_cmdline: str = "gmsh",
    dim: int = 2,
    msh_format: str = "msh2",
    extra_args: str = "",
    timeout_sec: int = 120,
) -> tuple[bytes, str]:
    """
    Convert Gmsh .geo text into .msh bytes by invoking local Gmsh CLI.
    Returns: (msh_bytes, log_text)
    """
    geo_text = geo_text or ""
    if not geo_text.strip():
        raise ValueError("Empty .geo content.")

    argv0 = _split_cmdline(gmsh_cmdline)
    if not argv0:
        raise ValueError("Gmsh command is empty.")

    exe = argv0[0]
    # Allow providing a folder containing gmsh.exe (common when unzipping portable builds)
    if os.path.isdir(exe):
        exe2 = os.path.join(exe, "gmsh.exe")
        if os.path.exists(exe2):
            argv0[0] = exe2
            exe = exe2

    if not (os.path.isabs(exe) and os.path.exists(exe)) and shutil.which(exe) is None:
        raise FileNotFoundError(
            f"Cannot find Gmsh executable: '{exe}'. Put gmsh in PATH or set full path to gmsh.exe."
        )

    dim = int(dim)
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3.")

    msh_format = (msh_format or "msh2").strip().lower()
    if msh_format not in ("msh2", "msh4"):
        raise ValueError("msh_format must be 'msh2' or 'msh4'.")

    extra_argv = _split_cmdline(extra_args)

    # Write geo -> temp, run gmsh -> temp msh, return bytes
    with tempfile.TemporaryDirectory() as td:
        geo_path = os.path.join(td, "model.geo")
        msh_path = os.path.join(td, "model.msh")

        with open(geo_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(geo_text)

        cmd = (
            argv0
            + [geo_path, f"-{dim}", "-format", msh_format, "-o", msh_path]
            + extra_argv
        )

        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(timeout_sec),
        )
        log = (
            "Command:\n"
            + " ".join(cmd)
            + "\n\n--- stdout ---\n"
            + (p.stdout or "")
            + "\n--- stderr ---\n"
            + (p.stderr or "")
        )
        if p.returncode != 0:
            raise RuntimeError(f"Gmsh failed (exit={p.returncode}).\n\n{log}")

        if not os.path.exists(msh_path):
            raise RuntimeError(f"Gmsh finished but did not create .msh.\n\n{log}")

        with open(msh_path, "rb") as f:
            msh_bytes = f.read()

    return msh_bytes, log


def open_geo_in_gmsh_gui(geo_text: str, *, gmsh_cmdline: str, geo_dim: int = 2, extra_args: str = "") -> tuple[bool, str]:
    """
    Launch the Gmsh GUI to open a .geo file for manual tweaking.
    Writes current geo_text to a temp file and opens it in Gmsh (non-blocking).
    Returns (ok, message).
    """
    geo_text = (geo_text or "").strip()
    if not geo_text:
        return False, "Empty .geo text."

    argv = _split_cmdline(gmsh_cmdline)
    if not argv:
        return False, "Gmsh command is empty. Please set a valid gmsh path in the sidebar."

    # Allow passing a folder containing gmsh.exe (portable builds)
    exe = argv[0]
    exe_unquoted = exe.strip("\"'")
    if os.path.isdir(exe_unquoted):
        exe2 = os.path.join(exe_unquoted, "gmsh.exe")
        if os.path.exists(exe2):
            argv[0] = exe2

    if not os.path.exists(argv[0]) and shutil.which(argv[0]) is None:
        return False, f"Cannot find Gmsh executable: '{argv[0]}'."

    # Optional extra args (e.g., -v 2). Keep GUI-friendly (no -format/-o here).
    extra = _split_cmdline(extra_args) if (extra_args and str(extra_args).strip()) else []

    try:
        td = tempfile.mkdtemp(prefix="dem_geo_")
        geo_path = os.path.join(td, f"model_{int(geo_dim)}d.geo")
        with open(geo_path, "w", encoding="utf-8") as f:
            f.write(geo_text + ("\n" if not geo_text.endswith("\n") else ""))

        args = argv + extra + [geo_path]

        # Non-blocking GUI launch
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=(os.name != "nt"),
            creationflags=creationflags,
        )
        return True, f"Opened in Gmsh: {geo_path}"
    except Exception as e:
        return False, f"Failed to launch Gmsh GUI: {e}"



def validate_geo_text(geo: str, *, require_gamma_t: bool = True, dim: int | None = None) -> tuple[bool, str]:
    """Text-level hard constraints for your pipeline (2D/3D)."""
    if geo is None or not geo.strip():
        return False, "geo is empty."

    if dim is None:
        if 'Physical Volume("Omega")' in geo:
            dim = 3
        elif 'Physical Surface("Omega")' in geo:
            dim = 2
        else:
            return False, 'Missing required token: Physical Surface("Omega") or Physical Volume("Omega")'

    dim = int(dim)
    if dim == 2:
        must = [
            'Physical Surface("Omega")',
            'Physical Curve("Gamma_u")',
        ]
        if require_gamma_t:
            must.append('Physical Curve("Gamma_t")')
    elif dim == 3:
        must = [
            'Physical Volume("Omega")',
            'Physical Surface("Gamma_u")',
        ]
        if require_gamma_t:
            must.append('Physical Surface("Gamma_t")')
    else:
        return False, "dim must be 2 or 3."
    for k in must:
        if k not in geo:
            return False, f"Missing required token: {k}"

    return True, "ok"

def upsert_lc_in_geo(geo: str, lc_value: float) -> str:
    """
    Ensure geo contains a single scalar 'lc' value and update it to lc_value.
    - If 'lc = ...;' exists, replace the first occurrence.
    - Else insert after SetFactory(...) if present, otherwise prepend.
    """
    geo = geo or ""
    # replace existing lc assignment
    if re.search(r"(?m)^\s*lc\s*=\s*[^;]+;\s*$", geo):
        geo = re.sub(r"(?m)^\s*lc\s*=\s*[^;]+;\s*$", f"lc = {lc_value};", geo, count=1)
        return geo

    # insert lc after SetFactory if found
    m = re.search(r'SetFactory\("OpenCASCADE"\);\s*', geo)
    if m:
        ins_pos = m.end()
        return geo[:ins_pos] + f"\nlc = {lc_value};\n" + geo[ins_pos:]

    # otherwise prepend
    return f'lc = {lc_value};\n' + geo


def llm_generate_geo_from_nl(
    nl_prompt: str,
    default_lc: float = 0.15,
    geo_dim: int = 2,
    *,
    base_geo: str | None = None,
    base_nl: str | None = None,
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    """
    Returns (chat_text, geo_text)
    """
    # Get provider and model from session state if not provided
    if provider is None:
        provider = st.session_state.get("llm_provider", "openai")
    if model is None:
        model = st.session_state.get("llm_model", _get_default_model(provider))

    geo_dim = int(geo_dim)
    if geo_dim not in (2, 3):
        geo_dim = 2

    if geo_dim == 2:
        hard_req = (
            "2D requirements:\n"
            "- Physical Surface(\"Omega\")\n"
            "- Physical Curve(\"Gamma_u\"), Physical Curve(\"Gamma_t\")\n"
        )
    else:
        hard_req = (
            "3D requirements:\n"
            "- Physical Volume(\"Omega\")\n"
            "- Physical Surface(\"Gamma_u\"), Physical Surface(\"Gamma_t\")\n"
        )

    # Build 3D-specific tetra enforcement instructions
    tetra_enforcement = ""
    if geo_dim == 3:
        tetra_enforcement = (
            "\n"
            "⚠️ CRITICAL FOR 3D: Tetrahedral mesh is MANDATORY. The solver will FAIL if Omega contains hexa/wedge/pyramid/prism elements.\n"
            "\n"
            "To ensure tetra-only volume mesh, you MUST include these settings in your .geo:\n"
            "```\n"
            "Mesh.RecombineAll = 0;\n"
            "Mesh.Recombine3DAll = 0;\n"
            "Mesh.Algorithm3D = 4;  // 4 = Frontal (tetra), 1 = Delaunay (tetra)\n"
            "```\n"
            "\n"
            "❌ DO NOT use:\n"
            "- Extrude with Layers{} (creates prism/wedge)\n"
            "- Mesh.RecombineAll = 1 (creates hexa)\n"
            "- Structured meshing that produces hexa/wedge\n"
            "\n"
            "✅ CORRECT approach for 3D:\n"
            "- Use BooleanUnion/BooleanDifference on 3D volumes (Box, Cylinder, Sphere, etc.)\n"
            "- Use Extrude WITHOUT Layers{}: `out[] = Extrude {0,0,t} { Surface{...}; };`\n"
            "- Always set Mesh.RecombineAll = 0 and Mesh.Algorithm3D = 4\n"
            "\n"
        )
    
    system = (
        "Output MUST be JSON: {\"chat\": \"...\", \"geo\": \"...\"}.\n"
        "No markdown, no code fences.\n"
        "Mesh element constraints (IMPORTANT for this solver):\n"
        "- 2D: Omega may be meshed with triangles or quads.\n"
        "- 3D: Omega MUST be a tetrahedral volume mesh (tetra elements). Do NOT generate hexa/wedge/pyramid-only volume meshes.\n"
        "- 3D boundary surfaces (Gamma_u/Gamma_t) may be triangles.\n"
        + tetra_enforcement +
        "Gmsh syntax notes:\n"
        "- OpenCASCADE primitives use: Box(tag) = {...}; Cylinder(tag) = {...}; Sphere(tag) = {...};\n"
        "- BoundingBox selectors use curly braces: Surface In BoundingBox{...}; Volume In BoundingBox{...};\n"
        "- Store selector results in arrays first: s[] = Surface In BoundingBox{...}; then use s[] in Physical definitions.\n"
        "- Boolean operations: v[] = BooleanUnion{ Volume{1}; Delete; }{ Volume{2}; Delete; };\n"
        "\n"
        + hard_req +
        "\n"
        "Boundary placement:\n"
        "- If Gamma_u/Gamma_t locations are specified in the user request, follow them exactly.\n"
        "- If not specified, choose reasonable locations and state them explicitly in `chat`.\n"
        "2D mesh sizing: if the user provides `lc`, it is sufficient to enforce it globally via Mesh.CharacteristicLengthMin = lc; Mesh.CharacteristicLengthMax = lc; lc should be a variable;\n"
    )

    base_geo = (base_geo or "").strip()
    base_nl = (base_nl or "").strip()

    user = "Task:\n"
    if base_geo:
        user += (
            "You are editing an existing .geo that already works.\n"
            "Keep everything not explicitly changed.\n\n"
            "Existing .geo (baseline):\n"
            + base_geo
            + "\n\n"
        )
    if base_nl:
        user += "Previous NL intent (for context):\n" + base_nl + "\n\n"
    user += (
        f"New natural language request:\n{nl_prompt}\n\n"
        f"Target geometry dimension: {geo_dim}D\n"
        f"Default mesh size: lc={default_lc}\n"
    )

    response_text = _call_llm(provider, model, system, user)
    chat, geo = _parse_llm_geo_json(response_text)

    return chat, geo




def llm_fix_geo_with_gmsh_log(
    geo_text: str,
    gmsh_log: str,
    *,
    geo_dim: int = 2,
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    """
    Returns (chat_text, geo_text)
    """
    # Get provider and model from session state if not provided
    if provider is None:
        provider = st.session_state.get("llm_provider", "openai")
    if model is None:
        model = st.session_state.get("llm_model", _get_default_model(provider))

    geo_dim = int(geo_dim)
    if geo_dim == 3:
        req = (
            "- Must have Physical Volume(\"Omega\")\n"
            "- Must have Physical Surface(\"Gamma_u\") and Physical Surface(\"Gamma_t\")\n"
        )
    else:
        req = (
            "- Must have Physical Surface(\"Omega\")\n"
            "- Must have Physical Curve(\"Gamma_u\") and Physical Curve(\"Gamma_t\")\n"
        )

    # Build 3D-specific tetra enforcement instructions
    tetra_enforcement = ""
    if geo_dim == 3:
        tetra_enforcement = (
            "\n"
            "⚠️ CRITICAL FOR 3D: Tetrahedral mesh is MANDATORY. The solver will FAIL if Omega contains hexa/wedge/pyramid/prism elements.\n"
            "\n"
            "To ensure tetra-only volume mesh, you MUST include these settings in your .geo:\n"
            "```\n"
            "Mesh.RecombineAll = 0;\n"
            "Mesh.Recombine3DAll = 0;\n"
            "Mesh.Algorithm3D = 4;  // 4 = Frontal (tetra), 1 = Delaunay (tetra)\n"
            "```\n"
            "\n"
            "❌ DO NOT use:\n"
            "- Extrude with Layers{} (creates prism/wedge)\n"
            "- Mesh.RecombineAll = 1 (creates hexa)\n"
            "- Structured meshing that produces hexa/wedge\n"
            "\n"
            "✅ CORRECT approach for 3D:\n"
            "- Use BooleanUnion/BooleanDifference on 3D volumes (Box, Cylinder, Sphere, etc.)\n"
            "- Use Extrude WITHOUT Layers{}: `out[] = Extrude {0,0,t} { Surface{...}; };`\n"
            "- Always set Mesh.RecombineAll = 0 and Mesh.Algorithm3D = 4\n"
            "\n"
        )
    
    system = (
        "Output MUST be JSON: {\"chat\": \"...\", \"geo\": \"...\"}.\n"
        "No markdown, no code fences.\n"
        "\n"
        "Mesh element constraints (IMPORTANT for this solver):\n"
        "- 2D: Omega may be meshed with triangles or quads.\n"
        "- 3D: Omega MUST be a tetrahedral volume mesh (tetra elements). Do NOT generate hexa/wedge/pyramid-only volume meshes.\n"
        "- 3D boundary surfaces (Gamma_u/Gamma_t) may be triangles.\n"
        + tetra_enforcement +
        "\n"
        + req
    )

    user = (
        "Here is the current .geo:\n"
        + (geo_text or "")
        + "\n\nHere is the Gmsh log:\n"
        + (gmsh_log or "")
    )

    response_text = _call_llm(provider, model, system, user)
    chat, geo = _parse_llm_geo_json(response_text)

    return chat, geo


def _get_default_model(provider: LLMProvider) -> str:
    """Get default model name for a provider."""
    defaults = {
        "openai": "gpt-5.2",  # Default to o3-mini (latest thinking model, previously used o3)
        "anthropic": "claude-3-5-sonnet-20241022",
        "google": "gemini-1.5-pro",
        "ollama": "llama3.1",
    }
    return defaults.get(provider, "o3-mini")


def _get_available_models(provider: LLMProvider) -> list[str]:
    """Get list of available models for a provider."""
    models = {
        "openai": [
            "gpt-5.2",
            "o3",  # Latest thinking model (most capable)
            "o3-mini",  # Latest thinking model (faster, cost-efficient)
            "o3-pro",  # Latest thinking model (most powerful)
            "o1-preview",  # Previous thinking model
            "o1-mini",  # Previous thinking model (smaller)
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "google": [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
        ],
        "ollama": [
            "llama3.1",
            "llama3",
            "mistral",
            "mixtral",
            "codellama",
            "qwen2.5",
        ],
    }
    return models.get(provider, ["gpt-4o"])


# ============================================================
# Gmsh mesh reading + physical group extraction
# ============================================================
def read_gmsh_msh_bytes(msh_bytes: bytes):
    # Write bytes to a temp file, then let meshio read from path (most compatible).
    with tempfile.NamedTemporaryFile(delete=False, suffix=".msh") as f:
        f.write(msh_bytes)
        tmp_path = f.name

    try:
        mesh = meshio.read(tmp_path)  # Let meshio auto-detect Gmsh format
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # Keep full coordinates; downstream will slice to 2D/3D based on Physical dim.
    pts = mesh.points.copy()
    field_data = mesh.field_data
    return mesh, pts, field_data


def get_phys_id(field_data, name: str, dim: int):
    if name not in field_data:
        raise KeyError(f"Physical group '{name}' not found. Available: {list(field_data.keys())}")
    pid, pdim = field_data[name]
    if int(pdim) != int(dim):
        raise ValueError(f"Physical group '{name}' dim={pdim}, expected dim={dim}.")
    return int(pid)


def extract_cells_by_phys(mesh, cell_type_prefix: str, phys_id: int):
    """
    Robust extraction by physical id.
    Uses mesh.cell_data['gmsh:physical'] which is aligned with mesh.cells blocks.
    Supports line/line3/triangle/triangle6... via prefix matching.
    """
    if not hasattr(mesh, "cell_data") or "gmsh:physical" not in mesh.cell_data:
        raise RuntimeError("No 'gmsh:physical' in mesh.cell_data. Export .msh with Physical Groups in Gmsh.")

    tags_blocks = mesh.cell_data["gmsh:physical"]  # list aligned with mesh.cells
    if len(tags_blocks) != len(mesh.cells):
        raise RuntimeError(f"cell_data blocks ({len(tags_blocks)}) != cell blocks ({len(mesh.cells)}).")

    conns = []
    for i, cell_block in enumerate(mesh.cells):
        ctype = cell_block.type
        if not ctype.startswith(cell_type_prefix):
            continue

        data = cell_block.data
        tags = np.asarray(tags_blocks[i]).reshape(-1)

        if data.shape[0] != tags.shape[0]:
            raise RuntimeError(
                f"Mismatch in block {i} type={ctype}: data has {data.shape[0]} elems, tags has {tags.shape[0]}."
            )

        mask = (tags == phys_id)
        if np.any(mask):
            conns.append(data[mask])

    if len(conns) == 0:
        return np.zeros((0, 0), dtype=int)
    return np.vstack(conns)


# ============================================================
# Quadrature rules (triangles + segments)
# ============================================================
def tri_rule(rule="3-point"):
    """
    Reference triangle vertices: (0,0), (1,0), (0,1)
    Return barycentric coords (nq,3) and weights (nq,) summing to area=1/2.
    """
    if rule == "1-point":
        bary = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float64)
        wref = np.array([0.5], dtype=np.float64)
        return bary, wref
    elif rule == "3-point":
        bary = np.array(
            [
                [2 / 3, 1 / 6, 1 / 6],
                [1 / 6, 2 / 3, 1 / 6],
                [1 / 6, 1 / 6, 2 / 3],
            ],
            dtype=np.float64,
        )
        wref = np.array([1 / 6, 1 / 6, 1 / 6], dtype=np.float64)  # sum=1/2
        return bary, wref
    else:
        raise ValueError("Unknown triangle rule")


def gauss_1d(n=2):
    """
    Gauss-Legendre on [-1,1]
    Return nodes xi and weights wi, sum(wi)=2.
    """
    if n == 1:
        xi = np.array([0.0], dtype=np.float64)
        wi = np.array([2.0], dtype=np.float64)
    elif n == 2:
        xi = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)], dtype=np.float64)
        wi = np.array([1.0, 1.0], dtype=np.float64)
    elif n == 3:
        xi = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)], dtype=np.float64)
        wi = np.array([5 / 9, 8 / 9, 5 / 9], dtype=np.float64)
    else:
        xi, wi = np.polynomial.legendre.leggauss(n)
        xi = xi.astype(np.float64)
        wi = wi.astype(np.float64)
    return xi, wi


def build_domain_quadrature(pts, tris, tri_rule_name="3-point"):
    """
    Build all quadrature points for Omega triangles.
    Returns Xq (Nq,2), Wq (Nq,1).
    """
    bary, wref = tri_rule(tri_rule_name)
    nq = bary.shape[0]

    Xq_list = []
    Wq_list = []

    for (i0, i1, i2) in tris:
        a = pts[i0]
        b = pts[i1]
        c = pts[i2]
        P = bary[:, [0]] * a + bary[:, [1]] * b + bary[:, [2]] * c  # (nq,2)
        area = 0.5 * abs(np.cross(b - a, c - a))
        W = wref * (2.0 * area)  # sum(W)=area
        Xq_list.append(P)
        Wq_list.append(W.reshape(nq, 1))

    Xq = np.vstack(Xq_list).astype(np.float32)
    Wq = np.vstack(Wq_list).astype(np.float32)
    return Xq, Wq


def gauss_2d_tensor(n: int):
    """
    Tensor-product Gauss-Legendre on [-1,1]^2.
    Returns:
      rs: (n*n, 2) points (r,s)
      w:  (n*n, 1) weights (sum=4)
    """
    xi, wi = np.polynomial.legendre.leggauss(int(n))
    rr, ss = np.meshgrid(xi, xi, indexing="xy")
    ww = np.outer(wi, wi)
    rs = np.stack([rr.reshape(-1), ss.reshape(-1)], axis=1).astype(np.float64)
    w = ww.reshape(-1, 1).astype(np.float64)
    return rs, w


def build_domain_quadrature_quads(pts, quads, gauss_n: int = 2):
    """
    Build quadrature points for bilinear quad4 elements on [-1,1]^2.
    quads: (nq,4) node indices, assumed ordering around the quad boundary.
    Returns Xq (Nq,2), Wq (Nq,1).
    """
    rs, wref = gauss_2d_tensor(int(gauss_n))  # (ng,2), (ng,1)
    r = rs[:, 0:1]
    s = rs[:, 1:2]
    ng = rs.shape[0]

    # shape funcs
    N1 = 0.25 * (1 - r) * (1 - s)
    N2 = 0.25 * (1 + r) * (1 - s)
    N3 = 0.25 * (1 + r) * (1 + s)
    N4 = 0.25 * (1 - r) * (1 + s)

    dNdr = np.concatenate(
        [
            -0.25 * (1 - s),
            0.25 * (1 - s),
            0.25 * (1 + s),
            -0.25 * (1 + s),
        ],
        axis=1,
    )  # (ng,4)
    dNds = np.concatenate(
        [
            -0.25 * (1 - r),
            -0.25 * (1 + r),
            0.25 * (1 + r),
            0.25 * (1 - r),
        ],
        axis=1,
    )  # (ng,4)

    Xq_list = []
    Wq_list = []

    for (i0, i1, i2, i3) in quads:
        X = pts[[i0, i1, i2, i3], :]  # (4,2)
        # mapping
        P = N1 * X[0] + N2 * X[1] + N3 * X[2] + N4 * X[3]  # (ng,2)

        dxdr = (dNdr @ X[:, 0:1])  # (ng,1)
        dydr = (dNdr @ X[:, 1:2])
        dxds = (dNds @ X[:, 0:1])
        dyds = (dNds @ X[:, 1:2])
        detJ = (dxdr * dyds - dxds * dydr).reshape(-1, 1)  # (ng,1)

        W = wref * np.abs(detJ)  # sum(W)=area
        Xq_list.append(P)
        Wq_list.append(W.astype(np.float64))

    Xq = np.vstack(Xq_list).astype(np.float32)
    Wq = np.vstack(Wq_list).astype(np.float32)
    return Xq, Wq


def _canonicalize_quad4_indices(pts: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Canonicalize a quad4 node ordering to be cyclic around the element (no crossing).

    Many Q4 formulas assume nodes are ordered around the boundary. If the ordering is
    scrambled (e.g., [0,1,3,2]) the isoparametric mapping is twisted and FEM results
    can be wildly wrong (often showing up on holed / complex geometries).
    """
    q = np.asarray(q).reshape(4)
    xy = pts[q, :2].astype(np.float64)  # (4,2)
    c = xy.mean(axis=0, keepdims=True)
    ang = np.arctan2(xy[:, 1] - c[0, 1], xy[:, 0] - c[0, 0])
    order = np.argsort(ang)  # CCW cyclic order
    q2 = q[order]
    return q2.astype(int)


def canonicalize_quads(pts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """
    Apply `_canonicalize_quad4_indices` to all quad elements.
    """
    if quads is None or quads.size == 0:
        return np.zeros((0, 4), dtype=int)
    q = np.asarray(quads)[:, :4]
    out = np.zeros_like(q, dtype=int)
    for i in range(q.shape[0]):
        out[i, :] = _canonicalize_quad4_indices(pts, q[i, :])
    return out


def triangulate_quads(quads: np.ndarray) -> np.ndarray:
    """
    Split quad4 connectivity into triangles for visualization/FEM (two per quad).
    quads: (n,4) -> tris: (2n,3)
    """
    if quads is None or quads.size == 0:
        return np.zeros((0, 3), dtype=int)
    q = quads[:, :4]
    t1 = q[:, [0, 1, 2]]
    t2 = q[:, [0, 2, 3]]
    return np.vstack([t1, t2]).astype(int)


def build_boundary_quadrature(pts, segs, gauss_n=2):
    """
    Build all Gauss points for boundary line segments.
    Returns Xg (Ng,2), Wg (Ng,1).
    """
    xi, wi = gauss_1d(gauss_n)
    ng = xi.shape[0]

    Xg_list = []
    Wg_list = []

    s = (xi + 1.0) * 0.5  # map xi -> s in [0,1]
    for (i0, i1) in segs:
        p0 = pts[i0]
        p1 = pts[i1]
        P = (1.0 - s)[:, None] * p0 + s[:, None] * p1  # (ng,2)
        L = np.linalg.norm(p1 - p0)
        W = wi * (L / 2.0)  # sum(W)=L
        Xg_list.append(P)
        Wg_list.append(W.reshape(ng, 1))

    Xg = np.vstack(Xg_list).astype(np.float32)
    Wg = np.vstack(Wg_list).astype(np.float32)
    return Xg, Wg


# ============================================================
# Shape-function integration data (for DEM w/o AD derivatives)
# ============================================================
def build_sf_domain_data_2d(
    pts: np.ndarray,
    tris_plot: np.ndarray,
    omega_info: dict,
    quads: np.ndarray,
    *,
    tri_rule_name: str,
    quad_gauss_n: int,
):
    """
    Build flattened Gauss-point data for 2D Omega suitable for computing gradients
    via shape functions (no autograd w.r.t. coordinates).

    Returns a dict of numpy arrays (float32/int64):
      - X:    (Ng,2) physical Gauss points
      - W:    (Ng,1) weights (already include |detJ| / area)
      - conn: (Ng,4) node indices (tri padded by repeating the 3rd node)
      - N:    (Ng,4) shape functions at Gauss points (tri has N4=0)
      - dNdx: (Ng,4) dN/dx at Gauss points (tri constant within elem)
      - dNdy: (Ng,4) dN/dy at Gauss points
    """
    pts = np.asarray(pts)[:, :2].astype(np.float64)
    tris_plot = np.asarray(tris_plot) if tris_plot is not None else np.zeros((0, 3), dtype=int)
    quads = np.asarray(quads) if quads is not None else np.zeros((0, 4), dtype=int)
    omega_tri_n = int(omega_info.get("Omega_triangles", 0)) if omega_info else int(tris_plot.shape[0])
    omega_tri = (
        np.asarray(tris_plot)[:omega_tri_n, :3].astype(int) if omega_tri_n > 0 else np.zeros((0, 3), dtype=int)
    )

    X_list, W_list, conn_list, N_list, dNdx_list, dNdy_list = [], [], [], [], [], []

    # ---- T3 triangles ----
    if omega_tri.size > 0:
        bary, wref = tri_rule(str(tri_rule_name))
        nq = int(bary.shape[0])
        for (i0, i1, i2) in omega_tri:
            a = pts[int(i0)]
            b = pts[int(i1)]
            c = pts[int(i2)]
            area = 0.5 * abs(np.cross(b - a, c - a))
            if not np.isfinite(area) or area <= 0.0:
                continue

            P = bary[:, [0]] * a + bary[:, [1]] * b + bary[:, [2]] * c  # (nq,2)
            W = (wref * (2.0 * area)).reshape(nq, 1)  # sum = area

            Ntri = np.zeros((nq, 4), dtype=np.float64)
            Ntri[:, 0:3] = bary

            x0, y0 = a[0], a[1]
            x1, y1 = b[0], b[1]
            x2, y2 = c[0], c[1]
            denom = 2.0 * area
            b0 = (y1 - y2) / denom
            c0 = (x2 - x1) / denom
            b1 = (y2 - y0) / denom
            c1 = (x0 - x2) / denom
            b2 = (y0 - y1) / denom
            c2 = (x1 - x0) / denom

            dNdx = np.zeros((nq, 4), dtype=np.float64)
            dNdy = np.zeros((nq, 4), dtype=np.float64)
            dNdx[:, 0] = b0
            dNdx[:, 1] = b1
            dNdx[:, 2] = b2
            dNdy[:, 0] = c0
            dNdy[:, 1] = c1
            dNdy[:, 2] = c2

            conn = np.tile(np.array([int(i0), int(i1), int(i2), int(i2)], dtype=np.int64)[None, :], (nq, 1))

            X_list.append(P)
            W_list.append(W)
            conn_list.append(conn)
            N_list.append(Ntri)
            dNdx_list.append(dNdx)
            dNdy_list.append(dNdy)

    # ---- Q4 quads ----
    if quads.size > 0:
        rs, wref = gauss_2d_tensor(int(quad_gauss_n))
        Nref, dNdr, dNds = _quad4_shape(rs)
        ng = int(rs.shape[0])
        for (i0, i1, i2, i3) in np.asarray(quads)[:, :4].astype(int):
            X = pts[[i0, i1, i2, i3], :]  # (4,2)
            P = (Nref @ X).reshape(ng, 2)
            dxdr = (dNdr @ X[:, 0:1]).reshape(ng, 1)
            dydr = (dNdr @ X[:, 1:2]).reshape(ng, 1)
            dxds = (dNds @ X[:, 0:1]).reshape(ng, 1)
            dyds = (dNds @ X[:, 1:2]).reshape(ng, 1)
            detJ = (dxdr * dyds - dxds * dydr).reshape(ng, 1)
            if not np.all(np.isfinite(detJ)):
                continue
            W = (wref * np.abs(detJ)).reshape(ng, 1)

            inv_det = 1.0 / (detJ + 1e-30)
            drdx = dyds * inv_det
            drdy = -dxds * inv_det
            dsdx = -dydr * inv_det
            dsdy = dxdr * inv_det

            dNdx = dNdr * drdx + dNds * dsdx
            dNdy = dNdr * drdy + dNds * dsdy

            conn = np.tile(np.array([i0, i1, i2, i3], dtype=np.int64)[None, :], (ng, 1))

            X_list.append(P)
            W_list.append(W)
            conn_list.append(conn)
            N_list.append(Nref)
            dNdx_list.append(dNdx)
            dNdy_list.append(dNdy)

    if len(X_list) == 0:
        return {
            "X": np.zeros((0, 2), dtype=np.float32),
            "W": np.zeros((0, 1), dtype=np.float32),
            "conn": np.zeros((0, 4), dtype=np.int64),
            "N": np.zeros((0, 4), dtype=np.float32),
            "dNdx": np.zeros((0, 4), dtype=np.float32),
            "dNdy": np.zeros((0, 4), dtype=np.float32),
        }

    return {
        "X": np.vstack(X_list).astype(np.float32),
        "W": np.vstack(W_list).astype(np.float32),
        "conn": np.vstack(conn_list).astype(np.int64),
        "N": np.vstack(N_list).astype(np.float32),
        "dNdx": np.vstack(dNdx_list).astype(np.float32),
        "dNdy": np.vstack(dNdy_list).astype(np.float32),
    }


def build_sf_boundary_segments_2d(pts: np.ndarray, segs: np.ndarray, *, gauss_n: int):
    """
    Build flattened Gauss-point data for 2D boundary line segments (Gamma_u/Gamma_t).
    Returns dict of numpy arrays:
      - X:    (Ng,2)
      - W:    (Ng,1) weights, sum per segment = length
      - conn: (Ng,2) node indices
      - N:    (Ng,2) segment shape functions at Gauss points
    """
    pts = np.asarray(pts)[:, :2].astype(np.float64)
    segs = np.asarray(segs) if segs is not None else np.zeros((0, 2), dtype=int)
    if segs.size == 0:
        return {
            "X": np.zeros((0, 2), dtype=np.float32),
            "W": np.zeros((0, 1), dtype=np.float32),
            "conn": np.zeros((0, 2), dtype=np.int64),
            "N": np.zeros((0, 2), dtype=np.float32),
        }

    xi, wi = gauss_1d(int(gauss_n))
    s = (xi + 1.0) * 0.5  # [0,1]
    N0 = (1.0 - s).reshape(-1, 1)
    N1 = s.reshape(-1, 1)
    Nref = np.hstack([N0, N1]).astype(np.float64)  # (ng,2)
    ng = int(Nref.shape[0])

    X_list, W_list, conn_list, N_list = [], [], [], []
    for (i0, i1) in np.asarray(segs)[:, :2].astype(int):
        p0 = pts[int(i0)]
        p1 = pts[int(i1)]
        L = np.linalg.norm(p1 - p0)
        if not np.isfinite(L) or L <= 0.0:
            continue
        P = N0 * p0 + N1 * p1  # (ng,2)
        W = (wi.reshape(-1, 1) * (L / 2.0)).astype(np.float64)  # sum=length
        conn = np.tile(np.array([int(i0), int(i1)], dtype=np.int64)[None, :], (ng, 1))
        X_list.append(P)
        W_list.append(W)
        conn_list.append(conn)
        N_list.append(Nref)

    if len(X_list) == 0:
        return {
            "X": np.zeros((0, 2), dtype=np.float32),
            "W": np.zeros((0, 1), dtype=np.float32),
            "conn": np.zeros((0, 2), dtype=np.int64),
            "N": np.zeros((0, 2), dtype=np.float32),
        }

    return {
        "X": np.vstack(X_list).astype(np.float32),
        "W": np.vstack(W_list).astype(np.float32),
        "conn": np.vstack(conn_list).astype(np.int64),
        "N": np.vstack(N_list).astype(np.float32),
    }


def build_domain_quadrature_tets(pts: np.ndarray, tets: np.ndarray):
    """
    Build a simple quadrature for 3D tetra volume integration.
    1-point rule at centroid: exact for linear fields.
    Returns Xq (Nq,3), Wq (Nq,1) where Wq are tetra volumes.
    """
    if tets is None or np.asarray(tets).size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    t = np.asarray(tets)[:, :4].astype(int)
    Xq_list = []
    Wq_list = []
    for (i0, i1, i2, i3) in t:
        a = pts[i0]
        b = pts[i1]
        c = pts[i2]
        d = pts[i3]
        centroid = (a + b + c + d) / 4.0
        vol = abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0
        if not np.isfinite(vol) or vol <= 0.0:
            continue
        Xq_list.append(centroid.reshape(1, 3))
        Wq_list.append(np.array([[vol]], dtype=np.float64))
    if len(Xq_list) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    Xq = np.vstack(Xq_list).astype(np.float32)
    Wq = np.vstack(Wq_list).astype(np.float32)
    return Xq, Wq


def build_surface_quadrature_tris(pts: np.ndarray, tris: np.ndarray, tri_rule_name: str = "3-point"):
    """
    Build quadrature points for 3D surface triangle integration.
    Uses the same barycentric rules as 2D triangles; weights map by triangle area.
    Returns Xq (Nq,3), Wq (Nq,1) where sum(Wq) = total area.
    """
    if tris is None or np.asarray(tris).size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    bary, wref = tri_rule(tri_rule_name)
    nq = bary.shape[0]
    Xq_list = []
    Wq_list = []
    tri_arr = np.asarray(tris)[:, :3].astype(int)
    for (i0, i1, i2) in tri_arr:
        a = pts[i0]
        b = pts[i1]
        c = pts[i2]
        P = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c  # (nq,3)
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        if not np.isfinite(area) or area <= 0.0:
            continue
        # wref sums to 1/2 on reference triangle => multiply by 2*area
        W = wref * (2.0 * area)
        Xq_list.append(P)
        Wq_list.append(W.reshape(nq, 1))
    if len(Xq_list) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    Xq = np.vstack(Xq_list).astype(np.float32)
    Wq = np.vstack(Wq_list).astype(np.float32)
    return Xq, Wq


# ============================================================
# Cache heavy load + quadrature (Improvement #1)
# ============================================================
@st.cache_data(show_spinner=False)
def cached_load_and_quadrature(msh_bytes: bytes, tri_rule_name: str, seg_gauss_n: int, quad_gauss_n: int):
    mesh, pts, field_data = read_gmsh_msh_bytes(msh_bytes)

    # Determine mesh dimension from Physical group "Omega" (2=surface, 3=volume).
    if "Omega" not in field_data:
        raise KeyError(f"Physical group 'Omega' not found. Available: {list(field_data.keys())}")
    pid_omega, omega_dim = field_data["Omega"]
    omega_dim = int(omega_dim)
    pid_omega = int(pid_omega)

    if omega_dim not in (2, 3):
        raise ValueError(f"Unsupported Omega dim={omega_dim}. Expected 2 or 3.")

    # Slice point coordinates to the correct dimension (2D/3D).
    pts = np.asarray(pts)[:, :omega_dim].copy()

    # Gamma_u/Gamma_t dims depend on Omega dim:
    # - 2D: Gamma_* are 1D curves
    # - 3D: Gamma_* are 2D surfaces
    gamma_dim = 1 if omega_dim == 2 else 2
    pid_gu = get_phys_id(field_data, "Gamma_u", dim=gamma_dim)
    pid_gt = None
    try:
        pid_gt = get_phys_id(field_data, "Gamma_t", dim=gamma_dim)
    except Exception:
        pid_gt = None

    # ----------------------------
    # 2D pipeline (existing)
    # ----------------------------
    if omega_dim == 2:
        tris = extract_cells_by_phys(mesh, "triangle", pid_omega)
        quads = extract_cells_by_phys(mesh, "quad", pid_omega)
        if quads.size > 0:
            # Ensure quad4 connectivity is cyclic (prevents twisted Q4 on complex geometries like holed plates)
            quads = canonicalize_quads(pts, quads)
        seg_u = extract_cells_by_phys(mesh, "line", pid_gu)
        seg_t = extract_cells_by_phys(mesh, "line", pid_gt) if pid_gt is not None else np.zeros((0, 2), dtype=int)

        # if (tris.size == 0) and (quads.size == 0):
        #     raise RuntimeError("No triangles/quads found in physical group 'Omega'.")
        # if seg_u.size == 0:
        #     raise RuntimeError("No line segments found in physical group 'Gamma_u'.")

        Xdom_list = []
        Wdom_list = []
        if tris.size > 0:
            Xt2, Wt2 = build_domain_quadrature(pts, tris, tri_rule_name=tri_rule_name)
            Xdom_list.append(Xt2)
            Wdom_list.append(Wt2)
        if quads.size > 0:
            Xq2, Wq2 = build_domain_quadrature_quads(pts, quads, gauss_n=int(quad_gauss_n))
            Xdom_list.append(Xq2)
            Wdom_list.append(Wq2)
        Xdom = np.vstack(Xdom_list).astype(np.float32)
        Wdom = np.vstack(Wdom_list).astype(np.float32)

        # For plotting (matplotlib tri* APIs), triangulate quads
        tris_plot = tris
        if quads.size > 0:
            tris_plot = np.vstack([tris, triangulate_quads(quads)]) if tris.size > 0 else triangulate_quads(quads)

        Xu, Wu = build_boundary_quadrature(pts, seg_u, gauss_n=int(seg_gauss_n))

        if seg_t.size > 0:
            Xt, Wt = build_boundary_quadrature(pts, seg_t, gauss_n=int(seg_gauss_n))
        else:
            Xt = np.zeros((1, 2), dtype=np.float32)
            Wt = np.zeros((1, 1), dtype=np.float32)

        omega_info = {
            "dim": 2,
            "Omega_triangles": int(tris.shape[0]) if tris.size else 0,
            "Omega_quads": int(quads.shape[0]) if quads.size else 0,
            "has_quads": bool(quads.size > 0),
        }

        return pts, tris_plot, quads, seg_u, seg_t, Xdom, Wdom, Xu, Wu, Xt, Wt, omega_info

    # ----------------------------
    # 3D pipeline (new, Poisson-first)
    # ----------------------------
    tets = extract_cells_by_phys(mesh, "tetra", pid_omega)
    # if tets.size == 0:
    #     raise RuntimeError("3D Omega requires tetra elements (cell type prefix 'tetra').")

    # Boundary surfaces: triangles/quads under Gamma_u / Gamma_t physical groups
    gu_tris = extract_cells_by_phys(mesh, "triangle", pid_gu)
    gu_quads = extract_cells_by_phys(mesh, "quad", pid_gu)
    if gu_quads.size > 0:
        gu_tris2 = triangulate_quads(gu_quads[:, :4].astype(int))
        gu_tris = np.vstack([gu_tris, gu_tris2]) if gu_tris.size > 0 else gu_tris2

    gt_tris = np.zeros((0, 3), dtype=int)
    gt_quads = np.zeros((0, 4), dtype=int)
    if pid_gt is not None:
        gt_tris = extract_cells_by_phys(mesh, "triangle", pid_gt)
        gt_quads = extract_cells_by_phys(mesh, "quad", pid_gt)
        if gt_quads.size > 0:
            gt_tris2 = triangulate_quads(gt_quads[:, :4].astype(int))
            gt_tris = np.vstack([gt_tris, gt_tris2]) if gt_tris.size > 0 else gt_tris2


    # Domain quadrature (tets) + boundary quadrature (surface tris)
    Xdom, Wdom = build_domain_quadrature_tets(pts, tets)
    Xu, Wu = build_surface_quadrature_tris(pts, gu_tris, tri_rule_name=tri_rule_name)

    if gt_tris.size > 0:
        Xt, Wt = build_surface_quadrature_tris(pts, gt_tris, tri_rule_name=tri_rule_name)
    else:
        Xt = np.zeros((1, 3), dtype=np.float32)
        Wt = np.zeros((1, 1), dtype=np.float32)

    # For 3D we store surface triangles in the "tris_plot" slot for lightweight preview.
    omega_info = {
        "dim": 3,
        "Omega_tets": int(tets.shape[0]),
        "Omega_tets_conn": np.asarray(tets)[:, :4].astype(int),
        "Gamma_u_tris": int(gu_tris.shape[0]),
        "Gamma_t_tris": int(gt_tris.shape[0]) if gt_tris is not None else 0,
        # Store surface connectivity for 3D preview (avoids re-parsing).
        "Gamma_u_tris_conn": gu_tris.astype(int),
        "Gamma_t_tris_conn": gt_tris.astype(int) if gt_tris is not None else np.zeros((0, 3), dtype=int),
    }

    # Return placeholders for 2D-specific arrays (quads/segments) to keep tuple shape stable.
    tris_plot = gu_tris.astype(int)
    quads_placeholder = np.zeros((0, 4), dtype=int)
    seg_u_placeholder = np.zeros((0, 2), dtype=int)
    seg_t_placeholder = np.zeros((0, 2), dtype=int)
    return pts, tris_plot, quads_placeholder, seg_u_placeholder, seg_t_placeholder, Xdom, Wdom, Xu, Wu, Xt, Wt, omega_info


# ============================================================
# Distance to Dirichlet boundary segments (for Hard BC)
# ============================================================
def point_segment_distance_sq_torch(P, A, B, tau: float | None = None):
    """
    P: (N,2)
    A,B: (M,2)
    Return min distance^2 from each P to any segment AB: (N,1)
    Vectorized over segments; may be heavy for very large M.

    If tau is provided and tau > 0, use a smooth soft-min (log-mean-exp) over
    segments to make the distance field differentiable across "closest segment"
    switching lines, which improves stress stability near Gamma_u.
    """
    P2 = P[:, None, :]  # (N,1,2)
    A2 = A[None, :, :]  # (1,M,2)
    B2 = B[None, :, :]  # (1,M,2)
    AB = B2 - A2
    AP = P2 - A2

    denom = (AB * AB).sum(dim=2, keepdim=True).clamp_min(1e-12)
    t = (AP * AB).sum(dim=2, keepdim=True) / denom
    t = t.clamp(0.0, 1.0)

    proj = A2 + t * AB
    d2 = ((P2 - proj) ** 2).sum(dim=2)

    # hard min (piecewise-smooth, non-differentiable on switching lines)
    if tau is None or float(tau) <= 0.0:
        min_d2, _ = torch.min(d2, dim=1, keepdim=True)
        return min_d2

    # soft min via log-mean-exp (avoids negative values at multi-min points)
    tau_t = torch.as_tensor(float(tau), device=d2.device, dtype=d2.dtype).clamp_min(1e-20)
    M = max(int(d2.shape[1]), 1)
    d2_soft = -tau_t * (torch.logsumexp(-d2 / tau_t, dim=1, keepdim=True) - math.log(M))
    return d2_soft


class DirichletDistance:
    """
    Helper to compute d(x,Gamma_u) for Hard BC.
    Stores Gamma_u segments endpoints as torch tensors.
    """

    def __init__(self, pts_np, segs_np, device, tau: float = 0.0):
        a = pts_np[segs_np[:, 0], :]
        b = pts_np[segs_np[:, 1], :]
        self.A = to_torch(a.astype(np.float32), device=device, requires_grad=False)
        self.B = to_torch(b.astype(np.float32), device=device, requires_grad=False)
        self.tau = float(tau)

    def distance(self, x, y, z=None):
        P = torch.cat([x, y], dim=1)  # (N,2)
        d2 = point_segment_distance_sq_torch(P, self.A, self.B, tau=self.tau)  # (N,1)
        d = torch.sqrt(d2.clamp_min(0.0) + 1e-12)
        return d


def point_triangle_distance_sq_pairwise_torch(
    P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> torch.Tensor:
    """
    Pairwise point-to-triangle squared distance.
    P,A,B,C: (N,3) aligned one-to-one (triangle i = (A[i],B[i],C[i]) for point P[i])
    Returns: (N,) squared distances.
    Based on "Real-Time Collision Detection" (Christer Ericson).
    """
    AB = B - A
    AC = C - A
    AP = P - A

    d1 = (AB * AP).sum(dim=1)
    d2 = (AC * AP).sum(dim=1)
    maskA = (d1 <= 0) & (d2 <= 0)
    dA = (AP * AP).sum(dim=1)

    BP = P - B
    d3 = (AB * BP).sum(dim=1)
    d4 = (AC * BP).sum(dim=1)
    maskB = (d3 >= 0) & (d4 <= d3)
    dB = (BP * BP).sum(dim=1)

    CP = P - C
    d5 = (AB * CP).sum(dim=1)
    d6 = (AC * CP).sum(dim=1)
    maskC = (d6 >= 0) & (d5 <= d6)
    dC = (CP * CP).sum(dim=1)

    vc = d1 * d4 - d3 * d2
    maskAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & (~maskA) & (~maskB)
    v = d1 / (d1 - d3 + 1e-20)
    projAB = A + v.unsqueeze(1) * AB
    dAB = ((P - projAB) ** 2).sum(dim=1)

    vb = d5 * d2 - d1 * d6
    maskAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & (~maskA) & (~maskC)
    w = d2 / (d2 - d6 + 1e-20)
    projAC = A + w.unsqueeze(1) * AC
    dAC = ((P - projAC) ** 2).sum(dim=1)

    va = d3 * d6 - d5 * d4
    maskBC = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & (~maskB) & (~maskC)
    BC = C - B
    u = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-20)
    projBC = B + u.unsqueeze(1) * BC
    dBC = ((P - projBC) ** 2).sum(dim=1)

    maskFace = ~(maskA | maskB | maskC | maskAB | maskAC | maskBC)
    n = torch.cross(AB, AC, dim=1)
    n2 = (n * n).sum(dim=1).clamp_min(1e-20)
    distPlane = (AP * n).sum(dim=1)
    dFace = (distPlane * distPlane) / n2

    d = torch.zeros_like(dA)
    d = torch.where(maskA, dA, d)
    d = torch.where(maskB, dB, d)
    d = torch.where(maskC, dC, d)
    d = torch.where(maskAB, dAB, d)
    d = torch.where(maskAC, dAC, d)
    d = torch.where(maskBC, dBC, d)
    d = torch.where(maskFace, dFace, d)
    return d


class DirichletDistance3D:
    """
    Hard-BC distance helper for 3D: d(x, Gamma_u surface).
    Uses k-nearest triangle centroids, then exact point-to-triangle distance.
    """

    def __init__(self, pts_np: np.ndarray, tris_np: np.ndarray, device, tau: float = 0.0, k: int = 16, chunk: int = 2048):
        tri = np.asarray(tris_np)[:, :3].astype(int)
        A = pts_np[tri[:, 0], :]
        B = pts_np[tri[:, 1], :]
        C = pts_np[tri[:, 2], :]
        cent = (A + B + C) / 3.0

        self.A = to_torch(A.astype(np.float32), device=device, requires_grad=False)
        self.B = to_torch(B.astype(np.float32), device=device, requires_grad=False)
        self.C = to_torch(C.astype(np.float32), device=device, requires_grad=False)
        self.cent = to_torch(cent.astype(np.float32), device=device, requires_grad=False)
        self.tau = float(tau)
        self.k = int(max(1, k))
        self.chunk = int(max(64, chunk))

    def distance(self, x, y, z=None):
        assert z is not None, "3D distance requires z"
        P = torch.cat([x, y, z], dim=1)  # (N,3)
        N = int(P.shape[0])
        M = int(self.cent.shape[0])
        kk = int(min(self.k, M))
        out = []
        for i in range(0, N, self.chunk):
            Pi = P[i : i + self.chunk]  # (c,3)
            # nearest centroids
            distc = torch.cdist(Pi, self.cent)  # (c,M)
            _, idx = torch.topk(distc, k=kk, dim=1, largest=False)  # (c,kk)
            Ai = self.A[idx]  # (c,kk,3)
            Bi = self.B[idx]
            Ci = self.C[idx]
            # flatten candidates and compute exact distances
            csz = int(Ai.shape[0])
            Aflat = Ai.reshape(-1, 3)
            Bflat = Bi.reshape(-1, 3)
            Cflat = Ci.reshape(-1, 3)
            # repeat points for each candidate triangle
            Prep = Pi[:, None, :].expand(-1, kk, -1).reshape(-1, 3)
            d2_flat = point_triangle_distance_sq_pairwise_torch(Prep, Aflat, Bflat, Cflat)  # (c*kk,)
            d2p = d2_flat.reshape(csz, kk)
            if self.tau <= 0.0:
                d2min, _ = torch.min(d2p, dim=1, keepdim=True)
            else:
                tau_t = torch.as_tensor(float(self.tau), device=d2p.device, dtype=d2p.dtype).clamp_min(1e-20)
                d2min = -tau_t * (torch.logsumexp(-d2p / tau_t, dim=1, keepdim=True) - math.log(kk))
            out.append(torch.sqrt(d2min.clamp_min(0.0) + 1e-12))
        return torch.cat(out, dim=0)


# ============================================================
# Model
# ============================================================
class MLP(nn.Module):
    def __init__(self, layers, activation="tanh"):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "silu":
            self.act = torch.nn.functional.silu
        elif activation == "gelu":
            self.act = torch.nn.functional.gelu
        else:
            raise ValueError("Unsupported activation")

        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


def _bspline_basis_1d(x: torch.Tensor, knots: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Compute B-spline basis functions N_{i,degree}(x) for a 1D input x.
    x: (N,1)
    knots: (M,)
    Returns: (N, n_basis) where n_basis = M - degree - 1
    """
    # Degree-0 basis
    x = x.squeeze(1)  # (N,)
    M = knots.numel()
    n_basis = M - degree - 1
    # (N, n_basis): knots[i] <= x < knots[i+1]
    left = knots[:-1]
    right = knots[1:]
    # broadcast compare
    x2 = x[:, None]
    Np = ((x2 >= left[None, :n_basis]) & (x2 < right[None, :n_basis])).to(x.dtype)
    # include right boundary at the very end
    Np = torch.where((x2 == knots[-1]) & (torch.arange(n_basis, device=x.device)[None, :] == (n_basis - 1)), torch.ones_like(Np), Np)

    for p in range(1, degree + 1):
        Nn = torch.zeros((x.shape[0], n_basis), device=x.device, dtype=x.dtype)
        for i in range(n_basis):
            denom1 = knots[i + p] - knots[i]
            denom2 = knots[i + p + 1] - knots[i + 1]
            a = 0.0
            b = 0.0
            if denom1.abs() > 1e-12:
                a = (x - knots[i]) / denom1
            if denom2.abs() > 1e-12:
                b = (knots[i + p + 1] - x) / denom2
            Ni = Np[:, i]
            Nip1 = Np[:, i + 1] if (i + 1) < n_basis else torch.zeros_like(Ni)
            Nn[:, i] = a * Ni + b * Nip1
        Np = Nn
    return Np


class KANLayer(nn.Module):
    """
    Minimal KAN-style layer:
      y_j = sum_d sum_k c[j,d,k] * B_k(x_d) + b_j
    where B_k are B-spline basis on a fixed knot grid.
    """

    def __init__(self, in_dim: int, out_dim: int, *, grid_size: int = 8, degree: int = 3, xmin: float = -1.0, xmax: float = 1.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.grid_size = int(grid_size)
        self.degree = int(degree)

        # uniform open knot vector
        # number of basis functions = grid_size + degree
        n_basis = self.grid_size + self.degree
        n_knots = n_basis + self.degree + 1
        # internal knots
        t = torch.linspace(float(xmin), float(xmax), self.grid_size + 1)
        # open clamped ends
        knots = torch.cat([t[0].repeat(self.degree), t, t[-1].repeat(self.degree)])
        if knots.numel() != n_knots:
            # fallback: ensure correct length
            knots = torch.linspace(float(xmin), float(xmax), n_knots)
        self.register_buffer("knots", knots)

        self.coeff = nn.Parameter(torch.zeros(self.out_dim, self.in_dim, n_basis))
        self.bias = nn.Parameter(torch.zeros(self.out_dim))
        nn.init.normal_(self.coeff, mean=0.0, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,in_dim)
        assert x.shape[1] == self.in_dim
        bases = []
        for d in range(self.in_dim):
            bd = _bspline_basis_1d(x[:, d : d + 1], self.knots, self.degree)  # (N,n_basis)
            bases.append(bd)
        B = torch.stack(bases, dim=1)  # (N,in_dim,n_basis)
        # einsum: (N,in_dim,n_basis) * (out,in_dim,n_basis) -> (N,out)
        y = torch.einsum("ndk,odk->no", B, self.coeff) + self.bias[None, :]
        return y


class KANNet(nn.Module):
    def __init__(
        self,
        layers: list[int],
        *,
        grid_size: int = 8,
        degree: int = 3,
        xmin: float = -1.0,
        xmax: float = 1.0,
        activation: str = "tanh",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                KANLayer(layers[i], layers[i + 1], grid_size=grid_size, degree=degree, xmin=xmin, xmax=xmax)
                for i in range(len(layers) - 1)
            ]
        )
        if activation == "tanh":
            self.act = torch.tanh
        elif activation == "silu":
            self.act = torch.nn.functional.silu
        elif activation == "gelu":
            self.act = torch.nn.functional.gelu
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


# ============================================================
# Material helpers
# ============================================================
def lame_from_E_nu(E: float, nu: float, plane: str):
    """
    plane: 'plane_strain' or 'plane_stress'
    Returns (lam, mu).
    """
    mu = E / (2.0 * (1.0 + nu))
    if plane == "plane_strain":
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    else:
        lam = E * nu / (1.0 - nu**2)
    return lam, mu


# ============================================================
# Energy densities
# ============================================================
def energy_poisson(u, x, y, f_expr: str | None):
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    psi = 0.5 * (u_x**2 + u_y**2)

    if f_expr and str(f_expr).strip():
        f = eval_expr_torch(f_expr, x=x, y=y)
    else:
        # Default: no body forcing
        f = torch.zeros_like(x)

    return psi, f * u


def energy_poisson_3d(u, x, y, z, f_expr: str | None):
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    psi = 0.5 * (u_x**2 + u_y**2 + u_z**2)

    if f_expr and str(f_expr).strip():
        f = eval_expr_torch(f_expr, x=x, y=y, z=z)
    else:
        # Default: no body forcing
        f = torch.zeros_like(x)
    return psi, f * u


def energy_Screened(u, x, y, f_expr: str | None, k_squared: float):
    """
    Screened equation energy density:
      -∇²u + k²u = f
    Energy functional: Π = ∫ 0.5|∇u|² dΩ + ∫ 0.5*k²*u² dΩ - ∫ f u dΩ + Wbc
    """
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    psi = 0.5 * (u_x**2 + u_y**2) + 0.5 * k_squared * (u**2)

    if f_expr and str(f_expr).strip():
        f = eval_expr_torch(f_expr, x=x, y=y)
    else:
        # Default: no body forcing
        f = torch.zeros_like(x)

    return psi, f * u


def energy_linear_elasticity(uvec, x, y, lam: float, mu: float):
    u = uvec[:, 0:1]
    v = uvec[:, 1:2]

    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    eps_xx = u_x
    eps_yy = v_y
    eps_xy = 0.5 * (u_y + v_x)

    tr = eps_xx + eps_yy
    eps2 = eps_xx**2 + eps_yy**2 + 2.0 * (eps_xy**2)
    psi = 0.5 * lam * (tr**2) + mu * eps2
    return psi


def energy_linear_elasticity_3d(uvec, x, y, z, lam: float, mu: float):
    """
    3D small-strain linear elasticity energy density:
      psi = 0.5*lam*tr(eps)^2 + mu*eps:eps
    where eps = sym(grad u), and eps:eps = sum_ij eps_ij^2.
    """
    u = uvec[:, 0:1]
    v = uvec[:, 1:2]
    w = uvec[:, 2:3]

    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_z = grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    w_x = grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_y = grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_z = grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]

    eps_xx = u_x
    eps_yy = v_y
    eps_zz = w_z
    eps_xy = 0.5 * (u_y + v_x)
    eps_yz = 0.5 * (v_z + w_y)
    eps_zx = 0.5 * (w_x + u_z)

    tr = eps_xx + eps_yy + eps_zz
    eps2 = eps_xx**2 + eps_yy**2 + eps_zz**2 + 2.0 * (eps_xy**2 + eps_yz**2 + eps_zx**2)
    psi = 0.5 * lam * (tr**2) + mu * eps2
    return psi


def energy_neo_hookean_2d(uvec, x, y, lam: float, mu: float):
    u = uvec[:, 0:1]
    v = uvec[:, 1:2]

    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    F11 = 1.0 + u_x
    F12 = u_y
    F21 = v_x
    F22 = 1.0 + v_y

    J = F11 * F22 - F12 * F21
    J_safe = torch.clamp(J, min=1e-8)

    C11 = F11**2 + F21**2
    C22 = F12**2 + F22**2
    I1 = C11 + C22

    lnJ = torch.log(J_safe)
    psi = 0.5 * mu * (I1 - 2.0) - mu * lnJ + 0.5 * lam * (lnJ**2)
    return psi


# ============================================================
# Dirichlet wrappers (Hard BC via distance-to-Gamma_u)
# ============================================================
def apply_dirichlet_scalar(nn_out, x, y, dir_mode: str, dist_obj, ubar_expr: str, ubar_const: float = 0.0, z=None):
    if "hard" not in dir_mode:
        return nn_out
    assert dist_obj is not None, "Hard BC requires Gamma_u segments (distance object)."
    if ubar_expr and str(ubar_expr).strip():
        ubar = eval_expr_torch(ubar_expr, x=x, y=y, z=z)
    else:
        ubar = torch.full_like(x, float(ubar_const))
    d = dist_obj.distance(x, y, z)
    return ubar + d * nn_out


def apply_dirichlet_vec(
    nn_out,
    x,
    y,
    dir_mode: str,
    dist_obj,
    ubarx_expr: str,
    ubary_expr: str,
    ubarz_expr: str | None = None,
    ubarx_const: float = 0.0,
    ubary_const: float = 0.0,
    ubarz_const: float = 0.0,
    z=None,
):
    if "hard" not in dir_mode:
        return nn_out
    assert dist_obj is not None, "Hard BC requires Gamma_u segments (distance object)."
    if ubarx_expr and str(ubarx_expr).strip():
        ubarx = eval_expr_torch(ubarx_expr, x=x, y=y, z=z)
    else:
        ubarx = torch.full_like(x, float(ubarx_const))
    if ubary_expr and str(ubary_expr).strip():
        ubary = eval_expr_torch(ubary_expr, x=x, y=y, z=z)
    else:
        ubary = torch.full_like(y, float(ubary_const))
    if z is None:
        ubar = torch.cat([ubarx, ubary], dim=1)
    else:
        if ubarz_expr and str(ubarz_expr).strip():
            ubarz = eval_expr_torch(ubarz_expr, x=x, y=y, z=z)
        else:
            ubarz = torch.full_like(z, float(ubarz_const))
        ubar = torch.cat([ubarx, ubary, ubarz], dim=1)
    d = dist_obj.distance(x, y, z)
    return ubar + d * nn_out


# ============================================================
# Training (mesh-based DEM)
# ============================================================
def train_dem_mesh(
    model,
    problem_type: str,
    epochs: int,
    lr: float,
    device,
    Xdom_full,
    Wdom_full,
    Xu_full,
    Wu_full,
    Xt_full,
    Wt_full,
    dist_obj: DirichletDistance | None,
    dom_batch: int,
    bnd_u_batch: int,
    bnd_t_batch: int,
    dir_mode: str,
    penalty_lambda: float,
    f_expr: str | None,
    g_expr: str | None,
    g_const: float,
    ubar_expr: str = "",
    ubar_const: float = 0.0,
    ubarx_expr: str = "",
    ubary_expr: str = "",
    ubarz_expr: str = "",
    ubarx_const: float = 0.0,
    ubary_const: float = 0.0,
    ubarz_const: float = 0.0,
    thickness: float = 1.0,
    plane_mode: str = "plane_stress",
    lam: float = 0.0,
    mu: float = 0.0,
    tx_expr: str | None = None,
    ty_expr: str | None = None,
    tz_expr: str | None = None,
    tx_const: float = 0.0,
    ty_const: float = 0.0,
    tz_const: float = 0.0,
    bx_expr: str | None = None,
    by_expr: str | None = None,
    bz_expr: str | None = None,
    bx_const: float = 0.0,
    by_const: float = 0.0,
    bz_const: float = 0.0,
    k_squared: float = 0.0,
    # --- Custom functional (optional) ---
    custom_out_dim: int = 1,
    custom_pi_omega_expr: str | None = None,
    custom_pi_gu_expr: str | None = None,
    custom_pi_gt_expr: str | None = None,
    custom_pi_gt_mode: Literal["extra", "replace"] = "extra",
    # --- shape-function mode (no autograd derivatives w.r.t. coordinates) ---
    derivative_method: str = "shape_function (mesh)",
    pts_nodes: np.ndarray | None = None,
    sf_dom_2d: dict | None = None,
    sf_gu_2d: dict | None = None,
    sf_gt_2d: dict | None = None,
    abort_on_spike: bool = True,
    spike_rtol: float = 1e-5,
    spike_warmup: int = 200,
    stable_rel_rtol: float = 1e-4,
    stable_rel_patience: int = 10,
    spike_eps: float = 1e-12,
    log_every=100,
):
    model.to(device)
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt = SOAP(model.parameters(), lr=lr, betas=(0.95, 0.95), weight_decay=0.01, precondition_frequency=10)
    hist = {"Pi": [], "Wint": [], "Wext": [], "Wext_body": [], "Wext_trac": [], "Wbc": []}
    progress = st.progress(0)
    status = st.empty()
    t0 = time.time()

    # Enable shape-function mode by default for 2D; fall back to autograd for 3D.
    use_shape = str(derivative_method or "").strip().lower().startswith("shape")
    try:
        if int(np.asarray(Xdom_full).shape[1]) != 2:
            use_shape = False
    except Exception:
        use_shape = False

    # Custom functional: use autograd mode (so we can expose u, ∇u, etc. in expressions).
    if str(problem_type) == "Custom (user-defined)":
        use_shape = False

    def _interp_from_nodes(u_nodes: torch.Tensor, conn: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        ue = u_nodes[conn]  # (B, nn, C)
        return torch.sum(ue * N.unsqueeze(-1), dim=1)  # (B, C)

    def _interp_grad_from_nodes(u_nodes: torch.Tensor, conn: torch.Tensor, dN: torch.Tensor) -> torch.Tensor:
        ue = u_nodes[conn]  # (B, nn, C)
        return torch.sum(ue * dN.unsqueeze(-1), dim=1)  # (B, C)

    if use_shape:
        # Missing/invalid cache -> fall back to autograd (prevents CUDA device-side asserts from bad connectivity).
        if pts_nodes is None or sf_dom_2d is None:
            use_shape = False
        else:
            try:
                n_nodes = int(np.asarray(pts_nodes).shape[0])
                ok1, _ = _sf_conn_is_valid(sf_dom_2d, n_nodes)
                ok2, _ = _sf_conn_is_valid(sf_gu_2d, n_nodes) if sf_gu_2d is not None else (True, "")
                ok3, _ = _sf_conn_is_valid(sf_gt_2d, n_nodes) if sf_gt_2d is not None else (True, "")
                if not (ok1 and ok2 and ok3):
                    use_shape = False
            except Exception:
                use_shape = False

    if use_shape:
        pts_nodes_np = np.asarray(pts_nodes).astype(np.float32)
        Xn_t = torch.tensor(pts_nodes_np[:, :2], device=device, dtype=torch.float32)  # (Nnodes,2)

        Xdom_full_t = torch.tensor(np.asarray(sf_dom_2d["X"], dtype=np.float32), device=device)
        Wdom_full_t = torch.tensor(np.asarray(sf_dom_2d["W"], dtype=np.float32), device=device)
        conn_dom_t = torch.tensor(np.asarray(sf_dom_2d["conn"], dtype=np.int64), device=device, dtype=torch.long)
        N_dom_t = torch.tensor(np.asarray(sf_dom_2d["N"], dtype=np.float32), device=device)
        dNdx_dom_t = torch.tensor(np.asarray(sf_dom_2d["dNdx"], dtype=np.float32), device=device)
        dNdy_dom_t = torch.tensor(np.asarray(sf_dom_2d["dNdy"], dtype=np.float32), device=device)

        sf_gu_2d = sf_gu_2d or {"X": np.zeros((0, 2), np.float32), "W": np.zeros((0, 1), np.float32), "conn": np.zeros((0, 2), np.int64), "N": np.zeros((0, 2), np.float32)}
        sf_gt_2d = sf_gt_2d or {"X": np.zeros((0, 2), np.float32), "W": np.zeros((0, 1), np.float32), "conn": np.zeros((0, 2), np.int64), "N": np.zeros((0, 2), np.float32)}

        Xu_full_t = torch.tensor(np.asarray(sf_gu_2d["X"], dtype=np.float32), device=device)
        Wu_full_t = torch.tensor(np.asarray(sf_gu_2d["W"], dtype=np.float32), device=device)
        conn_u_t = torch.tensor(np.asarray(sf_gu_2d["conn"], dtype=np.int64), device=device, dtype=torch.long)
        N_u_t = torch.tensor(np.asarray(sf_gu_2d["N"], dtype=np.float32), device=device)

        Xt_full_t = torch.tensor(np.asarray(sf_gt_2d["X"], dtype=np.float32), device=device)
        Wt_full_t = torch.tensor(np.asarray(sf_gt_2d["W"], dtype=np.float32), device=device)
        conn_t_t = torch.tensor(np.asarray(sf_gt_2d["conn"], dtype=np.int64), device=device, dtype=torch.long)
        N_t_t = torch.tensor(np.asarray(sf_gt_2d["N"], dtype=np.float32), device=device)

        Ndom = int(Xdom_full_t.shape[0])
        Nu = int(Xu_full_t.shape[0])
        Nt = int(Xt_full_t.shape[0])
    else:
        Ndom = Xdom_full.shape[0]
        Nu = Xu_full.shape[0]
        Nt = Xt_full.shape[0]

        Xdom_full_t = to_torch(Xdom_full, device=device, requires_grad=False)
        Wdom_full_t = to_torch(Wdom_full, device=device, requires_grad=False)

        Xu_full_t = to_torch(Xu_full, device=device, requires_grad=False)
        Wu_full_t = to_torch(Wu_full, device=device, requires_grad=False)

        Xt_full_t = to_torch(Xt_full, device=device, requires_grad=False)
        Wt_full_t = to_torch(Wt_full, device=device, requires_grad=False)

    prev_pi: float | None = None
    best_pi = math.inf
    best_epoch = 0
    best_state_cpu: dict[str, torch.Tensor] | None = None
    stable_rel_count = 0

    for ep in range(1, epochs + 1):
        opt.zero_grad()

        dom_idx = torch.randperm(int(Ndom), device=device)[: min(int(dom_batch), int(Ndom))]
        u_idx = (
            torch.randperm(int(Nu), device=device)[: min(int(bnd_u_batch), int(Nu))] if int(Nu) > 0 else torch.zeros((0,), device=device, dtype=torch.long)
        )
        t_idx = (
            torch.randperm(int(Nt), device=device)[: min(int(bnd_t_batch), int(Nt))] if int(Nt) > 0 else torch.zeros((0,), device=device, dtype=torch.long)
        )

        Xdom = Xdom_full_t[dom_idx].clone().detach().requires_grad_(not use_shape)
        Wdom = Wdom_full_t[dom_idx]

        Xu = Xu_full_t[u_idx].clone().detach().requires_grad_(False)
        Wu = Wu_full_t[u_idx]

        Xt = Xt_full_t[t_idx].clone().detach().requires_grad_(False)
        Wt = Wt_full_t[t_idx]

        dim = int(Xdom.shape[1])
        x = Xdom[:, 0:1]
        y = Xdom[:, 1:2]
        z = Xdom[:, 2:3] if dim == 3 else None
        w = Wdom

        xu = Xu[:, 0:1]
        yu = Xu[:, 1:2]
        zu = Xu[:, 2:3] if dim == 3 else None

        xt = Xt[:, 0:1]
        yt = Xt[:, 1:2]
        zt = Xt[:, 2:3] if dim == 3 else None

        Wint = torch.tensor(0.0, device=device)
        Wext = torch.tensor(0.0, device=device)
        Wext_body = torch.tensor(0.0, device=device)
        Wext_trac = torch.tensor(0.0, device=device)
        Wbc = torch.tensor(0.0, device=device)

        if problem_type == "Poisson (scalar)":
            if use_shape:
                # nodal scalar
                nn_nodes = model(Xn_t)  # (Nnodes,1)
                xn = Xn_t[:, 0:1]
                yn = Xn_t[:, 1:2]
                u_nodes = apply_dirichlet_scalar(nn_nodes, xn, yn, dir_mode, dist_obj, ubar_expr, z=None)  # (Nnodes,1)

                conn_b = conn_dom_t[dom_idx]
                N_b = N_dom_t[dom_idx]
                dNdx_b = dNdx_dom_t[dom_idx]
                dNdy_b = dNdy_dom_t[dom_idx]

                u_gp = _interp_from_nodes(u_nodes, conn_b, N_b)  # (B,1)
                ux = _interp_grad_from_nodes(u_nodes, conn_b, dNdx_b)
                uy = _interp_grad_from_nodes(u_nodes, conn_b, dNdy_b)
                psi_int = 0.5 * (ux**2 + uy**2)

                if f_expr and str(f_expr).strip():
                    f = eval_expr_torch(f_expr, x=x, y=y)
                else:
                    f = torch.zeros_like(x)
                dens_ext = f * u_gp

                Wint = torch.sum(w * psi_int)
                Wext_body = torch.sum(w * dens_ext)

                if int(Nt) > 0 and torch.any(Wt > 0):
                    conn_tb = conn_t_t[t_idx]
                    Nt_b = N_t_t[t_idx]
                    ut = _interp_from_nodes(u_nodes, conn_tb, Nt_b)
                    if g_expr and str(g_expr).strip():
                        g = eval_expr_torch(g_expr, x=xt, y=yt)
                    else:
                        g = torch.full_like(xt, float(g_const))
                    Wext_trac = torch.sum(Wt * (g * ut))

                Wext = Wext_body + Wext_trac

                if ("penalty" in dir_mode) and penalty_lambda > 0 and int(Nu) > 0 and torch.any(Wu > 0):
                    conn_ub = conn_u_t[u_idx]
                    Nu_b = N_u_t[u_idx]
                    ub_pred = _interp_from_nodes(u_nodes, conn_ub, Nu_b)
                    if ubar_expr and str(ubar_expr).strip():
                        ub_bar = eval_expr_torch(ubar_expr, x=xu, y=yu)
                    else:
                        ub_bar = torch.full_like(xu, float(ubar_const))
                    Wbc = penalty_lambda * torch.sum(Wu * (ub_pred - ub_bar) ** 2)

                Pi = Wint - Wext + Wbc
            else:
                # --- autograd (AD) coordinate-derivative mode ---
                if dim == 2:
                    nn_out = model(torch.cat([x, y], dim=1))  # (N,1)
                    u = apply_dirichlet_scalar(nn_out, x, y, dir_mode, dist_obj, ubar_expr, ubar_const=ubar_const, z=None)
                    psi_int, dens_ext = energy_poisson(u, x, y, f_expr)
                elif dim == 3:
                    nn_out = model(torch.cat([x, y, z], dim=1))  # (N,1)
                    u = apply_dirichlet_scalar(nn_out, x, y, dir_mode, dist_obj, ubar_expr, ubar_const=ubar_const, z=z)
                    psi_int, dens_ext = energy_poisson_3d(u, x, y, z, f_expr)
                else:
                    raise ValueError(f"Unsupported input dimension: {dim}")

                Wint = torch.sum(w * psi_int)
                Wext_body = torch.sum(w * dens_ext)

                # Optional Neumann/flux on Gamma_t: Π includes -∫ g u dΓ, so external work is ∫ g u dΓ
                if Nt > 0 and torch.any(Wt > 0):
                    if dim == 2:
                        ut_nn = model(torch.cat([xt, yt], dim=1))
                        ut = apply_dirichlet_scalar(ut_nn, xt, yt, dir_mode, dist_obj, ubar_expr, ubar_const=ubar_const, z=None)
                        if g_expr and str(g_expr).strip():
                            g = eval_expr_torch(g_expr, x=xt, y=yt)
                        else:
                            g = torch.full_like(xt, float(g_const))
                    else:
                        ut_nn = model(torch.cat([xt, yt, zt], dim=1))
                        ut = apply_dirichlet_scalar(ut_nn, xt, yt, dir_mode, dist_obj, ubar_expr, ubar_const=ubar_const, z=zt)
                        if g_expr and str(g_expr).strip():
                            g = eval_expr_torch(g_expr, x=xt, y=yt, z=zt)
                        else:
                            g = torch.full_like(xt, float(g_const))
                    Wext_trac = torch.sum(Wt * (g * ut))

                Wext = Wext_body + Wext_trac

                if ("penalty" in dir_mode) and penalty_lambda > 0:
                    if dim == 2:
                        ub_pred = model(torch.cat([xu, yu], dim=1))
                        ub_pred = (
                            apply_dirichlet_scalar(ub_pred, xu, yu, "hard", dist_obj, ubar_expr, ubar_const=ubar_const, z=None)
                            if ("hard" in dir_mode)
                            else ub_pred
                        )
                        if ubar_expr and str(ubar_expr).strip():
                            ub_bar = eval_expr_torch(ubar_expr, x=xu, y=yu)
                        else:
                            ub_bar = torch.full_like(xu, float(ubar_const))
                    else:
                        ub_pred = model(torch.cat([xu, yu, zu], dim=1))
                        if ubar_expr and str(ubar_expr).strip():
                            ub_bar = eval_expr_torch(ubar_expr, x=xu, y=yu, z=zu)
                        else:
                            ub_bar = torch.full_like(xu, float(ubar_const))
                    Wbc = penalty_lambda * torch.sum(Wu * (ub_pred - ub_bar) ** 2)

                Pi = Wint - Wext + Wbc

        elif problem_type == "Screened Poisson equation (2D)":
            if use_shape:
                # nodal scalar
                nn_nodes = model(Xn_t)  # (Nnodes,1)
                xn = Xn_t[:, 0:1]
                yn = Xn_t[:, 1:2]
                u_nodes = apply_dirichlet_scalar(nn_nodes, xn, yn, dir_mode, dist_obj, ubar_expr, z=None)  # (Nnodes,1)

                conn_b = conn_dom_t[dom_idx]
                N_b = N_dom_t[dom_idx]
                dNdx_b = dNdx_dom_t[dom_idx]
                dNdy_b = dNdy_dom_t[dom_idx]

                u_gp = _interp_from_nodes(u_nodes, conn_b, N_b)  # (B,1)
                ux = _interp_grad_from_nodes(u_nodes, conn_b, dNdx_b)
                uy = _interp_grad_from_nodes(u_nodes, conn_b, dNdy_b)
                psi_int = 0.5 * (ux**2 + uy**2) + 0.5 * float(k_squared) * (u_gp**2)

                if f_expr and str(f_expr).strip():
                    f = eval_expr_torch(f_expr, x=x, y=y)
                else:
                    f = torch.zeros_like(x)
                dens_ext = f * u_gp

                Wint = torch.sum(w * psi_int)
                Wext_body = torch.sum(w * dens_ext)

                if int(Nt) > 0 and torch.any(Wt > 0):
                    conn_tb = conn_t_t[t_idx]
                    Nt_b = N_t_t[t_idx]
                    ut = _interp_from_nodes(u_nodes, conn_tb, Nt_b)
                    if g_expr and str(g_expr).strip():
                        g = eval_expr_torch(g_expr, x=xt, y=yt)
                    else:
                        g = torch.full_like(xt, float(g_const))
                    Wext_trac = torch.sum(Wt * (g * ut))

                Wext = Wext_body + Wext_trac

                if ("penalty" in dir_mode) and penalty_lambda > 0 and int(Nu) > 0 and torch.any(Wu > 0):
                    conn_ub = conn_u_t[u_idx]
                    Nu_b = N_u_t[u_idx]
                    ub_pred = _interp_from_nodes(u_nodes, conn_ub, Nu_b)
                    if ubar_expr and str(ubar_expr).strip():
                        ub_bar = eval_expr_torch(ubar_expr, x=xu, y=yu)
                    else:
                        ub_bar = torch.full_like(xu, float(ubar_const))
                    Wbc = penalty_lambda * torch.sum(Wu * (ub_pred - ub_bar) ** 2)

                Pi = Wint - Wext + Wbc
            else:
                # --- autograd (AD) coordinate-derivative mode ---
                if dim != 2:
                    raise ValueError("Screened Poisson equation (2D) requires a 2D mesh.")
                nn_out = model(torch.cat([x, y], dim=1))  # (N,1)
                u = apply_dirichlet_scalar(nn_out, x, y, dir_mode, dist_obj, ubar_expr, z=None)
                psi_int, dens_ext = energy_Screened(u, x, y, f_expr, float(k_squared))

                Wint = torch.sum(w * psi_int)
                Wext_body = torch.sum(w * dens_ext)

                # Optional Neumann/flux on Gamma_t: Π includes -∫ g u dΓ, so external work is ∫ g u dΓ
                if Nt > 0 and torch.any(Wt > 0):
                    ut_nn = model(torch.cat([xt, yt], dim=1))
                    ut = apply_dirichlet_scalar(ut_nn, xt, yt, dir_mode, dist_obj, ubar_expr, z=None)
                    if g_expr and str(g_expr).strip():
                        g = eval_expr_torch(g_expr, x=xt, y=yt)
                    else:
                        g = torch.full_like(xt, float(g_const))
                    Wext_trac = torch.sum(Wt * (g * ut))

                Wext = Wext_body + Wext_trac

                if ("penalty" in dir_mode) and penalty_lambda > 0:
                    ub_pred = model(torch.cat([xu, yu], dim=1))
                    ub_pred = (
                        apply_dirichlet_scalar(ub_pred, xu, yu, "hard", dist_obj, ubar_expr, z=None)
                        if ("hard" in dir_mode)
                        else ub_pred
                    )
                    if ubar_expr and str(ubar_expr).strip():
                        ub_bar = eval_expr_torch(ubar_expr, x=xu, y=yu)
                    else:
                        ub_bar = torch.full_like(xu, float(ubar_const))
                    Wbc = penalty_lambda * torch.sum(Wu * (ub_pred - ub_bar) ** 2)

                Pi = Wint - Wext + Wbc

        elif problem_type == "Custom (user-defined)":
            # Custom functional with **built-in BC handling** (same as predefined problems):
            # - Dirichlet on Γu is enforced by Hard/Penalty settings (dir_mode, penalty_lambda, ubar*)
            # - Natural BC on Γt uses sidebar traction/Neumann inputs (tx/ty/tz)
            # The optional custom boundary terms π_u / π_t are treated as extra add-ons.
            cdim = int(custom_out_dim) if custom_out_dim is not None else 1
            cdim = int(max(1, min(3, cdim)))
            if int(dim) == 2:
                cdim = int(min(2, cdim))
            if int(dim) == 3 and cdim == 2:
                raise ValueError("Custom (user-defined): out_dim=2 is not supported for 3D meshes. Use out_dim=1 or out_dim=3.")
            expr_dom = str(custom_pi_omega_expr or "0").strip() or "0"
            expr_gu = str(custom_pi_gu_expr or "0").strip() or "0"
            expr_gt = str(custom_pi_gt_expr or "0").strip() or "0"

            def _d(s: torch.Tensor, v: torch.Tensor | None) -> torch.Tensor | None:
                if v is None:
                    return None
                return grad(s, v, grad_outputs=torch.ones_like(s), create_graph=True)[0]

            # Domain prediction
            nn_out = model(torch.cat([x, y], dim=1) if dim == 2 else torch.cat([x, y, z], dim=1))
            if cdim == 1:
                u_pred = apply_dirichlet_scalar(nn_out, x, y, dir_mode, dist_obj, ubar_expr, ubar_const=ubar_const, z=z)
                ux = _d(u_pred, x)
                uy = _d(u_pred, y)
                uz = _d(u_pred, z)
                pi_omega = eval_expr_torch_ext(
                    expr_dom,
                    x=x,
                    y=y,
                    z=z,
                    u=u_pred,
                    ux=ux,
                    uy=uy,
                    uz=uz,
                    lam=float(lam),
                    mu=float(mu),
                    E=float(E),
                    nu=float(nu),
                )
            else:
                uvec = apply_dirichlet_vec(
                    nn_out,
                    x,
                    y,
                    dir_mode,
                    dist_obj,
                    ubarx_expr,
                    ubary_expr,
                    ubarz_expr=(ubarz_expr if cdim >= 3 else None),
                    ubarx_const=ubarx_const,
                    ubary_const=ubary_const,
                    ubarz_const=ubarz_const,
                    z=z,
                )
                u1 = uvec[:, 0:1]
                v1 = uvec[:, 1:2] if cdim >= 2 else torch.zeros_like(u1)
                w1 = uvec[:, 2:3] if cdim >= 3 else None

                ux = _d(u1, x)
                uy = _d(u1, y)
                uz = _d(u1, z)
                vx = _d(v1, x)
                vy = _d(v1, y)
                vz = _d(v1, z)

                extra = {
                    "u": u1,
                    "v": v1,
                    "ux": ux,
                    "uy": uy,
                    "uz": uz,
                    "vx": vx,
                    "vy": vy,
                    "vz": vz,
                }
                if w1 is not None:
                    wx = _d(w1, x)
                    wy = _d(w1, y)
                    wz = _d(w1, z)
                    extra.update({"w": w1, "wx": wx, "wy": wy, "wz": wz})

                # 2D strain helpers (convenient for mechanics-style custom Π)
                if dim == 2:
                    extra["eps_xx"] = extra["ux"]
                    extra["eps_yy"] = extra["vy"]
                    extra["eps_xy"] = 0.5 * (extra["uy"] + extra["vx"])
                    extra["tr_eps"] = extra["eps_xx"] + extra["eps_yy"]

                pi_omega = eval_expr_torch_ext(
                    expr_dom,
                    x=x,
                    y=y,
                    z=z,
                    lam=float(lam),
                    mu=float(mu),
                    E=float(E),
                    nu=float(nu),
                    **extra,
                )

            # 1) Domain term
            Wint = torch.sum(w * pi_omega)
            Pi_extra_t = torch.tensor(0.0, device=device)
            Wext_body = torch.tensor(0.0, device=device)
            Wext_trac = torch.tensor(0.0, device=device)

            # 2) Built-in traction / Neumann on Γt (same as predefined)
            if int(Nt) > 0 and torch.any(Wt > 0):
                nn_t = model(torch.cat([xt, yt], dim=1) if dim == 2 else torch.cat([xt, yt, zt], dim=1))
                if cdim == 1:
                    u_t = apply_dirichlet_scalar(nn_t, xt, yt, dir_mode, dist_obj, ubar_expr, ubar_const=ubar_const, z=zt)
                    txv = eval_expr_torch(tx_expr, x=xt, y=yt, z=zt) if (tx_expr and str(tx_expr).strip()) else torch.full_like(xt, float(tx_const))
                    Wext_trac = torch.sum(Wt * (txv * u_t))
                else:
                    u_tvec = apply_dirichlet_vec(
                        nn_t,
                        xt,
                        yt,
                        dir_mode,
                        dist_obj,
                        ubarx_expr,
                        ubary_expr,
                        ubarz_expr=(ubarz_expr if cdim >= 3 else None),
                        ubarx_const=ubarx_const,
                        ubary_const=ubary_const,
                        ubarz_const=ubarz_const,
                        z=zt,
                    )
                    txv = eval_expr_torch(tx_expr, x=xt, y=yt, z=zt) if (tx_expr and str(tx_expr).strip()) else torch.full_like(xt, float(tx_const))
                    tyv = eval_expr_torch(ty_expr, x=xt, y=yt, z=zt) if (ty_expr and str(ty_expr).strip()) else torch.full_like(yt, float(ty_const))
                    if cdim >= 3:
                        tzv = eval_expr_torch(tz_expr, x=xt, y=yt, z=zt) if (tz_expr and str(tz_expr).strip()) else torch.full_like((zt if zt is not None else xt), float(tz_const))
                        Wext_trac = torch.sum(Wt * (txv * u_tvec[:, 0:1] + tyv * u_tvec[:, 1:2] + tzv * u_tvec[:, 2:3]))
                    else:
                        Wext_trac = torch.sum(Wt * (txv * u_tvec[:, 0:1] + tyv * u_tvec[:, 1:2]))

            # 2b) Replace mode: when user specifies π_t, use it *instead of* built-in traction (not extra)
            if custom_pi_gt_mode == "replace" and expr_gt != "0":
                Wext_trac = torch.tensor(0.0, device=device)

            # 3) Optional extra Γt term from user (added directly to Π)
            if int(Nt) > 0 and torch.any(Wt > 0) and (expr_gt != "0"):
                # expose u and tx/ty/tz variables for convenience
                nn_t2 = model(torch.cat([xt, yt], dim=1) if dim == 2 else torch.cat([xt, yt, zt], dim=1))
                if cdim == 1:
                    u_t2 = apply_dirichlet_scalar(nn_t2, xt, yt, dir_mode, dist_obj, ubar_expr, ubar_const=ubar_const, z=zt)
                    txv2 = eval_expr_torch(tx_expr, x=xt, y=yt, z=zt) if (tx_expr and str(tx_expr).strip()) else torch.full_like(xt, float(tx_const))
                    tyv2 = eval_expr_torch(ty_expr, x=xt, y=yt, z=zt) if (ty_expr and str(ty_expr).strip()) else torch.full_like(yt, float(ty_const))
                    tzv2 = eval_expr_torch(tz_expr, x=xt, y=yt, z=zt) if (zt is not None and tz_expr and str(tz_expr).strip()) else (torch.full_like(zt, float(tz_const)) if zt is not None else None)
                    pi_t = eval_expr_torch_ext(expr_gt, x=xt, y=yt, z=zt, u=u_t2, tx=txv2, ty=tyv2, tz=tzv2, lam=float(lam), mu=float(mu), E=float(E), nu=float(nu))
                else:
                    u_tvec2 = apply_dirichlet_vec(
                        nn_t2,
                        xt,
                        yt,
                        dir_mode,
                        dist_obj,
                        ubarx_expr,
                        ubary_expr,
                        ubarz_expr=(ubarz_expr if cdim >= 3 else None),
                        ubarx_const=ubarx_const,
                        ubary_const=ubary_const,
                        ubarz_const=ubarz_const,
                        z=zt,
                    )
                    txv2 = eval_expr_torch(tx_expr, x=xt, y=yt, z=zt) if (tx_expr and str(tx_expr).strip()) else torch.full_like(xt, float(tx_const))
                    tyv2 = eval_expr_torch(ty_expr, x=xt, y=yt, z=zt) if (ty_expr and str(ty_expr).strip()) else torch.full_like(yt, float(ty_const))
                    tzv2 = eval_expr_torch(tz_expr, x=xt, y=yt, z=zt) if (zt is not None and tz_expr and str(tz_expr).strip()) else (torch.full_like(zt, float(tz_const)) if zt is not None else None)
                    extra_t = {"u": u_tvec2[:, 0:1], "v": u_tvec2[:, 1:2]}
                    if cdim >= 3:
                        extra_t["w"] = u_tvec2[:, 2:3]
                    pi_t = eval_expr_torch_ext(expr_gt, x=xt, y=yt, z=zt, tx=txv2, ty=tyv2, tz=tzv2, lam=float(lam), mu=float(mu), E=float(E), nu=float(nu), **extra_t)
                Pi_extra_t = torch.sum(Wt * pi_t)

            # 4) Built-in penalty on Γu (same as predefined)
            if ("penalty" in dir_mode) and float(penalty_lambda) > 0 and int(Nu) > 0 and torch.any(Wu > 0):
                if cdim == 1:
                    ub_pred = model(torch.cat([xu, yu], dim=1) if dim == 2 else torch.cat([xu, yu, zu], dim=1))
                    if "hard" in dir_mode:
                        ub_pred = apply_dirichlet_scalar(ub_pred, xu, yu, "hard", dist_obj, ubar_expr, ubar_const=ubar_const, z=zu)
                    ub_bar = eval_expr_torch(ubar_expr, x=xu, y=yu, z=zu) if (ubar_expr and str(ubar_expr).strip()) else torch.full_like(xu, float(ubar_const))
                    Wbc = float(penalty_lambda) * torch.sum(Wu * (ub_pred - ub_bar) ** 2)
                else:
                    ub_pred = model(torch.cat([xu, yu], dim=1) if dim == 2 else torch.cat([xu, yu, zu], dim=1))
                    if "hard" in dir_mode:
                        ub_pred = apply_dirichlet_vec(
                            ub_pred,
                            xu,
                            yu,
                            "hard",
                            dist_obj,
                            ubarx_expr,
                            ubary_expr,
                            ubarz_expr=(ubarz_expr if cdim >= 3 else None),
                            ubarx_const=ubarx_const,
                            ubary_const=ubary_const,
                            ubarz_const=ubarz_const,
                            z=zu,
                        )
                    ubarx = eval_expr_torch(ubarx_expr, x=xu, y=yu, z=zu) if (ubarx_expr and str(ubarx_expr).strip()) else torch.full_like(xu, float(ubarx_const))
                    ubary = eval_expr_torch(ubary_expr, x=xu, y=yu, z=zu) if (ubary_expr and str(ubary_expr).strip()) else torch.full_like(yu, float(ubary_const))
                    ubar = torch.cat([ubarx, ubary], dim=1)
                    Wbc = float(penalty_lambda) * torch.sum(Wu * torch.sum((ub_pred[:, :2] - ubar) ** 2, dim=1, keepdim=True))

            # 5) Optional extra Γu term from user (added on top of penalty)
            if int(Nu) > 0 and torch.any(Wu > 0) and (expr_gu != "0"):
                nn_u2 = model(torch.cat([xu, yu], dim=1) if dim == 2 else torch.cat([xu, yu, zu], dim=1))
                if cdim == 1:
                    u_u = apply_dirichlet_scalar(nn_u2, xu, yu, dir_mode, dist_obj, ubar_expr, ubar_const=ubar_const, z=zu)
                    ubar_u = eval_expr_torch(ubar_expr, x=xu, y=yu, z=zu) if (ubar_expr and str(ubar_expr).strip()) else torch.full_like(xu, float(ubar_const))
                    pi_u = eval_expr_torch_ext(expr_gu, x=xu, y=yu, z=zu, u=u_u, ubar=ubar_u, lam=float(lam), mu=float(mu), E=float(E), nu=float(nu))
                else:
                    u_uvec = apply_dirichlet_vec(
                        nn_u2,
                        xu,
                        yu,
                        dir_mode,
                        dist_obj,
                        ubarx_expr,
                        ubary_expr,
                        ubarz_expr=(ubarz_expr if cdim >= 3 else None),
                        ubarx_const=ubarx_const,
                        ubary_const=ubary_const,
                        ubarz_const=ubarz_const,
                        z=zu,
                    )
                    extra_u = {"u": u_uvec[:, 0:1], "v": u_uvec[:, 1:2]}
                    if cdim >= 3:
                        extra_u["w"] = u_uvec[:, 2:3]
                    pi_u = eval_expr_torch_ext(expr_gu, x=xu, y=yu, z=zu, lam=float(lam), mu=float(mu), E=float(E), nu=float(nu), **extra_u)
                Wbc = Wbc + torch.sum(Wu * pi_u)

            # Final assembly matches predefined: Π = Wint - Wext + Wbc (+ extra boundary terms already accumulated in Wext/Wbc)
            Wext = Wext_body + Wext_trac  # Wext_body stays 0 for Custom for now
            Pi = Wint - Wext + Wbc + Pi_extra_t

        elif problem_type == "Linear Elasticity (2D)":
            if use_shape:
                nn_nodes = model(Xn_t)  # (Nnodes,2)
                xn = Xn_t[:, 0:1]
                yn = Xn_t[:, 1:2]
                u_nodes = apply_dirichlet_vec(nn_nodes, xn, yn, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=None, ubarx_const=ubarx_const, ubary_const=ubary_const, ubarz_const=0.0, z=None)  # (Nnodes,2)

                conn_b = conn_dom_t[dom_idx]
                N_b = N_dom_t[dom_idx]
                dNdx_b = dNdx_dom_t[dom_idx]
                dNdy_b = dNdy_dom_t[dom_idx]

                u_gp = _interp_from_nodes(u_nodes, conn_b, N_b)  # (B,2)
                du_dx = _interp_grad_from_nodes(u_nodes, conn_b, dNdx_b)  # (B,2)
                du_dy = _interp_grad_from_nodes(u_nodes, conn_b, dNdy_b)  # (B,2)

                u_x = du_dx[:, 0:1]
                v_x = du_dx[:, 1:2]
                u_y = du_dy[:, 0:1]
                v_y = du_dy[:, 1:2]

                eps_xx = u_x
                eps_yy = v_y
                eps_xy = 0.5 * (u_y + v_x)
                tr_eps = eps_xx + eps_yy
                eps2 = eps_xx**2 + eps_yy**2 + 2.0 * (eps_xy**2)
                psi = 0.5 * float(lam) * (tr_eps**2) + float(mu) * eps2
                Wint = torch.sum(w * psi)

                if bx_expr and str(bx_expr).strip():
                    bx = eval_expr_torch(bx_expr, x=x, y=y)
                else:
                    bx = torch.full_like(x, float(bx_const))
                if by_expr and str(by_expr).strip():
                    by = eval_expr_torch(by_expr, x=x, y=y)
                else:
                    by = torch.full_like(y, float(by_const))

                Wext_body = torch.sum(w * (bx * u_gp[:, 0:1] + by * u_gp[:, 1:2]))

                if int(Nt) > 0 and torch.any(Wt > 0):
                    conn_tb = conn_t_t[t_idx]
                    Nt_b = N_t_t[t_idx]
                    u_t_gp = _interp_from_nodes(u_nodes, conn_tb, Nt_b)
                    if tx_expr and str(tx_expr).strip():
                        txv = eval_expr_torch(tx_expr, x=xt, y=yt)
                    else:
                        txv = torch.full_like(xt, float(tx_const))
                    if ty_expr and str(ty_expr).strip():
                        tyv = eval_expr_torch(ty_expr, x=xt, y=yt)
                    else:
                        tyv = torch.full_like(yt, float(ty_const))
                    Wext_trac = torch.sum(Wt * (txv * u_t_gp[:, 0:1] + tyv * u_t_gp[:, 1:2]))

                Wext = Wext_body + Wext_trac

                if ("penalty" in dir_mode) and penalty_lambda > 0 and int(Nu) > 0 and torch.any(Wu > 0):
                    conn_ub = conn_u_t[u_idx]
                    Nu_b = N_u_t[u_idx]
                    u_u_gp = _interp_from_nodes(u_nodes, conn_ub, Nu_b)
                    if ubarx_expr and str(ubarx_expr).strip():
                        ubarx = eval_expr_torch(ubarx_expr, x=xu, y=yu)
                    else:
                        ubarx = torch.full_like(xu, float(ubarx_const))
                    if ubary_expr and str(ubary_expr).strip():
                        ubary = eval_expr_torch(ubary_expr, x=xu, y=yu)
                    else:
                        ubary = torch.full_like(yu, float(ubary_const))
                    ubar = torch.cat([ubarx, ubary], dim=1)
                    Wbc = penalty_lambda * torch.sum(Wu * torch.sum((u_u_gp - ubar) ** 2, dim=1, keepdim=True))

                Pi = Wint - Wext + Wbc
            else:
                nn_out = model(torch.cat([x, y], dim=1))  # (N,2)
                uvec = apply_dirichlet_vec(
                    nn_out, x, y, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=None, ubarx_const=ubarx_const, ubary_const=ubary_const, ubarz_const=0.0, z=None
                )

                psi = energy_linear_elasticity(uvec, x, y, lam=lam, mu=mu)
                Wint = torch.sum(w * psi)

                if bx_expr and bx_expr.strip():
                    bx = eval_expr_torch(bx_expr, x=x, y=y)
                else:
                    bx = torch.full_like(x, float(bx_const))
                if by_expr and by_expr.strip():
                    by = eval_expr_torch(by_expr, x=x, y=y)
                else:
                    by = torch.full_like(y, float(by_const))

                Wext_body = torch.sum(w * (bx * uvec[:, 0:1] + by * uvec[:, 1:2]))

                nn_t = model(torch.cat([xt, yt], dim=1))
                uvec_t = apply_dirichlet_vec(
                    nn_t, xt, yt, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=None, z=None
                )

                if tx_expr and tx_expr.strip():
                    txv = eval_expr_torch(tx_expr, x=xt, y=yt)
                else:
                    txv = torch.full_like(xt, float(tx_const))
                if ty_expr and ty_expr.strip():
                    tyv = eval_expr_torch(ty_expr, x=xt, y=yt)
                else:
                    tyv = torch.full_like(yt, float(ty_const))

                Wext_trac = torch.sum(Wt * (txv * uvec_t[:, 0:1] + tyv * uvec_t[:, 1:2]))
                Wext = Wext_body + Wext_trac

                if ("penalty" in dir_mode) and penalty_lambda > 0:
                    nn_u = model(torch.cat([xu, yu], dim=1))
                    uvec_u = apply_dirichlet_vec(
                        nn_u, xu, yu, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=None, z=None
                    )
                    if ubarx_expr and str(ubarx_expr).strip():
                        ubarx = eval_expr_torch(ubarx_expr, x=xu, y=yu)
                    else:
                        ubarx = torch.full_like(xu, float(ubarx_const))
                    if ubary_expr and str(ubary_expr).strip():
                        ubary = eval_expr_torch(ubary_expr, x=xu, y=yu)
                    else:
                        ubary = torch.full_like(yu, float(ubary_const))
                    ubar = torch.cat([ubarx, ubary], dim=1)
                    Wbc = penalty_lambda * torch.sum(Wu * torch.sum((uvec_u - ubar) ** 2, dim=1, keepdim=True))

                Pi = Wint - Wext + Wbc

        elif problem_type == "Neo-Hookean Hyperelasticity (2D)":
            if use_shape:
                nn_nodes = model(Xn_t)  # (Nnodes,2)
                xn = Xn_t[:, 0:1]
                yn = Xn_t[:, 1:2]
                u_nodes = apply_dirichlet_vec(nn_nodes, xn, yn, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=None, z=None)

                conn_b = conn_dom_t[dom_idx]
                N_b = N_dom_t[dom_idx]
                dNdx_b = dNdx_dom_t[dom_idx]
                dNdy_b = dNdy_dom_t[dom_idx]

                u_gp = _interp_from_nodes(u_nodes, conn_b, N_b)
                du_dx = _interp_grad_from_nodes(u_nodes, conn_b, dNdx_b)
                du_dy = _interp_grad_from_nodes(u_nodes, conn_b, dNdy_b)

                u_x = du_dx[:, 0:1]
                v_x = du_dx[:, 1:2]
                u_y = du_dy[:, 0:1]
                v_y = du_dy[:, 1:2]

                F11 = 1.0 + u_x
                F12 = u_y
                F21 = v_x
                F22 = 1.0 + v_y

                J = F11 * F22 - F12 * F21
                J_safe = torch.clamp(J, min=1e-8)

                C11 = F11**2 + F21**2
                C22 = F12**2 + F22**2
                I1 = C11 + C22

                lnJ = torch.log(J_safe)
                psi = 0.5 * float(mu) * (I1 - 2.0) - float(mu) * lnJ + 0.5 * float(lam) * (lnJ**2)
                Wint = torch.sum(w * psi)
            else:
                nn_out = model(torch.cat([x, y], dim=1))  # (N,2)
                uvec = apply_dirichlet_vec(
                    nn_out, x, y, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=None, ubarx_const=ubarx_const, ubary_const=ubary_const, ubarz_const=0.0, z=None
                )

                psi = energy_neo_hookean_2d(uvec, x, y, lam=lam, mu=mu)
                Wint = torch.sum(w * psi)

            if bx_expr and bx_expr.strip():
                bx = eval_expr_torch(bx_expr, x=x, y=y)
            else:
                bx = torch.full_like(x, float(bx_const))
            if by_expr and by_expr.strip():
                by = eval_expr_torch(by_expr, x=x, y=y)
            else:
                by = torch.full_like(y, float(by_const))

            if use_shape:
                Wext_body = torch.sum(w * (bx * u_gp[:, 0:1] + by * u_gp[:, 1:2]))
            else:
                Wext_body = torch.sum(w * (bx * uvec[:, 0:1] + by * uvec[:, 1:2]))

            if use_shape and int(Nt) > 0 and torch.any(Wt > 0):
                conn_tb = conn_t_t[t_idx]
                Nt_b = N_t_t[t_idx]
                uvec_t = _interp_from_nodes(u_nodes, conn_tb, Nt_b)
            else:
                nn_t = model(torch.cat([xt, yt], dim=1))
                uvec_t = apply_dirichlet_vec(nn_t, xt, yt, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=None, ubarx_const=ubarx_const, ubary_const=ubary_const, ubarz_const=0.0, z=None)

            if tx_expr and tx_expr.strip():
                txv = eval_expr_torch(tx_expr, x=xt, y=yt)
            else:
                txv = torch.full_like(xt, float(tx_const))
            if ty_expr and ty_expr.strip():
                tyv = eval_expr_torch(ty_expr, x=xt, y=yt)
            else:
                tyv = torch.full_like(yt, float(ty_const))

            if int(Nt) > 0 and torch.any(Wt > 0):
                Wext_trac = torch.sum(Wt * (txv * uvec_t[:, 0:1] + tyv * uvec_t[:, 1:2]))
            Wext = Wext_body + Wext_trac

            if ("penalty" in dir_mode) and penalty_lambda > 0:
                if use_shape and int(Nu) > 0 and torch.any(Wu > 0):
                    conn_ub = conn_u_t[u_idx]
                    Nu_b = N_u_t[u_idx]
                    uvec_u = _interp_from_nodes(u_nodes, conn_ub, Nu_b)
                else:
                    nn_u = model(torch.cat([xu, yu], dim=1))
                    uvec_u = apply_dirichlet_vec(nn_u, xu, yu, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=None, ubarx_const=ubarx_const, ubary_const=ubary_const, ubarz_const=0.0, z=None)
                ubarx = eval_expr_torch(ubarx_expr or "0", x=xu, y=yu)
                ubary = eval_expr_torch(ubary_expr or "0", x=xu, y=yu)
                ubar = torch.cat([ubarx, ubary], dim=1)
                Wbc = penalty_lambda * torch.sum(Wu * torch.sum((uvec_u - ubar) ** 2, dim=1, keepdim=True))

            Pi = Wint - Wext + Wbc

        elif problem_type == "Linear Elasticity (3D)":
            if dim != 3:
                raise ValueError("Linear Elasticity (3D) requires a 3D mesh and 3D quadrature.")

            nn_out = model(torch.cat([x, y, z], dim=1))  # (N,3)
            uvec = apply_dirichlet_vec(nn_out, x, y, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=ubarz_expr, ubarx_const=ubarx_const, ubary_const=ubary_const, ubarz_const=ubarz_const, z=z)

            psi = energy_linear_elasticity_3d(uvec, x, y, z, lam=float(lam), mu=float(mu))
            Wint = torch.sum(w * psi)

            # body force b in Omega
            if bx_expr and bx_expr.strip():
                bx = eval_expr_torch(bx_expr, x=x, y=y, z=z)
            else:
                bx = torch.full_like(x, float(bx_const))
            if by_expr and by_expr.strip():
                by = eval_expr_torch(by_expr, x=x, y=y, z=z)
            else:
                by = torch.full_like(y, float(by_const))
            if bz_expr and bz_expr.strip():
                bz = eval_expr_torch(bz_expr, x=x, y=y, z=z)
            else:
                bz = torch.full_like(z, float(bz_const))

            Wext_body = torch.sum(w * (bx * uvec[:, 0:1] + by * uvec[:, 1:2] + bz * uvec[:, 2:3]))

            # traction t on Gamma_t
            nn_t = model(torch.cat([xt, yt, zt], dim=1))
            uvec_t = apply_dirichlet_vec(nn_t, xt, yt, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=ubarz_expr, ubarx_const=ubarx_const, ubary_const=ubary_const, ubarz_const=ubarz_const, z=zt)

            if tx_expr and tx_expr.strip():
                txv = eval_expr_torch(tx_expr, x=xt, y=yt, z=zt)
            else:
                txv = torch.full_like(xt, float(tx_const))
            if ty_expr and ty_expr.strip():
                tyv = eval_expr_torch(ty_expr, x=xt, y=yt, z=zt)
            else:
                tyv = torch.full_like(yt, float(ty_const))
            if tz_expr and tz_expr.strip():
                tzv = eval_expr_torch(tz_expr, x=xt, y=yt, z=zt)
            else:
                tzv = torch.full_like(zt, float(tz_const))

            Wext_trac = torch.sum(Wt * (txv * uvec_t[:, 0:1] + tyv * uvec_t[:, 1:2] + tzv * uvec_t[:, 2:3]))
            Wext = Wext_body + Wext_trac

            # penalty on Gamma_u
            if ("penalty" in dir_mode) and penalty_lambda > 0:
                nn_u = model(torch.cat([xu, yu, zu], dim=1))
                uvec_u = apply_dirichlet_vec(nn_u, xu, yu, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=ubarz_expr, ubarx_const=ubarx_const, ubary_const=ubary_const, ubarz_const=ubarz_const, z=zu)
                if ubarx_expr and str(ubarx_expr).strip():
                    ubarx = eval_expr_torch(ubarx_expr, x=xu, y=yu, z=zu)
                else:
                    ubarx = torch.full_like(xu, float(ubarx_const))
                if ubary_expr and str(ubary_expr).strip():
                    ubary = eval_expr_torch(ubary_expr, x=xu, y=yu, z=zu)
                else:
                    ubary = torch.full_like(yu, float(ubary_const))
                if ubarz_expr and str(ubarz_expr).strip():
                    ubarz = eval_expr_torch(ubarz_expr, x=xu, y=yu, z=zu)
                else:
                    ubarz = torch.full_like(zu, float(ubarz_const))
                ubar = torch.cat([ubarx, ubary, ubarz], dim=1)
                Wbc = penalty_lambda * torch.sum(Wu * torch.sum((uvec_u - ubar) ** 2, dim=1, keepdim=True))

            Pi = Wint - Wext + Wbc

        # Scale 2D plane-strain linear elasticity energies by thickness (out-of-plane).
        if (problem_type == "Linear Elasticity (2D)") and (str(plane_mode) == "plane_strain"):
            t = float(thickness)
            Wint = t * Wint
            Wext_body = t * Wext_body
            Wext_trac = t * Wext_trac
            Wext = t * Wext
            Wbc = t * Wbc
            Pi = t * Pi

        Pi.backward()
        opt.step()

        pi_val = float(Pi.item())
        hist["Pi"].append(pi_val)
        hist["Wint"].append(float(Wint.item()))
        hist["Wext"].append(float(Wext.item()))
        hist["Wext_body"].append(float(Wext_body.item()))
        hist["Wext_trac"].append(float(Wext_trac.item()))
        hist["Wbc"].append(float(Wbc.item()))

        # Track best model (for rollback when training diverges)
        if math.isfinite(pi_val):
            improved = False
            if best_state_cpu is None:
                improved = True
            else:
                # Treat "tiny" improvements as noise to avoid resetting too often
                improve_thr = (abs(best_pi) + float(spike_eps)) * 1e-6
                improved = pi_val < (best_pi - improve_thr)
            if improved:
                best_pi = pi_val
                best_epoch = int(ep)
                best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Spike / divergence guard: if already stable then loss suddenly jumps, stop and rollback
        rel_change: float | None = None
        if prev_pi is not None:
            rel_change = abs(pi_val - prev_pi) / (abs(prev_pi) + float(spike_eps))

        # Extra stability detector: if rel(ΔΠ) is small for many epochs, treat as "stable"
        # (important for hard-BC only cases where best Π may keep improving slightly forever).
        if rel_change is not None and math.isfinite(rel_change):
            if rel_change < float(stable_rel_rtol):
                stable_rel_count += 1
            else:
                stable_rel_count = 0

        if abort_on_spike:
            stable = int(stable_rel_count) >= int(stable_rel_patience)
            should_abort = False
            reason = ""
            if not math.isfinite(pi_val):
                should_abort = True
                reason = "Π became NaN/Inf"
            elif rel_change is not None and ep > int(spike_warmup) and stable and rel_change > float(spike_rtol):
                should_abort = True
                reason = f"rel(ΔΠ)={rel_change:.2e} > {float(spike_rtol):.2e} after stability"

            if should_abort:
                # Rollback to best weights (if any) to keep final result good
                if best_state_cpu is not None:
                    model.load_state_dict(best_state_cpu)
                    model.to(device)
                    hist["restored_best"] = True
                    hist["best_epoch"] = int(best_epoch)
                    hist["best_Pi"] = float(best_pi)
                else:
                    hist["restored_best"] = False
                dt = time.time() - t0
                progress.progress(int(100 * ep / epochs))
                status.markdown(
                    f"**Aborted** at epoch {ep}/{epochs} | {reason} | "
                    f"rollback best@{best_epoch} (Π={best_pi:.3e}) | Time {dt:.1f}s"
                )
                hist["stopped_early"] = True
                hist["stop_epoch"] = int(ep)
                hist["stop_reason"] = f"abort_on_spike: {reason}"
                break

        prev_pi = pi_val

        if ep == 1 or ep == epochs or (ep % log_every) == 0:
            dt = time.time() - t0
            progress.progress(int(100 * ep / epochs))
            status.markdown(
                f"**Epoch** {ep}/{epochs} | **Π** {Pi.item():.3e} | "
                f"Wint {Wint.item():.3e} | Wext {Wext.item():.3e} "
                f"(body {Wext_body.item():.2e}, trac {Wext_trac.item():.2e}) | "
                f"Wbc {Wbc.item():.3e} | Time {dt:.1f}s"
            )

    progress.empty()
    status.empty()
    hist["time_sec"] = time.time() - t0
    return hist


def compute_stress_on_points(
    model,
    device,
    pts_xy: np.ndarray,
    problem_type: str,
    dir_mode: str,
    dist_obj,
    ubar_expr: str,
    ubarx_expr: str,
    ubary_expr: str,
    ubarz_expr: str = "0",
    lam: float = 0.0,
    mu: float = 0.0,
    plane_mode: str = "plane_strain",
):
    model.eval()

    X = to_torch(pts_xy.astype(np.float32), device=device, requires_grad=True)
    dim = int(X.shape[1])
    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3] if dim == 3 else None

    nn_out = model(torch.cat([x, y], dim=1) if dim == 2 else torch.cat([x, y, z], dim=1))

    if problem_type == "Poisson (scalar)":
        sigma = torch.zeros((X.shape[0], 3), device=device)
        mises = torch.zeros((X.shape[0], 1), device=device)
        return sigma.detach().cpu().numpy(), mises.detach().cpu().numpy()

    if problem_type == "Linear Elasticity (3D)":
        if dim != 3:
            raise ValueError("Linear Elasticity (3D) stress requires 3D points.")

        uvec = apply_dirichlet_vec(nn_out, x, y, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=ubarz_expr, z=z)
        u = uvec[:, 0:1]
        v = uvec[:, 1:2]
        w = uvec[:, 2:3]

        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_z = grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        w_x = grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_y = grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_z = grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]

        exx = u_x
        eyy = v_y
        ezz = w_z
        exy = 0.5 * (u_y + v_x)
        eyz = 0.5 * (v_z + w_y)
        ezx = 0.5 * (w_x + u_z)

        tr = exx + eyy + ezz

        sxx = lam * tr + 2.0 * mu * exx
        syy = lam * tr + 2.0 * mu * eyy
        szz = lam * tr + 2.0 * mu * ezz
        sxy = 2.0 * mu * exy
        syz = 2.0 * mu * eyz
        szx = 2.0 * mu * ezx

        sigma = torch.cat([sxx, syy, szz, sxy, syz, szx], dim=1)  # (N,6)

        # von Mises in 3D: sqrt(3/2 * s_dev:s_dev)
        mean_stress = (sxx + syy + szz) / 3.0
        sxx_d = sxx - mean_stress
        syy_d = syy - mean_stress
        szz_d = szz - mean_stress
        j2 = 0.5 * (sxx_d**2 + syy_d**2 + szz_d**2) + (sxy**2 + syz**2 + szx**2)
        mises = torch.sqrt(3.0 * j2 + 1e-12)
        return sigma.detach().cpu().numpy(), mises.detach().cpu().numpy()

    # 2D mechanics
    uvec = apply_dirichlet_vec(nn_out, x, y, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=None, z=None)
    u = uvec[:, 0:1]
    v = uvec[:, 1:2]

    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    if problem_type == "Linear Elasticity (2D)":
        exx = u_x
        eyy = v_y
        exy = 0.5 * (u_y + v_x)

        tr = exx + eyy

        sxx = lam * tr + 2.0 * mu * exx
        syy = lam * tr + 2.0 * mu * eyy
        sxy = 2.0 * mu * exy

        sigma = torch.cat([sxx, syy, sxy], dim=1)
        mises = torch.sqrt((sxx - syy) ** 2 + 3.0 * (sxy**2) + 1e-12)
        return sigma.detach().cpu().numpy(), mises.detach().cpu().numpy()

    elif problem_type == "Neo-Hookean Hyperelasticity (2D)":
        F11 = 1.0 + u_x
        F12 = u_y
        F21 = v_x
        F22 = 1.0 + v_y

        J = F11 * F22 - F12 * F21
        J_safe = torch.clamp(J, min=1e-8)
        lnJ = torch.log(J_safe)

        invF_T_11 = F22 / J_safe
        invF_T_12 = -F21 / J_safe
        invF_T_21 = -F12 / J_safe
        invF_T_22 = F11 / J_safe

        P11 = mu * (F11 - invF_T_11) + lam * lnJ * invF_T_11
        P12 = mu * (F12 - invF_T_12) + lam * lnJ * invF_T_12
        P21 = mu * (F21 - invF_T_21) + lam * lnJ * invF_T_21
        P22 = mu * (F22 - invF_T_22) + lam * lnJ * invF_T_22

        PFt11 = P11 * F11 + P12 * F12
        PFt12 = P11 * F21 + P12 * F22
        PFt21 = P21 * F11 + P22 * F12
        PFt22 = P21 * F21 + P22 * F22

        sxx = PFt11 / J_safe
        sxy = 0.5 * (PFt12 + PFt21) / J_safe
        syy = PFt22 / J_safe

        sigma = torch.cat([sxx, syy, sxy], dim=1)
        mises = torch.sqrt((sxx - syy) ** 2 + 3.0 * (sxy**2) + 1e-12)
        return sigma.detach().cpu().numpy(), mises.detach().cpu().numpy()

    else:
        sigma = torch.zeros((X.shape[0], 3), device=device)
        mises = torch.zeros((X.shape[0], 1), device=device)
        return sigma.detach().cpu().numpy(), mises.detach().cpu().numpy()


def compute_stress_on_sf_gauss_2d(
    model,
    device,
    pts_nodes: np.ndarray,
    sf_dom_2d: dict,
    problem_type: str,
    dir_mode: str,
    dist_obj,
    ubarx_expr: str,
    ubary_expr: str,
    lam: float,
    mu: float,
):
    """
    2D stress on domain Gauss points using shape-function gradients (no autograd).
    Returns (sigma, mises) at sf_dom_2d['X'] points.
    sigma: (Ng,3) [sxx, syy, sxy]
    mises: (Ng,1)
    """
    assert problem_type in ["Linear Elasticity (2D)", "Neo-Hookean Hyperelasticity (2D)"]
    sf = sf_dom_2d or {}
    Xg = np.asarray(sf.get("X", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    if Xg.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    conn = np.asarray(sf.get("conn"), dtype=np.int64)
    dNdx = np.asarray(sf.get("dNdx"), dtype=np.float32)
    dNdy = np.asarray(sf.get("dNdy"), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        Xn = torch.tensor(np.asarray(pts_nodes, dtype=np.float32)[:, :2], device=device, dtype=torch.float32)
        xn = Xn[:, 0:1]
        yn = Xn[:, 1:2]
        nn_nodes = model(Xn)  # (Nnodes,2)
        u_nodes = apply_dirichlet_vec(
            nn_nodes, xn, yn, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=None, z=None
        )  # (Nnodes,2)

        conn_t = torch.tensor(conn, device=device, dtype=torch.long)
        dNdx_t = torch.tensor(dNdx, device=device, dtype=torch.float32)
        dNdy_t = torch.tensor(dNdy, device=device, dtype=torch.float32)

        ue = u_nodes[conn_t]  # (Ng,4,2)
        du_dx = torch.sum(ue * dNdx_t.unsqueeze(-1), dim=1)  # (Ng,2)
        du_dy = torch.sum(ue * dNdy_t.unsqueeze(-1), dim=1)  # (Ng,2)

        u_x = du_dx[:, 0:1]
        v_x = du_dx[:, 1:2]
        u_y = du_dy[:, 0:1]
        v_y = du_dy[:, 1:2]

        if problem_type == "Linear Elasticity (2D)":
            exx = u_x
            eyy = v_y
            exy = 0.5 * (u_y + v_x)
            tr = exx + eyy
            sxx = float(lam) * tr + 2.0 * float(mu) * exx
            syy = float(lam) * tr + 2.0 * float(mu) * eyy
            sxy = 2.0 * float(mu) * exy
            sigma = torch.cat([sxx, syy, sxy], dim=1)
            mises = torch.sqrt((sxx - syy) ** 2 + 3.0 * (sxy**2) + 1e-12)
            return sigma.detach().cpu().numpy().astype(np.float32), mises.detach().cpu().numpy().astype(np.float32)

        # Neo-Hookean Hyperelasticity (2D): compute Cauchy stress via P*F^T/J (same as autograd path)
        F11 = 1.0 + u_x
        F12 = u_y
        F21 = v_x
        F22 = 1.0 + v_y
        J = F11 * F22 - F12 * F21
        J_safe = torch.clamp(J, min=1e-8)
        lnJ = torch.log(J_safe)

        invF_T_11 = F22 / J_safe
        invF_T_12 = -F21 / J_safe
        invF_T_21 = -F12 / J_safe
        invF_T_22 = F11 / J_safe

        P11 = float(mu) * (F11 - invF_T_11) + float(lam) * lnJ * invF_T_11
        P12 = float(mu) * (F12 - invF_T_12) + float(lam) * lnJ * invF_T_12
        P21 = float(mu) * (F21 - invF_T_21) + float(lam) * lnJ * invF_T_21
        P22 = float(mu) * (F22 - invF_T_22) + float(lam) * lnJ * invF_T_22

        PFt11 = P11 * F11 + P12 * F12
        PFt12 = P11 * F21 + P12 * F22
        PFt21 = P21 * F11 + P22 * F12
        PFt22 = P21 * F21 + P22 * F22

        sxx = PFt11 / J_safe
        sxy = 0.5 * (PFt12 + PFt21) / J_safe
        syy = PFt22 / J_safe

        sigma = torch.cat([sxx, syy, sxy], dim=1)
        mises = torch.sqrt((sxx - syy) ** 2 + 3.0 * (sxy**2) + 1e-12)
        return sigma.detach().cpu().numpy().astype(np.float32), mises.detach().cpu().numpy().astype(np.float32)


def project_point_field_to_nodes_idw(
    nodes_xy: np.ndarray,
    sample_xy: np.ndarray,
    sample_val: np.ndarray,
    device,
    k: int = 8,
    eps: float = 1e-12,
    chunk: int = 4096,
) -> np.ndarray:
    """
    Project sample values (e.g., stress at Xdom) onto mesh nodes via kNN inverse-distance weighting.
    Uses torch.cdist in chunks to keep memory stable and leverage GPU if available.

    nodes_xy: (N,2)
    sample_xy: (M,2)
    sample_val: (M,C)
    return: (N,C)
    """
    if sample_xy is None or sample_xy.size == 0:
        raise ValueError("sample_xy is empty; cannot project.")
    if sample_val is None or sample_val.size == 0:
        raise ValueError("sample_val is empty; cannot project.")

    nodes_t = to_torch(nodes_xy.astype(np.float32), device=device, requires_grad=False)
    sample_xy_t = to_torch(sample_xy.astype(np.float32), device=device, requires_grad=False)
    sample_val_t = to_torch(sample_val.astype(np.float32), device=device, requires_grad=False)

    M = int(sample_xy_t.shape[0])
    kk = int(min(max(k, 1), M))
    out_chunks = []
    for i in range(0, int(nodes_t.shape[0]), int(chunk)):
        ni = nodes_t[i : i + int(chunk)]
        dist = torch.cdist(ni, sample_xy_t)  # (chunk,M)
        d_k, idx = torch.topk(dist, k=kk, dim=1, largest=False)  # (chunk,kk)
        w = 1.0 / (d_k + float(eps))
        w = w / (w.sum(dim=1, keepdim=True) + float(eps))
        v_k = sample_val_t[idx]  # (chunk,kk,C)
        out = (w.unsqueeze(-1) * v_k).sum(dim=1)  # (chunk,C)
        out_chunks.append(out)

    out_t = torch.cat(out_chunks, dim=0)
    return out_t.detach().cpu().numpy()

# ============================================================
# FEM reference (T3) for Poisson / Linear Elasticity (small strain)
# ============================================================

def eval_expr_numpy_via_torch(expr: str, x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None, device="cpu"):
    """Use your torch-safe eval to evaluate expr on numpy arrays (no grad)."""
    if expr is None or (isinstance(expr, str) and expr.strip() == ""):
        expr = "0"
    xt = to_torch(x.reshape(-1, 1).astype(np.float32), device=device, requires_grad=False)
    yt = to_torch(y.reshape(-1, 1).astype(np.float32), device=device, requires_grad=False)
    zt = None
    if z is not None:
        zt = to_torch(z.reshape(-1, 1).astype(np.float32), device=device, requires_grad=False)
    with torch.no_grad():
        out = eval_expr_torch(expr, x=xt, y=yt, z=zt)
    return out.detach().cpu().numpy().reshape(-1)


def _dirichlet_nodes_from_segs(seg_u: np.ndarray):
    if seg_u is None or seg_u.size == 0:
        return np.array([], dtype=int)
    return np.unique(seg_u[:, :2].reshape(-1))


def _dirichlet_nodes_from_tris(tris_u: np.ndarray):
    if tris_u is None or np.asarray(tris_u).size == 0:
        return np.array([], dtype=int)
    t = np.asarray(tris_u)[:, :3].reshape(-1)
    return np.unique(t.astype(int))


def fem_poisson_T3(pts, tris, seg_u, ubar_expr: str, f_expr: str | None, device_for_eval="cpu"):
    """
    Solve: -Δu = f in Ω, u = ubar on Gamma_u
    (No Neumann term here; if you need, we can add Gamma_t flux similarly.)
    """
    n = pts.shape[0]
    rows, cols, data = [], [], []
    b = np.zeros(n, dtype=np.float64)

    X = pts[:, 0]; Y = pts[:, 1]

    # element assembly (T3)
    for (i0, i1, i2) in tris:
        x0, y0 = pts[i0]; x1, y1 = pts[i1]; x2, y2 = pts[i2]
        area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
        if area < 1e-14:
            continue

        # grad of shape functions: N = a + b x + c y
        b0 = (y1 - y2) / (2*area); c0 = (x2 - x1) / (2*area)
        b1 = (y2 - y0) / (2*area); c1 = (x0 - x2) / (2*area)
        b2 = (y0 - y1) / (2*area); c2 = (x1 - x0) / (2*area)

        B = np.array([[b0, b1, b2],
                      [c0, c1, c2]], dtype=np.float64)  # (2,3)

        Ke = area * (B.T @ B)  # ∫ gradNi·gradNj dΩ

        # load vector via 1-point centroid (good enough; you can switch to 3-pt if desired)
        xc = (x0 + x1 + x2) / 3.0
        yc = (y0 + y1 + y2) / 3.0
        if f_expr and str(f_expr).strip():
            fc = float(eval_expr_numpy_via_torch(f_expr, np.array([xc]), np.array([yc]), device=device_for_eval)[0])
        else:
            fc = 0.0
        fe = fc * area * (1.0/3.0) * np.ones(3, dtype=np.float64)

        idx = [i0, i1, i2]
        for a in range(3):
            b[idx[a]] += fe[a]
            for c in range(3):
                rows.append(idx[a]); cols.append(idx[c]); data.append(Ke[a, c])

    K = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # Dirichlet strong impose
    dbc_nodes = _dirichlet_nodes_from_segs(seg_u)
    u = np.zeros(n, dtype=np.float64)
    if dbc_nodes.size > 0:
        ubar_vals = eval_expr_numpy_via_torch(ubar_expr or "0", X[dbc_nodes], Y[dbc_nodes], device=device_for_eval)
        u[dbc_nodes] = ubar_vals

        # modify RHS: b = b - K[:,dbc]*u_d
        b = b - K[:, dbc_nodes] @ u[dbc_nodes]

        # enforce rows/cols
        mask_free = np.ones(n, dtype=bool)
        mask_free[dbc_nodes] = False
        Kff = K[mask_free][:, mask_free]
        bf = b[mask_free]
        uf = spla.spsolve(Kff, bf)
        u[mask_free] = uf
    else:
        u[:] = spla.spsolve(K, b)

    return u.reshape(-1, 1)


def fem_Screened_T3(pts, tris, seg_u, seg_t, ubar_expr: str, f_expr: str | None, k_squared: float, g_expr: str | None, g_const: float, device_for_eval="cpu"):
    """
    Solve: -Δu + k²u = f in Ω, u = ubar on Gamma_u, ∂u/∂n = g on Gamma_t
    Weak form: ∫ ∇u·∇v dΩ + k²∫ u v dΩ = ∫ f v dΩ + ∫ g v dΓ
    """
    n = pts.shape[0]
    rows, cols, data = [], [], []
    b = np.zeros(n, dtype=np.float64)

    X = pts[:, 0]; Y = pts[:, 1]

    # element assembly (T3)
    for (i0, i1, i2) in tris:
        x0, y0 = pts[i0]; x1, y1 = pts[i1]; x2, y2 = pts[i2]
        area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
        if area < 1e-14:
            continue

        # grad of shape functions: N = a + b x + c y
        b0 = (y1 - y2) / (2*area); c0 = (x2 - x1) / (2*area)
        b1 = (y2 - y0) / (2*area); c1 = (x0 - x2) / (2*area)
        b2 = (y0 - y1) / (2*area); c2 = (x1 - x0) / (2*area)

        B = np.array([[b0, b1, b2],
                      [c0, c1, c2]], dtype=np.float64)  # (2,3)

        # Stiffness matrix: ∫ gradNi·gradNj dΩ
        Ke = area * (B.T @ B)

        # Mass matrix: ∫ Ni·Nj dΩ (for T3: M_e = (area/12) * [[2,1,1],[1,2,1],[1,1,2]])
        Me = (area / 12.0) * np.array([[2.0, 1.0, 1.0],
                                        [1.0, 2.0, 1.0],
                                        [1.0, 1.0, 2.0]], dtype=np.float64)

        # Total element matrix: K + k²*M
        Ke = Ke + float(k_squared) * Me

        # load vector via 1-point centroid
        xc = (x0 + x1 + x2) / 3.0
        yc = (y0 + y1 + y2) / 3.0
        if f_expr and str(f_expr).strip():
            fc = float(eval_expr_numpy_via_torch(f_expr, np.array([xc]), np.array([yc]), device=device_for_eval)[0])
        else:
            fc = 0.0
        fe = fc * area * (1.0/3.0) * np.ones(3, dtype=np.float64)

        idx = [i0, i1, i2]
        for a in range(3):
            b[idx[a]] += fe[a]
            for c in range(3):
                rows.append(idx[a]); cols.append(idx[c]); data.append(Ke[a, c])

    # Neumann boundary term on Gamma_t: ∫ g v dΓ
    if seg_t is not None and seg_t.size > 0:
        for (j0, j1) in seg_t:
            x0, y0 = pts[j0]; x1, y1 = pts[j1]
            length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            if length < 1e-14:
                continue
            xc = (x0 + x1) / 2.0
            yc = (y0 + y1) / 2.0
            if g_expr and str(g_expr).strip():
                gv = float(eval_expr_numpy_via_torch(g_expr, np.array([xc]), np.array([yc]), device=device_for_eval)[0])
            else:
                gv = float(g_const)
            # P1 on segment: consistent load -> gv*length/2 on each node
            add = gv * length / 2.0
            b[int(j0)] += add
            b[int(j1)] += add

    K = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # Dirichlet strong impose
    dbc_nodes = _dirichlet_nodes_from_segs(seg_u)
    u = np.zeros(n, dtype=np.float64)
    if dbc_nodes.size > 0:
        ubar_vals = eval_expr_numpy_via_torch(ubar_expr or "0", X[dbc_nodes], Y[dbc_nodes], device=device_for_eval)
        u[dbc_nodes] = ubar_vals

        # modify RHS: b = b - K[:,dbc]*u_d
        b = b - K[:, dbc_nodes] @ u[dbc_nodes]

        # enforce rows/cols
        mask_free = np.ones(n, dtype=bool)
        mask_free[dbc_nodes] = False
        Kff = K[mask_free][:, mask_free]
        bf = b[mask_free]
        uf = spla.spsolve(Kff, bf)
        u[mask_free] = uf
    else:
        u[:] = spla.spsolve(K, b)

    return u.reshape(-1, 1)


def fem_poisson_TET4_3d(
    pts: np.ndarray,
    tets: np.ndarray,
    tris_u: np.ndarray,
    tris_t: np.ndarray | None,
    *,
    ubar_expr: str,
    f_expr: str | None,
    g_expr: str | None,
    g_const: float,
    device_for_eval="cpu",
):
    """
    3D Poisson FEM (tetra P1):
      -Δu = f in Ω (Omega volume)
      u = ubar on Gamma_u (Physical Surface)
      ∂u/∂n = g on Gamma_t (Physical Surface)  [weakly: adds ∫ g v dΓ to RHS]
    """
    P = np.asarray(pts, dtype=np.float64)
    T = np.asarray(tets, dtype=int)[:, :4]
    n = int(P.shape[0])

    rows, cols, data = [], [], []
    b = np.zeros(n, dtype=np.float64)

    # ---- element assembly ----
    for (i0, i1, i2, i3) in T:
        idx = np.array([i0, i1, i2, i3], dtype=int)
        Xe = P[idx, :]  # (4,3)
        # volume
        v6 = float(np.dot(Xe[1] - Xe[0], np.cross(Xe[2] - Xe[0], Xe[3] - Xe[0])))
        vol = abs(v6) / 6.0
        if not np.isfinite(vol) or vol < 1e-18:
            continue

        # gradients via inverse of [1 x y z]
        M = np.ones((4, 4), dtype=np.float64)
        M[:, 1:] = Xe
        try:
            invM = np.linalg.inv(M)
        except Exception:
            continue
        grads = invM[1:4, :].T  # (4,3): grad N_i = [b_i,c_i,d_i]

        Ke = vol * (grads @ grads.T)  # (4,4)

        # body force load (centroid 1-point)
        xc = float(Xe[:, 0].mean())
        yc = float(Xe[:, 1].mean())
        zc = float(Xe[:, 2].mean())
        if f_expr and str(f_expr).strip():
            fc = float(
                eval_expr_numpy_via_torch(
                    f_expr, np.array([xc]), np.array([yc]), np.array([zc]), device=device_for_eval
                )[0]
            )
        else:
            fc = 0.0
        fe = fc * vol * 0.25 * np.ones(4, dtype=np.float64)

        for a in range(4):
            ia = int(idx[a])
            b[ia] += fe[a]
            for c in range(4):
                rows.append(ia)
                cols.append(int(idx[c]))
                data.append(float(Ke[a, c]))

    K = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # ---- Neumann/flux on Gamma_t: add ∫ g v dΓ to RHS ----
    if tris_t is not None and np.asarray(tris_t).size > 0:
        TT = np.asarray(tris_t)[:, :3].astype(int)
        for (j0, j1, j2) in TT:
            Xf = P[[j0, j1, j2], :]  # (3,3)
            area = 0.5 * float(np.linalg.norm(np.cross(Xf[1] - Xf[0], Xf[2] - Xf[0])))
            if not np.isfinite(area) or area < 1e-18:
                continue
            xc = float(Xf[:, 0].mean())
            yc = float(Xf[:, 1].mean())
            zc = float(Xf[:, 2].mean())
            if g_expr and str(g_expr).strip():
                gv = float(
                    eval_expr_numpy_via_torch(
                        g_expr, np.array([xc]), np.array([yc]), np.array([zc]), device=device_for_eval
                    )[0]
                )
            else:
                gv = float(g_const)
            # P1 on triangle: consistent load with 1-pt -> gv*area/3 on each node
            add = gv * area / 3.0
            b[int(j0)] += add
            b[int(j1)] += add
            b[int(j2)] += add

    # ---- Dirichlet strong impose on Gamma_u nodes ----
    dbc_nodes = _dirichlet_nodes_from_tris(tris_u)
    u = np.zeros(n, dtype=np.float64)
    if dbc_nodes.size > 0:
        ubar_vals = eval_expr_numpy_via_torch(
            ubar_expr or "0",
            P[dbc_nodes, 0],
            P[dbc_nodes, 1],
            P[dbc_nodes, 2],
            device=device_for_eval,
        )
        u[dbc_nodes] = ubar_vals

        # RHS adjustment: b = b - K[:,dbc]*u_d
        b = b - K[:, dbc_nodes] @ u[dbc_nodes]

        mask_free = np.ones(n, dtype=bool)
        mask_free[dbc_nodes] = False
        Kff = K[mask_free][:, mask_free]
        bf = b[mask_free]
        uf = spla.spsolve(Kff, bf)
        u[mask_free] = uf
    else:
        u[:] = spla.spsolve(K, b)

    return u.reshape(-1, 1)


def fem_Screened_mixed_T3_Q4(
    pts,
    tris_plot,
    omega_triangles_n: int,
    quads,
    seg_u,
    seg_t,
    ubar_expr: str,
    f_expr: str | None,
    k_squared: float,
    g_expr: str | None,
    g_const: float,
    *,
    gauss_n_quad: int = 2,
    device_for_eval="cpu",
):
    """
    Mixed FEM for Screened:
    - Use T3 on the *original* triangles: tris_plot[:omega_triangles_n]
    - Use Q4 on quad elements
    Solve: -Δu + k²u = f in Ω, u = ubar on Gamma_u, ∂u/∂n = g on Gamma_t
    """
    n = pts.shape[0]
    rows, cols, data = [], [], []
    b = np.zeros(n, dtype=np.float64)
    Xn = pts[:, 0]
    Yn = pts[:, 1]

    # --- T3 contribution ---
    if omega_triangles_n > 0:
        tris = tris_plot[:omega_triangles_n]
        for (i0, i1, i2) in tris:
            x0, y0 = pts[i0]; x1, y1 = pts[i1]; x2, y2 = pts[i2]
            area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
            if area < 1e-14:
                continue
            b0 = (y1 - y2) / (2*area); c0 = (x2 - x1) / (2*area)
            b1 = (y2 - y0) / (2*area); c1 = (x0 - x2) / (2*area)
            b2 = (y0 - y1) / (2*area); c2 = (x1 - x0) / (2*area)
            B = np.array([[b0, b1, b2],
                          [c0, c1, c2]], dtype=np.float64)
            Ke = area * (B.T @ B)  # Stiffness matrix

            # Mass matrix for T3
            Me = (area / 12.0) * np.array([[2.0, 1.0, 1.0],
                                            [1.0, 2.0, 1.0],
                                            [1.0, 1.0, 2.0]], dtype=np.float64)
            Ke = Ke + float(k_squared) * Me  # K + k²*M

            xc = (x0 + x1 + x2) / 3.0
            yc = (y0 + y1 + y2) / 3.0
            if f_expr and str(f_expr).strip():
                fc = float(eval_expr_numpy_via_torch(f_expr, np.array([xc]), np.array([yc]), device=device_for_eval)[0])
            else:
                fc = 0.0
            fe = fc * area * (1.0/3.0) * np.ones(3, dtype=np.float64)
            idx = [int(i0), int(i1), int(i2)]
            for a in range(3):
                b[idx[a]] += fe[a]
                for c in range(3):
                    rows.append(idx[a]); cols.append(idx[c]); data.append(Ke[a, c])

    # --- Q4 contribution ---
    if quads is not None and quads.size > 0:
        rs, w2 = gauss_2d_tensor(int(gauss_n_quad))
        N, dNdr, dNds = _quad4_shape(rs)
        ng = rs.shape[0]
        for (i0, i1, i2, i3) in quads[:, :4]:
            idx = [int(i0), int(i1), int(i2), int(i3)]
            Xe = pts[idx, :]
            dxdr = dNdr @ Xe[:, 0:1]; dydr = dNdr @ Xe[:, 1:2]
            dxds = dNds @ Xe[:, 0:1]; dyds = dNds @ Xe[:, 1:2]
            detJ = (dxdr * dyds - dxds * dydr).reshape(-1, 1)
            if np.any(np.abs(detJ) < 1e-14):
                continue
            drdx = (dyds / detJ)
            drdy = (-dxds / detJ)
            dsdx = (-dydr / detJ)
            dsdy = (dxdr / detJ)
            dNdx = drdx * dNdr + dsdx * dNds
            dNdy = drdy * dNdr + dsdy * dNds

            Ke = np.zeros((4, 4), dtype=np.float64)
            Me = np.zeros((4, 4), dtype=np.float64)
            fe = np.zeros(4, dtype=np.float64)
            Pg = N @ Xe
            xg = Pg[:, 0]; yg = Pg[:, 1]
            if f_expr and str(f_expr).strip():
                fv = eval_expr_numpy_via_torch(f_expr, xg, yg, device=device_for_eval).astype(np.float64)
            else:
                fv = np.zeros_like(xg, dtype=np.float64)
            wJ = (w2[:, 0] * np.abs(detJ[:, 0])).astype(np.float64)
            for k in range(ng):
                g = np.stack([dNdx[k, :], dNdy[k, :]], axis=0)
                Ke += (g.T @ g) * wJ[k]  # Stiffness matrix contribution
                Me += (N[k, :].reshape(-1, 1) @ N[k, :].reshape(1, -1)) * wJ[k]  # Mass matrix contribution
                fe += N[k, :] * fv[k] * wJ[k]
            Ke = Ke + float(k_squared) * Me  # K + k²*M
            for a in range(4):
                b[idx[a]] += fe[a]
                for c in range(4):
                    rows.append(idx[a]); cols.append(idx[c]); data.append(Ke[a, c])

    # Neumann boundary term on Gamma_t
    if seg_t is not None and seg_t.size > 0:
        for (j0, j1) in seg_t:
            x0, y0 = pts[j0]; x1, y1 = pts[j1]
            length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            if length < 1e-14:
                continue
            xc = (x0 + x1) / 2.0
            yc = (y0 + y1) / 2.0
            if g_expr and str(g_expr).strip():
                gv = float(eval_expr_numpy_via_torch(g_expr, np.array([xc]), np.array([yc]), device=device_for_eval)[0])
            else:
                gv = float(g_const)
            add = gv * length / 2.0
            b[int(j0)] += add
            b[int(j1)] += add

    K = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    dbc_nodes = _dirichlet_nodes_from_segs(seg_u)
    u = np.zeros(n, dtype=np.float64)
    if dbc_nodes.size > 0:
        ubar_vals = eval_expr_numpy_via_torch(ubar_expr or "0", Xn[dbc_nodes], Yn[dbc_nodes], device=device_for_eval)
        u[dbc_nodes] = ubar_vals
        b = b - K[:, dbc_nodes] @ u[dbc_nodes]
        mask_free = np.ones(n, dtype=bool)
        mask_free[dbc_nodes] = False
        Kff = K[mask_free][:, mask_free]
        bf = b[mask_free]
        uf = spla.spsolve(Kff, bf)
        u[mask_free] = uf
    else:
        u[:] = spla.spsolve(K, b)

    return u.reshape(-1, 1)


def fem_linear_elasticity_T3(pts, tris, seg_u, seg_t,
                             lam: float, mu: float, plane_mode: str,
                             ubarx_expr: str, ubary_expr: str,
                             bx_expr: str | None, by_expr: str | None, bx_const: float, by_const: float,
                             tx_expr: str | None, ty_expr: str | None, tx_const: float, ty_const: float,
                             gauss_n_seg: int = 2,
                             device_for_eval="cpu"):
    """
    Small-strain linear elasticity FEM with T3 triangles.
    Unknown: [u_x, u_y] at nodes (2n dofs).
    Body force in Omega, traction on Gamma_t, strong Dirichlet on Gamma_u.
    """
    n = pts.shape[0]
    ndof = 2*n
    rows, cols, data = [], [], []
    f = np.zeros(ndof, dtype=np.float64)

    # constitutive D
    if plane_mode == "plane_strain":
        # D = lam*1⊗1 + 2mu*I_sym
        # in Voigt [xx, yy, xy]
        D = np.array([[lam+2*mu, lam,       0],
                      [lam,      lam+2*mu,  0],
                      [0,        0,         mu]], dtype=np.float64)
    else:
        # plane stress: use E,nu equivalent from lam,mu:
        # mu = E/(2(1+nu)), lam = E*nu/(1-nu^2)  (your code)
        # So E = 2mu(1+nu); solve nu from lam = E*nu/(1-nu^2)
        # robust: derive E,nu numerically
        # nu from mu,lam (plane stress): lam = 2mu*nu/(1-nu)  -> nu = lam/(lam+2mu)
        nu_eff = lam / (lam + 2*mu + 1e-15)
        E_eff = 2*mu*(1+nu_eff)
        coef = E_eff / (1 - nu_eff**2 + 1e-15)
        D = coef * np.array([[1,      nu_eff, 0],
                             [nu_eff, 1,      0],
                             [0,      0,      (1-nu_eff)/2]], dtype=np.float64)

    X = pts[:, 0]; Y = pts[:, 1]

    # element assembly
    for (i0, i1, i2) in tris:
        x0, y0 = pts[i0]; x1, y1 = pts[i1]; x2, y2 = pts[i2]
        area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
        if area < 1e-14:
            continue

        b0 = (y1 - y2) / (2*area); c0 = (x2 - x1) / (2*area)
        b1 = (y2 - y0) / (2*area); c1 = (x0 - x2) / (2*area)
        b2 = (y0 - y1) / (2*area); c2 = (x1 - x0) / (2*area)

        # B matrix (3 x 6)
        B = np.array([
            [b0, 0,  b1, 0,  b2, 0],
            [0,  c0, 0,  c1, 0,  c2],
            [c0, b0, c1, b1, c2, b2]
        ], dtype=np.float64)

        Ke = area * (B.T @ D @ B)  # (6,6)

        # body force centroid
        xc = (x0 + x1 + x2) / 3.0
        yc = (y0 + y1 + y2) / 3.0
        if bx_expr and str(bx_expr).strip():
            bxc = float(eval_expr_numpy_via_torch(bx_expr, np.array([xc]), np.array([yc]), device=device_for_eval)[0])
        else:
            bxc = float(bx_const)
        if by_expr and str(by_expr).strip():
            byc = float(eval_expr_numpy_via_torch(by_expr, np.array([xc]), np.array([yc]), device=device_for_eval)[0])
        else:
            byc = float(by_const)

        # fe = ∫ N^T b dΩ  (constant b at centroid) => b*area/3 each node
        fe = (area/3.0) * np.array([bxc, byc, bxc, byc, bxc, byc], dtype=np.float64)

        idx_nodes = [i0, i1, i2]
        idx_dofs = [2*i0, 2*i0+1, 2*i1, 2*i1+1, 2*i2, 2*i2+1]

        for a in range(6):
            f[idx_dofs[a]] += fe[a]
            ra = idx_dofs[a]
            for c in range(6):
                rows.append(ra); cols.append(idx_dofs[c]); data.append(Ke[a, c])

    # traction on Gamma_t via segment Gauss
    if seg_t is not None and seg_t.size > 0:
        xi, wi = gauss_1d(int(gauss_n_seg))
        s = (xi + 1.0) * 0.5
        for (i0, i1) in seg_t[:, :2]:
            p0 = pts[i0]; p1 = pts[i1]
            L = np.linalg.norm(p1 - p0)
            if L < 1e-14:
                continue
            # Gauss points
            Pg = (1.0 - s)[:, None] * p0 + s[:, None] * p1  # (ng,2)
            xg = Pg[:, 0]; yg = Pg[:, 1]

            if tx_expr and str(tx_expr).strip():
                txv = eval_expr_numpy_via_torch(tx_expr, xg, yg, device=device_for_eval)
            else:
                txv = tx_const * np.ones_like(xg)
            if ty_expr and str(ty_expr).strip():
                tyv = eval_expr_numpy_via_torch(ty_expr, xg, yg, device=device_for_eval)
            else:
                tyv = ty_const * np.ones_like(yg)

            # line shape functions on segment: N0=1-s, N1=s
            N0 = (1.0 - s); N1 = s
            wJ = wi * (L/2.0)

            # add to nodal force
            fx0 = np.sum(wJ * N0 * txv)
            fy0 = np.sum(wJ * N0 * tyv)
            fx1 = np.sum(wJ * N1 * txv)
            fy1 = np.sum(wJ * N1 * tyv)

            f[2*i0]   += fx0
            f[2*i0+1] += fy0
            f[2*i1]   += fx1
            f[2*i1+1] += fy1

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # Dirichlet strong impose on nodes in seg_u
    dbc_nodes = _dirichlet_nodes_from_segs(seg_u)
    u = np.zeros(ndof, dtype=np.float64)

    if dbc_nodes.size > 0:
        ux_d = eval_expr_numpy_via_torch(ubarx_expr or "0", X[dbc_nodes], Y[dbc_nodes], device=device_for_eval)
        uy_d = eval_expr_numpy_via_torch(ubary_expr or "0", X[dbc_nodes], Y[dbc_nodes], device=device_for_eval)

        dbc_dofs = np.zeros(2*dbc_nodes.size, dtype=int)
        dbc_vals = np.zeros(2*dbc_nodes.size, dtype=np.float64)
        dbc_dofs[0::2] = 2*dbc_nodes
        dbc_dofs[1::2] = 2*dbc_nodes + 1
        dbc_vals[0::2] = ux_d
        dbc_vals[1::2] = uy_d

        u[dbc_dofs] = dbc_vals

        f = f - K[:, dbc_dofs] @ u[dbc_dofs]

        mask_free = np.ones(ndof, dtype=bool)
        mask_free[dbc_dofs] = False
        Kff = K[mask_free][:, mask_free]
        ff = f[mask_free]
        uf = spla.spsolve(Kff, ff)
        u[mask_free] = uf
    else:
        u[:] = spla.spsolve(K, f)

    U = u.reshape(n, 2)

    # element stress (constant per element), then simple nodal averaging
    sigma_elem = np.zeros((tris.shape[0], 3), dtype=np.float64)  # [sxx, syy, sxy]
    for e, (i0, i1, i2) in enumerate(tris):
        x0, y0 = pts[i0]; x1, y1 = pts[i1]; x2, y2 = pts[i2]
        area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
        if area < 1e-14:
            continue
        b0 = (y1 - y2) / (2*area); c0 = (x2 - x1) / (2*area)
        b1 = (y2 - y0) / (2*area); c1 = (x0 - x2) / (2*area)
        b2 = (y0 - y1) / (2*area); c2 = (x1 - x0) / (2*area)
        B = np.array([
            [b0, 0,  b1, 0,  b2, 0],
            [0,  c0, 0,  c1, 0,  c2],
            [c0, b0, c1, b1, c2, b2]
        ], dtype=np.float64)
        ue = np.array([U[i0,0], U[i0,1], U[i1,0], U[i1,1], U[i2,0], U[i2,1]], dtype=np.float64)
        eps = B @ ue  # [exx, eyy, gxy]
        # convert gamma_xy to exy in constitutive already consistent with Voigt shear = gamma_xy
        sig = D @ eps
        sigma_elem[e, :] = sig

    # nodal average
    sigma_n = np.zeros((n, 3), dtype=np.float64)
    cnt = np.zeros((n, 1), dtype=np.float64)
    for e, (i0, i1, i2) in enumerate(tris):
        for ii in (i0, i1, i2):
            sigma_n[ii, :] += sigma_elem[e, :]
            cnt[ii, 0] += 1.0
    cnt = np.maximum(cnt, 1.0)
    sigma_n /= cnt

    sxx = sigma_n[:, 0:1]
    syy = sigma_n[:, 1:2]
    sxy = sigma_n[:, 2:3]
    mises = np.sqrt((sxx - syy)**2 + 3.0*(sxy**2) + 1e-12)

    return U, sigma_n, mises


def fem_linear_elasticity_TET4_3d(
    pts: np.ndarray,
    tets: np.ndarray,
    tris_u: np.ndarray,
    tris_t: np.ndarray | None,
    *,
    lam: float,
    mu: float,
    ubarx_expr: str,
    ubary_expr: str,
    ubarz_expr: str,
    bx_expr: str | None,
    by_expr: str | None,
    bz_expr: str | None,
    bx_const: float,
    by_const: float,
    bz_const: float,
    tx_expr: str | None,
    ty_expr: str | None,
    tz_expr: str | None,
    tx_const: float,
    ty_const: float,
    tz_const: float,
    device_for_eval="cpu",
):
    """
    3D small-strain linear elasticity FEM (Tet4 / P1).
    Unknown: displacement u=[ux,uy,uz] at nodes (3n dofs).
    Loads:
      - body force b in Omega
      - traction t on Gamma_t (Physical Surface triangles)
      - strong Dirichlet u=ubar on Gamma_u nodes (Physical Surface triangles)
    Returns:
      U: (N,3) nodal displacement
      sigma_n: (N,6) nodal stress in Voigt [sxx,syy,szz,sxy,syz,szx]
      mises: (N,1) nodal von Mises
    """
    P = np.asarray(pts, dtype=np.float64)
    T = np.asarray(tets, dtype=int)[:, :4]
    n = int(P.shape[0])
    ndof = 3 * n

    # constitutive matrix C for isotropic 3D in Voigt [xx,yy,zz,xy,yz,zx]
    lam = float(lam)
    mu = float(mu)
    C = np.array(
        [
            [lam + 2 * mu, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mu, lam, 0, 0, 0],
            [lam, lam, lam + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ],
        dtype=np.float64,
    )

    rows, cols, data = [], [], []
    f = np.zeros(ndof, dtype=np.float64)

    # ---- element assembly ----
    for (i0, i1, i2, i3) in T:
        idx = np.array([i0, i1, i2, i3], dtype=int)
        Xe = P[idx, :]  # (4,3)
        v6 = float(np.dot(Xe[1] - Xe[0], np.cross(Xe[2] - Xe[0], Xe[3] - Xe[0])))
        vol = abs(v6) / 6.0
        if not np.isfinite(vol) or vol < 1e-18:
            continue

        M = np.ones((4, 4), dtype=np.float64)
        M[:, 1:] = Xe
        try:
            invM = np.linalg.inv(M)
        except Exception:
            continue
        grads = invM[1:4, :].T  # (4,3): grad N_i = [b_i,c_i,d_i]

        # B matrix (6 x 12)
        B = np.zeros((6, 12), dtype=np.float64)
        for a in range(4):
            bx, by, bz = grads[a, 0], grads[a, 1], grads[a, 2]
            col = 3 * a
            # exx, eyy, ezz
            B[0, col + 0] = bx
            B[1, col + 1] = by
            B[2, col + 2] = bz
            # gxy, gyz, gzx (engineering shear)
            B[3, col + 0] = by
            B[3, col + 1] = bx
            B[4, col + 1] = bz
            B[4, col + 2] = by
            B[5, col + 0] = bz
            B[5, col + 2] = bx

        Ke = vol * (B.T @ C @ B)  # (12,12)

        # body force at centroid (1-pt)
        xc = float(Xe[:, 0].mean())
        yc = float(Xe[:, 1].mean())
        zc = float(Xe[:, 2].mean())
        if bx_expr and str(bx_expr).strip():
            bxc = float(eval_expr_numpy_via_torch(bx_expr, np.array([xc]), np.array([yc]), np.array([zc]), device=device_for_eval)[0])
        else:
            bxc = float(bx_const)
        if by_expr and str(by_expr).strip():
            byc = float(eval_expr_numpy_via_torch(by_expr, np.array([xc]), np.array([yc]), np.array([zc]), device=device_for_eval)[0])
        else:
            byc = float(by_const)
        if bz_expr and str(bz_expr).strip():
            bzc = float(eval_expr_numpy_via_torch(bz_expr, np.array([xc]), np.array([yc]), np.array([zc]), device=device_for_eval)[0])
        else:
            bzc = float(bz_const)

        # consistent body load: ∫ N^T b dV -> b * vol/4 per node
        fe_node = (vol / 4.0) * np.array([bxc, byc, bzc], dtype=np.float64)
        fe = np.tile(fe_node, 4)  # (12,)

        dofs = np.array(
            [3 * i0, 3 * i0 + 1, 3 * i0 + 2, 3 * i1, 3 * i1 + 1, 3 * i1 + 2, 3 * i2, 3 * i2 + 1, 3 * i2 + 2, 3 * i3, 3 * i3 + 1, 3 * i3 + 2],
            dtype=int,
        )

        f[dofs] += fe
        for a in range(12):
            ra = int(dofs[a])
            for c in range(12):
                rows.append(ra)
                cols.append(int(dofs[c]))
                data.append(float(Ke[a, c]))

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # ---- traction on Gamma_t: ∫ N^T t dΓ over surface triangles (1-pt centroid) ----
    if tris_t is not None and np.asarray(tris_t).size > 0:
        TT = np.asarray(tris_t)[:, :3].astype(int)
        for (j0, j1, j2) in TT:
            Xf = P[[j0, j1, j2], :]  # (3,3)
            area = 0.5 * float(np.linalg.norm(np.cross(Xf[1] - Xf[0], Xf[2] - Xf[0])))
            if not np.isfinite(area) or area < 1e-18:
                continue
            xc = float(Xf[:, 0].mean())
            yc = float(Xf[:, 1].mean())
            zc = float(Xf[:, 2].mean())
            if tx_expr and str(tx_expr).strip():
                txv = float(eval_expr_numpy_via_torch(tx_expr, np.array([xc]), np.array([yc]), np.array([zc]), device=device_for_eval)[0])
            else:
                txv = float(tx_const)
            if ty_expr and str(ty_expr).strip():
                tyv = float(eval_expr_numpy_via_torch(ty_expr, np.array([xc]), np.array([yc]), np.array([zc]), device=device_for_eval)[0])
            else:
                tyv = float(ty_const)
            if tz_expr and str(tz_expr).strip():
                tzv = float(eval_expr_numpy_via_torch(tz_expr, np.array([xc]), np.array([yc]), np.array([zc]), device=device_for_eval)[0])
            else:
                tzv = float(tz_const)

            # P1 triangle load, centroid 1-pt: each node gets t*area/3
            add = (area / 3.0) * np.array([txv, tyv, tzv], dtype=np.float64)
            f[3 * int(j0) : 3 * int(j0) + 3] += add
            f[3 * int(j1) : 3 * int(j1) + 3] += add
            f[3 * int(j2) : 3 * int(j2) + 3] += add

    # ---- Dirichlet strong impose on Gamma_u nodes ----
    dbc_nodes = _dirichlet_nodes_from_tris(tris_u)
    u = np.zeros(ndof, dtype=np.float64)
    if dbc_nodes.size > 0:
        ux_d = eval_expr_numpy_via_torch(
            ubarx_expr or "0", P[dbc_nodes, 0], P[dbc_nodes, 1], P[dbc_nodes, 2], device=device_for_eval
        )
        uy_d = eval_expr_numpy_via_torch(
            ubary_expr or "0", P[dbc_nodes, 0], P[dbc_nodes, 1], P[dbc_nodes, 2], device=device_for_eval
        )
        uz_d = eval_expr_numpy_via_torch(
            ubarz_expr or "0", P[dbc_nodes, 0], P[dbc_nodes, 1], P[dbc_nodes, 2], device=device_for_eval
        )

        dbc_dofs = np.zeros(3 * dbc_nodes.size, dtype=int)
        dbc_vals = np.zeros(3 * dbc_nodes.size, dtype=np.float64)
        dbc_dofs[0::3] = 3 * dbc_nodes
        dbc_dofs[1::3] = 3 * dbc_nodes + 1
        dbc_dofs[2::3] = 3 * dbc_nodes + 2
        dbc_vals[0::3] = ux_d
        dbc_vals[1::3] = uy_d
        dbc_vals[2::3] = uz_d

        u[dbc_dofs] = dbc_vals
        f = f - K[:, dbc_dofs] @ u[dbc_dofs]

        mask_free = np.ones(ndof, dtype=bool)
        mask_free[dbc_dofs] = False
        Kff = K[mask_free][:, mask_free]
        ff = f[mask_free]
        uf = spla.spsolve(Kff, ff)
        u[mask_free] = uf
    else:
        u[:] = spla.spsolve(K, f)

    U = u.reshape(n, 3)

    # ---- stress recovery: constant per tet -> nodal averaging ----
    sigma_elem = np.zeros((T.shape[0], 6), dtype=np.float64)
    for e, (i0, i1, i2, i3) in enumerate(T):
        idx = np.array([i0, i1, i2, i3], dtype=int)
        Xe = P[idx, :]
        v6 = float(np.dot(Xe[1] - Xe[0], np.cross(Xe[2] - Xe[0], Xe[3] - Xe[0])))
        vol = abs(v6) / 6.0
        if not np.isfinite(vol) or vol < 1e-18:
            continue
        M = np.ones((4, 4), dtype=np.float64)
        M[:, 1:] = Xe
        try:
            invM = np.linalg.inv(M)
        except Exception:
            continue
        grads = invM[1:4, :].T  # (4,3)
        B = np.zeros((6, 12), dtype=np.float64)
        for a in range(4):
            bx, by, bz = grads[a, 0], grads[a, 1], grads[a, 2]
            col = 3 * a
            B[0, col + 0] = bx
            B[1, col + 1] = by
            B[2, col + 2] = bz
            B[3, col + 0] = by
            B[3, col + 1] = bx
            B[4, col + 1] = bz
            B[4, col + 2] = by
            B[5, col + 0] = bz
            B[5, col + 2] = bx

        ue = U[idx, :].reshape(-1)  # (12,)
        eps = B @ ue  # Voigt strain [exx,eyy,ezz,gxy,gyz,gzx]
        sig = C @ eps
        sigma_elem[e, :] = sig

    sigma_n = np.zeros((n, 6), dtype=np.float64)
    cnt = np.zeros((n, 1), dtype=np.float64)
    for e, (i0, i1, i2, i3) in enumerate(T):
        for ii in (i0, i1, i2, i3):
            sigma_n[int(ii), :] += sigma_elem[e, :]
            cnt[int(ii), 0] += 1.0
    cnt = np.maximum(cnt, 1.0)
    sigma_n /= cnt

    sxx = sigma_n[:, 0:1]
    syy = sigma_n[:, 1:2]
    szz = sigma_n[:, 2:3]
    sxy = sigma_n[:, 3:4]
    syz = sigma_n[:, 4:5]
    szx = sigma_n[:, 5:6]
    mean = (sxx + syy + szz) / 3.0
    sxx_d = sxx - mean
    syy_d = syy - mean
    szz_d = szz - mean
    mises = np.sqrt(
        1.5 * (sxx_d**2 + syy_d**2 + szz_d**2 + 2.0 * (sxy**2 + syz**2 + szx**2)) + 1e-12
    )

    return U.astype(np.float32), sigma_n.astype(np.float32), mises.astype(np.float32)


def _quad4_shape(rs: np.ndarray):
    """
    rs: (ng,2) in [-1,1]^2.
    Returns:
      N: (ng,4)
      dNdr: (ng,4)
      dNds: (ng,4)
    """
    r = rs[:, 0:1]
    s = rs[:, 1:2]
    N = np.concatenate(
        [
            0.25 * (1 - r) * (1 - s),
            0.25 * (1 + r) * (1 - s),
            0.25 * (1 + r) * (1 + s),
            0.25 * (1 - r) * (1 + s),
        ],
        axis=1,
    )
    dNdr = np.concatenate(
        [
            -0.25 * (1 - s),
            0.25 * (1 - s),
            0.25 * (1 + s),
            -0.25 * (1 + s),
        ],
        axis=1,
    )
    dNds = np.concatenate(
        [
            -0.25 * (1 - r),
            -0.25 * (1 + r),
            0.25 * (1 + r),
            0.25 * (1 - r),
        ],
        axis=1,
    )
    return N, dNdr, dNds


def fem_poisson_mixed_T3_Q4(
    pts,
    tris_plot,
    omega_triangles_n: int,
    quads,
    seg_u,
    ubar_expr: str,
    f_expr: str | None,
    *,
    gauss_n_quad: int = 2,
    device_for_eval="cpu",
):
    """
    Mixed FEM for Poisson:
    - Use T3 on the *original* triangles: tris_plot[:omega_triangles_n]
    - Use Q4 on quad elements
    """
    n = pts.shape[0]
    rows, cols, data = [], [], []
    b = np.zeros(n, dtype=np.float64)
    Xn = pts[:, 0]
    Yn = pts[:, 1]

    # --- T3 contribution ---
    if omega_triangles_n > 0:
        tris = tris_plot[:omega_triangles_n]
        for (i0, i1, i2) in tris:
            x0, y0 = pts[i0]; x1, y1 = pts[i1]; x2, y2 = pts[i2]
            area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
            if area < 1e-14:
                continue
            b0 = (y1 - y2) / (2*area); c0 = (x2 - x1) / (2*area)
            b1 = (y2 - y0) / (2*area); c1 = (x0 - x2) / (2*area)
            b2 = (y0 - y1) / (2*area); c2 = (x1 - x0) / (2*area)
            B = np.array([[b0, b1, b2],
                          [c0, c1, c2]], dtype=np.float64)
            Ke = area * (B.T @ B)
            xc = (x0 + x1 + x2) / 3.0
            yc = (y0 + y1 + y2) / 3.0
            if f_expr and str(f_expr).strip():
                fc = float(eval_expr_numpy_via_torch(f_expr, np.array([xc]), np.array([yc]), device=device_for_eval)[0])
            else:
                # Default: no body forcing (match DEM default)
                fc = 0.0
            fe = fc * area * (1.0/3.0) * np.ones(3, dtype=np.float64)
            idx = [int(i0), int(i1), int(i2)]
            for a in range(3):
                b[idx[a]] += fe[a]
                for c in range(3):
                    rows.append(idx[a]); cols.append(idx[c]); data.append(Ke[a, c])

    # --- Q4 contribution ---
    if quads is not None and quads.size > 0:
        rs, w2 = gauss_2d_tensor(int(gauss_n_quad))
        N, dNdr, dNds = _quad4_shape(rs)
        ng = rs.shape[0]
        for (i0, i1, i2, i3) in quads[:, :4]:
            idx = [int(i0), int(i1), int(i2), int(i3)]
            Xe = pts[idx, :]
            dxdr = dNdr @ Xe[:, 0:1]; dydr = dNdr @ Xe[:, 1:2]
            dxds = dNds @ Xe[:, 0:1]; dyds = dNds @ Xe[:, 1:2]
            detJ = (dxdr * dyds - dxds * dydr).reshape(-1, 1)
            if np.any(np.abs(detJ) < 1e-14):
                continue
            # Inverse Jacobian for mapping x(r,s):
            #   [dx/dr dx/ds; dy/dr dy/ds]^{-1} = [dr/dx dr/dy; ds/dx ds/dy]
            drdx = (dyds / detJ)
            drdy = (-dxds / detJ)
            dsdx = (-dydr / detJ)
            dsdy = (dxdr / detJ)
            # Chain rule:
            #   dN/dx = dN/dr * dr/dx + dN/ds * ds/dx
            #   dN/dy = dN/dr * dr/dy + dN/ds * ds/dy
            dNdx = drdx * dNdr + dsdx * dNds
            dNdy = drdy * dNdr + dsdy * dNds

            Ke = np.zeros((4, 4), dtype=np.float64)
            fe = np.zeros(4, dtype=np.float64)
            Pg = N @ Xe
            xg = Pg[:, 0]; yg = Pg[:, 1]
            if f_expr and str(f_expr).strip():
                fv = eval_expr_numpy_via_torch(f_expr, xg, yg, device=device_for_eval).astype(np.float64)
            else:
                # Default: no body forcing (match DEM default)
                fv = np.zeros_like(xg, dtype=np.float64)
            wJ = (w2[:, 0] * np.abs(detJ[:, 0])).astype(np.float64)
            for k in range(ng):
                g = np.stack([dNdx[k, :], dNdy[k, :]], axis=0)
                Ke += (g.T @ g) * wJ[k]
                fe += N[k, :] * fv[k] * wJ[k]
            for a in range(4):
                b[idx[a]] += fe[a]
                for c in range(4):
                    rows.append(idx[a]); cols.append(idx[c]); data.append(Ke[a, c])

    K = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    dbc_nodes = _dirichlet_nodes_from_segs(seg_u)
    u = np.zeros(n, dtype=np.float64)
    if dbc_nodes.size > 0:
        ubar_vals = eval_expr_numpy_via_torch(ubar_expr or "0", Xn[dbc_nodes], Yn[dbc_nodes], device=device_for_eval)
        u[dbc_nodes] = ubar_vals
        b = b - K[:, dbc_nodes] @ u[dbc_nodes]
        mask_free = np.ones(n, dtype=bool)
        mask_free[dbc_nodes] = False
        Kff = K[mask_free][:, mask_free]
        bf = b[mask_free]
        uf = spla.spsolve(Kff, bf)
        u[mask_free] = uf
    else:
        u[:] = spla.spsolve(K, b)
    return u.reshape(-1, 1).astype(np.float32)


def fem_linear_elasticity_mixed_T3_Q4(
    pts,
    tris_plot,
    omega_triangles_n: int,
    quads,
    seg_u,
    seg_t,
    lam: float,
    mu: float,
    plane_mode: str,
    ubarx_expr: str,
    ubary_expr: str,
    bx_expr: str | None,
    by_expr: str | None,
    bx_const: float,
    by_const: float,
    tx_expr: str | None,
    ty_expr: str | None,
    tx_const: float,
    ty_const: float,
    *,
    gauss_n_quad: int = 2,
    gauss_n_seg: int = 2,
    device_for_eval="cpu",
):
    """
    Mixed FEM for 2D linear elasticity:
    - T3 on original triangles
    - Q4 on quads
    Returns (U, sigma_n, mises_n).
    """
    n = pts.shape[0]
    ndof = 2 * n
    rows, cols, data = [], [], []
    f = np.zeros(ndof, dtype=np.float64)
    Xn = pts[:, 0]; Yn = pts[:, 1]

    if plane_mode == "plane_strain":
        D = np.array([[lam+2*mu, lam,       0],
                      [lam,      lam+2*mu,  0],
                      [0,        0,         mu]], dtype=np.float64)
    else:
        nu_eff = lam / (lam + 2*mu + 1e-15)
        E_eff = 2*mu*(1+nu_eff)
        coef = E_eff / (1 - nu_eff**2 + 1e-15)
        D = coef * np.array([[1,      nu_eff, 0],
                             [nu_eff, 1,      0],
                             [0,      0,      (1-nu_eff)/2]], dtype=np.float64)

    # --- T3 contribution ---
    if omega_triangles_n > 0:
        tris = tris_plot[:omega_triangles_n]
        for (i0, i1, i2) in tris:
            x0, y0 = pts[i0]; x1, y1 = pts[i1]; x2, y2 = pts[i2]
            area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
            if area < 1e-14:
                continue
            b0 = (y1 - y2) / (2*area); c0 = (x2 - x1) / (2*area)
            b1 = (y2 - y0) / (2*area); c1 = (x0 - x2) / (2*area)
            b2 = (y0 - y1) / (2*area); c2 = (x1 - x0) / (2*area)
            B = np.array([
                [b0, 0,  b1, 0,  b2, 0],
                [0,  c0, 0,  c1, 0,  c2],
                [c0, b0, c1, b1, c2, b2]
            ], dtype=np.float64)
            Ke = area * (B.T @ D @ B)
            xc = (x0 + x1 + x2) / 3.0
            yc = (y0 + y1 + y2) / 3.0
            if bx_expr and str(bx_expr).strip():
                bxc = float(eval_expr_numpy_via_torch(bx_expr, np.array([xc]), np.array([yc]), device=device_for_eval)[0])
            else:
                bxc = float(bx_const)
            if by_expr and str(by_expr).strip():
                byc = float(eval_expr_numpy_via_torch(by_expr, np.array([xc]), np.array([yc]), device=device_for_eval)[0])
            else:
                byc = float(by_const)
            fe = (area/3.0) * np.array([bxc, byc, bxc, byc, bxc, byc], dtype=np.float64)
            idx_dofs = [2*i0, 2*i0+1, 2*i1, 2*i1+1, 2*i2, 2*i2+1]
            for a in range(6):
                f[idx_dofs[a]] += fe[a]
                ra = idx_dofs[a]
                for c in range(6):
                    rows.append(ra); cols.append(idx_dofs[c]); data.append(Ke[a, c])

    # --- Q4 contribution ---
    if quads is not None and quads.size > 0:
        rs, w2 = gauss_2d_tensor(int(gauss_n_quad))
        N, dNdr, dNds = _quad4_shape(rs)
        ng = rs.shape[0]
        for (i0, i1, i2, i3) in quads[:, :4]:
            idx = [int(i0), int(i1), int(i2), int(i3)]
            Xe = pts[idx, :]
            dxdr = dNdr @ Xe[:, 0:1]; dydr = dNdr @ Xe[:, 1:2]
            dxds = dNds @ Xe[:, 0:1]; dyds = dNds @ Xe[:, 1:2]
            detJ = (dxdr * dyds - dxds * dydr).reshape(-1, 1)
            if np.any(np.abs(detJ) < 1e-14):
                continue
            drdx = (dyds / detJ)
            drdy = (-dxds / detJ)
            dsdx = (-dydr / detJ)
            dsdy = (dxdr / detJ)
            dNdx = drdx * dNdr + dsdx * dNds
            dNdy = drdy * dNdr + dsdy * dNds

            Ke = np.zeros((8, 8), dtype=np.float64)
            fe = np.zeros(8, dtype=np.float64)

            Pg = N @ Xe
            xg = Pg[:, 0]; yg = Pg[:, 1]
            if bx_expr and str(bx_expr).strip():
                bxv = eval_expr_numpy_via_torch(bx_expr, xg, yg, device=device_for_eval).astype(np.float64)
            else:
                bxv = float(bx_const) * np.ones_like(xg, dtype=np.float64)
            if by_expr and str(by_expr).strip():
                byv = eval_expr_numpy_via_torch(by_expr, xg, yg, device=device_for_eval).astype(np.float64)
            else:
                byv = float(by_const) * np.ones_like(yg, dtype=np.float64)

            wJ = (w2[:, 0] * np.abs(detJ[:, 0])).astype(np.float64)
            for k in range(ng):
                B = np.zeros((3, 8), dtype=np.float64)
                for a in range(4):
                    B[0, 2*a]     = dNdx[k, a]
                    B[1, 2*a+1]   = dNdy[k, a]
                    B[2, 2*a]     = dNdy[k, a]
                    B[2, 2*a+1]   = dNdx[k, a]
                Ke += (B.T @ D @ B) * wJ[k]
                for a in range(4):
                    fe[2*a]   += N[k, a] * bxv[k] * wJ[k]
                    fe[2*a+1] += N[k, a] * byv[k] * wJ[k]

            idx_dofs = [2*idx[0],2*idx[0]+1,2*idx[1],2*idx[1]+1,2*idx[2],2*idx[2]+1,2*idx[3],2*idx[3]+1]
            for a in range(8):
                f[idx_dofs[a]] += fe[a]
                ra = idx_dofs[a]
                for c in range(8):
                    rows.append(ra); cols.append(idx_dofs[c]); data.append(Ke[a, c])

    # traction on Gamma_t via segment Gauss (endpoint distribution)
    if seg_t is not None and seg_t.size > 0:
        xi, wi = gauss_1d(int(gauss_n_seg))
        s = (xi + 1.0) * 0.5
        for (j0, j1) in seg_t[:, :2]:
            j0 = int(j0); j1 = int(j1)
            p0 = pts[j0]; p1 = pts[j1]
            L = np.linalg.norm(p1 - p0)
            if L < 1e-14:
                continue
            Pg = (1.0 - s)[:, None] * p0 + s[:, None] * p1
            xg = Pg[:, 0]; yg = Pg[:, 1]
            if tx_expr and str(tx_expr).strip():
                txv = eval_expr_numpy_via_torch(tx_expr, xg, yg, device=device_for_eval)
            else:
                txv = float(tx_const) * np.ones_like(xg)
            if ty_expr and str(ty_expr).strip():
                tyv = eval_expr_numpy_via_torch(ty_expr, xg, yg, device=device_for_eval)
            else:
                tyv = float(ty_const) * np.ones_like(yg)
            N0 = (1.0 - s); N1 = s
            wJ = wi * (L/2.0)
            f[2*j0]   += np.sum(wJ * N0 * txv)
            f[2*j0+1] += np.sum(wJ * N0 * tyv)
            f[2*j1]   += np.sum(wJ * N1 * txv)
            f[2*j1+1] += np.sum(wJ * N1 * tyv)

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # Dirichlet strong impose on nodes in seg_u
    dbc_nodes = _dirichlet_nodes_from_segs(seg_u)
    u = np.zeros(ndof, dtype=np.float64)
    if dbc_nodes.size > 0:
        ux_d = eval_expr_numpy_via_torch(ubarx_expr or "0", Xn[dbc_nodes], Yn[dbc_nodes], device=device_for_eval)
        uy_d = eval_expr_numpy_via_torch(ubary_expr or "0", Xn[dbc_nodes], Yn[dbc_nodes], device=device_for_eval)
        dbc_dofs = np.zeros(2*dbc_nodes.size, dtype=int)
        dbc_vals = np.zeros(2*dbc_nodes.size, dtype=np.float64)
        dbc_dofs[0::2] = 2*dbc_nodes
        dbc_dofs[1::2] = 2*dbc_nodes + 1
        dbc_vals[0::2] = ux_d
        dbc_vals[1::2] = uy_d
        u[dbc_dofs] = dbc_vals
        f = f - K[:, dbc_dofs] @ u[dbc_dofs]
        mask_free = np.ones(ndof, dtype=bool)
        mask_free[dbc_dofs] = False
        Kff = K[mask_free][:, mask_free]
        ff = f[mask_free]
        uf = spla.spsolve(Kff, ff)
        u[mask_free] = uf
    else:
        u[:] = spla.spsolve(K, f)

    U = u.reshape(n, 2)

    # very simple stress recovery: average constant element stress to nodes
    sigma_n = np.zeros((n, 3), dtype=np.float64)
    cnt = np.zeros((n, 1), dtype=np.float64)

    # triangles
    if omega_triangles_n > 0:
        tris = tris_plot[:omega_triangles_n]
        for (i0, i1, i2) in tris:
            x0, y0 = pts[i0]; x1, y1 = pts[i1]; x2, y2 = pts[i2]
            area = 0.5 * abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0))
            if area < 1e-14:
                continue
            b0 = (y1 - y2) / (2*area); c0 = (x2 - x1) / (2*area)
            b1 = (y2 - y0) / (2*area); c1 = (x0 - x2) / (2*area)
            b2 = (y0 - y1) / (2*area); c2 = (x1 - x0) / (2*area)
            B = np.array([
                [b0, 0,  b1, 0,  b2, 0],
                [0,  c0, 0,  c1, 0,  c2],
                [c0, b0, c1, b1, c2, b2]
            ], dtype=np.float64)
            ue = np.array([U[i0,0], U[i0,1], U[i1,0], U[i1,1], U[i2,0], U[i2,1]], dtype=np.float64)
            epsv = B @ ue
            sig = D @ epsv
            for ii in (i0, i1, i2):
                sigma_n[ii, :] += sig
                cnt[ii, 0] += 1.0

    # quads at center (r=s=0)
    if quads is not None and quads.size > 0:
        rs0 = np.array([[0.0, 0.0]], dtype=np.float64)
        N0, dNdr0, dNds0 = _quad4_shape(rs0)
        for (i0, i1, i2, i3) in quads[:, :4]:
            idx = [int(i0), int(i1), int(i2), int(i3)]
            Xe = pts[idx, :]
            dxdr = dNdr0 @ Xe[:, 0:1]; dydr = dNdr0 @ Xe[:, 1:2]
            dxds = dNds0 @ Xe[:, 0:1]; dyds = dNds0 @ Xe[:, 1:2]
            detJ = float(dxdr * dyds - dxds * dydr)
            if abs(detJ) < 1e-14:
                continue
            drdx = float(dyds / detJ)
            drdy = float(-dxds / detJ)
            dsdx = float(-dydr / detJ)
            dsdy = float(dxdr / detJ)
            dNdx = drdx * dNdr0 + dsdx * dNds0  # (1,4)
            dNdy = drdy * dNdr0 + dsdy * dNds0
            B = np.zeros((3, 8), dtype=np.float64)
            for a in range(4):
                B[0, 2*a]     = dNdx[0, a]
                B[1, 2*a+1]   = dNdy[0, a]
                B[2, 2*a]     = dNdy[0, a]
                B[2, 2*a+1]   = dNdx[0, a]
            ue = np.array([U[idx[0],0], U[idx[0],1], U[idx[1],0], U[idx[1],1], U[idx[2],0], U[idx[2],1], U[idx[3],0], U[idx[3],1]], dtype=np.float64)
            epsv = B @ ue
            sig = D @ epsv
            for ii in idx:
                sigma_n[ii, :] += sig
                cnt[ii, 0] += 1.0

    cnt = np.maximum(cnt, 1.0)
    sigma_n /= cnt
    sxx = sigma_n[:, 0:1]; syy = sigma_n[:, 1:2]; sxy = sigma_n[:, 2:3]
    mises = np.sqrt((sxx - syy)**2 + 3.0*(sxy**2) + 1e-12)
    return U, sigma_n, mises

# ============================================================
# Visualization helpers
# ============================================================
def plot_mesh_preview_3d_surfaces(
    pts: np.ndarray,
    gu_tris: np.ndarray | None,
    gt_tris: np.ndarray | None,
    *,
    tets_conn: np.ndarray | None = None,
    max_faces: int = 4000,
    max_points: int = 6000,
):
    """
    3D mesh preview (lightweight):
    - scatter a subset of points
    - render Gamma_u / Gamma_t as translucent triangle surfaces (if provided)
    """
    fig = plt.figure(figsize=(6.2, 5.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    P = np.asarray(pts)
    if P.size == 0:
        return fig

    # point cloud (subsample for speed)
    npts = int(P.shape[0])
    if npts > int(max_points):
        idxp = np.linspace(0, npts - 1, int(max_points), dtype=int)
        Pp = P[idxp]
    else:
        Pp = P
    ax.scatter(Pp[:, 0], Pp[:, 1], Pp[:, 2], s=1.0, c="0.65", alpha=0.35, linewidths=0)

    def _add_tris(tris: np.ndarray | None, *, color: str, label: str):
        if tris is None:
            return
        T = np.asarray(tris)
        if T.size == 0:
            return
        T = T[:, :3].astype(int)
        nt = int(T.shape[0])
        if nt > int(max_faces):
            idxt = np.linspace(0, nt - 1, int(max_faces), dtype=int)
            T = T[idxt]
        faces = P[T]  # (nf,3,3)
        poly = Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors=(0, 0, 0, 0.15),
            linewidths=0.2,
            alpha=0.25,
        )
        ax.add_collection3d(poly)
        # legend proxy
        poly.set_label(label)

    _add_tris(gu_tris, color="tab:red", label="Gamma_u")
    _add_tris(gt_tris, color="tab:green", label="Gamma_t")

    # Always try to show the remaining outer surface (outer surface minus Gamma_u/Gamma_t),
    # so 3D bodies (e.g., a cylinder side wall) are visible even when Gamma_u/Gamma_t exist.

    has_gu = gu_tris is not None and np.asarray(gu_tris).size > 0
    has_gt = gt_tris is not None and np.asarray(gt_tris).size > 0
    if tets_conn is not None and np.asarray(tets_conn).size > 0:
        surf = np.asarray(extract_boundary_tris_from_tets(np.asarray(tets_conn)))[:, :3].astype(int)
        if surf.size > 0:
            # build normalized "keys" (sorted vertex ids) to subtract Gamma_u/Gamma_t from outer surface
            surf_s = np.sort(surf, axis=1).astype(np.int64)
            surf_v = surf_s.view(np.dtype((np.void, surf_s.dtype.itemsize * surf_s.shape[1]))).reshape(-1)

            def _keys_void(tris_in: np.ndarray | None) -> np.ndarray:
                Tin = np.asarray(tris_in) if tris_in is not None else np.zeros((0, 3), dtype=np.int64)
                if Tin.size == 0:
                    return np.zeros((0,), dtype=surf_v.dtype)
                Tin = Tin[:, :3].astype(np.int64)
                Tin_s = np.sort(Tin, axis=1)
                return Tin_s.view(np.dtype((np.void, Tin_s.dtype.itemsize * Tin_s.shape[1]))).reshape(-1)

            gu_v = _keys_void(gu_tris) if has_gu else np.zeros((0,), dtype=surf_v.dtype)
            gt_v = _keys_void(gt_tris) if has_gt else np.zeros((0,), dtype=surf_v.dtype)

            keep = np.ones((surf.shape[0],), dtype=bool)
            if gu_v.size > 0:
                keep &= ~np.isin(surf_v, gu_v)
            if gt_v.size > 0:
                keep &= ~np.isin(surf_v, gt_v)
            other = surf[keep]

            if other.size > 0:
                _add_tris(other, color="tab:blue", label="Other boundary (outer surface)")



    # bounds + aspect
    xmin, ymin, zmin = np.min(P[:, 0]), np.min(P[:, 1]), np.min(P[:, 2])
    xmax, ymax, zmax = np.max(P[:, 0]), np.max(P[:, 1]), np.max(P[:, 2])
    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(float(ymin), float(ymax))
    ax.set_zlim(float(zmin), float(zmax))
    try:
        ax.set_box_aspect((float(xmax - xmin), float(ymax - ymin), float(zmax - zmin)))
    except Exception:
        pass

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # Put legend on the right side to avoid covering the mesh.
    # Get all legend handles to determine number of items
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.1),
        borderaxespad=0.0,
        frameon=True,
        ncol=len(handles) if len(handles) > 0 else 1,  # Horizontal layout: place items side by side
    )
    # Reserve space on the right for the legend (avoid clipping).
    fig.tight_layout(rect=[0.0, 0.0, 1.00, 1.0])
    return fig


def plot_3d_scalar_pointcloud_and_slices(
    pts: np.ndarray,
    u: np.ndarray,
    *,
    max_points_3d: int = 20000,
    slice_frac: float = 0.03,
    x0: float | None = None,
    y0: float | None = None,
    z0: float | None = None,
):
    """
    3D scalar result visualization:
    - 3D colored point cloud (downsample)
    - 3 orthogonal slices as 2D scatter: XY@z0, XZ@y0, YZ@x0
    """
    P = np.asarray(pts)
    U = np.asarray(u).reshape(-1)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("pts must be (N,3) for 3D plotting.")
    if P.shape[0] != U.shape[0]:
        raise ValueError("u must align with pts (same N).")

    xmin, ymin, zmin = np.min(P[:, 0]), np.min(P[:, 1]), np.min(P[:, 2])
    xmax, ymax, zmax = np.max(P[:, 0]), np.max(P[:, 1]), np.max(P[:, 2])
    lx = float(max(xmax - xmin, 1e-12))
    ly = float(max(ymax - ymin, 1e-12))
    lz = float(max(zmax - zmin, 1e-12))
    L = max(lx, ly, lz)

    if x0 is None:
        x0 = float(np.median(P[:, 0]))
    if y0 is None:
        y0 = float(np.median(P[:, 1]))
    if z0 is None:
        z0 = float(np.median(P[:, 2]))

    th = float(max(slice_frac, 1e-4)) * float(L)
    mx = np.abs(P[:, 0] - float(x0)) <= th
    my = np.abs(P[:, 1] - float(y0)) <= th
    mz = np.abs(P[:, 2] - float(z0)) <= th

    # 3D downsample
    n = int(P.shape[0])
    if n > int(max_points_3d):
        idx3 = np.random.choice(n, size=int(max_points_3d), replace=False)
    else:
        idx3 = np.arange(n, dtype=int)

    # Slice indices (cap to keep fast)
    def _cap(idx: np.ndarray, cap: int = 25000):
        ii = np.where(idx)[0]
        if ii.size > cap:
            return np.random.choice(ii, size=cap, replace=False)
        return ii

    idx_xy = _cap(mz)   # XY slice @ z0
    idx_xz = _cap(my)   # XZ slice @ y0
    idx_yz = _cap(mx)   # YZ slice @ x0

    # 3D cloud
    fig3 = plt.figure(figsize=(6.6, 5.4), dpi=120)
    ax3 = fig3.add_subplot(111, projection="3d")
    sc3 = ax3.scatter(P[idx3, 0], P[idx3, 1], P[idx3, 2], c=U[idx3], s=3.0, cmap="viridis", alpha=0.9, linewidths=0)
    ax3.set_title(f"3D u point cloud (N={idx3.size})")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    try:
        ax3.set_box_aspect((lx, ly, lz))
    except Exception:
        pass
    plt.colorbar(sc3, ax=ax3, shrink=0.7, pad=0.08)
    plt.tight_layout()

    # 2D slices
    figS, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=120)
    ax_xy, ax_xz, ax_yz = axes

    s1 = ax_xy.scatter(P[idx_xy, 0], P[idx_xy, 1], c=U[idx_xy], s=6, cmap="viridis")
    ax_xy.set_title(f"XY slice @ z≈{float(z0):.3g} (±{th:.3g})")
    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
    ax_xy.set_aspect("equal", adjustable="box")
    plt.colorbar(s1, ax=ax_xy)

    s2 = ax_xz.scatter(P[idx_xz, 0], P[idx_xz, 2], c=U[idx_xz], s=6, cmap="viridis")
    ax_xz.set_title(f"XZ slice @ y≈{float(y0):.3g} (±{th:.3g})")
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    ax_xz.set_aspect("equal", adjustable="box")
    plt.colorbar(s2, ax=ax_xz)

    s3 = ax_yz.scatter(P[idx_yz, 1], P[idx_yz, 2], c=U[idx_yz], s=6, cmap="viridis")
    ax_yz.set_title(f"YZ slice @ x≈{float(x0):.3g} (±{th:.3g})")
    ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z")
    ax_yz.set_aspect("equal", adjustable="box")
    plt.colorbar(s3, ax=ax_yz)

    plt.tight_layout()
    return fig3, figS


def extract_boundary_tris_from_tets(tets: np.ndarray) -> np.ndarray:
    """
    Extract outer surface triangles from tetra connectivity.
    Returns (nf,3) triangle indices (sorted per face, orientation not guaranteed).
    """
    T = np.asarray(tets)
    if T.size == 0:
        return np.zeros((0, 3), dtype=int)
    T = T[:, :4].astype(int)
    # faces per tet
    f0 = T[:, [0, 1, 2]]
    f1 = T[:, [0, 1, 3]]
    f2 = T[:, [0, 2, 3]]
    f3 = T[:, [1, 2, 3]]
    F = np.vstack([f0, f1, f2, f3])
    Fs = np.sort(F, axis=1)
    uniq, counts = np.unique(Fs, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    return boundary.astype(int)


def plot_3d_scalar_surface_cloud(
    pts: np.ndarray,
    tris: np.ndarray,
    u: np.ndarray,
    *,
    max_faces: int = 12000,
):
    """
    Surface "cloud map" for 3D scalar field on boundary triangles.
    Colors each triangle by mean(u at its vertices).
    """
    P = np.asarray(pts)
    T = np.asarray(tris)
    U = np.asarray(u).reshape(-1)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("pts must be (N,3).")
    if T.size == 0:
        fig = plt.figure(figsize=(6.4, 5.2), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("3D surface cloud map (no surface triangles)")
        return fig

    T = T[:, :3].astype(int)
    nf = int(T.shape[0])
    if nf > int(max_faces):
        idxt = np.random.choice(nf, size=int(max_faces), replace=False)
        T = T[idxt]

    faces = P[T]  # (nf,3,3)
    face_u = U[T].mean(axis=1)

    from matplotlib import cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=float(np.min(face_u)), vmax=float(np.max(face_u)))
    cmap = cm.get_cmap("viridis")
    colors = cmap(norm(face_u))

    fig = plt.figure(figsize=(6.8, 5.6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(
        faces,
        facecolors=colors,
        edgecolors=(0, 0, 0, 0.10),
        linewidths=0.15,
        alpha=0.95,
    )
    ax.add_collection3d(poly)

    xmin, ymin, zmin = np.min(P[:, 0]), np.min(P[:, 1]), np.min(P[:, 2])
    xmax, ymax, zmax = np.max(P[:, 0]), np.max(P[:, 1]), np.max(P[:, 2])
    ax.set_xlim(float(xmin), float(xmax))
    ax.set_ylim(float(ymin), float(ymax))
    ax.set_zlim(float(zmin), float(zmax))
    try:
        ax.set_box_aspect((float(xmax - xmin), float(ymax - ymin), float(zmax - zmin)))
    except Exception:
        pass

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"3D surface cloud map (tris={int(T.shape[0])})")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.08)
    plt.tight_layout()
    return fig


def plot_mesh_with_boundaries_fast(pts, tris, seg_u, seg_t, quads=None, omega_triangles_n: int | None = None):
    """
    Faster + deterministic boundary colors.
    Simple view fitting - just use the actual geometry bounds.
    """
    # Keep preview compact in Streamlit "wide" layout
    fig, ax = plt.subplots(figsize=(3.8, 3.8), dpi=120)

    # mesh
    # - Triangles: draw via triplot (ONLY the true Omega triangles, not triangulated quads)
    # - Quads: draw as quad outlines (so quad meshes don't show diagonal splits)
    if tris is not None and np.asarray(tris).size > 0:
        tris_arr = np.asarray(tris)
        if omega_triangles_n is not None:
            ntri = int(max(0, min(int(omega_triangles_n), int(tris_arr.shape[0]))))
            tris_arr = tris_arr[:ntri]  # only real triangles
        if tris_arr.size > 0:
            ax.triplot(pts[:, 0], pts[:, 1], tris_arr, linewidth=0.3, alpha=0.6, zorder=1)

    if quads is not None and np.asarray(quads).size > 0:
        q = np.asarray(quads)[:, :4].astype(int)
        # build edge segments (nq*4, 2, 2)
        edges = np.stack(
            [
                q[:, [0, 1]],
                q[:, [1, 2]],
                q[:, [2, 3]],
                q[:, [3, 0]],
            ],
            axis=1,
        ).reshape(-1, 2)
        segs_xy = pts[edges]  # (nseg,2,2)
        lc_q = LineCollection(segs_xy, colors="0.35", linewidths=0.6, alpha=0.8, zorder=1)
        ax.add_collection(lc_q)

    # Gamma_u (Dirichlet) in RED
    if seg_u is not None and seg_u.size > 0:
        segs_u_xy = pts[seg_u[:, :2]]  # (nseg, 2, 2)
        lc_u = LineCollection(
            segs_u_xy,
            colors="tab:red",
            linewidths=2.5,
            zorder=3,
        )
        ax.add_collection(lc_u)

    # Gamma_t (Traction) in GREEN
    if seg_t is not None and seg_t.size > 0:
        segs_t_xy = pts[seg_t[:, :2]]  # (nseg, 2, 2)
        lc_t = LineCollection(
            segs_t_xy,
            colors="tab:green",
            linewidths=2.5,
            zorder=3,
        )
        ax.add_collection(lc_t)

    # Force matplotlib to update view limits based on all drawn elements
    # LineCollection doesn't auto-update limits, so we need to do it manually

    
    # Get the auto-scaled limits to verify they're correct
    xlim_auto = ax.get_xlim()
    ylim_auto = ax.get_ylim()
    
    # If limits are default (0,1), recalculate from actual data
    if pts is not None and np.asarray(pts).size > 0:
        pts_arr = np.asarray(pts)
        if pts_arr.shape[0] > 0:
            xmin_data = float(np.min(pts_arr[:, 0]))
            xmax_data = float(np.max(pts_arr[:, 0]))
            ymin_data = float(np.min(pts_arr[:, 1]))
            ymax_data = float(np.max(pts_arr[:, 1]))
            
            # Check if auto limits are wrong (e.g., default 0-1)
            if (xlim_auto[0] == 0.0 and xlim_auto[1] == 1.0 and 
                (xmin_data < 0 or xmax_data > 1 or abs(xmin_data) > 0.1 or abs(xmax_data - 1) > 0.1)):
                # Auto limits are wrong, use data limits
                sx = max(xmax_data - xmin_data, 1e-12)
                sy = max(ymax_data - ymin_data, 1e-12)
                pad_x = sx
                pad_y = sy
                ax.set_xlim(xmin_data, xmax_data)
                ax.set_ylim(ymin_data, ymax_data)
            elif (ylim_auto[0] == 0.0 and ylim_auto[1] == 1.0 and 
                  (ymin_data < 0 or ymax_data > 1 or abs(ymin_data) > 0.1 or abs(ymax_data - 1) > 0.1)):
                # Auto limits are wrong, use data limits
                sx = max(xmax_data - xmin_data, 1e-12)
                sy = max(ymax_data - ymin_data, 1e-12)
                pad_x = sx
                pad_y = sy
                ax.set_xlim(xmin_data, xmax_data)
                ax.set_ylim(ymin_data, ymax_data)
    
    # Set equal aspect - same as contour plots
    ax.set_aspect("equal", adjustable="box")
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2, linestyle="--")
    
    # legend (use explicit handles, so colors never get messed up)
    # Place legend above the plot, outside the plot area
    handles = []
    if seg_u is not None and seg_u.size > 0:
        handles.append(Line2D([0], [0], color="tab:red", lw=2.5, label="Gamma_u"))
    if seg_t is not None and seg_t.size > 0:
        handles.append(Line2D([0], [0], color="tab:green", lw=2.5, label="Gamma_t"))
    if handles:
        # Put legend above the plot, well above the plot area to avoid overlap
        # ncol=len(handles) ensures horizontal (left-right) layout
        legend = ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.42),  # Higher position to avoid overlap
            frameon=True,
            ncol=len(handles),  # Horizontal layout: place items side by side
        )
        # Ensure legend doesn't overlap with plot content
        fig.canvas.draw()  # Update layout
    
    
    return fig
    
@torch.no_grad()
def eval_on_points(model, device, pts_xy, out_dim, dir_mode, dist_obj, ubar_expr, ubarx_expr, ubary_expr, ubarz_expr="0", ubar_const=0.0, ubarx_const=0.0, ubary_const=0.0, ubarz_const=0.0):
    X = to_torch(pts_xy.astype(np.float32), device=device, requires_grad=False)
    dim = int(X.shape[1])
    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3] if dim == 3 else None
    nn_out = model(torch.cat([x, y], dim=1) if dim == 2 else torch.cat([x, y, z], dim=1))
    if out_dim == 1:
        u = apply_dirichlet_scalar(nn_out, x, y, dir_mode, dist_obj, ubar_expr, ubar_const=ubar_const, z=z)
        return u.cpu().numpy()
    else:
        uv = apply_dirichlet_vec(nn_out, x, y, dir_mode, dist_obj, ubarx_expr, ubary_expr, ubarz_expr=ubarz_expr, ubarx_const=ubarx_const, ubary_const=ubary_const, ubarz_const=ubarz_const, z=z)
        return uv.cpu().numpy()


def plot_stress_on_mesh(pts, tris, sigma, mises, title_prefix="Stress"):
    X = pts[:, 0]
    Y = pts[:, 1]
    sxx = sigma[:, 0]
    syy = sigma[:, 1]
    sxy = sigma[:, 2]
    vm = mises[:, 0]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    t1 = axes[0].tricontourf(X, Y, tris, sxx, levels=60)
    axes[0].set_title(r"$\sigma_{xx}$")
    axes[0].set_aspect("equal", adjustable="box")
    plt.colorbar(t1, ax=axes[0])

    t2 = axes[1].tricontourf(X, Y, tris, syy, levels=60)
    axes[1].set_title(r"$\sigma_{yy}$")
    axes[1].set_aspect("equal", adjustable="box")
    plt.colorbar(t2, ax=axes[1])

    t3 = axes[2].tricontourf(X, Y, tris, sxy, levels=60)
    axes[2].set_title(r"$\sigma_{xy}$")
    axes[2].set_aspect("equal", adjustable="box")
    plt.colorbar(t3, ax=axes[2])

    t4 = axes[3].tricontourf(X, Y, tris, vm, levels=60)
    axes[3].set_title(r"von Mises")
    axes[3].set_aspect("equal", adjustable="box")
    plt.colorbar(t4, ax=axes[3])

    fig.suptitle(title_prefix, y=1.02)
    plt.tight_layout()
    return fig


def plot_field_on_mesh(pts, tris, nodal_values, title, cmap=None, is_scalar=True):
    tri = tris
    X = pts[:, 0]
    Y = pts[:, 1]

    if is_scalar:
        Z = nodal_values[:, 0]
        fig, ax = plt.subplots(figsize=(7, 6))
        tpc = ax.tricontourf(X, Y, tri, Z, levels=60)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(tpc, ax=ax)
        plt.tight_layout()
        return fig
    else:
        U = nodal_values[:, 0]
        V = nodal_values[:, 1]
        Umag = np.sqrt(U**2 + V**2)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        t1 = axes[0].tricontourf(X, Y, tri, U, levels=60)
        axes[0].set_title("u_x")
        axes[0].set_aspect("equal", adjustable="box")
        plt.colorbar(t1, ax=axes[0])

        t2 = axes[1].tricontourf(X, Y, tri, V, levels=60)
        axes[1].set_title("u_y")
        axes[1].set_aspect("equal", adjustable="box")
        plt.colorbar(t2, ax=axes[1])

        t3 = axes[2].tricontourf(X, Y, tri, Umag, levels=60)
        axes[2].set_title("|u|")
        axes[2].set_aspect("equal", adjustable="box")
        plt.colorbar(t3, ax=axes[2])

        fig.suptitle(title, y=1.02)
        plt.tight_layout()
        return fig


# ============================================================
# Sidebar UI
# ============================================================
phys_omega = "Omega"
phys_gu = "Gamma_u"
phys_gt = "Gamma_t"

with st.sidebar:
    st.header("⚙️ Settings")

    # ------------------------------------------------------------
    # LLM Provider & API Key (session-only)
    # ------------------------------------------------------------
    with st.expander("🔑 LLM API Configuration (optional)", expanded=False):
        # Provider selection
        provider = st.selectbox(
            "LLM Provider",
            options=["openai", "anthropic", "google", "ollama"],
            index=0,
            key="llm_provider",
            help="Select the LLM provider to use for geometry generation.",
        )
        
        # Model selection (depends on provider)
        available_models = _get_available_models(provider)
        default_model = _get_default_model(provider)
        current_model = st.session_state.get("llm_model", default_model)
        if current_model not in available_models:
            current_model = default_model
        
        model = st.selectbox(
            "Model",
            options=available_models,
            index=available_models.index(current_model) if current_model in available_models else 0,
            key="llm_model",
            help="Select the specific model to use.",
        )
        
        # API Key/URL input (depends on provider)
        if provider == "openai":
            st.text_input(
                "OPENAI_API_KEY (session-only)",
                type="password",
                key="openai_api_key_ui",
                help=(
                    "If set here, it is used only for your current browser session and is not saved to disk. "
                    "On Streamlit Community Cloud, the app owner can also set it in Settings → Secrets."
                ),
            )
        elif provider == "anthropic":
            st.text_input(
                "ANTHROPIC_API_KEY (session-only)",
                type="password",
                key="anthropic_api_key_ui",
                help=(
                    "If set here, it is used only for your current browser session and is not saved to disk. "
                    "On Streamlit Community Cloud, the app owner can also set it in Settings → Secrets."
                ),
            )
        elif provider == "google":
            st.text_input(
                "GOOGLE_API_KEY (session-only)",
                type="password",
                key="google_api_key_ui",
                help=(
                    "If set here, it is used only for your current browser session and is not saved to disk. "
                    "On Streamlit Community Cloud, the app owner can also set it in Settings → Secrets. "
                    "Get your API key from: https://makersuite.google.com/app/apikey"
                ),
            )
        elif provider == "ollama":
            st.text_input(
                "OLLAMA_BASE_URL (session-only)",
                value="http://localhost:11434",
                key="ollama_base_url_ui",
                help=(
                    "Base URL for your local Ollama instance. Default: http://localhost:11434. "
                    "Make sure Ollama is running locally."
                ),
            )
            st.info("💡 Make sure Ollama is installed and running locally. Install from: https://ollama.ai")

    # ------------------------------------------------------------
    # Geometry (Gmsh CLI)
    # ------------------------------------------------------------
    _llm_ready = bool(_get_llm_api_key(st.session_state.get("llm_provider", "openai")))
    if not _llm_ready:
        current_provider = st.session_state.get("llm_provider", "openai")
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "ollama": "OLLAMA_BASE_URL",
        }
        env_var = env_var_map.get(current_provider, "OPENAI_API_KEY")
        st.warning(
            f"{env_var} not set. LLM geometry generation/repair is disabled. "
            f"Set it via Streamlit Secrets ({env_var}) or environment variable."
        )
    with st.expander("📐 Geometry ", expanded=False):

        st.subheader("🔳 Mesh size (lc)")
        auto_mesh = st.checkbox("Auto re-mesh when lc changes", value=True)

        def _on_lc_change():
            lc_val = float(st.session_state["lc_ui"])

            # 1) update geo
            if "geo_text" in st.session_state and st.session_state["geo_text"].strip():
                st.session_state["geo_text"] = upsert_lc_in_geo(st.session_state["geo_text"], lc_val)

            # 2) optional: auto run gmsh
            if auto_mesh:
                geo_text = st.session_state.get("geo_text", "")
                pt = st.session_state.get("problem_type", "Linear Elasticity (2D)")
                require_gt = (pt not in ["Poisson (scalar)", "Screened Poisson equation (2D)"])
                dim0 = int(st.session_state.get("geo_dim", 2))
                ok_geo, msg = validate_geo_text(geo_text, require_gamma_t=require_gt, dim=dim0)
                if not ok_geo:
                    st.session_state["last_auto_mesh_error"] = f".geo invalid: {msg}"
                    return

                try:
                    msh_bytes, gmsh_log = gmsh_geo_to_msh_bytes(
                        geo_text,
                        gmsh_cmdline=st.session_state["gmsh_cmdline"],
                        dim=dim0,
                        msh_format=st.session_state["geo_msh_format"],
                        extra_args=st.session_state["gmsh_extra_args"],
                        timeout_sec=180,
                    )
                    st.session_state["generated_msh_bytes"] = msh_bytes
                    st.session_state["generated_msh_name"] = "generated_from_geo.msh"
                    st.session_state["last_auto_mesh_log"] = gmsh_log
                    st.session_state.pop("mesh_sig", None)  # force reload
                    # Auto-load so mesh preview updates immediately (no manual click).
                    st.session_state["auto_load_mesh_after_gen"] = True
                    st.session_state.pop("last_auto_mesh_error", None)
                except Exception as e:
                    st.session_state["last_auto_mesh_error"] = str(e)

        lc_ui = st.select_slider(
            "lc (global mesh size)",
            options=[0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
            value=float(st.session_state.get("lc_ui", 0.15)),
            key="lc_ui",
            on_change=_on_lc_change,
        )

        if "gmsh_cmdline" not in st.session_state:
            st.session_state["gmsh_cmdline"] = "gmsh"
        if "gmsh_extra_args" not in st.session_state:
            st.session_state["gmsh_extra_args"] = ""
        if "geo_msh_format" not in st.session_state:
            st.session_state["geo_msh_format"] = "msh2"

        gmsh_cmdline = st.text_input(
            "Gmsh command / path",
            key="gmsh_cmdline",
            help=(
                "Example: gmsh (recommended). "
                "Local Windows example: C:\\Program Files\\Gmsh\\gmsh.exe. "
                "On Streamlit Cloud you must use 'gmsh' (installed via packages.txt); "
                "Windows paths like C:\\... do not exist on the server."
            ),
        )
        geo_dim = st.selectbox("Mesh dimension", [2, 3], index=0, key="geo_dim")
        _msh_fmt_opts = ["msh2", "msh4"]
        geo_msh_format = st.selectbox(
            "MSH format",
            _msh_fmt_opts,
            index=_msh_fmt_opts.index(st.session_state["geo_msh_format"])
            if st.session_state.get("geo_msh_format") in _msh_fmt_opts
            else 0,
            key="geo_msh_format",
        )
        geo_extra_args = st.text_input("Extra Gmsh args (optional)", key="gmsh_extra_args")

        st.subheader("📦 Mesh (.msh)")
        msh_file = st.file_uploader("Upload .msh", type=["msh"])
        st.write("**Physical names (fixed):**")
        st.code("2D: Omega (Surface), Gamma_u (Curve), Gamma_t (Curve)\n3D: Omega (Volume),  Gamma_u (Surface), Gamma_t (Surface)")
        show_mesh_preview = st.checkbox("Show mesh preview", value=True)

        st.subheader("∫ Quadrature (mesh-based)")
        tri_rule_name = st.selectbox("Triangle rule", ["1-point", "3-point"], index=0)
        quad_gauss_n = st.selectbox("Quad Gauss (n×n)", [1, 2, 3, 4], index=1)
        seg_gauss_n = st.selectbox("Boundary Gauss points", [1, 2, 3], index=1)
        do_fem_ref = st.checkbox("Compute FEM reference (for comparison)", value=True)

        st.subheader("∇ Derivatives (DEM)")
        derivative_method = st.selectbox(
            "How to compute ∇u / strains at Gauss points",
            ["shape_function (mesh)", "autograd (AD)"],
            index=0,
            help="Default: shape_function uses element shape functions (B-matrix) and avoids autograd w.r.t. coordinates. 3D currently falls back to autograd.",
        )



    # ------------------------------------------------------------
    # Problem selection (kept visible in boundary page)
    # ------------------------------------------------------------
    _loaded_mesh_dim = st.session_state.get("mesh_dim", None)
    _dim_for_problem_ui = int(st.session_state.get("geo_dim", 2))
    if _loaded_mesh_dim is not None and int(_loaded_mesh_dim) != int(_dim_for_problem_ui):
        st.warning(
            f"Target mesh dim is **{int(_dim_for_problem_ui)}D** but currently loaded mesh is **{int(_loaded_mesh_dim)}D**. "
            "Generate/upload a new mesh to match before training."
        )

    if int(_dim_for_problem_ui) == 3:
        _problem_opts = ["Poisson (scalar)", "Linear Elasticity (3D)", "Custom (user-defined)"]
    else:
        _problem_opts = ["Poisson (scalar)", "Screened Poisson equation (2D)", "Linear Elasticity (2D)", "Neo-Hookean Hyperelasticity (2D)", "Custom (user-defined)"]

    _prev_problem = str(st.session_state.get("problem_type", "") or "")
    if _prev_problem not in _problem_opts and "problem_type" in st.session_state:
        del st.session_state["problem_type"]

    # ------------------------------------------------------------
    # Boundary conditions + source terms
    # ------------------------------------------------------------
    with st.expander("🧱 Boundary conditions", expanded=False):
        # Mesh loading UI placeholder (filled later, after ensure_mesh_loaded() is defined)
        bc_mesh_load_ui = st.empty()
        # Problem is selected in "Material & Loads" (keep this section focused on BCs).
        problem_type = str(
            st.session_state.get(
                "problem_type",
                _prev_problem if _prev_problem in _problem_opts else (_problem_opts[1] if len(_problem_opts) > 1 else _problem_opts[0]),
            )
        )

        st.subheader("Dirichlet on Gamma_u")
        hard_bc = st.checkbox("Hard BC (distance-based)", value=True)
        penalty_bc = st.checkbox("Penalty BC", value=True)
        penalty_lambda = st.select_slider(
            "Penalty λ",
            options=[0.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
            value=1e6,
        )

        dir_mode = "none"
        if hard_bc and penalty_bc:
            dir_mode = "hard+penalty"
        elif hard_bc:
            dir_mode = "hard"
        elif penalty_bc:
            dir_mode = "penalty"

        dist_tau = 0.0
        if hard_bc:
            smooth_dist = st.checkbox(
                "Smooth distance (soft-min over segments)",
                value=True,
                help="Makes d(x, Γu) differentiable across 'closest segment' switching lines; improves stress near Γu.",
            )
            if smooth_dist:
                lc_val = float(st.session_state.get("lc_ui", 0.15))
                _tau_opts = [float((f * lc_val) ** 2) for f in (0.05, 0.10, 0.25, 0.50, 1.00)]
                dist_tau = float(
                    st.select_slider(
                        "soft-min τ (squared-distance units)",
                        options=_tau_opts,
                        value=float((0.25 * lc_val) ** 2),
                        help="Start around (0.25·lc)^2; smaller → closer to hard min but less smooth.",
                    )
                )

        # Default values
        f_expr = ""
        g_expr = ""
        g_const = 0.0
        k_squared = 0.0
        ubar_expr = ""
        ubar_const = 0.0
        ubarx_expr = ""
        ubary_expr = ""
        ubarz_expr = ""
        ubarx_const = 0.0
        ubary_const = 0.0
        ubarz_const = 0.0
        # Traction defaults (kept here so training/export always has values)
        tx_const = 10.0
        ty_const = 0.0
        tz_const = 0.0
        tx_expr = ""
        ty_expr = ""
        tz_expr = ""

        # Body force defaults (must exist for all problem types; used in train_dem_mesh call)
        bx_const = 0.0
        by_const = 0.0
        bz_const = 0.0
        bx_expr = ""
        by_expr = ""
        bz_expr = ""

        if problem_type == "Poisson (scalar)":
            st.subheader("Poisson terms")
            with st.expander("f, g and ū", expanded=True):
                f_expr = st.text_input("f(x,y,z) in Ω =", value="0", help="Body forcing. Use x,y,z (z ignored for 2D).")
                g_const = st.slider("Constant g on Gamma_t", -50.0, 50.0, 0.0, 0.5, help="Neumann/flux value (scalar).")
                g_expr = st.text_input("Or expr g(x,y,z) on Gamma_t =", value="", help="If set, overrides constant g.")
                ubar_const = st.slider("Constant ū on Gamma_u", -50.0, 50.0, 0.0, 0.5, help="Dirichlet boundary value (scalar).")
                ubar_expr = st.text_input("Or expr ū(x,y,z) on Gamma_u =", value="", help="If set, overrides constant ū.")
        elif problem_type == "Screened Poisson equation (2D)":
            st.subheader("Screened Poisson equation terms")
            with st.expander("f, k², g and ū", expanded=True):
                f_expr = st.text_input("f(x,y) in Ω =", value="0", help="Body forcing.")
                k_squared = st.slider("k² (wave number squared)", 0.0, 100.0, 1.0, 0.1, help="Screened Poisson equation parameter k².")
                g_const = st.slider("Constant g on Gamma_t", -50.0, 50.0, 0.0, 0.5, help="Neumann/flux value (scalar).")
                g_expr = st.text_input("Or expr g(x,y) on Gamma_t =", value="", help="If set, overrides constant g.")
                ubar_const = st.slider("Constant ū on Gamma_u", -50.0, 50.0, 0.0, 0.5, help="Dirichlet boundary value (scalar).")
                ubar_expr = st.text_input("Or expr ū(x,y) on Gamma_u =", value="", help="If set, overrides constant ū.")
        elif problem_type == "Custom (user-defined)":
            st.subheader("Custom Dirichlet ū on Gamma_u")
            _codim = int(st.session_state.get("custom_out_dim", 1))
            if _codim <= 1:
                ubar_const = st.slider("Constant ū on Gamma_u", -50.0, 50.0, 0.0, 0.5)
                ubar_expr = st.text_input("Or expr ū(x,y,z) on Gamma_u =", value="")
            else:
                ubarx_const = st.slider("Constant ūx", -10.0, 10.0, 0.0, 0.1)
                ubary_const = st.slider("Constant ūy", -10.0, 10.0, 0.0, 0.1)
                ubarx_expr = st.text_input("Or expr ūx(x,y,z) =", value="")
                ubary_expr = st.text_input("Or expr ūy(x,y,z) =", value="")
                if _codim >= 3:
                    ubarz_const = st.slider("Constant ūz", -10.0, 10.0, 0.0, 0.1)
                    ubarz_expr = st.text_input("Or expr ūz(x,y,z) =", value="")
        else:
            st.subheader("Mechanics Dirichlet ū on Gamma_u")
            # (No expander) — keep always visible
            if problem_type == "Linear Elasticity (3D)":
                ubarx_const = st.slider("Constant ūx", -10.0, 10.0, 0.0, 0.1)
                ubary_const = st.slider("Constant ūy", -10.0, 10.0, 0.0, 0.1)
                ubarz_const = st.slider("Constant ūz", -10.0, 10.0, 0.0, 0.1)
                ubarx_expr = st.text_input("Or expr ūx(x,y,z) =", value="", help="If set, overrides constant ūx.")
                ubary_expr = st.text_input("Or expr ūy(x,y,z) =", value="", help="If set, overrides constant ūy.")
                ubarz_expr = st.text_input("Or expr ūz(x,y,z) =", value="", help="If set, overrides constant ūz.")
            else:
                ubarx_const = st.slider("Constant ūx", -10.0, 10.0, 0.0, 0.1)
                ubary_const = st.slider("Constant ūy", -10.0, 10.0, 0.0, 0.1)
                ubarx_expr = st.text_input("Or expr ūx(x,y) =", value="", help="If set, overrides constant ūx.")
                ubary_expr = st.text_input("Or expr ūy(x,y) =", value="", help="If set, overrides constant ūy.")

        # Traction / natural BC on Gamma_t
        # - Mechanics problems: vector traction (tx,ty[,tz])
        # - Custom (user-defined): show here too; variables are exposed to π_t as tx/ty/tz
        _custom_out_dim_ui = int(st.session_state.get("custom_out_dim", 1) or 1) if problem_type == "Custom (user-defined)" else 0

        if problem_type in ["Linear Elasticity (2D)", "Neo-Hookean Hyperelasticity (2D)", "Linear Elasticity (3D)"]:
            st.subheader("Traction on Gamma_t")
            if problem_type == "Linear Elasticity (3D)":
                with st.expander("t(x,y,z) = (t_x, t_y, t_z)", expanded=True):
                    tx_const = st.slider("Constant t_x", -10.0, 10.0, 0.0, 0.1)
                    ty_const = st.slider("Constant t_y", -10.0, 10.0, 0.0, 0.1)
                    tz_const = st.slider("Constant t_z", -10.0, 10.0, 0.0, 0.1)
                    tx_expr = st.text_input("Or expr t_x(x,y,z) =", value="")
                    ty_expr = st.text_input("Or expr t_y(x,y,z) =", value="")
                    tz_expr = st.text_input("Or expr t_z(x,y,z) =", value="")
            else:
                with st.expander("t(x,y) = (t_x, t_y)", expanded=True):
                    tx_const = st.slider("Constant t_x", -10.0, 10.0, 10.0, 0.1)
                    ty_const = st.slider("Constant t_y", -10.0, 10.0, 0.0, 0.1)
                    tx_expr = st.text_input("Or expr t_x(x,y) =", value="")
                    ty_expr = st.text_input("Or expr t_y(x,y) =", value="")

        elif problem_type == "Custom (user-defined)":
            st.subheader("Traction / Neumann on Gamma_t (Custom)")
            st.caption("These values are exposed as variables in π_t: `tx`, `ty`, `tz`.")

            # For 2D meshes: out_dim is 1 or 2. For 3D meshes: out_dim is 1 or 3.
            _mesh_dim_ui = int(st.session_state.get("mesh_dim", int(st.session_state.get("geo_dim", 2) or 2)) or 2)
            if _mesh_dim_ui == 2:
                if int(_custom_out_dim_ui) >= 2:
                    with st.expander("t(x,y) = (t_x, t_y)", expanded=True):
                        tx_const = st.slider("Constant t_x", -10.0, 10.0, 10.0, 0.1)
                        ty_const = st.slider("Constant t_y", -10.0, 10.0, 0.0, 0.1)
                        tx_expr = st.text_input("Or expr t_x(x,y) =", value="")
                        ty_expr = st.text_input("Or expr t_y(x,y) =", value="")
                else:
                    with st.expander("t(x,y) scalar (use as `tx` in π_t)", expanded=True):
                        tx_const = st.slider("Constant t", -50.0, 50.0, 0.0, 0.5)
                        tx_expr = st.text_input("Or expr t(x,y) =", value="")
            else:
                if int(_custom_out_dim_ui) >= 3:
                    with st.expander("t(x,y,z) = (t_x, t_y, t_z)", expanded=True):
                        tx_const = st.slider("Constant t_x", -10.0, 10.0, 0.0, 0.1)
                        ty_const = st.slider("Constant t_y", -10.0, 10.0, 0.0, 0.1)
                        tz_const = st.slider("Constant t_z", -10.0, 10.0, 0.0, 0.1)
                        tx_expr = st.text_input("Or expr t_x(x,y,z) =", value="")
                        ty_expr = st.text_input("Or expr t_y(x,y,z) =", value="")
                        tz_expr = st.text_input("Or expr t_z(x,y,z) =", value="")
                else:
                    with st.expander("t(x,y,z) scalar (use as `tx` in π_t)", expanded=True):
                        tx_const = st.slider("Constant t", -50.0, 50.0, 0.0, 0.5)
                        tx_expr = st.text_input("Or expr t(x,y,z) =", value="")

        # Body force (kept for built-in mechanics problems)
        if problem_type in ["Linear Elasticity (2D)", "Neo-Hookean Hyperelasticity (2D)", "Linear Elasticity (3D)"]:
            st.subheader("Body force in Omega")
            if problem_type == "Linear Elasticity (3D)":
                with st.expander("b(x,y,z) = (b_x, b_y, b_z)", expanded=False):
                    bx_const = st.slider("Constant b_x", -10.0, 10.0, 0.0, 0.1)
                    by_const = st.slider("Constant b_y", -10.0, 10.0, 0.0, 0.1)
                    bz_const = st.slider("Constant b_z", -10.0, 10.0, 0.0, 0.1)
                    bx_expr = st.text_input("Or expr b_x(x,y,z) =", value="")
                    by_expr = st.text_input("Or expr b_y(x,y,z) =", value="")
                    bz_expr = st.text_input("Or expr b_z(x,y,z) =", value="")
            else:
                with st.expander("b(x,y) = (b_x, b_y)", expanded=False):
                    bx_const = st.slider("Constant b_x", -10.0, 10.0, 0.0, 0.1)
                    by_const = st.slider("Constant b_y", -10.0, 10.0, 0.0, 0.1)
                    bx_expr = st.text_input("Or expr b_x(x,y) =", value="")
                    by_expr = st.text_input("Or expr b_y(x,y) =", value="")

    # ------------------------------------------------------------
    # Material + loads
    # ------------------------------------------------------------
    with st.expander("🧬 Material", expanded=False):
        problem_type = st.selectbox(
            "Problem",
            _problem_opts,
            index=_problem_opts.index(_prev_problem) if _prev_problem in _problem_opts else (1 if len(_problem_opts) > 1 else 0),
            key="problem_type",
        )

        # Defaults
        lam = mu = 0.0
        plane_mode = "plane_strain"
        thickness = 1.0
        E = 100.0
        nu = 0.3
        # Custom functional defaults
        custom_out_dim = int(st.session_state.get("custom_out_dim", 1))
        custom_pi_omega_expr = str(st.session_state.get("custom_pi_omega_expr", "") or "")
        custom_pi_gu_expr = str(st.session_state.get("custom_pi_gu_expr", "") or "")
        custom_pi_gt_expr = str(st.session_state.get("custom_pi_gt_expr", "") or "")
        custom_pi_gt_mode = str(st.session_state.get("custom_pi_gt_mode", "extra") or "extra")
        if custom_pi_gt_mode not in ("extra", "replace"):
            custom_pi_gt_mode = "extra"

        if problem_type in ["Linear Elasticity (2D)", "Neo-Hookean Hyperelasticity (2D)"]:
            st.subheader("Material (E, ν)")
            plane_mode = st.selectbox("Mode", ["plane_strain", "plane_stress"], index=1)
            if plane_mode == "plane_strain":
                thickness = st.number_input("Thickness (out-of-plane)", min_value=1e-6, value=1.0, step=0.1)
            else:
                thickness = 1.0
            E = st.slider("E", 1.0, 300.0, 100.0, 1.0)
            nu = st.slider("ν", 0.0, 0.49, 0.30, 0.01)
            lam, mu = lame_from_E_nu(float(E), float(nu), plane_mode)
        elif problem_type == "Linear Elasticity (3D)":
            st.subheader("Material (E, ν) — 3D")
            plane_mode = "3d"
            thickness = 1.0
            E = st.slider("E", 1.0, 300.0, 100.0, 1.0)
            nu = st.slider("ν", 0.0, 0.49, 0.30, 0.01)
            lam = float(E) * float(nu) / ((1.0 + float(nu)) * (1.0 - 2.0 * float(nu)))
            mu = float(E) / (2.0 * (1.0 + float(nu)))
        elif problem_type == "Custom (user-defined)":
            st.subheader("Custom functional (advanced)")
            st.caption(
                "For Custom: boundary conditions are handled the same way as predefined problems "
                "(Γu via Hard/Penalty in Boundary conditions; Γt via the traction/Neumann inputs). "
                "The Γu/Γt expressions below are *extra optional terms*—usually keep them as 0 to avoid double-counting."
            )

            # Output dimension controls the NN output: 1→scalar u, 2→(u,v), 3→(u,v,w)
            # For 2D meshes: allow scalar or 2-vector. For 3D meshes: allow scalar or 3-vector.
            _od_opts = [1, 3] if int(_dim_for_problem_ui) == 3 else [1, 2]
            custom_out_dim = int(
                st.selectbox(
                    "Output dimension (NN output)",
                    _od_opts,
                    index=_od_opts.index(custom_out_dim) if custom_out_dim in _od_opts else 0,
                    key="custom_out_dim",
                    help="This sets the neural network output dimension. It also controls which variables exist (u / v / w).",
                )
            )

            st.markdown("**How it is used**")
            st.markdown(
                "- **Default (recommended)**: you only provide the *internal* term `π_Ω` (energy density in Ω).\n"
                "- **Dirichlet on Γu** is enforced by the existing Boundary-conditions settings (Hard / Penalty + `ū`).\n"
                "- **Neumann/traction on Γt** is enforced by the existing Boundary-conditions traction inputs (`tx/ty/tz`).\n"
                "- `π_u` / `π_t` below are **extra optional add-ons** (advanced). Keep them as `0` unless you know you need extra terms."
            )

            with st.expander("Examples", expanded=False):
                st.markdown("**Poisson-like (scalar, 2D)**")
                st.code(
                    "π_Ω: 0.5*(ux**2 + uy**2) - 1.0*u\n"
                    "π_u: 1e6*(u-ubar)**2   # optional penalty on Γu\n"
                    "π_t: 0",
                    language="text",
                )
                st.markdown("**Linear-elasticity-like (vector, 2D)**")
                st.code(
                    "π_Ω: 0.5*lam*(tr_eps**2) + mu*(eps_xx**2 + eps_yy**2 + 2*eps_xy**2)\n"
                    "π_u: 0\n"
                    "π_t: 0   # Γt traction is set in Boundary conditions (tx/ty)",
                    language="text",
                )

            with st.expander("Available symbols in expressions", expanded=False):
                st.markdown(
                    "- Coordinates: `x`, `y`, `z` (z is None/ignored for 2D meshes)\n"
                    "- Scalars: `u`, and its grads `ux`, `uy`, `uz`\n"
                    "- Vectors (when output dim ≥ 2): `v`, grads `vx`, `vy`, `vz`\n"
                    "- Vectors (when output dim ≥ 3): `w`, grads `wx`, `wy`, `wz`\n"
                    "- 2D strain helpers (when output dim ≥ 2 and mesh dim==2): `eps_xx`, `eps_yy`, `eps_xy`, `tr_eps`\n"
                    "- On Γt only (optional): `tx`, `ty`, `tz` from the Boundary-condition traction inputs\n"
                    "- On Γu only (optional): scalar `ubar` (if out_dim=1) or `ubarx/ubary/ubarz` (if out_dim≥2)\n"
                    "- Material params from this panel (if you reference them): `lam`, `mu`, `E`, `nu`\n"
                    "- Functions: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `tanh`, `sinh`, `cosh`, `where`, `maximum`, `minimum`, `pi`"
                )

            st.text_area(
                "π_Ω(x,y,z,...) integrand in Ω",
                value=custom_pi_omega_expr or "0",
                key="custom_pi_omega_expr",
                height=90,
                help="Domain integrand. Example (Poisson): 0.5*(ux**2+uy**2) - f*u  (here you would hardcode f or extend vars).",
            )

            with st.expander("Advanced: extra boundary terms (usually keep 0)", expanded=False):
                st.caption(
                    "These are **optional add-ons**. If you already set Γu/Γt in Boundary conditions, "
                    "do NOT repeat the same terms here in **Extra** mode (otherwise you will double-count boundary work/penalty)."
                )
                pi_gt_mode = st.radio(
                    "π_t (Γt) mode",
                    options=["extra", "replace"],
                    index=0,
                    key="custom_pi_gt_mode",
                    help=(
                        "**Extra** (default): π_t is added on top of built-in traction (tx/ty/tz). "
                        "**Replace**: π_t fully specifies the Γt external work; built-in traction is skipped."
                    ),
                    horizontal=True,
                )
                st.caption(
                    "**Extra**: π_t adds to built-in traction. **Replace**: π_t overrides built-in; use π_t alone for Γt."
                )
                st.markdown(
                    "**External work term hint (Γt):** Typical traction work: `-(tx*u)` (scalar) or `-(tx*u + ty*v + tz*w)` (vector). "
                    "`tx`/`ty`/`tz` come from Boundary conditions.\n\n"
                )
                st.text_area(
                    "π_u(x,y,z,...) extra integrand on Γu",
                    value=custom_pi_gu_expr or "0",
                    key="custom_pi_gu_expr",
                    height=70,
                    help="Added on top of built-in Hard/Penalty BC on Gamma_u. Keep 0 for normal use.",
                )
                st.text_area(
                    "π_t(x,y,z,...) integrand on Γt",
                    value=custom_pi_gt_expr or "0",
                    key="custom_pi_gt_expr",
                    height=70,
                    help=(
                        "Extra mode: add-on to built-in traction. Replace mode: full Γt external work (no tx/ty/tz integral)."
                    ),
                )

    # ------------------------------------------------------------
    # Network
    # ------------------------------------------------------------
    with st.expander("🧩 Network", expanded=False):
        net_type = st.selectbox("Network type", ["MLP", "KAN"], index=0)
        num_layers = st.slider("Hidden Layers", 2, 10, 4)
        num_neurons = st.slider("Neurons per Layer", 10, 300, 64, 8)
        activation = st.selectbox("Activation", ["tanh", "silu", "gelu"], index=0)
        kan_grid = st.slider("KAN grid size", 4, 32, 8, 1) if net_type == "KAN" else 8
        kan_degree = st.select_slider("KAN spline degree", options=[1, 2, 3], value=3) if net_type == "KAN" else 3

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    with st.expander("🧪 Training", expanded=False):
        learning_rate = st.slider("Learning Rate", 1e-4, 2e-2, 1e-3, 1e-4, format="%.4f")
        num_epochs = st.slider("Epochs", 100, 10000, 800, 100)
        log_every = st.slider("Log every", 10, 500, 100, 10)

        st.subheader("⏹️ Abort on spike (rollback best)")
        abort_on_spike = st.checkbox("Abort on spike (rollback best)", value=True)
        spike_rtol = st.number_input(
            "Spike rtol (abort if rel change > this)",
            min_value=0.0,
            value=1e-5,
            format="%.1e",
        )
        spike_warmup = st.slider("Spike warmup (epochs)", 0, 5000, 200, 50)
        stable_rel_rtol = st.number_input(
            "Stable rel tol (rel change < this)",
            min_value=0.0,
            value=1e-4,
            format="%.1e",
        )
        stable_rel_patience = st.slider("Stable rel patience (epochs)", 1, 500, 10, 1)

        st.subheader("🧮 Batching (speed/memory)")
        dom_batch = st.slider("Domain quad batch", 500, 200000, 30000, 500)
        bnd_u_batch = st.slider("Gamma_u batch", 50, 80000, 8000, 50)
        bnd_t_batch = st.slider("Gamma_t batch", 50, 80000, 8000, 50)

        st.subheader("🖥️ Runtime")
        seed = st.number_input("Random Seed", min_value=0, max_value=100000, value=1234, step=1)
        use_cuda = st.checkbox("Use CUDA if available", value=True)
        device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        st.write(f"**Device:** `{device}`")

# ============================================================
# Main info panel
# ============================================================
# Shared summary values (used by both main + Details)
in_dim = int(st.session_state.get("mesh_dim", 2))
if problem_type in ["Poisson (scalar)", "Screened Poisson equation (2D)"]:
    out_dim = 1
elif problem_type == "Linear Elasticity (3D)":
    out_dim = 3
elif problem_type == "Custom (user-defined)":
    out_dim = int(st.session_state.get("custom_out_dim", 1) or 1)
else:
    out_dim = 2
layers = [in_dim] + [num_neurons] * num_layers + [out_dim]
_dir_mode_eff = st.session_state.get("dir_mode_effective", dir_mode)

with st.expander("Details", expanded=False):
    st.markdown(f"**Problem**: {problem_type}")

    # Problem description (compact)
    if problem_type == "Poisson (scalar)":
        st.markdown("Dirichlet boundary: **Gamma_u**.")
    elif problem_type == "Screened Poisson equation (2D)":
        st.markdown("Screened Poisson equation: $-\\nabla^2 u + k^2 u = f$. Dirichlet: **Gamma_u**. Neumann: **Gamma_t**.")
    elif problem_type == "Linear Elasticity (2D)":
        st.markdown("Linear elasticity (2D). Dirichlet: **Gamma_u**. Traction: **Gamma_t**.")
    elif problem_type == "Linear Elasticity (3D)":
        st.markdown("Linear elasticity (3D). Dirichlet: **Gamma_u**. Traction: **Gamma_t**.")
    elif problem_type == "Custom (user-defined)":
        st.markdown("Custom functional (user-defined). Dirichlet: **Gamma_u**. Boundary: **Gamma_t** (optional).")
    else:
        st.markdown("Neo-Hookean (compressible, 2D demo). Dirichlet: **Gamma_u**. Traction: **Gamma_t**.")

    # Equation (optional)
    _show_equation = st.toggle("Show equation", value=False, key="ui_show_equation")
    if _show_equation:
        if problem_type == "Poisson (scalar)":
            st.latex(
                r"\Pi(u)=\int_{\Omega}\frac12\|\nabla u\|^2\,d\Omega - \int_{\Omega} f u\,d\Omega - \int_{\Gamma_t} g u\,d\Gamma + W_{bc}"
            )
            st.markdown("**Strong form**")
            st.latex(r"-\Delta u = f \quad \text{in }\Omega")
            st.latex(r"u=\bar u \quad \text{on }\Gamma_u")
            st.latex(r"\frac{\partial u}{\partial n}=g \quad \text{on }\Gamma_t")
        elif problem_type == "Screened Poisson equation (2D)":
            st.latex(
                r"\Pi(u)=\int_{\Omega}\frac12\|\nabla u\|^2\,d\Omega + \int_{\Omega}\frac12 k^2 u^2\,d\Omega - \int_{\Omega} f u\,d\Omega - \int_{\Gamma_t} g u\,d\Gamma + W_{bc}"
            )
            st.markdown("**Strong form**")
            st.latex(r"-\Delta u + k^2 u = f \quad \text{in }\Omega")
            st.latex(r"u=\bar u \quad \text{on }\Gamma_u")
            st.latex(r"\frac{\partial u}{\partial n}=g \quad \text{on }\Gamma_t")
        elif problem_type == "Linear Elasticity (2D)":
            st.latex(
                r"\Pi(\mathbf{u})=\int_{\Omega}\psi(\varepsilon)\,d\Omega - \int_{\Omega}\mathbf{b}\cdot\mathbf{u}\,d\Omega - \int_{\Gamma_t}\mathbf{t}\cdot\mathbf{u}\,d\Gamma + W_{bc}"
            )
            st.markdown("**Strong form**")
            st.latex(r"-\nabla\cdot\boldsymbol{\sigma}(\mathbf{u})=\mathbf{b}\quad \text{in }\Omega")
            st.latex(r"\mathbf{u}=\bar{\mathbf{u}}\quad \text{on }\Gamma_u")
            st.latex(r"\boldsymbol{\sigma}\cdot(\mathbf{u})\,\mathbf{n}=\mathbf{t}\quad \text{on }\Gamma_t")
        elif problem_type == "Custom (user-defined)":
            st.latex(r"\Pi=\int_{\Omega}\pi_{\Omega}\,d\Omega + \int_{\Gamma_u}\pi_u\,d\Gamma + \int_{\Gamma_t}\pi_t\,d\Gamma")
            st.markdown("**Strong form**")
            st.caption("Depends on the chosen functional. In this app, boundary conditions are enforced via the existing Boundary-conditions controls (Γu: Hard/Penalty, Γt: traction/Neumann inputs).")
        else:
            st.latex(
                r"\Pi(\mathbf{u})=\int_{\Omega}\psi(\mathbf{F})\,d\Omega - \int_{\Omega}\mathbf{b}\cdot\mathbf{u}\,d\Omega - \int_{\Gamma_t}\mathbf{t}\cdot\mathbf{u}\,d\Gamma + W_{bc}"
            )
            st.markdown("**Strong form**")
            st.latex(r"-\nabla\cdot\mathbf{P}(\mathbf{u})=\mathbf{b}\quad \text{in }\Omega")
            st.latex(r"\mathbf{u}=\bar{\mathbf{u}}\quad \text{on }\Gamma_u")
            st.latex(r"\mathbf{P}(\mathbf{u})\,\cdot\mathbf{n}=\mathbf{t}\quad \text{on }\Gamma_t")

    # Model info (chatGPT-like summary)
    st.markdown("**Model Info**")
    _line2_parts = [f"Act: `{activation}`", f"BC: `{_dir_mode_eff}`"]
    if problem_type == "Screened Poisson equation (2D)":
        _line2_parts.append(f"k²={k_squared:.3g}")
    elif problem_type != "Poisson (scalar)":
        _line2_parts.append(f"E={float(E):.2f}, ν={float(nu):.2f}")
        _line2_parts.append(f"λ={float(lam):.3g}, μ={float(mu):.3g}")

    st.markdown(f"- Arch: `{layers}`")
    st.markdown("- " + "  |  ".join(_line2_parts))
    st.markdown(f"- Quad: tri={tri_rule_name}, seg={int(seg_gauss_n)}, quad={int(quad_gauss_n)}")
    st.markdown(f"- Mode: `{plane_mode}`  |  Input dim: `{in_dim}`  |  Output dim: `{out_dim}`")

st.markdown("---")

# ============================================================
# GEO text “chat” panel -> generate & load mesh (optional)
# ============================================================
with st.expander("Ready when you are（LLM for geometry generation）", expanded=True):
    # ------------------------------------------------------------
    # Chat sessions (ChatGPT-like)
    # ------------------------------------------------------------
    if "geo_chats" not in st.session_state:
        now = time.time()
        st.session_state["geo_chats"] = [
            {
                "id": f"chat_{int(now*1000)}",
                "title": time.strftime("New chat %m-%d %H:%M", time.localtime(now)),
                "created_ts": now,
                "messages": [],
                "geo_text": "",
                "last_llm_nl": "",
            }
        ]
        st.session_state["geo_chat_active_id"] = st.session_state["geo_chats"][0]["id"]

    if "geo_chat_active_id" not in st.session_state:
        st.session_state["geo_chat_active_id"] = st.session_state["geo_chats"][0]["id"]

    def _geo_chat_find(chat_id: str):
        for i, c in enumerate(st.session_state.get("geo_chats", [])):
            if str(c.get("id")) == str(chat_id):
                return i, c
        return None, None

    def _geo_chat_save_current():
        idx, c = _geo_chat_find(st.session_state.get("geo_chat_active_id"))
        if c is None:
            return
        # Keep active chat state in sync
        c["messages"] = list(st.session_state.get("geo_messages", []))
        c["geo_text"] = str(st.session_state.get("geo_text", "") or "")
        c["last_llm_nl"] = str(st.session_state.get("last_llm_nl", "") or "")
        st.session_state["geo_chats"][idx] = c

    def _geo_chat_set_active(chat_id: str):
        _geo_chat_save_current()
        idx, c = _geo_chat_find(chat_id)
        if c is None:
            return
        st.session_state["geo_chat_active_id"] = c["id"]
        st.session_state["geo_messages"] = c.get("messages", []) or []
        st.session_state["geo_text"] = c.get("geo_text", "") or ""
        st.session_state["last_llm_nl"] = c.get("last_llm_nl", "") or ""

    def _geo_chat_new():
        _geo_chat_save_current()
        now = time.time()
        new_chat = {
            "id": f"chat_{int(now*1000)}",
            "title": time.strftime("New chat %m-%d %H:%M", time.localtime(now)),
            "created_ts": now,
            "messages": [],
            "geo_text": "",
            "last_llm_nl": "",
        }
        st.session_state["geo_chats"] = [new_chat] + list(st.session_state.get("geo_chats", []))
        _geo_chat_set_active(new_chat["id"])
        st.rerun()

    def _geo_chat_autotitle_from_prompt(prompt: str, *, max_chars: int = 18) -> str:
        """
        Make a short, single-line chat title from the user's first prompt.
        Keeps it compact for the sidebar chat list.
        """
        s = str(prompt or "").strip()
        s = re.sub(r"\s+", " ", s)
        if not s:
            return ""
        # Prefer a short prefix; add ellipsis if truncated
        if len(s) > int(max_chars):
            return s[: int(max_chars)].rstrip() + "…"
        return s

    def _geo_chat_maybe_autotitle(prompt: str):
        """
        If the active chat still has the default 'New chat ...' title, rename it
        using the first user prompt.
        """
        idx, c = _geo_chat_find(st.session_state.get("geo_chat_active_id"))
        if c is None:
            return
        title0 = str(c.get("title", "") or "")
        is_default = title0.strip().lower().startswith("new chat")
        # Only auto-title when we are starting a fresh chat (no messages yet) or still default title.
        if not is_default and (c.get("messages") or []):
            return
        new_title = _geo_chat_autotitle_from_prompt(prompt, max_chars=18)
        if new_title:
            c["title"] = new_title
            st.session_state["geo_chats"][idx] = c

    # Ensure active chat is loaded into session_state
    _active_id = str(st.session_state.get("geo_chat_active_id"))
    _, _active_chat = _geo_chat_find(_active_id)
    if _active_chat is None:
        st.session_state["geo_chat_active_id"] = st.session_state["geo_chats"][0]["id"]
        _active_id = str(st.session_state.get("geo_chat_active_id"))
        _, _active_chat = _geo_chat_find(_active_id)
    if "geo_messages" not in st.session_state:
        st.session_state["geo_messages"] = _active_chat.get("messages", []) if _active_chat else []
    if "geo_text" not in st.session_state:
        st.session_state["geo_text"] = _active_chat.get("geo_text", "") if _active_chat else ""
    if "last_llm_nl" not in st.session_state:
        st.session_state["last_llm_nl"] = _active_chat.get("last_llm_nl", "") if _active_chat else ""

    # ---- default lc comes from sidebar ----
    default_lc = float(st.session_state.get("lc_ui", 0.15))
    default_geo = ("")

    # Layout: left chat list, right chat
    left, right = st.columns([1, 3], vertical_alignment="top")
    with left:
        if st.button("New chat", use_container_width=True, key="btn_geo_new_chat"):
            _geo_chat_new()

        chat_titles = [c.get("title", c.get("id", "chat")) for c in st.session_state.get("geo_chats", [])]
        chat_ids = [c.get("id") for c in st.session_state.get("geo_chats", [])]
        current_idx = chat_ids.index(st.session_state.get("geo_chat_active_id")) if st.session_state.get("geo_chat_active_id") in chat_ids else 0
        sel_id = st.radio(
            "Chats",
            options=chat_ids,
            index=current_idx,
            format_func=lambda cid: st.session_state["geo_chats"][chat_ids.index(cid)].get("title", str(cid)),
            key="geo_chat_picker",
            label_visibility="collapsed",
        )
        if str(sel_id) != str(st.session_state.get("geo_chat_active_id")):
            _geo_chat_set_active(str(sel_id))
            st.rerun()

    with right:
        # Minimal header; keep it compact like ChatGPT
        st.caption("Enter your request and press Enter: auto-generate/edit `.geo`, then auto-generate/load `.msh` (auto-repair on failure: 5 rounds × 2 repairs = 10 attempts).")
        if not _llm_ready:
            current_provider = st.session_state.get("llm_provider", "openai")
            env_var_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
                "ollama": "OLLAMA_BASE_URL",
            }
            env_var = env_var_map.get(current_provider, "OPENAI_API_KEY")
            st.info(
                f"LLM is disabled because `{env_var}` is not configured. "
                f"For Streamlit Cloud: put `{env_var}=\"...\"` in Secrets. "
                f"For other deployments: set environment variable `{env_var}`."
            )

        # Chat messages
        if not st.session_state.get("geo_messages"):
            st.caption("Example: `1x1 plate; left fixed (Gamma_u); right traction (Gamma_t); lc=0.1`")

        else:
            for m in st.session_state["geo_messages"]:
                role = m.get("role", "assistant")
                avatar = "🧑" if role == "user" else "🤖"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(m.get("content", ""))

        # Chat input (single, global chat input in Streamlit)
        user_msg = st.chat_input("Describe geometry + BC… (press Enter to send)", disabled=(not _llm_ready))
        prompt_to_send = (user_msg.strip() if isinstance(user_msg, str) else None)

    # ------------------------------------------------------------
    # Boundary placement directive (hidden settings; keep behavior)
    # ------------------------------------------------------------
    with st.expander("Chat settings", expanded=False):
        cbc1, cbc2, cbc3 = st.columns([1, 1, 1])
        with cbc1:
            use_geo_context = st.checkbox(
                "Use current .geo as baseline (LLM edits)",
                value=bool(st.session_state.get("use_geo_context", True)),
                key="use_geo_context",
            )

        if int(geo_dim) == 2:
            _bc_opts = ["auto", "xmin (left)", "xmax (right)", "ymin (bottom)", "ymax (top)"]
        else:
            _bc_opts = ["auto", "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]

        with cbc2:
            bc_gamma_u = st.selectbox("Gamma_u location", _bc_opts, index=0, key=f"bc_gamma_u_{int(geo_dim)}d")
        with cbc3:
            bc_gamma_t = st.selectbox("Gamma_t location", _bc_opts, index=0, key=f"bc_gamma_t_{int(geo_dim)}d")

    use_geo_context = bool(st.session_state.get("use_geo_context", True))
    bc_gamma_u = st.session_state.get(f"bc_gamma_u_{int(geo_dim)}d", "auto")
    bc_gamma_t = st.session_state.get(f"bc_gamma_t_{int(geo_dim)}d", "auto")

    def _bc_directive_text() -> str:
        if (bc_gamma_u == "auto") and (bc_gamma_t == "auto"):
            return (
                "\n\nBoundary placement directive:\n"
                "- Gamma_u: choose a reasonable boundary and state it explicitly.\n"
                "- Gamma_t: choose a reasonable boundary and state it explicitly.\n"
            )
        return (
            "\n\nBoundary placement directive:\n"
            f"- Gamma_u location: {bc_gamma_u}\n"
            f"- Gamma_t location: {bc_gamma_t}\n"
        )

    if prompt_to_send and (not _llm_ready):
        # Should be unreachable if chat_input is disabled, but keep it safe.
        current_provider = st.session_state.get("llm_provider", "openai")
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "ollama": "OLLAMA_BASE_URL",
        }
        env_var = env_var_map.get(current_provider, "OPENAI_API_KEY")
        st.session_state["geo_messages"].append(
            {"role": "assistant", "content": f"❌ {env_var} is not configured, so LLM calls are disabled."}
        )
        prompt_to_send = None

    if prompt_to_send:
        _geo_chat_maybe_autotitle(prompt_to_send)
        st.session_state["geo_messages"].append({"role": "user", "content": prompt_to_send})
        # Inject explicit boundary placement to avoid sticky defaults
        nl_for_llm = (prompt_to_send or "") + _bc_directive_text()
        base_geo_for_llm = st.session_state.get("geo_text", "") if bool(use_geo_context) else ""
        try:
            with st.spinner("LLM is generating .geo..."):
                chat_text, geo_candidate = llm_generate_geo_from_nl(
                    nl_for_llm,
                    default_lc=float(st.session_state.get("lc_ui", 0.15)),
                    geo_dim=int(geo_dim),
                    base_geo=base_geo_for_llm,
                    base_nl=st.session_state.get("last_llm_nl", ""),
                )
        except Exception as e:
            st.session_state["geo_messages"].append({"role": "assistant", "content": f"❌ LLM call failed: {e}"})
            st.rerun()

        ok_geo, msg = validate_geo_text(geo_candidate, require_gamma_t=(problem_type not in ["Poisson (scalar)", "Screened Poisson equation (2D)"]), dim=int(geo_dim))
        if not ok_geo:
            st.session_state["geo_messages"].append({"role": "assistant", "content": f"❌ LLM .geo validation failed: {msg}"})
            if chat_text and chat_text.strip():
                st.session_state["geo_messages"].append({"role": "assistant", "content": chat_text})
        else:
            st.session_state["geo_text"] = geo_candidate
            # keep "last_llm_nl" as the user's raw message (avoid accumulating directives)
            st.session_state["last_llm_nl"] = prompt_to_send
            if chat_text and chat_text.strip():
                st.session_state["geo_messages"].append({"role": "assistant", "content": chat_text})
            st.session_state["geo_messages"].append({"role": "assistant", "content": "✅ Generated `.geo` has been written into the editor below."})
            # Auto run "Generate .msh and load" after pressing Enter in LLM chat input.
            st.session_state["auto_geo_to_msh"] = True

        _geo_chat_save_current()
        st.rerun()


    # ------------------------------------------------------------
    # Tools are hidden by default (keep only chat UI visible)
    # ------------------------------------------------------------
    tools = st.expander("🧰 Tools (.geo editor / Gmsh / repair)", expanded=False)
    with tools:
        geo_text = st.text_area(
            "`.geo` editor",
            value=st.session_state.get("geo_text", default_geo),
            height=200,
        )
        st.session_state["geo_text"] = geo_text
        _geo_chat_save_current()

    # ------------------------------------------------------------
    # Import a .geo edited in Gmsh
    # ------------------------------------------------------------
    with tools:
        with st.expander("📥 Import tweaked .geo from file (edited in Gmsh)", expanded=False):
            st.caption("In Gmsh: File → Save As… → `*.geo`, then upload it here to replace the editor content.")
            geo_up = st.file_uploader("Upload a .geo file", type=["geo"], key="geo_upload_file")
            c_imp1, c_imp2 = st.columns([1, 1])
            with c_imp1:
                do_import_geo = st.button("⬅️ Load uploaded .geo into editor", use_container_width=True, key="btn_import_geo")
            with c_imp2:
                st.caption("This overwrites the current `.geo` editor content.")

            if do_import_geo:
                if geo_up is None:
                    st.warning("Please upload a .geo file first.")
                else:
                    try:
                        b = geo_up.getvalue()
                        txt = b.decode("utf-8", errors="replace")
                        if not txt.strip():
                            st.warning("Uploaded .geo is empty.")
                        else:
                            st.session_state["geo_text"] = txt if txt.endswith("\n") else (txt + "\n")
                            st.session_state["geo_messages"].append({"role": "assistant", "content": f"✅ Imported `{geo_up.name}` into the `.geo` editor."})
                            _geo_chat_save_current()
                            st.rerun()
                    except Exception as e:
                        st.warning(f"Failed to import .geo: {e}")

    with tools:
        c_open1, c_open2, c_open3 = st.columns([1, 1, 1])
        with c_open1:
            do_open_geo = st.button("🧩 Open current .geo in Gmsh (GUI)", use_container_width=True, key="btn_open_geo_gmsh")
        with c_open2:
            st.caption("Tip: tweak geometry/Physical groups in Gmsh, then export .msh and upload it back here.")
        with c_open3:
            pass

        if do_open_geo:
            ok_open, msg_open = open_geo_in_gmsh_gui(
                st.session_state.get("geo_text", ""),
                gmsh_cmdline=str(gmsh_cmdline),
                geo_dim=int(geo_dim),
                extra_args=str(geo_extra_args),
            )
            if ok_open:
                st.success(msg_open)
            else:
                st.error(msg_open)

    # ------------------------------------------------------------
    # Manual repair/regenerate using pasted error log
    # ------------------------------------------------------------
    if "geo_manual_error_log" not in st.session_state:
        st.session_state["geo_manual_error_log"] = ""
    with tools:
        with st.expander("🛠️ Manual repair/regenerate .geo (paste error log)", expanded=False):
            st.caption("If the LLM-generated geometry is wrong, paste the Gmsh error log / your description here and let the model iteratively fix the current `.geo`. If needed, regenerate a brand-new `.geo` based on the error.")
            manual_log = st.text_area(
                "Error log / description",
                key="geo_manual_error_log",
                height=160,
                placeholder="Paste Gmsh log here, e.g.\nError   : Curve loop is not closed\n...\nOr describe the issue, e.g. 'Gamma_t is not selected on the right face'.",
            )
            bfix1, bfix2, bfix3 = st.columns([1, 1, 1])
        with bfix1:
            do_manual_fix = st.button("🔧 Repair current .geo from error log", use_container_width=True, key="btn_manual_fix_geo")
        with bfix2:
            do_manual_regen = st.button("🧬 Regenerate a new .geo from error log", use_container_width=True, key="btn_manual_regen_geo")
        with bfix3:
            do_manual_clear = st.button("🧹 Clear error input", use_container_width=True, key="btn_manual_clear_geo_log")

        if do_manual_clear:
            st.session_state["geo_manual_error_log"] = ""
            st.rerun()

        def _truncate(s: str, n: int = 6000) -> str:
            s = (s or "").strip()
            if len(s) <= n:
                return s
            return s[-n:]

        require_gt = (problem_type not in ["Poisson (scalar)", "Screened Poisson equation (2D)"])

        if do_manual_fix:
            if not str(manual_log).strip():
                st.warning("Please paste an error log/description first.")
            else:
                geo_in = st.session_state.get("geo_text", "") or ""
                # Include boundary placement directive so repairs can move Gamma_u/Gamma_t if needed
                log_in = _truncate(str(manual_log) + _bc_directive_text(), 8000)
                # Manual repair: 2 repairs per round, 5 rounds (same as auto-repair)
                _repairs_per_round = 2
                _max_regeneration_rounds = 5
                repaired_ok = False
                
                for round_num in range(1, _max_regeneration_rounds + 1):
                    if round_num > 1:
                        # Regenerate: completely fresh start, only use original prompt (no error info)
                        st.session_state["geo_messages"].append(
                            {"role": "assistant", "content": f"🔄 Round {round_num}/{_max_regeneration_rounds}: Regenerating from scratch (using original prompt only)..."}
                        )
                        try:
                            with st.spinner(f"Regenerating .geo (round {round_num}/{_max_regeneration_rounds})..."):
                                # Use only the original prompt, no error information (complete restart)
                                original_nl = st.session_state.get("last_llm_nl", "")
                                if original_nl and original_nl.strip():
                                    regen_prompt = original_nl + _bc_directive_text()
                                else:
                                    # Fallback if no original prompt available
                                    regen_prompt = (
                                        f"Generate a correct {int(geo_dim)}D Gmsh .geo file.\n"
                                        "Ensure it generates a valid mesh with proper Physical groups (Omega, Gamma_u, Gamma_t)."
                                    )
                                    regen_prompt = regen_prompt + _bc_directive_text()
                                chat_regen, geo_in = llm_generate_geo_from_nl(
                                    regen_prompt,
                                    default_lc=float(st.session_state.get("lc_ui", 0.15)),
                                    geo_dim=int(geo_dim),
                                    base_geo="",  # Start fresh - no previous .geo
                                    base_nl="",   # Start fresh - no previous NL context
                                )
                                if chat_regen and chat_regen.strip():
                                    st.session_state["geo_messages"].append({"role": "assistant", "content": chat_regen})
                                log_in = _truncate(str(manual_log) + _bc_directive_text(), 8000)
                        except Exception as e_regen:
                            st.session_state["geo_messages"].append(
                                {"role": "assistant", "content": f"❌ Regeneration failed: {str(e_regen)}"}
                            )
                            continue
                    
                    # Repair attempts within this round
                    for repair_attempt in range(1, _repairs_per_round + 1):
                        with st.spinner(f"Round {round_num}/{_max_regeneration_rounds}, Repair {repair_attempt}/{_repairs_per_round}..."):
                            chat_fix, geo_fixed = llm_fix_geo_with_gmsh_log(
                                geo_in,
                                log_in + f"\n\nRound {round_num}/{_max_regeneration_rounds}, Repair {repair_attempt}/{_repairs_per_round}. Return a FULL corrected .geo.",
                                geo_dim=int(geo_dim),
                            )
                        ok2, msg2 = validate_geo_text(geo_fixed, require_gamma_t=require_gt, dim=int(geo_dim))
                        st.session_state["geo_messages"].append(
                            {"role": "user", "content": f"[Manual repair] Round {round_num}, Repair {repair_attempt}/{_repairs_per_round}\n\n{_truncate(str(manual_log), 1200)}"}
                        )
                        if chat_fix and chat_fix.strip():
                            st.session_state["geo_messages"].append({"role": "assistant", "content": chat_fix})
                        if ok2:
                            st.session_state["geo_text"] = geo_fixed
                            st.session_state["geo_messages"].append({"role": "assistant", "content": f"✅ Repair succeeded (Round {round_num}, Repair {repair_attempt}): written back to the `.geo` editor."})
                            _geo_chat_save_current()
                            repaired_ok = True
                            st.rerun()
                        else:
                            st.session_state["geo_messages"].append({"role": "assistant", "content": f"❌ Round {round_num}, Repair {repair_attempt}: Invalid .geo: {msg2}"})
                            geo_in = geo_fixed or geo_in
                            log_in = _truncate(str(manual_log) + f"\n\nValidation failed: {msg2}\nPlease fix accordingly.", 8000)
                    
                    if repaired_ok:
                        break

                if not repaired_ok:
                    st.error(f"Repair failed after {_max_regeneration_rounds} rounds × {_repairs_per_round} repairs = {_max_regeneration_rounds * _repairs_per_round} attempts. Try 'Regenerate .geo' (or paste a fuller Gmsh log).")

        if do_manual_regen:
            if not str(manual_log).strip():
                st.warning("Please paste an error log/description first.")
            else:
                with st.spinner("LLM regenerating a new .geo from error log..."):
                    prompt = (
                        f"Please regenerate a correct {int(geo_dim)}D Gmsh .geo that satisfies the Physical group requirements.\n"
                        f"Error log / description (may be partial):\n{_truncate(str(manual_log), 8000)}\n\n"
                        "If the error suggests missing Physical entities, fix them. Prefer a minimal, robust geometry."
                    )
                    prompt = prompt + _bc_directive_text()
                    chat_new, geo_new = llm_generate_geo_from_nl(
                        prompt,
                        default_lc=float(st.session_state.get("lc_ui", 0.15)),
                        geo_dim=int(geo_dim),
                        base_geo="",
                        base_nl=st.session_state.get("last_llm_nl", ""),
                    )
                ok3, msg3 = validate_geo_text(geo_new, require_gamma_t=require_gt, dim=int(geo_dim))
                st.session_state["geo_messages"].append(
                    {"role": "user", "content": f"[Manual regenerate]\n\n{_truncate(str(manual_log), 1200)}"}
                )
                if chat_new and chat_new.strip():
                    st.session_state["geo_messages"].append({"role": "assistant", "content": chat_new})
                if ok3:
                    st.session_state["geo_text"] = geo_new
                    st.session_state["geo_messages"].append({"role": "assistant", "content": "✅ Regenerated a new `.geo` and wrote it into the editor."})
                    _geo_chat_save_current()
                    st.rerun()
                else:
                    st.session_state["geo_messages"].append({"role": "assistant", "content": f"❌ Regenerated `.geo` is still invalid: {msg3}"})
                    st.error(f"Regenerated .geo invalid: {msg3}")

    with tools:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            do_gen = st.button("🧱 Generate .msh and load", key="btn_geo_gen")
        with c2:
            do_clear = st.button("🧹 Clear generated mesh", key="btn_geo_clear")
        with c3:
            st.download_button(
                "⬇️ Download .geo",
                data=(geo_text or "").encode("utf-8"),
                file_name="model.geo",
                mime="text/plain",
            )

    if "generated_msh_bytes" in st.session_state:
        msh_bytes0 = st.session_state["generated_msh_bytes"]
        msh_name0 = st.session_state.get("generated_msh_name", "generated_from_geo.msh")
        st.success(f"Generated mesh is available in session: **{msh_name0}** ({len(msh_bytes0):,} bytes)")
        st.download_button(
            "⬇️ Download generated .msh",
            data=msh_bytes0,
            file_name=msh_name0,
            mime="application/octet-stream",
        )

    if do_clear:
        st.session_state.pop("generated_msh_bytes", None)
        st.session_state.pop("generated_msh_name", None)
        st.session_state.pop("mesh_sig", None)
        # Also reset LLM chat/context to start a new conversation.
        st.session_state.pop("geo_text", None)
        st.session_state.pop("nl_prompt", None)
        st.session_state.pop("last_llm_nl", None)
        st.session_state["geo_messages"] = []
        _geo_chat_save_current()
        st.rerun()

    _auto_gen = bool(st.session_state.pop("auto_geo_to_msh", False))
    _auto_retry = bool(st.session_state.pop("auto_retry_mesh_error", False))
    if do_gen or _auto_gen or _auto_retry:
        # NOTE: gmsh_cmdline / gmsh_extra_args are bound to sidebar widgets via keys.
        # Do not manually write them into session_state here, otherwise Streamlit raises
        # StreamlitAPIException ("cannot set session_state for a widget key").
        _repairs_per_round = 2  # Repair attempts per regeneration round
        _max_regeneration_rounds = 5  # Maximum number of regeneration rounds
        if _auto_retry:
            mesh_err = st.session_state.pop("mesh_error_msg", "Unknown mesh error")
            st.session_state["geo_messages"].append({
                "role": "user", 
                "content": f"Auto-retry: Mesh load failed with error: {mesh_err}. Please regenerate a correct .geo file."
            })
        elif _auto_gen:
            st.session_state["geo_messages"].append({"role": "user", "content": "Auto: generate .msh and load from the newly generated .geo."})
        else:
            st.session_state["geo_messages"].append({"role": "user", "content": "Generate mesh from current .geo text."})
        # sync lc from UI into geo_text before validate+run gmsh
        geo_text = upsert_lc_in_geo(geo_text, float(st.session_state.get("lc_ui", 0.15)))
        st.session_state["geo_text"] = geo_text
        
        # For auto-retry, skip validation and Gmsh run, go directly to regeneration
        if _auto_retry:
            # Set up error message for regeneration
            mesh_err = st.session_state.get("mesh_error_msg", "Mesh load/build error")
            err_text = mesh_err
            initial_nl = st.session_state.get("last_llm_nl", "")
            if not initial_nl:
                initial_nl = "Regenerate a correct .geo file that produces a valid mesh."
            repaired_ok = False
            st.warning(f"Mesh error detected. Trying automatic regeneration (LLM, {_max_regeneration_rounds} rounds × {_repairs_per_round} repairs = up to {_max_regeneration_rounds * _repairs_per_round} attempts)...")
        else:
            # ---- validate before running gmsh ----
            ok_geo, msg = validate_geo_text(geo_text, require_gamma_t=(problem_type not in ["Poisson (scalar)", "Screened Poisson equation (2D)"]), dim=int(geo_dim))
            if not ok_geo:
                st.error(f".geo validation failed: {msg}")
                st.session_state["geo_messages"].append({"role": "assistant", "content": f"❌ .geo validation failed: {msg}"})
                st.stop()

            def _run_once(geo_in: str):
                msh_bytes, gmsh_log = gmsh_geo_to_msh_bytes(
                    geo_in,
                    gmsh_cmdline=gmsh_cmdline,
                    dim=int(geo_dim),
                    msh_format=str(geo_msh_format),
                    extra_args=str(geo_extra_args),
                    timeout_sec=180,
                )
                return msh_bytes, gmsh_log

            try:
                with st.spinner("Running Gmsh to generate .msh..."):
                    msh_bytes, gmsh_log = _run_once(geo_text)

            except Exception as e1:
                err_text = str(e1)
                st.warning(f"Gmsh failed. Trying automatic repair (LLM, {_max_regeneration_rounds} rounds × {_repairs_per_round} repairs = up to {_max_regeneration_rounds * _repairs_per_round} attempts)...")

                initial_nl = st.session_state.get("last_llm_nl", "")
                repaired_ok = False

        # Regeneration/repair loop (for both auto-retry and Gmsh failures)
        if _auto_retry or 'repaired_ok' in locals():
            # Outer loop: regeneration rounds (restart with fresh .geo generation)
            # For auto-retry, always start with regeneration (round 1)
            for round_num in range(1, _max_regeneration_rounds + 1):
                if round_num > 1 or _auto_retry:
                    # Regenerate: completely fresh start, only use original prompt (no error info)
                    st.session_state["geo_messages"].append(
                        {"role": "assistant", "content": f"🔄 Round {round_num}/{_max_regeneration_rounds}: Regenerating from scratch (using original prompt only)..."}
                    )
                    try:
                        with st.spinner(f"Regenerating .geo (round {round_num}/{_max_regeneration_rounds})..."):
                            # Use only the original prompt, no error information (complete restart)
                            if initial_nl and initial_nl.strip():
                                regen_prompt = initial_nl + _bc_directive_text()
                            else:
                                # Fallback if no original prompt available
                                regen_prompt = (
                                    f"Generate a correct {int(geo_dim)}D Gmsh .geo file.\n"
                                    "Ensure it generates a valid mesh with proper Physical groups (Omega, Gamma_u, Gamma_t)."
                                )
                                regen_prompt = regen_prompt + _bc_directive_text()
                            chat_regen, geo_text = llm_generate_geo_from_nl(
                                regen_prompt,
                                default_lc=float(st.session_state.get("lc_ui", 0.15)),
                                geo_dim=int(geo_dim),
                                base_geo="",  # Start fresh - no previous .geo
                                base_nl="",   # Start fresh - no previous NL context
                            )
                            if chat_regen and chat_regen.strip():
                                st.session_state["geo_messages"].append({"role": "assistant", "content": chat_regen})
                            geo_text = upsert_lc_in_geo(geo_text, float(st.session_state.get("lc_ui", 0.15)))
                            st.session_state["geo_text"] = geo_text
                    except Exception as e_regen:
                        st.session_state["geo_messages"].append(
                            {"role": "assistant", "content": f"❌ Regeneration failed: {str(e_regen)}"}
                        )
                        continue

                # Inner loop: repair attempts within this round
                geo_try = geo_text
                gmsh_log_for_llm = err_text
                last_err = err_text

                for repair_attempt in range(1, _repairs_per_round + 1):
                    try:
                        with st.spinner(f"Round {round_num}/{_max_regeneration_rounds}, Repair {repair_attempt}/{_repairs_per_round}..."):
                            chat_fix, geo_fixed = llm_fix_geo_with_gmsh_log(
                                geo_try,
                                _truncate(
                                    gmsh_log_for_llm
                                    + _bc_directive_text()
                                    + f"\n\nRound {round_num}/{_max_regeneration_rounds}, Repair {repair_attempt}/{_repairs_per_round}. Fix the issues and return full .geo.",
                                    8000,
                                ),
                                geo_dim=int(geo_dim),
                            )

                        ok2, msg2 = validate_geo_text(geo_fixed, require_gamma_t=(problem_type != "Poisson (scalar)"), dim=int(geo_dim))
                        if not ok2:
                            st.session_state["geo_messages"].append(
                                {"role": "assistant", "content": f"❌ Round {round_num}, Repair {repair_attempt}/{_repairs_per_round}: Invalid .geo: {msg2}"}
                            )
                            geo_try = geo_fixed or geo_try
                            gmsh_log_for_llm = last_err + f"\n\nValidation failed: {msg2}"
                            continue

                        with st.spinner(f"Testing repaired .geo (Round {round_num}, Repair {repair_attempt}/{_repairs_per_round})..."):
                            msh_bytes, gmsh_log = _run_once(geo_fixed)

                        # success - store msh_bytes and gmsh_log for later use
                        st.session_state["geo_text"] = geo_fixed
                        if chat_fix and chat_fix.strip():
                            st.session_state["geo_messages"].append({"role": "assistant", "content": chat_fix})
                        st.session_state["geo_messages"].append(
                            {"role": "assistant", "content": f"✅ Success! Round {round_num}, Repair {repair_attempt}/{_repairs_per_round} succeeded."}
                        )
                        repaired_ok = True
                        # Store msh_bytes and gmsh_log in session_state so they're available after the loop
                        st.session_state["_temp_msh_bytes"] = msh_bytes
                        st.session_state["_temp_gmsh_log"] = gmsh_log
                        break

                    except Exception as e2:
                        last_err = str(e2)
                        st.session_state["geo_messages"].append(
                            {
                                "role": "assistant",
                                "content": f"❌ Round {round_num}, Repair {repair_attempt}/{_repairs_per_round} failed:\n{_truncate(last_err, 1800)}",
                            }
                        )
                        gmsh_log_for_llm = last_err
                        geo_try = geo_fixed if 'geo_fixed' in locals() else geo_try

                if repaired_ok:
                    break

            if not repaired_ok:
                st.error(
                    f"❌ Automatic repair failed after {_max_regeneration_rounds} rounds × {_repairs_per_round} repairs = {_max_regeneration_rounds * _repairs_per_round} total attempts. "
                    "Use the manual repair panel to paste the full log and retry."
                )
                st.stop()
        
        # ---- success path ----
        # Get msh_bytes and gmsh_log from either direct generation or repair loop
        if 'repaired_ok' in locals() and repaired_ok:
            # Success from repair loop - use stored values
            msh_bytes = st.session_state.pop("_temp_msh_bytes", None)
            gmsh_log = st.session_state.pop("_temp_gmsh_log", "")
        elif not _auto_retry:
            # Success from direct generation - use existing values
            pass
        else:
            # Should not reach here, but handle gracefully
            st.error("Unexpected state in mesh generation")
            st.stop()
        
        if msh_bytes is not None:
            st.session_state["generated_msh_bytes"] = msh_bytes
            st.session_state["generated_msh_name"] = "generated_from_geo.msh"
            st.session_state.pop("mesh_sig", None)  # force reload
            # Auto-load the mesh so the preview appears immediately (no need to click Start Training).
            st.session_state["auto_load_mesh_after_gen"] = True

            st.session_state["geo_messages"].append(
                {"role": "assistant", "content": f"✅ Mesh generated: {len(msh_bytes):,} bytes\n\n```text\n{gmsh_log}\n```"}
            )
            _geo_chat_save_current()
            st.rerun()



# ============================================================
# Load mesh + precompute quadrature
# ============================================================
def ensure_mesh_loaded():
    # Source A: uploaded .msh
    file_name = None
    file_bytes = None
    if msh_file is not None:
        file_name = msh_file.name
        file_bytes = msh_file.getvalue()
    # Source B: generated from .geo
    elif "generated_msh_bytes" in st.session_state:
        file_name = st.session_state.get("generated_msh_name", "generated_from_geo.msh")
        file_bytes = st.session_state["generated_msh_bytes"]
    else:
        st.warning("Please upload a .msh file in the sidebar, or generate one from .geo text.")
        return False

    # (Improvement #4) lighter & stable signature
    sig = (
        file_name,
        stable_fingerprint(file_bytes),
        tri_rule_name,
        int(quad_gauss_n),
        int(seg_gauss_n),
        str(device),
        dir_mode,  # because dist_obj depends on this
        float(dist_tau),  # because dist_obj depends on this (for smooth hard-BC distance)
    )

    if "mesh_sig" in st.session_state and st.session_state["mesh_sig"] == sig:
        return True

    # If a previous CUDA kernel hit a device-side assert, CUDA enters an error state and the *next*
    # CUDA API call may fail "asynchronously" at an unrelated line. Detect early and provide guidance.
    if str(device).startswith("cuda"):
        try:
            torch.cuda.synchronize()
        except Exception as e:
            st.error(
                "CUDA is in an error state (often caused by an earlier out-of-bounds index on GPU). "
                "Please **restart the app** (or temporarily switch to CPU) and rerun.\n\n"
                f"Details: {e}"
            )
            return False

    # (Improvement #5) spinner feedback
    with st.spinner("Parsing .msh and building quadrature (cached when possible)..."):
        try:
            pts, tris, quads, seg_u, seg_t, Xdom, Wdom, Xu, Wu, Xt, Wt, omega_info = cached_load_and_quadrature(
                file_bytes, tri_rule_name, int(seg_gauss_n), int(quad_gauss_n)
            )
        except Exception as e:
            err_msg = str(e)
            st.error(f"Mesh load/build error: {err_msg}")
            
            # Auto-retry with LLM regeneration if:
            # 1. Mesh was generated from .geo (not uploaded .msh)
            # 2. Error is mesh-related (empty arrays, no domain elements, etc.)
            # 3. We have .geo text (either from LLM or manually pasted)
            is_mesh_error = any(keyword in err_msg.lower() for keyword in [
                "need at least one array",
                "no domain elements",
                "empty",
                "concatenate",
                "no triangles",
                "no quads",
                "vstack",
                "at least one array",
                "zero-size array",
                "reduction operation",
                "which has no identity",
                "cannot concatenate",
                "empty array",
                "no elements found",
                "no cells found"
            ])
            is_generated_mesh = "generated_msh_bytes" in st.session_state or file_name == st.session_state.get("generated_msh_name")
            # Check if we have .geo text (from LLM or manual paste)
            has_geo_text = bool(st.session_state.get("geo_text", "").strip())
            
            # For any mesh error from generated mesh, always try to regenerate
            if is_mesh_error and is_generated_mesh and has_geo_text:
                st.warning("🔄 Mesh error detected. Automatically regenerating .geo with LLM...")
                # Set flag to trigger automatic regeneration
                st.session_state["auto_retry_mesh_error"] = True
                st.session_state["mesh_error_msg"] = err_msg
                # If no LLM history, use the current .geo as context
                if not st.session_state.get("last_llm_nl", ""):
                    # Extract a description from the .geo if possible, or use a generic message
                    geo_text = st.session_state.get("geo_text", "")
                    if geo_text:
                        # Try to extract a simple description from comments or structure
                        # Look for comments in the .geo file
                        lines = geo_text.split('\n')
                        comments = [line.strip() for line in lines if line.strip().startswith('//')]
                        if comments:
                            # Use the first meaningful comment as description
                            desc = comments[0].replace('//', '').strip()[:100]
                            st.session_state["last_llm_nl"] = f"{desc}. Previous error: {err_msg[:150]}"
                        else:
                            st.session_state["last_llm_nl"] = f"Generate a correct .geo file that produces a valid mesh. Previous error: {err_msg[:200]}"
                # Trigger immediate rerun to start regeneration
                # Use st.rerun() to immediately trigger the regeneration logic
                st.rerun()
                return False
            elif is_generated_mesh and has_geo_text:
                # Even if error keywords don't match, if it's a generated mesh with .geo text,
                # try to regenerate (catch-all for mesh errors)
                st.warning("🔄 Mesh error detected. Automatically regenerating .geo with LLM...")
                st.session_state["auto_retry_mesh_error"] = True
                st.session_state["mesh_error_msg"] = err_msg
                if not st.session_state.get("last_llm_nl", ""):
                    geo_text = st.session_state.get("geo_text", "")
                    if geo_text:
                        lines = geo_text.split('\n')
                        comments = [line.strip() for line in lines if line.strip().startswith('//')]
                        if comments:
                            desc = comments[0].replace('//', '').strip()[:100]
                            st.session_state["last_llm_nl"] = f"{desc}. Previous error: {err_msg[:150]}"
                        else:
                            st.session_state["last_llm_nl"] = f"Generate a correct .geo file that produces a valid mesh. Previous error: {err_msg[:200]}"
                st.rerun()
                return False
            else:
                return False
    st.session_state["omega_info"] = omega_info
    mesh_dim = int(omega_info.get("dim", 2))
    st.session_state["mesh_dim"] = mesh_dim

    if mesh_dim == 3:
        # 3D mesh (Omega is a Volume): currently supports Poisson (scalar) and Linear Elasticity (3D).
        if problem_type not in ["Poisson (scalar)", "Linear Elasticity (3D)", "Custom (user-defined)"]:
            st.error(
                "3D mesh detected (Omega is a Volume). Current build supports **3D Poisson (scalar)**, "
                "**Linear Elasticity (3D)**, and **Custom (user-defined)**."
            )
            return False

    if mesh_dim == 2:
        if seg_t.size == 0 and problem_type != "Poisson (scalar)":
            st.warning("No 'Gamma_t' segments found. Traction work will be ~0.")
    else:
        # 3D path: seg_t is placeholder; traction surfaces are handled via Xt/Wt if present.
        if Xt is not None and np.asarray(Xt).shape[0] <= 1 and problem_type != "Poisson (scalar)":
            st.warning("No 'Gamma_t' surfaces found. Traction work will be ~0.")

    # Hard BC distance: 2D uses segments; 3D uses Gamma_u surface triangles.
    if "hard" in dir_mode:
        if mesh_dim == 2:
            try:
                dist_obj = DirichletDistance(pts, seg_u, device=device, tau=float(dist_tau))
            except Exception as e:
                # If CUDA is poisoned, this might be where it surfaces. Give actionable message.
                st.error(
                    "Failed to build Dirichlet distance object on the selected device. "
                    "If you see `device-side assert triggered`, restart the app or switch to CPU.\n\n"
                    f"Details: {e}"
                )
                return False
        else:
            gu_tris = np.asarray(omega_info.get("Gamma_u_tris_conn", np.zeros((0, 3), dtype=int)))
            if gu_tris.size == 0:
                st.warning("3D hard BC requested, but Gamma_u surface triangles are missing. Falling back to penalty BC.")
                dist_obj = None
                st.session_state["dir_mode_effective"] = "penalty"
            else:
                try:
                    dist_obj = DirichletDistance3D(pts, gu_tris, device=device, tau=float(dist_tau), k=16, chunk=2048)
                except Exception as e:
                    st.error(
                        "Failed to build 3D Dirichlet distance object on the selected device. "
                        "If you see `device-side assert triggered`, restart the app or switch to CPU.\n\n"
                        f"Details: {e}"
                    )
                    return False
                st.session_state["dir_mode_effective"] = dir_mode
    else:
        dist_obj = None
        st.session_state["dir_mode_effective"] = dir_mode

    # Quick sanity check: distance on Gamma_u quadrature points should be ~0 in hard mode
    if dist_obj is not None and "hard" in st.session_state.get("dir_mode_effective", ""):
        try:
            Xu_np = np.asarray(Xu)
            if Xu_np.size > 0:
                Xu_s = Xu_np[: min(2000, Xu_np.shape[0])]
                Xu_t = to_torch(Xu_s.astype(np.float32), device=device, requires_grad=False)
                dx = Xu_t[:, 0:1]
                dy = Xu_t[:, 1:2]
                dz = Xu_t[:, 2:3] if mesh_dim == 3 else None
                with torch.no_grad():
                    dtest = dist_obj.distance(dx, dy, dz).detach().cpu().numpy().reshape(-1)
                st.info(f"Hard-BC distance sanity check on Gamma_u samples: mean={float(dtest.mean()):.3e}, max={float(dtest.max()):.3e}")
        except Exception:
            pass

    st.session_state["mesh_sig"] = sig
    st.session_state["pts"] = pts
    st.session_state["tris"] = tris
    st.session_state["quads"] = quads
    st.session_state["seg_u"] = seg_u
    st.session_state["seg_t"] = seg_t
    st.session_state["Xdom"] = Xdom
    st.session_state["Wdom"] = Wdom
    st.session_state["Xu"] = Xu
    st.session_state["Wu"] = Wu
    st.session_state["Xt"] = Xt
    st.session_state["Wt"] = Wt
    st.session_state["dist_obj"] = dist_obj
    # Shape-function integration data (2D): needed for DEM without autograd derivatives.
    # Keep Xdom/Xu/Xt for compatibility; sf_* include connectivity + dN/dx.
    try:
        if int(mesh_dim) == 2:
            st.session_state["sf_dom_2d"] = build_sf_domain_data_2d(
                pts,
                tris,
                omega_info,
                quads,
                tri_rule_name=str(tri_rule_name),
                quad_gauss_n=int(quad_gauss_n),
            )
            st.session_state["sf_gu_2d"] = build_sf_boundary_segments_2d(pts, seg_u, gauss_n=int(seg_gauss_n))
            st.session_state["sf_gt_2d"] = (
                build_sf_boundary_segments_2d(pts, seg_t, gauss_n=int(seg_gauss_n))
                if (seg_t is not None and np.asarray(seg_t).size > 0)
                else {"X": np.zeros((0, 2), dtype=np.float32), "W": np.zeros((0, 1), dtype=np.float32), "conn": np.zeros((0, 2), dtype=np.int64), "N": np.zeros((0, 2), dtype=np.float32)}
            )
            # Validate connectivity bounds; invalid indices can trigger CUDA device-side asserts later.
            n_nodes = int(np.asarray(pts).shape[0])
            ok1, msg1 = _sf_conn_is_valid(st.session_state.get("sf_dom_2d", None), n_nodes)
            ok2, msg2 = _sf_conn_is_valid(st.session_state.get("sf_gu_2d", None), n_nodes)
            ok3, msg3 = _sf_conn_is_valid(st.session_state.get("sf_gt_2d", None), n_nodes)
            if not (ok1 and ok2 and ok3):
                st.warning(
                    "Shape-function cache invalid for current mesh (will fall back to autograd). "
                    f"sf_dom_2d={msg1}, sf_gu_2d={msg2}, sf_gt_2d={msg3}"
                )
                st.session_state.pop("sf_dom_2d", None)
                st.session_state.pop("sf_gu_2d", None)
                st.session_state.pop("sf_gt_2d", None)
        else:
            st.session_state.pop("sf_dom_2d", None)
            st.session_state.pop("sf_gu_2d", None)
            st.session_state.pop("sf_gt_2d", None)
    except Exception:
        # Don't block mesh loading; autograd mode still works.
        st.session_state.pop("sf_dom_2d", None)
        st.session_state.pop("sf_gu_2d", None)
        st.session_state.pop("sf_gt_2d", None)

    if mesh_dim == 2:
        if omega_info.get("has_quads", False):
            st.info("Omega contains quads: visualization uses internal triangulation; integration uses quad Gauss points.")
        st.success(
            f"Mesh loaded (2D): pts={pts.shape[0]}, plot-tris={tris.shape[0]}, "
            f"Omega(tri={omega_info.get('Omega_triangles',0)}, quad={omega_info.get('Omega_quads',0)}), "
            f"|Gamma_u| segs={seg_u.shape[0]}, |Gamma_t| segs={seg_t.shape[0] if seg_t.size else 0}\n"
            f"Quadrature: dom pts={Xdom.shape[0]}, Gamma_u pts={Xu.shape[0]}, Gamma_t pts={Xt.shape[0]}"
        )
    else:
        st.success(
            f"Mesh loaded (3D): pts={pts.shape[0]}, Omega(tets={omega_info.get('Omega_tets',0)}), "
            f"Gamma_u(tris={omega_info.get('Gamma_u_tris',0)}), Gamma_t(tris={omega_info.get('Gamma_t_tris',0)})\n"
            f"Quadrature: dom pts={Xdom.shape[0]}, Gamma_u pts={Xu.shape[0]}, Gamma_t pts={Xt.shape[0]}"
        )
    return True


# Auto-load after .geo→.msh generation (LLM or manual), so geometry preview shows immediately.
if bool(st.session_state.pop("auto_load_mesh_after_gen", False)):
    if "bc_mesh_load_ui" in locals():
        with bc_mesh_load_ui.container():
            ensure_mesh_loaded()
    else:
        ensure_mesh_loaded()


if "bc_mesh_load_ui" in locals():
    # Render mesh load controls inside the Boundary conditions sidebar section.
    with bc_mesh_load_ui.container():
        st.subheader("Load mesh")
        has_source = (msh_file is not None) or ("generated_msh_bytes" in st.session_state)
        if not has_source:
            st.caption("Upload a `.msh` in **Mesh (.msh)**, or generate one from `.geo` first.")
        else:
            colL, colR = st.columns([1, 1])
            with colL:
                load_clicked = st.button("Load / Reload", use_container_width=True, key="btn_load_mesh_bc")
            with colR:
                st.caption("Parses `.msh` + builds quadrature.")

            if load_clicked:
                st.session_state.pop("mesh_sig", None)  # force reload

            # Only load when user explicitly clicks (keeps main page clean).
            if load_clicked:
                ensure_mesh_loaded()

# Mesh preview: auto-load when preview is enabled (so upload shows immediately)
if show_mesh_preview:
    _has_mesh_source = (msh_file is not None) or ("generated_msh_bytes" in st.session_state)
    _ok_preview = False
    if _has_mesh_source:
        _ok_preview = bool(ensure_mesh_loaded())
        # If auto-retry is triggered, rerun to start regeneration
        if not _ok_preview and st.session_state.get("auto_retry_mesh_error", False):
            st.info("🔄 Mesh error detected. LLM regeneration will be triggered automatically. Please wait...")
            st.rerun()
    if (not _has_mesh_source) or (not _ok_preview):
        _ok_preview = False

if show_mesh_preview and ("mesh_sig" in st.session_state):
    md = int(st.session_state.get("mesh_dim", 2))
    if md == 2:
        figm = plot_mesh_with_boundaries_fast(
            st.session_state["pts"],
            st.session_state["tris"],
            st.session_state["seg_u"],
            st.session_state["seg_t"] if st.session_state["seg_t"].size else np.zeros((0, 2), dtype=int),
            quads=st.session_state.get("quads", np.zeros((0, 4), dtype=int)),
            omega_triangles_n=int(st.session_state.get("omega_info", {}).get("Omega_triangles", 0)),
        )
        st.pyplot(figm, use_container_width=False)
        plt.close(figm)
    else:
        info = st.session_state.get("omega_info", {})
        gu_tris = info.get("Gamma_u_tris_conn", np.zeros((0, 3), dtype=int))
        gt_tris = info.get("Gamma_t_tris_conn", np.zeros((0, 3), dtype=int))
        tets_conn = info.get("Omega_tets_conn", None)
        fig3m = plot_mesh_preview_3d_surfaces(
            st.session_state["pts"],
            gu_tris,
            gt_tris,
            tets_conn=np.asarray(tets_conn) if tets_conn is not None else None,
            max_faces=3500,
            max_points=7000,
        )
        st.pyplot(fig3m, use_container_width=False)
        plt.close(fig3m)


# ============================================================
# Train button
# ============================================================
train_clicked = st.button("🚀 Start Training", type="primary", key="btn_train")
if train_clicked:
    # Prefer writing mesh-load status into the Boundary conditions sidebar section.
    if "bc_mesh_load_ui" in locals():
        with bc_mesh_load_ui.container():
            ok_mesh = ensure_mesh_loaded()
    else:
        ok_mesh = ensure_mesh_loaded()
    if not ok_mesh:
        # If auto-retry is triggered, don't stop - let the regeneration logic run
        if st.session_state.get("auto_retry_mesh_error", False):
            st.info("🔄 Mesh error detected. LLM regeneration will be triggered automatically. Please wait...")
            st.rerun()  # Rerun to trigger the regeneration logic
        else:
            st.stop()

    set_seed(int(seed))

    if net_type == "MLP":
        model = MLP(layers, activation=activation).to(device)
    else:
        model = KANNet(layers, grid_size=int(kan_grid), degree=int(kan_degree), activation=activation).to(device)
    st.success(f"Model created: **{total_params(model):,}** params on **{device}**")

    # fail-fast expression checks
    try:
        mesh_dim = int(st.session_state.get("mesh_dim", 2))
        xtest = to_torch(np.array([[st.session_state["pts"][:, 0].mean()]], dtype=np.float32), device=device, requires_grad=False)
        ytest = to_torch(np.array([[st.session_state["pts"][:, 1].mean()]], dtype=np.float32), device=device, requires_grad=False)
        ztest = None
        if mesh_dim == 3:
            ztest = to_torch(np.array([[st.session_state["pts"][:, 2].mean()]], dtype=np.float32), device=device, requires_grad=False)

        if problem_type in ["Poisson (scalar)", "Screened Poisson equation (2D)"]:
            if ubar_expr and ubar_expr.strip():
                _ = eval_expr_torch(ubar_expr, x=xtest, y=ytest, z=ztest)
            if f_expr.strip():
                _ = eval_expr_torch(f_expr, x=xtest, y=ytest, z=ztest)
            if g_expr.strip():
                _ = eval_expr_torch(g_expr, x=xtest, y=ytest, z=ztest)
        elif problem_type == "Custom (user-defined)":
            cdim = int(custom_out_dim) if custom_out_dim is not None else 1
            cdim = int(max(1, min(3, cdim)))
            if int(mesh_dim) == 2:
                cdim = int(min(2, cdim))
            if int(mesh_dim) == 3 and cdim == 2:
                raise ValueError("Custom (user-defined): out_dim=2 is not supported for 3D meshes. Use out_dim=1 or out_dim=3.")
            z0 = torch.zeros_like(xtest)
            extra = {
                "u": z0,
                "ux": z0,
                "uy": z0,
                "uz": z0 if ztest is not None else None,
                # Traction/Neumann variables (available on Γt in real training, but define here for validation)
                "tx": z0,
                "ty": z0,
                "tz": z0 if ztest is not None else None,
                "lam": float(lam),
                "mu": float(mu),
                "E": float(E),
                "nu": float(nu),
            }
            if cdim >= 2:
                extra.update({"v": z0, "vx": z0, "vy": z0, "vz": z0 if ztest is not None else None})
                extra.update({"eps_xx": z0, "eps_yy": z0, "eps_xy": z0, "tr_eps": z0})
            if cdim >= 3:
                extra.update({"w": z0, "wx": z0, "wy": z0, "wz": z0 if ztest is not None else None})

            if str(custom_pi_omega_expr or "").strip():
                _ = eval_expr_torch_ext(str(custom_pi_omega_expr), x=xtest, y=ytest, z=ztest, **extra)
            if str(custom_pi_gu_expr or "").strip():
                _ = eval_expr_torch_ext(str(custom_pi_gu_expr), x=xtest, y=ytest, z=ztest, **extra)
            if str(custom_pi_gt_expr or "").strip():
                _ = eval_expr_torch_ext(str(custom_pi_gt_expr), x=xtest, y=ytest, z=ztest, **extra)
        else:
            if problem_type == "Linear Elasticity (3D)":
                if ubarx_expr and ubarx_expr.strip():
                    _ = eval_expr_torch(ubarx_expr, x=xtest, y=ytest, z=ztest)
                if ubary_expr and ubary_expr.strip():
                    _ = eval_expr_torch(ubary_expr, x=xtest, y=ytest, z=ztest)
                if ubarz_expr and ubarz_expr.strip():
                    _ = eval_expr_torch(ubarz_expr, x=xtest, y=ytest, z=ztest)
                if tx_expr.strip():
                    _ = eval_expr_torch(tx_expr, x=xtest, y=ytest, z=ztest)
                if ty_expr.strip():
                    _ = eval_expr_torch(ty_expr, x=xtest, y=ytest, z=ztest)
                if tz_expr.strip():
                    _ = eval_expr_torch(tz_expr, x=xtest, y=ytest, z=ztest)
                if bx_expr.strip():
                    _ = eval_expr_torch(bx_expr, x=xtest, y=ytest, z=ztest)
                if by_expr.strip():
                    _ = eval_expr_torch(by_expr, x=xtest, y=ytest, z=ztest)
                if bz_expr.strip():
                    _ = eval_expr_torch(bz_expr, x=xtest, y=ytest, z=ztest)
            else:
                if ubarx_expr and ubarx_expr.strip():
                    _ = eval_expr_torch(ubarx_expr, x=xtest, y=ytest)
                if ubary_expr and ubary_expr.strip():
                    _ = eval_expr_torch(ubary_expr, x=xtest, y=ytest)
                if tx_expr.strip():
                    _ = eval_expr_torch(tx_expr, x=xtest, y=ytest)
                if ty_expr.strip():
                    _ = eval_expr_torch(ty_expr, x=xtest, y=ytest)
                if bx_expr.strip():
                    _ = eval_expr_torch(bx_expr, x=xtest, y=ytest)
                if by_expr.strip():
                    _ = eval_expr_torch(by_expr, x=xtest, y=ytest)
    except Exception as e:
        st.error(f"Expression error: {e}")
        st.stop()

    Xdom_full = st.session_state["Xdom"]
    Wdom_full = st.session_state["Wdom"]
    Xu_full = st.session_state["Xu"]
    Wu_full = st.session_state["Wu"]
    Xt_full = st.session_state["Xt"]
    Wt_full = st.session_state["Wt"]
    dist_obj = st.session_state["dist_obj"]
    dir_mode_eff = st.session_state.get("dir_mode_effective", dir_mode)

    with st.spinner("Training..."):
        logs = train_dem_mesh(
            model=model,
            problem_type=problem_type,
            epochs=int(num_epochs),
            lr=float(learning_rate),
            device=device,
            Xdom_full=Xdom_full,
            Wdom_full=Wdom_full,
            Xu_full=Xu_full,
            Wu_full=Wu_full,
            Xt_full=Xt_full,
            Wt_full=Wt_full,
            dist_obj=dist_obj,
            dom_batch=int(dom_batch),
            bnd_u_batch=int(bnd_u_batch),
            bnd_t_batch=int(bnd_t_batch),
            dir_mode=dir_mode_eff,
            penalty_lambda=float(penalty_lambda),
            f_expr=f_expr if f_expr.strip() else None,
            g_expr=g_expr if g_expr.strip() else None,
            g_const=float(g_const),
            ubar_expr=ubar_expr,
            ubar_const=float(ubar_const),
            ubarx_expr=ubarx_expr,
            ubary_expr=ubary_expr,
            ubarz_expr=ubarz_expr,
            ubarx_const=float(ubarx_const),
            ubary_const=float(ubary_const),
            ubarz_const=float(ubarz_const),
            thickness=float(thickness),
            plane_mode=str(plane_mode),
            lam=float(lam),
            mu=float(mu),
            tx_expr=tx_expr if tx_expr.strip() else None,
            ty_expr=ty_expr if ty_expr.strip() else None,
            tz_expr=tz_expr if tz_expr.strip() else None,
            tx_const=float(tx_const),
            ty_const=float(ty_const),
            tz_const=float(tz_const),
            bx_expr=bx_expr if bx_expr.strip() else None,
            by_expr=by_expr if by_expr.strip() else None,
            bz_expr=bz_expr if bz_expr.strip() else None,
            bx_const=float(bx_const),
            by_const=float(by_const),
            bz_const=float(bz_const),
            k_squared=float(k_squared) if problem_type == "Screened Poisson equation (2D)" else 0.0,
            custom_out_dim=int(custom_out_dim),
            custom_pi_omega_expr=str(custom_pi_omega_expr or ""),
            custom_pi_gu_expr=str(custom_pi_gu_expr or ""),
            custom_pi_gt_expr=str(custom_pi_gt_expr or ""),
            custom_pi_gt_mode=custom_pi_gt_mode,
            derivative_method=str(derivative_method),
            pts_nodes=st.session_state.get("pts", None),
            sf_dom_2d=st.session_state.get("sf_dom_2d", None),
            sf_gu_2d=st.session_state.get("sf_gu_2d", None),
            sf_gt_2d=st.session_state.get("sf_gt_2d", None),
            abort_on_spike=bool(abort_on_spike),
            spike_rtol=float(spike_rtol),
            spike_warmup=int(spike_warmup),
            stable_rel_rtol=float(stable_rel_rtol),
            stable_rel_patience=int(stable_rel_patience),
            log_every=int(log_every),
        )

    pts = st.session_state["pts"]
    tris = st.session_state["tris"]
    mesh_dim = int(st.session_state.get("mesh_dim", 2))
    if problem_type in ["Poisson (scalar)", "Screened Poisson equation (2D)"]:
        out_dim = 1
    elif problem_type == "Linear Elasticity (3D)":
        out_dim = 3
    elif problem_type == "Custom (user-defined)":
        out_dim = int(st.session_state.get("custom_out_dim", 1) or 1)
    else:
        out_dim = 2

    U = eval_on_points(
        model=model,
        device=device,
        pts_xy=pts,
        out_dim=out_dim,
        dir_mode=dir_mode_eff,
        dist_obj=dist_obj,
        ubar_expr=ubar_expr,
        ubar_const=float(ubar_const),
        ubarx_expr=ubarx_expr,
        ubary_expr=ubary_expr,
        ubarz_expr=ubarz_expr,
        ubarx_const=float(ubarx_const),
        ubary_const=float(ubary_const),
        ubarz_const=float(ubarz_const),
    )

    # Persist model state for post-run visualization (e.g., stress field) without retraining.
    model_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    st.session_state["last_result"] = {
        "mesh_sig": st.session_state.get("mesh_sig"),
        "problem_type": problem_type,
        "mesh_dim": int(mesh_dim),
        "out_dim": int(out_dim),
        # Persist the exact plotting mesh used for evaluation so results remain displayable
        # even if the user later loads/remeshes a different geometry.
        "pts_plot": np.asarray(pts).copy(),
        "tris_plot": np.asarray(tris).copy(),
        "omega_info": st.session_state.get("omega_info", {}),
        "logs": logs,
        "U": U.astype(np.float32),
        "model_state": model_state_cpu,
        "model_meta": {
            "net_type": str(net_type),
            "layers": list(layers),
            "activation": str(activation),
            "kan_grid": int(kan_grid),
            "kan_degree": int(kan_degree),
            "derivative_method": str(derivative_method),
            "dir_mode": str(dir_mode_eff),
            "ubar_expr": str(ubar_expr),
            "ubarx_expr": str(ubarx_expr),
            "ubary_expr": str(ubary_expr),
            "ubarz_expr": str(ubarz_expr),
            "plane_mode": str(plane_mode),
            "lam": float(lam),
            "mu": float(mu),
            "tx_expr": str(tx_expr),
            "ty_expr": str(ty_expr),
            "tz_expr": str(tz_expr),
            "bx_expr": str(bx_expr),
            "by_expr": str(by_expr),
            "bz_expr": str(bz_expr),
        },
    }
    st.rerun()


# ============================================================
# Results panel (persisted; changing sliders won't retrain)
# ============================================================
res = st.session_state.get("last_result", None)
if res is not None:
    if res.get("mesh_sig") != st.session_state.get("mesh_sig"):
        st.info("Detected a new mesh/settings signature. Previous results are kept, but may not match the current mesh. Retrain if needed.")

    logs = res.get("logs", None)
    U = res.get("U", None)
    res_problem = res.get("problem_type", "")
    mesh_dim = int(res.get("mesh_dim", 2))
    out_dim = int(res.get("out_dim", 1))

    if logs:
        st.success(f"✅ Training done in {logs['time_sec']:.2f} s")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Π", f"{logs['Pi'][-1]:.4e}")
        m2.metric("Final Wint", f"{logs['Wint'][-1]:.4e}")
        m3.metric("Final Wext", f"{logs['Wext'][-1]:.4e}")
        m4.metric("Final Wbc", f"{logs['Wbc'][-1]:.4e}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(logs["Pi"], linewidth=2, label="Π (total)")
        ax.plot(np.abs(np.array(logs["Wint"])) + 1e-16, linewidth=2, label="Wint")
        ax.plot(np.abs(np.array(logs["Wext"])) + 1e-16, linewidth=2, label="|Wext|")
        ax.plot(np.abs(np.array(logs["Wext_body"])) + 1e-16, linewidth=2, label="|Wext_body|")
        ax.plot(np.abs(np.array(logs["Wext_trac"])) + 1e-16, linewidth=2, label="|Wext_trac|")
        ax.plot(np.abs(np.array(logs["Wbc"])) + 1e-16, linewidth=2, label="Wbc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Energy Components During Training")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ------------------------------------------------------------
        # Export to ParaView (.vtu)
        # ------------------------------------------------------------
        with st.expander("⬇️ Export results (ParaView .vtu)", expanded=False):
            include_stress = False
            if res_problem in ["Linear Elasticity (2D)", "Neo-Hookean Hyperelasticity (2D)", "Linear Elasticity (3D)"]:
                include_stress = st.checkbox(
                    "Include stress fields (Stress, vonMises) in VTU (may take time)",
                    value=True,
                    key="pv_include_stress_dem",
                )

            # Prefer exporting true quad cells if current mesh matches; otherwise fall back to saved plotting mesh.
            if res.get("mesh_sig") == st.session_state.get("mesh_sig"):
                pts_e = st.session_state.get("pts", res.get("pts_plot", None))
                tris_e = st.session_state.get("tris", res.get("tris_plot", None))
                quads_e = st.session_state.get("quads", np.zeros((0, 4), dtype=int))
                omega_info_e = st.session_state.get("omega_info", res.get("omega_info", {}))
            else:
                pts_e = res.get("pts_plot", None)
                tris_e = res.get("tris_plot", None)
                quads_e = np.zeros((0, 4), dtype=int)
                omega_info_e = res.get("omega_info", {}) or {}

            if pts_e is None or U is None:
                st.warning("No result mesh/field to export.")
                m = None
            else:
                pts_e = np.asarray(pts_e)
                Uarr = np.asarray(U)

                try:
                    if int(mesh_dim) == 2:
                        omega_tri_n = int(omega_info_e.get("Omega_triangles", 0))
                        if tris_e is None:
                            st.warning("No triangle connectivity found to export 2D VTU.")
                            raise RuntimeError("Missing tris_e")
                        m = build_paraview_mesh_2d(
                            pts=np.asarray(pts_e),
                            tris_plot=np.asarray(tris_e),
                            quads=np.asarray(quads_e) if quads_e is not None else None,
                            omega_triangles_n=int(omega_tri_n),
                        )
                    else:
                        tets_conn = omega_info_e.get("Omega_tets_conn", None)
                        if tets_conn is None or np.asarray(tets_conn).size == 0:
                            st.warning("No tetra connectivity found (Omega_tets_conn) to export 3D VTU.")
                            raise RuntimeError("Missing Omega_tets_conn")
                        m = build_paraview_mesh_3d(pts=np.asarray(pts_e), tets_conn=np.asarray(tets_conn))
                except Exception as e:
                    st.warning(f"Failed to build export mesh: {e}")
                    m = None

                if m is not None:
                    if Uarr.ndim != 2 or int(Uarr.shape[0]) != int(m.points.shape[0]):
                        st.warning("Field/mesh size mismatch; please retrain on the current mesh to export.")
                    else:
                        sigma_node = None
                        mises_node = None

                        if include_stress:
                            # reuse cached stress if already computed during visualization/export
                            sigma_node = res.get("sigma_node", None)
                            mises_node = res.get("mises_node", None)

                            if sigma_node is None or mises_node is None:
                                meta = res.get("model_meta", {}) or {}
                                state = res.get("model_state", None)
                                if state is None:
                                    st.warning("No saved model state found for stress export. Please retrain once.")
                                else:
                                    try:
                                        layers0 = list(meta.get("layers", []))
                                        net_type0 = str(meta.get("net_type", "MLP"))
                                        act0 = str(meta.get("activation", "tanh"))
                                        if net_type0 == "MLP":
                                            m_vis = MLP(layers0, activation=act0).to(device)
                                        else:
                                            m_vis = KANNet(
                                                layers0,
                                                grid_size=int(meta.get("kan_grid", 8)),
                                                degree=int(meta.get("kan_degree", 3)),
                                                activation=act0,
                                            ).to(device)
                                        m_vis.load_state_dict(state, strict=True)
                                        m_vis.eval()

                                        deriv0 = str(meta.get("derivative_method", "autograd")).strip().lower()
                                        use_shape0 = deriv0.startswith("shape") and int(mesh_dim) == 2
                                        with st.spinner(
                                            f"Computing stress for VTU export ({'shape-function' if use_shape0 else 'autograd'})..."
                                        ):
                                            if use_shape0:
                                                sf_dom = st.session_state.get("sf_dom_2d", None)
                                                if sf_dom is None:
                                                    sf_dom = build_sf_domain_data_2d(
                                                        pts=pts_e,
                                                        tris_plot=st.session_state.get("tris", tris_e),
                                                        omega_info=st.session_state.get("omega_info", omega_info_e),
                                                        quads=st.session_state.get("quads", np.zeros((0, 4), dtype=int)),
                                                        tri_rule_name=str(tri_rule_name),
                                                        quad_gauss_n=int(quad_gauss_n),
                                                    )
                                                Xstress = np.asarray(sf_dom.get("X", pts_e))
                                                sigma_dom, mises_dom = compute_stress_on_sf_gauss_2d(
                                                    model=m_vis,
                                                    device=device,
                                                    pts_nodes=pts_e,
                                                    sf_dom_2d=sf_dom,
                                                    problem_type=res_problem,
                                                    dir_mode=str(meta.get("dir_mode", "")),
                                                    dist_obj=st.session_state.get("dist_obj", None),
                                                    ubarx_expr=str(meta.get("ubarx_expr", "0")),
                                                    ubary_expr=str(meta.get("ubary_expr", "0")),
                                                    lam=float(meta.get("lam", 0.0)),
                                                    mu=float(meta.get("mu", 0.0)),
                                                )
                                            else:
                                                Xdom_full = st.session_state.get("Xdom", None)
                                                Xstress = Xdom_full if (Xdom_full is not None) else pts_e
                                                # avoid huge autograd batches
                                                if Xstress is not None and int(Xstress.shape[0]) > 30000:
                                                    idx = np.random.choice(int(Xstress.shape[0]), size=30000, replace=False)
                                                    Xstress = Xstress[idx]
                                                sigma_dom, mises_dom = compute_stress_on_points(
                                                    model=m_vis,
                                                    device=device,
                                                    pts_xy=np.asarray(Xstress),
                                                    problem_type=res_problem,
                                                    dir_mode=str(meta.get("dir_mode", "")),
                                                    dist_obj=st.session_state.get("dist_obj", None),
                                                    ubar_expr=str(meta.get("ubar_expr", "0")),
                                                    ubarx_expr=str(meta.get("ubarx_expr", "0")),
                                                    ubary_expr=str(meta.get("ubary_expr", "0")),
                                                    ubarz_expr=str(meta.get("ubarz_expr", "0")),
                                                    lam=float(meta.get("lam", 0.0)),
                                                    mu=float(meta.get("mu", 0.0)),
                                                    plane_mode=str(meta.get("plane_mode", "plane_strain")),
                                                )

                                            sigma_node = project_point_field_to_nodes_idw(
                                                nodes_xy=pts_e, sample_xy=np.asarray(Xstress), sample_val=sigma_dom, device=device, k=8
                                            )
                                            mises_node = project_point_field_to_nodes_idw(
                                                nodes_xy=pts_e, sample_xy=np.asarray(Xstress), sample_val=mises_dom, device=device, k=8
                                            )

                                        # cache into the persisted result dict for later reuse
                                        try:
                                            st.session_state["last_result"]["sigma_node"] = np.asarray(sigma_node).astype(np.float32)
                                            st.session_state["last_result"]["mises_node"] = np.asarray(mises_node).astype(np.float32)
                                            res["sigma_node"] = st.session_state["last_result"]["sigma_node"]
                                            res["mises_node"] = st.session_state["last_result"]["mises_node"]
                                        except Exception:
                                            pass

                                    except Exception as e:
                                        st.warning(f"Stress export failed: {e}")

                        _disp_name = "u" if int(out_dim) == 1 else "displacement"
                        add_paraview_point_fields(m, U=Uarr, sigma=sigma_node, mises=mises_node, displacement_name=_disp_name)

                        data = _meshio_vtu_bytes(m)
                        st.download_button(
                            "Download DEM result (.vtu)",
                            data=data,
                            file_name="dem_result.vtu",
                            mime="application/octet-stream",
                            use_container_width=True,
                        )

    if U is not None:
        st.markdown("---")
        # Use the plotting mesh stored with the result to avoid shape mismatches after remeshing.
        pts = res.get("pts_plot", st.session_state.get("pts", None))
        tris = res.get("tris_plot", st.session_state.get("tris", None))
        can_plot = True
        if pts is None or tris is None:
            st.warning("No mesh data found for visualization.")
            can_plot = False
        else:
            pts = np.asarray(pts)
            tris = np.asarray(tris)
            U_arr = np.asarray(U)
            if int(U_arr.shape[0]) != int(pts.shape[0]):
                st.warning(
                    f"Result field size mismatch: U has {int(U_arr.shape[0])} nodes but mesh has {int(pts.shape[0])} nodes. "
                    f"This usually happens after remeshing/loading a different mesh. Please retrain."
                )
                # Skip plotting to avoid matplotlib TriContour errors.
                can_plot = False

        if can_plot and mesh_dim == 2:
            if out_dim == 1:
                figU = plot_field_on_mesh(pts, tris, U, title=f"{res_problem} — DEM solution on mesh nodes", is_scalar=True)
                st.pyplot(figU)
                plt.close(figU)
            else:
                figUV = plot_field_on_mesh(pts, tris, U, title=f"{res_problem} — DEM solution on mesh nodes", is_scalar=False)
                st.pyplot(figUV)
                plt.close(figUV)

            # Stress field (2D mechanics only) — rebuild trained model from saved state
            if res_problem in ["Linear Elasticity (2D)", "Neo-Hookean Hyperelasticity (2D)"]:
                meta = res.get("model_meta", {}) or {}
                state = res.get("model_state", None)
                if state is None:
                    st.warning("No saved model state found for stress visualization. Please retrain once.")
                else:
                    try:
                        layers0 = list(meta.get("layers", []))
                        net_type0 = str(meta.get("net_type", "MLP"))
                        act0 = str(meta.get("activation", "tanh"))
                        if net_type0 == "MLP":
                            m_vis = MLP(layers0, activation=act0).to(device)
                        else:
                            m_vis = KANNet(
                                layers0,
                                grid_size=int(meta.get("kan_grid", 8)),
                                degree=int(meta.get("kan_degree", 3)),
                                activation=act0,
                            ).to(device)
                        m_vis.load_state_dict(state, strict=True)
                        m_vis.eval()

                        deriv0 = str(meta.get("derivative_method", "autograd")).strip().lower()
                        use_shape0 = deriv0.startswith("shape")
                        with st.spinner(f"Computing stress field ({'shape-function' if use_shape0 else 'autograd'})..."):
                            if use_shape0:
                                sf_dom = st.session_state.get("sf_dom_2d", None)
                                if sf_dom is None:
                                    sf_dom = build_sf_domain_data_2d(
                                        pts=pts,
                                        tris_plot=st.session_state.get("tris", tris),
                                        omega_info=st.session_state.get("omega_info", {}),
                                        quads=st.session_state.get("quads", np.zeros((0, 4), dtype=int)),
                                        tri_rule_name=str(tri_rule_name),
                                        quad_gauss_n=int(quad_gauss_n),
                                    )
                                Xstress = np.asarray(sf_dom.get("X", pts))
                                sigma_dom, mises_dom = compute_stress_on_sf_gauss_2d(
                                    model=m_vis,
                                    device=device,
                                    pts_nodes=pts,
                                    sf_dom_2d=sf_dom,
                                    problem_type=res_problem,
                                    dir_mode=str(meta.get("dir_mode", "")),
                                    dist_obj=st.session_state.get("dist_obj", None),
                                    ubarx_expr=str(meta.get("ubarx_expr", "0")),
                                    ubary_expr=str(meta.get("ubary_expr", "0")),
                                    lam=float(meta.get("lam", 0.0)),
                                    mu=float(meta.get("mu", 0.0)),
                                )
                                if Xstress is not None and int(Xstress.shape[0]) > 20000:
                                    idx = np.random.choice(int(Xstress.shape[0]), size=20000, replace=False)
                                    Xstress = Xstress[idx]
                                    sigma_dom = sigma_dom[idx]
                                    mises_dom = mises_dom[idx]
                                    st.info("Stress is evaluated on a random subset of Gauss points (20k) for speed/memory.")
                            else:
                                Xdom_full = st.session_state.get("Xdom", None)
                                Xstress = Xdom_full if (Xdom_full is not None) else pts
                                if Xstress is not None and int(Xstress.shape[0]) > 20000:
                                    idx = np.random.choice(int(Xstress.shape[0]), size=20000, replace=False)
                                    Xstress = Xstress[idx]
                                    st.info("Stress is evaluated on a random subset of Xdom (20k points) for speed/memory.")

                                sigma_dom, mises_dom = compute_stress_on_points(
                                    model=m_vis,
                                    device=device,
                                    pts_xy=Xstress,
                                    problem_type=res_problem,
                                    dir_mode=str(meta.get("dir_mode", "")),
                                    dist_obj=st.session_state.get("dist_obj", None),
                                    ubar_expr=str(meta.get("ubar_expr", "0")),
                                    ubarx_expr=str(meta.get("ubarx_expr", "0")),
                                    ubary_expr=str(meta.get("ubary_expr", "0")),
                                    ubarz_expr=str(meta.get("ubarz_expr", "0")),
                                    lam=float(meta.get("lam", 0.0)),
                                    mu=float(meta.get("mu", 0.0)),
                                    plane_mode=str(meta.get("plane_mode", "plane_strain")),
                                )

                            sigma = project_point_field_to_nodes_idw(
                                nodes_xy=pts, sample_xy=Xstress, sample_val=sigma_dom, device=device, k=8
                            )
                            mises = project_point_field_to_nodes_idw(
                                nodes_xy=pts, sample_xy=Xstress, sample_val=mises_dom, device=device, k=8
                            )

                        figS = plot_stress_on_mesh(pts, tris, sigma, mises, title_prefix=f"{res_problem} Stress")
                        st.pyplot(figS)
                        plt.close(figS)
                    except Exception as e:
                        st.warning(f"Stress visualization failed: {e}")
        else:
            if out_dim == 1:
                st.markdown("### 3D Cloud Map + Slices")
                st.markdown("#### 3D Surface Cloud Map (recommended)")
                show_surface = st.checkbox("Show outer-surface cloud map (extract from tetra boundary)", value=True, key="show_surface_res")
                if show_surface:
                    max_faces = st.slider("Max surface triangles", 1000, 80000, 12000, 1000, key="max_faces_res")
                    tets_conn = st.session_state.get("omega_info", {}).get("Omega_tets_conn", None)
                    if tets_conn is None or np.asarray(tets_conn).size == 0:
                        st.warning("Tetra connectivity not found (Omega_tets_conn). Please reload the 3D mesh.")
                    else:
                        surf_tris = extract_boundary_tris_from_tets(np.asarray(tets_conn))
                        fig_surf = plot_3d_scalar_surface_cloud(
                            pts,
                            surf_tris,
                            np.asarray(U)[:, 0],
                            max_faces=int(max_faces),
                        )
                        st.pyplot(fig_surf)
                        plt.close(fig_surf)

                st.markdown("#### Point cloud + orthogonal slices (extra)")
                xmin, ymin, zmin = float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), float(np.min(pts[:, 2]))
                xmax, ymax, zmax = float(np.max(pts[:, 0])), float(np.max(pts[:, 1])), float(np.max(pts[:, 2]))
                cx, cy, cz = float(np.median(pts[:, 0])), float(np.median(pts[:, 1])), float(np.median(pts[:, 2]))

                cA, cB, cC = st.columns([1, 1, 2])
                with cA:
                    max_points_3d = st.slider("Max 3D point-cloud points", 2000, 80000, 20000, 1000, key="max_points_3d_res")
                with cB:
                    slice_frac = st.slider("Slice thickness (fraction of max bbox edge)", 0.002, 0.10, 0.03, 0.002, key="slice_frac_res")
                with cC:
                    cX, cY, cZ = st.columns(3)
                    with cX:
                        x0 = st.slider("YZ slice position x0", xmin, xmax, cx, key="x0_res")
                    with cY:
                        y0 = st.slider("XZ slice position y0", ymin, ymax, cy, key="y0_res")
                    with cZ:
                        z0 = st.slider("XY slice position z0", zmin, zmax, cz, key="z0_res")

                fig_cloud, fig_slices = plot_3d_scalar_pointcloud_and_slices(
                    pts,
                    np.asarray(U)[:, 0],
                    max_points_3d=int(max_points_3d),
                    slice_frac=float(slice_frac),
                    x0=float(x0),
                    y0=float(y0),
                    z0=float(z0),
                )
                st.pyplot(fig_cloud)
                plt.close(fig_cloud)
                st.pyplot(fig_slices)
                plt.close(fig_slices)
            elif out_dim == 3 and res_problem == "Linear Elasticity (3D)":
                st.markdown("### 3D Displacement Field (DEM)")

                # Choose displacement scalar to visualize
                disp_opt = st.selectbox("Display", ["|u|", "u_x", "u_y", "u_z"], index=0, key="disp3d_opt")
                Uarr = np.asarray(U)
                if disp_opt == "|u|":
                    s_disp = np.sqrt(Uarr[:, 0] ** 2 + Uarr[:, 1] ** 2 + Uarr[:, 2] ** 2)
                elif disp_opt == "u_x":
                    s_disp = Uarr[:, 0]
                elif disp_opt == "u_y":
                    s_disp = Uarr[:, 1]
                else:
                    s_disp = Uarr[:, 2]

                tets_conn = st.session_state.get("omega_info", {}).get("Omega_tets_conn", None)
                if tets_conn is None or np.asarray(tets_conn).size == 0:
                    st.warning("Tetra connectivity not found (Omega_tets_conn). Please reload the 3D mesh.")
                else:
                    surf_tris = extract_boundary_tris_from_tets(np.asarray(tets_conn))
                    max_faces = st.slider("Max surface triangles (displacement)", 1000, 80000, 12000, 1000, key="max_faces_u3d")
                    fig_u_surf = plot_3d_scalar_surface_cloud(pts, surf_tris, s_disp, max_faces=int(max_faces))
                    st.pyplot(fig_u_surf)
                    plt.close(fig_u_surf)

                # Slices for displacement
                xmin, ymin, zmin = float(np.min(pts[:, 0])), float(np.min(pts[:, 1])), float(np.min(pts[:, 2]))
                xmax, ymax, zmax = float(np.max(pts[:, 0])), float(np.max(pts[:, 1])), float(np.max(pts[:, 2]))
                cx, cy, cz = float(np.median(pts[:, 0])), float(np.median(pts[:, 1])), float(np.median(pts[:, 2]))
                cA, cB, cC = st.columns([1, 1, 2])
                with cA:
                    max_points_3d = st.slider("Max 3D point-cloud points (displacement)", 2000, 80000, 20000, 1000, key="max_pts_u3d")
                with cB:
                    slice_frac = st.slider("Slice thickness (fraction of max bbox edge)", 0.002, 0.10, 0.03, 0.002, key="slice_frac_u3d")
                with cC:
                    cX, cY, cZ = st.columns(3)
                    with cX:
                        x0 = st.slider("YZ slice position x0", xmin, xmax, cx, key="x0_u3d")
                    with cY:
                        y0 = st.slider("XZ slice position y0", ymin, ymax, cy, key="y0_u3d")
                    with cZ:
                        z0 = st.slider("XY slice position z0", zmin, zmax, cz, key="z0_u3d")

                fig_cloud_u, fig_slices_u = plot_3d_scalar_pointcloud_and_slices(
                    pts,
                    s_disp,
                    max_points_3d=int(max_points_3d),
                    slice_frac=float(slice_frac),
                    x0=float(x0),
                    y0=float(y0),
                    z0=float(z0),
                )
                st.pyplot(fig_cloud_u)
                plt.close(fig_cloud_u)
                st.pyplot(fig_slices_u)
                plt.close(fig_slices_u)

                # ----------------------------
                # Stress field (3D)
                # ----------------------------
                st.markdown("### 3D Stress Field (DEM)")
                meta = res.get("model_meta", {}) or {}
                state = res.get("model_state", None)
                if state is None:
                    st.warning("No saved model state found for stress visualization. Please retrain once.")
                else:
                    compute_stress = st.checkbox("Compute stress (autograd) and visualize", value=False, key="do_stress3d")
                    if compute_stress:
                        # rebuild trained model
                        layers0 = list(meta.get("layers", []))
                        net_type0 = str(meta.get("net_type", "MLP"))
                        act0 = str(meta.get("activation", "tanh"))
                        if net_type0 == "MLP":
                            m_vis = MLP(layers0, activation=act0).to(device)
                        else:
                            m_vis = KANNet(
                                layers0,
                                grid_size=int(meta.get("kan_grid", 8)),
                                degree=int(meta.get("kan_degree", 3)),
                                activation=act0,
                            ).to(device)
                        m_vis.load_state_dict(state, strict=True)
                        m_vis.eval()

                        Xdom_full = st.session_state.get("Xdom", None)
                        Xstress = Xdom_full if (Xdom_full is not None) else pts
                        max_eval = st.slider("Max stress evaluation points", 2000, 60000, 20000, 2000, key="max_stress_pts3d")
                        if Xstress is not None and int(Xstress.shape[0]) > int(max_eval):
                            idx = np.random.choice(int(Xstress.shape[0]), size=int(max_eval), replace=False)
                            Xstress = Xstress[idx]

                        with st.spinner("Computing 3D stress on sample points..."):
                            sigma_dom, mises_dom = compute_stress_on_points(
                                model=m_vis,
                                device=device,
                                pts_xy=Xstress,
                                problem_type="Linear Elasticity (3D)",
                                dir_mode=str(meta.get("dir_mode", "")),
                                dist_obj=st.session_state.get("dist_obj", None),
                                ubar_expr=str(meta.get("ubar_expr", "0")),
                                ubarx_expr=str(meta.get("ubarx_expr", "0")),
                                ubary_expr=str(meta.get("ubary_expr", "0")),
                                ubarz_expr=str(meta.get("ubarz_expr", "0")),
                                lam=float(meta.get("lam", 0.0)),
                                mu=float(meta.get("mu", 0.0)),
                                plane_mode=str(meta.get("plane_mode", "3d")),
                            )

                            sigma_node = project_point_field_to_nodes_idw(
                                nodes_xy=pts, sample_xy=Xstress, sample_val=sigma_dom, device=device, k=8
                            )
                            mises_node = project_point_field_to_nodes_idw(
                                nodes_xy=pts, sample_xy=Xstress, sample_val=mises_dom, device=device, k=8
                            )

                        stress_opt = st.selectbox(
                            "Stress to display",
                            ["von Mises", "σxx", "σyy", "σzz", "σxy", "σyz", "σzx"],
                            index=0,
                            key="stress3d_opt",
                        )
                        if stress_opt == "von Mises":
                            s_stress = np.asarray(mises_node)[:, 0]
                        else:
                            comp_map = {"σxx": 0, "σyy": 1, "σzz": 2, "σxy": 3, "σyz": 4, "σzx": 5}
                            s_stress = np.asarray(sigma_node)[:, comp_map[stress_opt]]

                        if tets_conn is not None and np.asarray(tets_conn).size > 0:
                            surf_tris = extract_boundary_tris_from_tets(np.asarray(tets_conn))
                            max_faces_s = st.slider("Max surface triangles (stress)", 1000, 80000, 12000, 1000, key="max_faces_s3d")
                            fig_s_surf = plot_3d_scalar_surface_cloud(pts, surf_tris, s_stress, max_faces=int(max_faces_s))
                            st.pyplot(fig_s_surf)
                            plt.close(fig_s_surf)

                        fig_cloud_s, fig_slices_s = plot_3d_scalar_pointcloud_and_slices(
                            pts,
                            s_stress,
                            max_points_3d=int(max_points_3d),
                            slice_frac=float(slice_frac),
                            x0=float(x0),
                            y0=float(y0),
                            z0=float(z0),
                        )
                        st.pyplot(fig_cloud_s)
                        plt.close(fig_cloud_s)
                        st.pyplot(fig_slices_s)
                        plt.close(fig_slices_s)
            else:
                st.info("3D visualization is available for Poisson (scalar) and Linear Elasticity (3D).")

        # ============================================================
        # FEM reference (2D only)
        # ============================================================
        if do_fem_ref and mesh_dim == 3 and problem_type == "Poisson (scalar)":
            st.markdown("## 🧱 FEM Reference (3D Poisson, Tet4)")
            info = st.session_state.get("omega_info", {})
            tets_conn = info.get("Omega_tets_conn", None)
            gu_tris = info.get("Gamma_u_tris_conn", np.zeros((0, 3), dtype=int))
            gt_tris = info.get("Gamma_t_tris_conn", np.zeros((0, 3), dtype=int))

            if tets_conn is None or np.asarray(tets_conn).size == 0:
                st.warning("No tetra connectivity found for FEM (Omega_tets_conn). Reload the 3D mesh.")
            else:
                fem_sig = (
                    st.session_state.get("mesh_sig"),
                    str(f_expr or ""),
                    str(ubar_expr or ""),
                    str(g_expr or ""),
                    float(g_const),
                    str(device),
                )
                if st.session_state.get("fem3d_sig") != fem_sig:
                    with st.spinner("Solving 3D FEM Poisson reference (Tet4)..."):
                        u_fem3d = fem_poisson_TET4_3d(
                            pts=st.session_state["pts"],
                            tets=np.asarray(tets_conn),
                            tris_u=np.asarray(gu_tris),
                            tris_t=np.asarray(gt_tris) if np.asarray(gt_tris).size > 0 else None,
                            ubar_expr=ubar_expr,
                            f_expr=f_expr if str(f_expr).strip() else None,
                            g_expr=g_expr if str(g_expr).strip() else None,
                            g_const=float(g_const),
                            device_for_eval=str(device),
                        )
                    st.session_state["fem3d_u"] = u_fem3d.astype(np.float32)
                    st.session_state["fem3d_sig"] = fem_sig
                else:
                    u_fem3d = st.session_state.get("fem3d_u", None)

                if u_fem3d is not None:
                    # error vs DEM on nodes
                    eL2 = rel_l2_error(np.asarray(U)[:, 0], np.asarray(u_fem3d)[:, 0])
                    st.write(f"**Relative L2 error (global):** {float(eL2):.3e}")

                    with st.expander("⬇️ Export FEM reference (ParaView .vtu)", expanded=False):
                        try:
                            m_fem3d = build_paraview_mesh_3d(
                                pts=np.asarray(st.session_state["pts"]),
                                tets_conn=np.asarray(tets_conn),
                            )
                            add_paraview_point_fields(m_fem3d, U=np.asarray(u_fem3d), displacement_name="u")
                            data_fem3d = _meshio_vtu_bytes(m_fem3d)
                            st.download_button(
                                "Download FEM reference (.vtu)",
                                data=data_fem3d,
                                file_name="fem_reference.vtu",
                                mime="application/octet-stream",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.warning(f"FEM VTU export failed: {e}")

                    st.markdown("### FEM 3D Cloud Map")
                    max_faces_fem = st.slider("FEM max surface triangles", 1000, 80000, 12000, 1000, key="fem3d_max_faces")
                    surf_tris = extract_boundary_tris_from_tets(np.asarray(tets_conn))
                    fig_fem_surf = plot_3d_scalar_surface_cloud(
                        st.session_state["pts"],
                        surf_tris,
                        np.asarray(u_fem3d)[:, 0],
                        max_faces=int(max_faces_fem),
                    )
                    st.pyplot(fig_fem_surf)
                    plt.close(fig_fem_surf)

                    st.markdown("### FEM 3D Slices (extra)")
                    pts3 = st.session_state["pts"]
                    xmin, ymin, zmin = float(np.min(pts3[:, 0])), float(np.min(pts3[:, 1])), float(np.min(pts3[:, 2]))
                    xmax, ymax, zmax = float(np.max(pts3[:, 0])), float(np.max(pts3[:, 1])), float(np.max(pts3[:, 2]))
                    cx, cy, cz = float(np.median(pts3[:, 0])), float(np.median(pts3[:, 1])), float(np.median(pts3[:, 2]))
                    cA, cB, cC = st.columns([1, 1, 2])
                    with cA:
                        max_points_3d = st.slider("FEM max 3D point-cloud points", 2000, 80000, 20000, 1000, key="fem3d_max_pts")
                    with cB:
                        slice_frac = st.slider("FEM slice thickness (fraction of max bbox edge)", 0.002, 0.10, 0.03, 0.002, key="fem3d_slice_frac")
                    with cC:
                        cX, cY, cZ = st.columns(3)
                        with cX:
                            x0 = st.slider("FEM YZ slice x0", xmin, xmax, cx, key="fem3d_x0")
                        with cY:
                            y0 = st.slider("FEM XZ slice y0", ymin, ymax, cy, key="fem3d_y0")
                        with cZ:
                            z0 = st.slider("FEM XY slice z0", zmin, zmax, cz, key="fem3d_z0")

                    fig_cloud_f, fig_slices_f = plot_3d_scalar_pointcloud_and_slices(
                        pts3,
                        np.asarray(u_fem3d)[:, 0],
                        max_points_3d=int(max_points_3d),
                        slice_frac=float(slice_frac),
                        x0=float(x0),
                        y0=float(y0),
                        z0=float(z0),
                    )
                    st.pyplot(fig_cloud_f)
                    plt.close(fig_cloud_f)
                    st.pyplot(fig_slices_f)
                    plt.close(fig_slices_f)

        if do_fem_ref and mesh_dim == 3 and problem_type == "Linear Elasticity (3D)":
            st.markdown("## 🧱 FEM Reference (3D Linear Elasticity, Tet4)")
            info = st.session_state.get("omega_info", {})
            tets_conn = info.get("Omega_tets_conn", None)
            gu_tris = info.get("Gamma_u_tris_conn", np.zeros((0, 3), dtype=int))
            gt_tris = info.get("Gamma_t_tris_conn", np.zeros((0, 3), dtype=int))

            if tets_conn is None or np.asarray(tets_conn).size == 0:
                st.warning("No tetra connectivity found for FEM (Omega_tets_conn). Reload the 3D mesh.")
            else:
                fem_sig = (
                    st.session_state.get("mesh_sig"),
                    float(lam),
                    float(mu),
                    str(ubarx_expr or ""),
                    str(ubary_expr or ""),
                    str(ubarz_expr or ""),
                    str(bx_expr or ""),
                    str(by_expr or ""),
                    str(bz_expr or ""),
                    float(bx_const),
                    float(by_const),
                    float(bz_const),
                    str(tx_expr or ""),
                    str(ty_expr or ""),
                    str(tz_expr or ""),
                    float(tx_const),
                    float(ty_const),
                    float(tz_const),
                    str(device),
                )

                if st.session_state.get("fem3d_el_sig") != fem_sig:
                    with st.spinner("Solving 3D FEM linear elasticity reference (Tet4)..."):
                        U_fem3d, sigma_fem3d, mises_fem3d = fem_linear_elasticity_TET4_3d(
                            pts=np.asarray(st.session_state["pts"]),
                            tets=np.asarray(tets_conn),
                            tris_u=np.asarray(gu_tris),
                            tris_t=np.asarray(gt_tris) if np.asarray(gt_tris).size > 0 else None,
                            lam=float(lam),
                            mu=float(mu),
                            ubarx_expr=ubarx_expr,
                            ubary_expr=ubary_expr,
                            ubarz_expr=ubarz_expr,
                            bx_expr=bx_expr if str(bx_expr).strip() else None,
                            by_expr=by_expr if str(by_expr).strip() else None,
                            bz_expr=bz_expr if str(bz_expr).strip() else None,
                            bx_const=float(bx_const),
                            by_const=float(by_const),
                            bz_const=float(bz_const),
                            tx_expr=tx_expr if str(tx_expr).strip() else None,
                            ty_expr=ty_expr if str(ty_expr).strip() else None,
                            tz_expr=tz_expr if str(tz_expr).strip() else None,
                            tx_const=float(tx_const),
                            ty_const=float(ty_const),
                            tz_const=float(tz_const),
                            device_for_eval=str(device),
                        )
                    st.session_state["fem3d_el_u"] = np.asarray(U_fem3d).astype(np.float32)
                    st.session_state["fem3d_el_sigma"] = np.asarray(sigma_fem3d).astype(np.float32)
                    st.session_state["fem3d_el_mises"] = np.asarray(mises_fem3d).astype(np.float32)
                    st.session_state["fem3d_el_sig"] = fem_sig
                else:
                    U_fem3d = st.session_state.get("fem3d_el_u", None)
                    sigma_fem3d = st.session_state.get("fem3d_el_sigma", None)
                    mises_fem3d = st.session_state.get("fem3d_el_mises", None)

                if U_fem3d is not None:
                    # error vs DEM on nodes (displacement)
                    if U is not None and np.asarray(U).ndim == 2 and np.asarray(U).shape[1] >= 3:
                        eL2 = rel_l2_error(np.asarray(U)[:, :3], np.asarray(U_fem3d)[:, :3])
                        st.write(f"**Relative L2 error (global, displacement):** {float(eL2):.3e}")

                    with st.expander("⬇️ Export FEM reference (ParaView .vtu)", expanded=False):
                        try:
                            m_fem3d = build_paraview_mesh_3d(
                                pts=np.asarray(st.session_state["pts"]),
                                tets_conn=np.asarray(tets_conn),
                            )
                            add_paraview_point_fields(
                                m_fem3d,
                                U=np.asarray(U_fem3d),
                                sigma=np.asarray(sigma_fem3d) if sigma_fem3d is not None else None,
                                mises=np.asarray(mises_fem3d) if mises_fem3d is not None else None,
                            )
                            data_fem3d = _meshio_vtu_bytes(m_fem3d)
                            st.download_button(
                                "Download FEM reference (.vtu)",
                                data=data_fem3d,
                                file_name="fem_reference.vtu",
                                mime="application/octet-stream",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.warning(f"FEM VTU export failed: {e}")

                    # visualize displacement magnitude and vonMises on outer surface
                    try:
                        tets_conn0 = np.asarray(tets_conn)
                        surf_tris = extract_boundary_tris_from_tets(tets_conn0)
                        umag = np.linalg.norm(np.asarray(U_fem3d)[:, :3], axis=1)
                        max_faces_fem = st.slider(
                            "FEM max surface triangles (3D elasticity)",
                            1000,
                            80000,
                            12000,
                            1000,
                            key="fem3d_el_max_faces",
                        )
                        st.markdown("### FEM 3D Surface — |u|")
                        fig_u_surf = plot_3d_scalar_surface_cloud(
                            np.asarray(st.session_state["pts"]),
                            surf_tris,
                            umag,
                            max_faces=int(max_faces_fem),
                        )
                        st.pyplot(fig_u_surf)
                        plt.close(fig_u_surf)

                        if mises_fem3d is not None:
                            st.markdown("### FEM 3D Surface — von Mises")
                            fig_m_surf = plot_3d_scalar_surface_cloud(
                                np.asarray(st.session_state["pts"]),
                                surf_tris,
                                np.asarray(mises_fem3d).reshape(-1),
                                max_faces=int(max_faces_fem),
                            )
                            st.pyplot(fig_m_surf)
                            plt.close(fig_m_surf)
                    except Exception as e:
                        st.warning(f"FEM 3D visualization failed: {e}")

        if do_fem_ref and mesh_dim == 2:
            has_quads = bool(st.session_state.get("omega_info", {}).get("has_quads", False))
            st.markdown("## 🧱 FEM Reference (T3/Q4) — for DEM verification")

            pts = st.session_state["pts"]
            tris_plot = st.session_state["tris"]
            quads = st.session_state.get("quads", np.zeros((0, 4), dtype=int))
            omega_tri_n = int(st.session_state.get("omega_info", {}).get("Omega_triangles", 0))
            seg_u = st.session_state["seg_u"]
            seg_t = st.session_state["seg_t"] if st.session_state["seg_t"].size else np.zeros((0, 2), dtype=int)

            with st.spinner("Solving FEM reference..."):
                if problem_type == "Poisson (scalar)":
                    if has_quads:
                        u_fem = fem_poisson_mixed_T3_Q4(
                            pts=pts,
                            tris_plot=tris_plot,
                            omega_triangles_n=omega_tri_n,
                            quads=quads,
                            seg_u=seg_u,
                            ubar_expr=ubar_expr,
                            f_expr=f_expr if f_expr.strip() else None,
                            gauss_n_quad=int(quad_gauss_n),
                            device_for_eval=str(device),
                        )
                    else:
                        u_fem = fem_poisson_T3(
                            pts=pts, tris=tris_plot, seg_u=seg_u,
                            ubar_expr=ubar_expr,
                            f_expr=f_expr if f_expr.strip() else None,
                            device_for_eval=str(device)
                        )
                    figUf = plot_field_on_mesh(pts, tris_plot, u_fem, title="FEM Solution u (T3/Q4) on Mesh Nodes", is_scalar=True)
                    st.pyplot(figUf)
                    plt.close(figUf)

                    # optional: compare error on nodes (relative)
                    eL2 = rel_l2_error(U[:, 0], u_fem[:, 0])
                    st.write(f"**Relative L2 error (global):** {float(eL2):.3e}")
                elif problem_type == "Screened Poisson equation (2D)":
                    if has_quads:
                        u_fem = fem_Screened_mixed_T3_Q4(
                            pts=pts,
                            tris_plot=tris_plot,
                            omega_triangles_n=omega_tri_n,
                            quads=quads,
                            seg_u=seg_u,
                            seg_t=seg_t,
                            ubar_expr=ubar_expr,
                            f_expr=f_expr if f_expr.strip() else None,
                            k_squared=float(k_squared),
                            g_expr=g_expr if g_expr.strip() else None,
                            g_const=float(g_const),
                            gauss_n_quad=int(quad_gauss_n),
                            device_for_eval=str(device),
                        )
                    else:
                        u_fem = fem_Screened_T3(
                            pts=pts,
                            tris=tris_plot,
                            seg_u=seg_u,
                            seg_t=seg_t,
                            ubar_expr=ubar_expr,
                            f_expr=f_expr if f_expr.strip() else None,
                            k_squared=float(k_squared),
                            g_expr=g_expr if g_expr.strip() else None,
                            g_const=float(g_const),
                            device_for_eval=str(device)
                        )
                    figUf = plot_field_on_mesh(pts, tris_plot, u_fem, title="FEM Solution u (T3/Q4) on Mesh Nodes", is_scalar=True)
                    st.pyplot(figUf)
                    plt.close(figUf)

                    # optional: compare error on nodes (relative)
                    eL2 = rel_l2_error(U[:, 0], u_fem[:, 0])
                    st.write(f"**Relative L2 error (global):** {float(eL2):.3e}")

                    with st.expander("⬇️ Export FEM reference (ParaView .vtu)", expanded=False):
                        try:
                            m_fem2d = build_paraview_mesh_2d(
                                pts=np.asarray(pts),
                                tris_plot=np.asarray(tris_plot),
                                quads=np.asarray(quads) if quads is not None else None,
                                omega_triangles_n=int(omega_tri_n),
                            )
                            add_paraview_point_fields(m_fem2d, U=np.asarray(u_fem), displacement_name="u")
                            data_fem2d = _meshio_vtu_bytes(m_fem2d)
                            st.download_button(
                                "Download FEM reference (.vtu)",
                                data=data_fem2d,
                                file_name="fem_reference.vtu",
                                mime="application/octet-stream",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.warning(f"FEM VTU export failed: {e}")

                elif problem_type == "Linear Elasticity (2D)":
                    if has_quads:
                        U_fem, sigma_fem, mises_fem = fem_linear_elasticity_mixed_T3_Q4(
                            pts=pts,
                            tris_plot=tris_plot,
                            omega_triangles_n=omega_tri_n,
                            quads=quads,
                            seg_u=seg_u,
                            seg_t=seg_t,
                            lam=float(lam), mu=float(mu), plane_mode=plane_mode,
                            ubarx_expr=ubarx_expr, ubary_expr=ubary_expr,
                            bx_expr=bx_expr if bx_expr.strip() else None,
                            by_expr=by_expr if by_expr.strip() else None,
                            bx_const=float(bx_const), by_const=float(by_const),
                            tx_expr=tx_expr if tx_expr.strip() else None,
                            ty_expr=ty_expr if ty_expr.strip() else None,
                            tx_const=float(tx_const), ty_const=float(ty_const),
                            gauss_n_quad=int(quad_gauss_n),
                            gauss_n_seg=int(seg_gauss_n),
                            device_for_eval=str(device),
                        )
                    else:
                        U_fem, sigma_fem, mises_fem = fem_linear_elasticity_T3(
                            pts=pts, tris=tris_plot,
                            seg_u=seg_u, seg_t=seg_t,
                            lam=float(lam), mu=float(mu), plane_mode=plane_mode,
                            ubarx_expr=ubarx_expr, ubary_expr=ubary_expr,
                            bx_expr=bx_expr if bx_expr.strip() else None,
                            by_expr=by_expr if by_expr.strip() else None,
                            bx_const=float(bx_const), by_const=float(by_const),
                            tx_expr=tx_expr if tx_expr.strip() else None,
                            ty_expr=ty_expr if ty_expr.strip() else None,
                            tx_const=float(tx_const), ty_const=float(ty_const),
                            gauss_n_seg=int(seg_gauss_n),
                            device_for_eval=str(device),
                        )

                    figUVf = plot_field_on_mesh(pts, tris_plot, U_fem, title="FEM Displacement (T3/Q4) on Mesh Nodes", is_scalar=False)
                    st.pyplot(figUVf)
                    plt.close(figUVf)

                    figSf = plot_stress_on_mesh(pts, tris_plot, sigma_fem, mises_fem, title_prefix="FEM Stress (T3/Q4, small strain)")
                    st.pyplot(figSf)
                    plt.close(figSf)

                    # Relative L2 error on displacement vector
                    eL2 = rel_l2_error(U[:, :2], U_fem[:, :2])
                    st.write(f"**Relative L2 error (global, displacement):** {float(eL2):.3e}")

                    with st.expander("⬇️ Export FEM reference (ParaView .vtu)", expanded=False):
                        try:
                            m_fem2d = build_paraview_mesh_2d(
                                pts=np.asarray(pts),
                                tris_plot=np.asarray(tris_plot),
                                quads=np.asarray(quads) if quads is not None else None,
                                omega_triangles_n=int(omega_tri_n),
                            )
                            add_paraview_point_fields(
                                m_fem2d,
                                U=np.asarray(U_fem),
                                sigma=np.asarray(sigma_fem),
                                mises=np.asarray(mises_fem),
                            )
                            data_fem2d = _meshio_vtu_bytes(m_fem2d)
                            st.download_button(
                                "Download FEM reference (.vtu)",
                                data=data_fem2d,
                                file_name="fem_reference.vtu",
                                mime="application/octet-stream",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.warning(f"FEM VTU export failed: {e}")

                else:
                    st.info("Neo-Hookean FEM reference (Newton) is not included yet. Use Linear Elasticity FEM to validate first.")

        if do_fem_ref and mesh_dim != 2 and not (mesh_dim == 3 and problem_type == "Poisson (scalar)"):
            st.info("FEM reference is currently available for 2D meshes only (except 3D Poisson Tet4).")


st.markdown("---")
st.markdown("**DEM (Gmsh)** | Upload .msh (Omega/Gamma_u/Gamma_t) + Mesh Gauss Quadrature + Hard/Penalty Dirichlet") 