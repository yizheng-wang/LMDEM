# DEM (Streamlit) — Deep Energy Method based on LLM

This is a Streamlit app for DEM (Deep Energy Method): load/generate a Gmsh mesh (`.msh`), train on mesh quadrature points, and export results to ParaView (`.vtu`).

The author is Yizheng Wang, email: wang-yz19@tsinghua.org.cn
---

### 1) Install & Run

In the project directory (where `DEM.py` and `requirements.txt` are located):

```bash
python -m pip install -r requirements.txt
streamlit run DEM.py
```

If you want to install this repo as a Python package for reuse/customization:

```bash
pip install -e .
```

> Using a virtual environment (venv/conda) is recommended.

---

### 2) OpenAI API Key (required for LLM geometry generation/repair)

#### Local (Windows PowerShell)

```powershell
setx OPENAI_API_KEY "YOUR_KEY"
```

Re-open the terminal for it to take effect.

#### Local (macOS/Linux)

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

#### Streamlit Cloud / Deployment

Use platform Secrets / environment variables. Do **not** commit keys to your repo.

- **Streamlit Community Cloud**: App page → `Settings` → `Secrets`, add:

```toml
OPENAI_API_KEY="YOUR_KEY"
```

- **Other deployments**: set environment variable `OPENAI_API_KEY="..."`

---

### 3) Gmsh Setup (CLI / GUI)

This project converts `.geo → .msh` by invoking the **local Gmsh executable (CLI)**.

Official download: [Gmsh download](https://gmsh.info/#Download)

> Note: deployed environments are usually headless (no GUI). Gmsh GUI is typically unavailable on servers.

#### Windows (PATH)

- Install Gmsh (or unzip the portable build)
- Add the directory containing `gmsh.exe` to your system `PATH`

Example (PowerShell, persistent):

```powershell
setx PATH "$env:PATH;C:\Program Files\Gmsh"
```

#### macOS (Homebrew)

```bash
brew install gmsh
```

#### Ubuntu/Debian

```bash
sudo apt-get update && sudo apt-get install -y gmsh
```

In the app sidebar you can also set `gmsh_cmdline` to either `gmsh` or an absolute path (e.g. `C:\Program Files\Gmsh\gmsh.exe`).

---

### 4) Input / Output

- **Input**
  - Upload a `.msh` (recommended / most reliable)
  - Or generate `.msh` from `.geo` (LLM-assisted)
- **Output**
  - Export results as `.vtu` (ParaView)
  - For mechanics problems, `.vtu` can include `Stress` / `vonMises` (if enabled)
