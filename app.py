# smart_docs_app.py
# ================================================================
# Codenection Docs Assistant (Google Docs style with AI)
# ================================================================

import os
import re
import subprocess
import numpy as np
import faiss
import requests
import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Preformatted
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
import io, re, unicodedata
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ================================================================
# 1. Repo ingestion (gitingest)
# ================================================================
def run_gitingest(repo_url, output_file="digest.txt"):
    command = ["python", "-m", "gitingest", repo_url]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return f.read()
    return None

# ================================================================
# 2. Prepare RAG (load + chunk + embed)
# ================================================================
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#MODEL_DIR = "./transformer_model"
MODEL_DIR = "sentence-transformers/all-MiniLM-L6-v2"  # or your model
model = SentenceTransformer(MODEL_DIR, device="cpu")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_DIR) and len(os.listdir(MODEL_DIR)) != 0:
        return SentenceTransformer(MODEL_DIR, device="cpu")
    else:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(MODEL_DIR)
        return model

model = load_model()
# Check if local model directory exists
if len(os.listdir(MODEL_DIR)) != 0:
    print(f"Loading model from local directory: {MODEL_DIR}")
    model = SentenceTransformer(MODEL_DIR)
else:
    print(f"Downloading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.save(MODEL_DIR)
    print(f"Model saved locally at {MODEL_DIR}")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_faiss_index(text_data):
    chunks = chunk_text(text_data)
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return chunks, index

def retrieve(query, chunks, index, top_k=3):
    query_vec = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

# ================================================================
# 3. AI Models (Gemini)
# ================================================================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ask_with_context(query, chunks, index, top_k=3):
    context = "\n\n".join(retrieve(query, chunks, index, top_k=top_k))
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    You are a friendly AI assistant helping with repository understanding.
    Use the context below to answer the question.

    Context:
    {context}

    Question:
    {query}
    """
    response = gemini_model.generate_content(
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        request_options={"timeout": 300}
    )
    return response.text

def documentation(text_data):
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    You are an expert technical writer. Generate complete documentation 
    for the following project based on the information provided.
    The documentation should be clear, structured, and professional.

    Codebase:
    {text_data}

    Include mermaid diagrams wherever necessary using the following syntax:
    
    Markdown text
    ```mermaid
    (mermaid codes)
    ```
    Markdown text

    ONLY USE PLAIN MERMAID SYNTAX.
    
    DO NOT PUT ANY PARENTHESES INSIDE THE SYNTAX, SINCE IT WILL BREAK THE CODE.

    For example:
    A --> B[Stuff (explanation)]

    Will cause the mermaid code to break. Just write it as:
    A --> B[Stuff explanation]
    """
    
    response = gemini_model.generate_content(
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        request_options={"timeout": 300}
    )
    return response.text

def save_docs(content, filepath="UPDATED_DOCS.md"):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

def summarize_doc(doc_text):
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"Summarize this documentation in a concise TL;DR format:\n\n{doc_text}"
    response = gemini_model.generate_content([prompt])
    return response.text

def detect_doc_drift(doc_text, diff):
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Compare this documentation with the following code diff. 
    Highlight which sections of the documentation may be outdated.

    Documentation:
    {doc_text}

    Diff:
    {diff}
    """
    response = gemini_model.generate_content([prompt])
    return response.text

def get_repo_info(url: str):
    """Fetches GitHub repository info."""
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", url)
    if not match:
        return None
    owner, repo = match.groups()
    repo_api = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        repo_resp = requests.get(repo_api)
        repo_resp.raise_for_status()
        repo_data = repo_resp.json()
        default_branch = repo_data["default_branch"]
        commits_api = f"https://api.github.com/repos/{owner}/{repo}/commits/{default_branch}"
        commits_resp = requests.get(commits_api)
        commits_resp.raise_for_status()
        commit_data = commits_resp.json()
        latest_commit = commit_data["sha"]
        return {
            "owner": owner,
            "repo": repo,
            "branch": default_branch,
            "last_commit": latest_commit
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching repo info from GitHub API: {e}")
        return None
    
def is_repo_updated_github(info, last_known_commit):
    """Checks if a repo has new commits."""
    if not info:
        return None
    url = f"https://api.github.com/repos/{info['owner']}/{info['repo']}/commits/{info['branch']}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        latest_commit = response.json()["sha"]
        return last_known_commit != latest_commit, latest_commit
    except requests.exceptions.RequestException as e:
        st.error(f"Error checking for updates: {e}")
        return None, None

def update_documentation_from_diff(commit_msg, diff, original_doc):
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    """Updates documentation based on git diffs."""
    prompt = f"""
    You are an expert technical writer. Update the project documentation based on the following GitHub changes.

    Commit message:
    {commit_msg}

    Diff:
    {diff}

    Original Documentation:
    {original_doc}

    Analyze the changes and update the Original Documentation only where relevant.
    CHANGE THE ORIGINAL DOCUMENTATION, BASED  OF THE DIFF AND COMMIT MESSAGE. OUTPUT THE CHANGED ORIGINAL DOCUMENTATION.
    Ignore trivial changes like formatting or comments.
    Do not include any placeholders or instructions to the user in the final output.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"
    
def custom_documentation(custom_req, current_doc):
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    """In case users ask for modification too the current documentation."""

    prompt = f"""
    You are an expert technical writer. Update the current project documentation based on the following custom documentation request.

    Custom Documentation Request:
    {custom_req}

    Current Project Documentation:
    {current_doc}

    Analyze the changes and update the documentation only where relevant. DO NOT CHANGE ANYTHING ELSE.
    Do not include any placeholders or instructions to the user in the final output.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def get_git_changes(owner, repo, branch="main"):
    # Get the latest commit
    commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
    commit_data = requests.get(commit_url).json()

    commit_msg = commit_data["commit"]["message"]

    # Get diff by fetching the commit URL with diff headers
    diff_url = commit_data["html_url"] + ".diff"
    diff_data = requests.get(diff_url).text

    return commit_msg, diff_data

MERMAID_JS = """
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true, theme: "default" });
</script>
"""

def render_mermaid_from_markdown(md_text: str):
    # Split text into markdown + mermaid blocks
    parts = re.split(r"(```mermaid\n.*?\n```)", md_text, flags=re.DOTALL)

    for part in parts:
        if part.strip().startswith("```mermaid"):
            # Extract just the diagram code
            diagram = re.sub(r"```mermaid\n|\n```", "", part).strip()
            html = f"""
            <!doctype html>
            <html>
              <head>{MERMAID_JS}</head>
              <body>
                <pre class="mermaid">{diagram}</pre>
              </body>
            </html>
            """
            st.components.v1.html(html, height=500, scrolling=True)
        elif part.strip():
            # Render normal markdown
            st.markdown(part)

# ================================================================
# 4. Streamlit UI
# ================================================================
st.set_page_config(page_title="Smart Docs Assistant", layout="wide")

# Init session state
if "projects" not in st.session_state:
    st.session_state.projects = {}

if "selected_project" not in st.session_state:
    st.session_state.selected_project = None

if "show_add_modal" not in st.session_state:
    st.session_state.show_add_modal = False

if "project_data" not in st.session_state:
    # project_data holds {proj_name: {"text": ..., "chunks": ..., "index": ..., "file": ...}}
    st.session_state.project_data = {}

if "last_known_commit" not in st.session_state:
    st.session_state.last_known_commit = ""

if "prev_docs_content" not in st.session_state:
    st.session_state.prev_docs_content = ""

with st.sidebar:
    # Sidebar
    st.sidebar.title("📂 Projects")

    # Display existing projects
    to_delete = None
    for proj, info in st.session_state.projects.items():
        cols = st.sidebar.columns([4, 1])
        if cols[0].button(proj, key=f"proj-{proj}"):
            st.session_state.selected_project = proj
        if cols[1].button("❌", key=f"del-{proj}"):
            to_delete = proj

    # Handle deletion
    if to_delete:
        del st.session_state.projects[to_delete]
        if to_delete in st.session_state.project_data:
            del st.session_state.project_data[to_delete]
        if st.session_state.selected_project == to_delete:
            st.session_state.selected_project = list(st.session_state.projects.keys())[0] if st.session_state.projects else None

    # Add project button
    if st.sidebar.button("➕ Add Project"):
        st.session_state.show_add_modal = True

    # Modal (sidebar form) for new repo
    if st.session_state.show_add_modal:
        with st.sidebar.form("add_repo_form", clear_on_submit=True):
            st.write("### Add New Repository")
            new_proj_name = st.text_input("Project Name")
            new_repo_url = st.text_input("Repository URL (GitHub link)")
            submitted = st.form_submit_button("Add")
            if submitted:
                if new_proj_name and new_repo_url:
                    st.sidebar.info("⏳ Ingesting repository...")
                    text_data = run_gitingest(new_repo_url)
                    print(text_data)

                    if text_data:
                        filepath = f"{new_proj_name}.md"
                        # Generate docs
                        docs = documentation(text_data)
                        save_docs(docs, filepath)

                        # Build FAISS index
                        chunks, index = build_faiss_index(text_data)

                        # Save in session state
                        st.session_state.projects[new_proj_name] = {"repo": new_repo_url, "file": filepath}
                        st.session_state.project_data[new_proj_name] = {
                            "text": text_data,
                            "chunks": chunks,
                            "index": index,
                            "file": filepath,
                            "docs": docs,
                        }
                        st.session_state.selected_project = new_proj_name
                        st.sidebar.success(f"✅ Added project: {new_proj_name}")
                        st.rerun()
                    else:
                        st.sidebar.error("⚠️ Failed to ingest repository.")
                    st.session_state.show_add_modal = False
                else:
                    st.sidebar.error("⚠️ Please fill in all fields.")

# ================================================================
# App Branding (Sticky Header: Logo + Name)
# ================================================================
st.markdown(
    """
    <style>
    .app-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #0e1117; /* match Streamlit dark mode */
        padding: 12px 0;
        z-index: 1000;
        border-bottom: 1px solid #333;
    }
    .app-header-content {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    .app-header h1 {
        font-size: 28px;
        color: #f5f5f5;
        margin: 0;
    }
    /* Push content down so it isn’t hidden behind header */
    .main > div {
        padding-top: 90px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Fixed header using st.image + st.markdown inside placeholders
header_col1, header_col2 = st.columns([1, 10])
with header_col1:
    st.image("logo_docupilot.png", width=100)
with header_col2:
    st.markdown("<h1 style='color:#f5f5f5; margin:0;'>DocuPilot</h1>", unsafe_allow_html=True)

# ================================================================
# Main layout (3 panels)
# ================================================================
col1, col2 = st.columns([2, 1])


# ================== Markdown-to-PDF with Internal Links ==================
# --------- Utility: make safe anchor IDs ----------
def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)

def generate_docs_pdf(doc_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=18
    )
    
    styles = getSampleStyleSheet()
    if "CustomCode" not in styles:
        styles.add(ParagraphStyle(name='CustomCode', fontName="Courier", fontSize=9, leading=12, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    Story = []

    # -------- Pass 1: Collect headings as anchors --------
    anchors = [("top", "Table of Contents", 1)]  # always include top
    for line in doc_text.split("\n"):
        if line.startswith("# "):
            anchors.append((slugify(line[2:]), line[2:], 1))
        elif line.startswith("## "):
            anchors.append((slugify(line[3:]), line[3:], 2))
        elif line.startswith("### "):
            anchors.append((slugify(line[4:]), line[4:], 3))

    all_anchors = {a[0] for a in anchors}

    # -------- Inline formatter --------
    def format_inline_md(text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)  # bold
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)  # italic
        text = re.sub(r"\[(.+?)\]\((https?://[^\s]+)\)", r'<a href="\2" color="blue">\1</a>', text)  # external links

        def repl_internal(m):
            label, target = m.group(1), slugify(m.group(2))
            if target in all_anchors:
                return f'<a href="#{target}" color="blue">{label}</a>'
            else:
                return f'<a href="#top" color="blue">{label}</a>'  # fallback
        text = re.sub(r"\[(.+?)\]\(#([^)]+)\)", repl_internal, text)
        return text

    # -------- Add TOC --------
    Story.append(Paragraph('<a name="top"/><a name="table-of-contents"/>', styles["BodyText"]))
    Story.append(Paragraph("Table of Contents", styles["Title"]))
    Story.append(Spacer(1, 12))
    for anchor, title, level in anchors[1:]:  # skip 'top'
        indent = "&nbsp;&nbsp;&nbsp;" * (level - 1)
        toc_entry = f'{indent}<a href="#{anchor}" color="blue">{title}</a>'
        Story.append(Paragraph(toc_entry, styles["BodyText"]))
    Story.append(Spacer(1, 24))

    # -------- Parse content --------
    lines = doc_text.split("\n")
    bullet_buffer, number_buffer, code_buffer = [], [], []
    inside_code = False
    skip_toc = False

    def flush_lists():
        nonlocal bullet_buffer, number_buffer
        if bullet_buffer:
            lst = ListFlowable(
                [ListItem(Paragraph(format_inline_md(item), styles["BodyText"])) for item in bullet_buffer],
                bulletType='bullet'
            )
            Story.append(lst); Story.append(Spacer(1, 6)); bullet_buffer = []
        if number_buffer:
            lst = ListFlowable(
                [ListItem(Paragraph(format_inline_md(item), styles["BodyText"])) for item in number_buffer],
                bulletType='1'
            )
            Story.append(lst); Story.append(Spacer(1, 6)); number_buffer = []

    def flush_code():
        nonlocal code_buffer
        if code_buffer:
            Story.append(Preformatted("\n".join(code_buffer), styles["CustomCode"]))
            Story.append(Spacer(1, 6)); code_buffer = []

    for line in lines:
        stripped = line.rstrip()
        
        if stripped.lower().startswith("# table of contents") or stripped.lower().startswith("## table of contents"):
            skip_toc = True
            continue

        # Stop skipping once we hit the next header
        if skip_toc and (stripped.startswith("# ") or stripped.startswith("## ") or stripped.startswith("### ")):
            skip_toc = False

        if skip_toc:
            continue  # ignore all lines inside original TOC

        if stripped.startswith("```"):
            inside_code = not inside_code
            if not inside_code:
                flush_code()
            continue

        if inside_code:
            code_buffer.append(stripped)
            continue

        if not stripped:
            flush_lists(); Story.append(Spacer(1, 12)); continue

        # Headers
        if stripped.startswith("# "):
            flush_lists(); flush_code()
            anchor = slugify(stripped[2:])
            Story.append(Paragraph(f'<a name="{anchor}"/>{format_inline_md(stripped[2:])}', styles["Heading1"]))
            Story.append(Spacer(1, 12))

        elif stripped.startswith("## "):
            flush_lists(); flush_code()
            anchor = slugify(stripped[3:])
            Story.append(Paragraph(f'<a name="{anchor}"/>{format_inline_md(stripped[3:])}', styles["Heading2"]))
            Story.append(Spacer(1, 8))

        elif stripped.startswith("### "):
            flush_lists(); flush_code()
            anchor = slugify(stripped[4:])
            Story.append(Paragraph(f'<a name="{anchor}"/>{format_inline_md(stripped[4:])}', styles["Heading3"]))
            Story.append(Spacer(1, 6))

        # Lists
        elif stripped.startswith(("- ", "* ")):
            bullet_buffer.append(stripped[2:])
        elif re.match(r"^\d+\.\s+", stripped):
            number_buffer.append(stripped[stripped.find(" ")+1:])

        # Normal paragraph
        else:
            flush_lists(); flush_code()
            Story.append(Paragraph(format_inline_md(stripped), styles["BodyText"]))
            Story.append(Spacer(1, 6))

    flush_lists(); flush_code()

    doc.build(Story)
    buffer.seek(0)
    return buffer.getvalue()

# Center: Documentation Workspace
with col1:
    tab1, tab2 = st.tabs(["Documentation Editing", "Finalised Documentation"])

    with tab1:
        if st.session_state.selected_project:
            proj_name = st.session_state.selected_project
            proj_info = st.session_state.projects[proj_name]
            st.markdown(f"## 📘 {proj_name} Documentation")

            filepath = proj_info["file"]

            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    doc_text = f.read()
            else:
                doc_text = st.session_state.project_data[proj_name]["docs"]

            # Editable doc
            edited_text = st.text_area("Edit Documentation", value=doc_text, height=450)

            col2a, col2b, col2c, col2d = st.columns([1, 1, 1, 1])
            with col2a:
                if st.button("💾 Save"):
                    save_docs(edited_text, filepath=filepath)
                    st.success("✅ Documentation saved!")

            with col2b:
                if st.button("📌 TL;DR"):
                    with st.spinner("Summarizing..."):
                        summary = summarize_doc(edited_text)
                    #st.info(summary)

            with col2c:
                if st.button("⚠️ Check Drift"):
                    repo_url = st.session_state.projects[proj_name]["repo"]
                    repo_info = get_repo_info(repo_url)
                    commit_msg, diff = get_git_changes(repo_info['owner'], repo_info['repo'])
                    with st.spinner("Analyzing..."):
                        drift = detect_doc_drift(edited_text, diff)
                    #st.warning(drift)

            with col2d:
                if st.button("Check new commits"):
                    with st.spinner("Checking GitHub for new commits..."):
                        repo_url = st.session_state.projects[proj_name]["repo"]
                        repo_info = get_repo_info(repo_url)                    
                        if repo_info:
                            if not st.session_state.last_known_commit:
                                st.session_state.last_known_commit = repo_info['last_commit']
                                st.info("No previous commit stored. Storing current commit for future checks.")
                            else:
                                is_updated, latest_commit = is_repo_updated_github(repo_info, st.session_state.last_known_commit)
                                if is_updated:
                                    st.warning("⚠️ New commits detected on GitHub!")
                                    st.session_state.last_known_commit = latest_commit
                                    run_gitingest(repo_url)
                                    st.rerun()
                                else:
                                    st.success("✅ The repository is up-to-date with GitHub.")
            
            st.markdown("Do you want to make any changes to the documentation? Input your requested changes below!")
            # custom_req = st.text_input("Custom Modification:")

            with st.form("my_form", clear_on_submit=True):
                custom_req = st.text_input("Custom Modification:")
                submitted = st.form_submit_button("Submit")

            if submitted:
                updated_docs = custom_documentation(custom_req, doc_text)
                st.session_state.project_data[proj_name]["docs"] = updated_docs
                doc_text = updated_docs

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(updated_docs)

                st.rerun()

        else:
            st.info("No project selected. Please add a project.")
        
        with tab2:
            with st.expander("Export Documentation..."):
                # PDF Export (using structured export)
                if st.button("as .pdf"):
                    pdf_bytes = generate_docs_pdf(doc_text)

                    st.download_button(
                        label="📥 Export as .pdf",
                        data=pdf_bytes,
                        file_name="finalised_documentation.pdf",
                        mime="application/pdf"
                    )

                # Markdown Export (unchanged)
                if st.button("as .md"):
                    st.download_button(
                        label="⬇️ Download Markdown",
                        data=doc_text,
                        file_name="documentation.md",
                        mime="text/markdown"
                    )
            
            try:
                print(edited_text)
                render_mermaid_from_markdown(edited_text)
            except NameError:
                st.markdown("No documentation is produced yet.")

# Right: Smart Assistant
with col2:
    st.markdown("### 🤖 Ask the Docs")

    if st.session_state.selected_project:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": "👋 Hi, I’m DocuPilot Assistant! Ask me anything about your documentation or repository."}
            ]


        # --- Display Chat History ABOVE input ---
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(
                    f"""
                    <div style="
                        background-color:#007BFF;
                        color:black;
                        padding:10px 14px;
                        border-radius:18px;
                        margin-bottom:6px;
                        width:fit-content;
                        max-width:80%;
                        margin-left:auto;
                        text-align:left;
                        font-size:15px;
                    ">
                        👤 {chat['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:  # assistant
                st.markdown(
                    f"""
                    <div style="
                        background-color:#28a745;
                        color:black;
                        padding:10px 14px;
                        border-radius:18px;
                        margin-bottom:6px;
                        width:fit-content;
                        max-width:80%;
                        margin-right:auto;
                        text-align:left;
                        font-size:15px;
                    ">
                        🤖 {chat['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # --- Input bar with rounded Send button (like ChatGPT) ---
        with st.form("chat_form", clear_on_submit=True):
            st.markdown(
                """
                <style>
                .chat-input-container {
                    display: flex;
                    align-items: center;
                    border: 1px solid #ccc;
                    border-radius: 25px;
                    padding: 5px 10px;
                    background-color: #f9f9f9;
                }
                .chat-input {
                    flex-grow: 1;
                    border: none;
                    background: transparent;
                    padding: 8px;
                    font-size: 15px;
                    outline: none;
                }
                .send-btn {
                    background-color: #007BFF;
                    color: white;
                    border: none;
                    border-radius: 50%;
                    width: 36px;
                    height: 36px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    font-size: 16px;
                }
                .send-btn:hover {
                    background-color: #0056b3;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            cols = st.columns([12, 1])
            user_query = cols[0].text_input(
                "Type your message...",
                key="chat_input",
                label_visibility="collapsed",
                placeholder="Ask Docs Assistant..."
            )
            send = cols[1].form_submit_button("▶", use_container_width=True)

            if send and user_query.strip():
                # Save user message
                st.session_state.chat_history.append({"role": "user", "content": user_query})

                # Temporary "typing..." bubble
                placeholder = st.empty()
                placeholder.markdown(
                    """
                    <div style="
                        background-color:#28a745;
                        color:black;
                        padding:10px 14px;
                        border-radius:18px;
                        margin-bottom:6px;
                        width:fit-content;
                        max-width:80%;
                        margin-right:auto;
                        text-align:left;
                        font-size:15px;
                    ">
                        🤖 Assistant is typing...
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Generate assistant response
                proj_name = st.session_state.selected_project
                pdata = st.session_state.project_data.get(proj_name)
                if pdata:
                    response = ask_with_context(user_query, pdata["chunks"], pdata["index"])
                else:
                    response = "⚠️ No data available for this project."

                # Replace typing bubble with actual response
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                placeholder.empty()
                st.rerun()
        
        try:
            st.warning(drift)
        except Exception:
            pass

        try:
            st.info(summary)
        except Exception:
            pass
