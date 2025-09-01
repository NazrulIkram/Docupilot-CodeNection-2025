import streamlit as st
import subprocess
import os
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import requests
import re
import git
from dotenv import load_dotenv

# ================================================================
# 0. Configuration and Initialization
# ================================================================

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(layout="wide", page_title="iFAST Documentation Assistant")

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Please create a .env file with your API key.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Use Streamlit's caching to avoid re-running expensive functions
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel("gemini-1.5-flash")

# Initialize models
embedding_model = get_embedding_model()
gemini_model = get_gemini_model()


# ================================================================
# 1. Backend Functions (Refactored)
# ================================================================

def run_gitingest(repo_url, repo_dir="repo_content"):
    """Clones a repo and ingests its content into digest.txt."""
    if os.path.exists(repo_dir):
        st.info("Repository directory already exists. Pulling latest changes...")
        try:
            repo = git.Repo(repo_dir)
            repo.remotes.origin.pull()
        except git.GitCommandError as e:
            st.error(f"Error pulling changes: {e}")
            return None
    else:
        st.info(f"Cloning repo from {repo_url}...")
        try:
            git.Repo.clone_from(repo_url, repo_dir)
        except git.GitCommandError as e:
            st.error(f"Error cloning repo: {e}")
            return None

    st.info("Running gitingest to create digest.txt...")
    command = ["python", "-m", "gitingest", repo_url]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error with gitingest: {e.stderr}")
        return False

def chunk_text(text, chunk_size=800, overlap=100):
    """Simple text chunking function."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@st.cache_resource
def create_faiss_index(chunks):
    """Creates embeddings and a FAISS index from text chunks."""
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve(query, index, chunks, top_k=3):
    """Retrieves top-k similar chunks for a query."""
    query_vec = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

def ask_with_context(query, index, chunks, top_k=3):
    """Performs RAG to answer a query."""
    context = "\n\n".join(retrieve(query, index, chunks, top_k=top_k))
    prompt = f"""
    You are an AI assistant helping with repository understanding.
    Use the context below to answer the question.

    Context:
    {context}

    Question:
    {query}

    
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def generate_full_documentation(text_data):
    """Generates complete documentation from a codebase."""
    prompt = f"""
    You are an expert technical writer. Generate complete, professional documentation for the following project.
    The documentation should be clear, structured with markdown headings, and easy to read.
    Do not include any placeholders or instructions to the user in the final output.

    Codebase:
    {text_data}
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def update_documentation_from_diff(commit_msg, diff):
    """Updates documentation based on git diffs."""
    prompt = f"""
    You are an expert technical writer. Update the project documentation based on the following GitHub changes.

    Commit message:
    {commit_msg}

    Diff:
    {diff}

    Analyze the changes and update the documentation only where relevant.
    Ignore trivial changes like formatting or comments.
    Format your response in Markdown, ready to be added to the documentation.
    Do not include any placeholders or instructions to the user in the final output.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"
    
def custom_documentation(custom_req, current_doc):
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

def get_git_changes(repo_dir="repo_content"):
    """Get the latest commit message and diff from the repo."""
    try:
        repo = git.Repo(repo_dir)
        commit_msg = repo.head.commit.message.strip()
        diff = repo.git.diff("HEAD~1", "HEAD")
        return commit_msg, diff
    except Exception as e:
        st.error(f"Error getting git changes: {e}")
        return None, None

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


# ================================================================
# 2. Streamlit UI
# ================================================================

#st.set_page_config(layout="wide", page_title="iFAST Documentation Assistant")

st.title("iFAST Documentation Assistant")
st.subheader("Smarter, Faster, Maintainable Documentation for the Real World")

# Main user input area
with st.sidebar:
    st.header("Project Setup")
    repo_url = st.text_input(
        "Enter GitHub Repository URL:",
        "https://github.com/coderamp-labs/gitingest"
    )
    if st.button("Ingest/Update Repository", use_container_width=True):
        if repo_url:
            with st.spinner("Processing repository..."):
                if run_gitingest(repo_url):
                    st.success("Repository ingested and ready!")
                    st.session_state.repo_ingested = True
                    # Invalidate caches to re-create index
                    st.cache_data.clear()
                    st.cache_resource.clear()
                else:
                    st.session_state.repo_ingested = False
        else:
            st.error("Please enter a valid GitHub URL.")

# Initialize session state for the app
if 'repo_ingested' not in st.session_state:
    st.session_state.repo_ingested = False
if 'docs_content' not in st.session_state:
    st.session_state.docs_content = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_known_commit' not in st.session_state:
    st.session_state.last_known_commit = ""
if 'prev_docs_content' not in st.session_state:
    st.session_state.prev_docs_content = ""

if st.session_state.repo_ingested:
    st.success("Repository is ready!")
    
    # Read the ingested data for the entire app
    with open("digest.txt", "r", encoding="utf-8") as f:
        text_data = f.read()
    
    # Create chunks and FAISS index
    chunks = chunk_text(text_data)
    faiss_index = create_faiss_index(chunks)

    tab1, tab2, tab3 = st.tabs(["💬 Q&A Chatbot", "📄 Full Documentation", "🔄 Maintenance & Updates"])

    with tab1:
        st.header("AI-Powered Q&A Chatbot")
        st.markdown("Ask questions about the repository code and get instant answers based on the ingested codebase.")
        
        # Display chat messages from history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the repository..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response
            with st.spinner("Thinking..."):
                response = ask_with_context(prompt, faiss_index, chunks)
            
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    with tab2:
        st.header("Auto-Generated Documentation")
        
        if st.button("Generate Full Documentation", use_container_width=True) and st.session_state.docs_content == "":
            with st.spinner("Generating comprehensive documentation... This may take a moment."):
                docs = generate_full_documentation(text_data)
                st.session_state.docs_content = docs
            st.success("Documentation generated!")
        
        docs = st.session_state.docs_content
        
        if st.session_state.docs_content:
            st.markdown(st.session_state.docs_content)

            st.markdown("Do you want to make any changes to the documentation? Input your requested changes below!")
            # custom_req = st.text_input("Custom Modification:")

            with st.form("my_form", clear_on_submit=True):
                custom_req = st.text_input("Custom Modification:")
                submitted = st.form_submit_button("Submit")

            if submitted:
                updated_docs = custom_documentation(custom_req, docs)
                st.session_state.docs_content = updated_docs
                st.session_state.prev_docs_content = docs
                docs = updated_docs
                st.rerun()
            
            if st.button("Restore previous documentation version"):
                st.session_state.docs_content = st.session_state.prev_docs_content
                st.rerun()
                
    with tab3:
        st.header("Maintenance & Updates")
        st.markdown("This section helps you keep your documentation in sync with the latest code changes.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check for Latest Commits", use_container_width=True):
                with st.spinner("Checking GitHub for new commits..."):
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
                            else:
                                st.success("✅ The repository is up-to-date with GitHub.")

        with col2:
            if st.button("Generate Doc Update from Diffs", use_container_width=True):
                with st.spinner("Analyzing recent code changes..."):
                    commit_msg, diff = get_git_changes()
                    if commit_msg and diff:
                        st.info("Generating documentation update from last commit diff...")
                        update_text = update_documentation_from_diff(commit_msg, diff)
                        st.session_state.update_text = update_text
                        st.success("Update generated!")

        if 'update_text' in st.session_state and st.session_state.update_text:
            st.subheader("Suggested Documentation Update")
            st.markdown(st.session_state.update_text)

else:
    st.info("Please enter a GitHub URL in the sidebar and click 'Ingest/Update Repository' to begin.")
