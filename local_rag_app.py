"""
Local PageIndex RAG App - No SaaS required
Uses locally generated tree structures + OpenAI for retrieval and answering
Supports single document and global knowledge base modes
"""

import streamlit as st
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import pymupdf
from pageindex.llm import chat

load_dotenv()

OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
DEFAULT_BEDROCK_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-west-1"

BEDROCK_MODELS = [
    {
        "label": "Claude Opus 4.5 (Bedrock, EU profile)",
        "model_id": "anthropic.claude-opus-4-5-20251101-v1:0",
        "inference_profile_arn": "arn:aws:bedrock:eu-west-1:502456974089:inference-profile/eu.anthropic.claude-opus-4-5-20251101-v1:0",
    },
    {
        "label": "Claude Sonnet 4.5 (Bedrock, EU profile)",
        "model_id": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "inference_profile_arn": "arn:aws:bedrock:eu-west-1:502456974089:inference-profile/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    },
    {
        "label": "Claude Haiku 4.5 (Bedrock, EU profile)",
        "model_id": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "inference_profile_arn": "arn:aws:bedrock:eu-west-1:502456974089:inference-profile/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    },
]

DEFAULT_PROVIDER = (os.getenv("PAGEINDEX_PROVIDER") or "openai").lower()
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-sonnet-4-5-20250929-v1:0"

OPENAI_MODELS = [
    "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o", "gpt-5.2-mini", "gpt-5.2",
    "o1-mini", "o1", "o3-mini", "o3", "o4-mini"
]

# --- Helper Functions ---

def load_tree_structure(json_path):
    """Load a PageIndex tree structure from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def flatten_nodes(structure, nodes=None):
    """Flatten tree structure into a list of all nodes"""
    if nodes is None:
        nodes = []
    for item in structure:
        node_copy = {k: v for k, v in item.items() if k != 'nodes'}
        nodes.append(node_copy)
        if 'nodes' in item:
            flatten_nodes(item['nodes'], nodes)
    return nodes

def create_node_map(structure):
    """Create a mapping from node_id to node info"""
    nodes = flatten_nodes(structure)
    return {node['node_id']: node for node in nodes if 'node_id' in node}

def extract_text_from_pdf(pdf_path, start_page, end_page):
    """Extract text from specific pages of a PDF"""
    doc = pymupdf.open(pdf_path)
    text = ""
    for page_num in range(start_page - 1, min(end_page, len(doc))):
        page = doc.load_page(page_num)
        text += f"\n--- Page {page_num + 1} ---\n"
        text += page.get_text()
    doc.close()
    return text

def supports_reasoning_effort(model):
    """Check if model supports reasoning_effort parameter"""
    reasoning_models = [
        # o-series reasoning models
        "o1", "o1-mini", "o1-pro",
        "o3", "o3-mini", "o3-pro",
        "o4-mini",
        # gpt-5.x series with reasoning
        "gpt-5.2", "gpt-5.2-mini",
    ]
    return any(rm in model.lower() for rm in reasoning_models)

def call_llm(
    prompt,
    model="gpt-4o",
    provider="openai",
    reasoning_effort=None,
    aws_region=None,
    aws_profile=None,
    bedrock_inference_profile_arn=None,
    bedrock_max_tokens=None,
):
    """Call LLM with optional reasoning effort and Bedrock settings"""
    messages = [{"role": "user", "content": prompt}]
    text, _finish_reason = chat(
        messages=messages,
        model=model,
        provider=provider,
        api_key=OPENAI_API_KEY,
        region=aws_region,
        profile=aws_profile,
        inference_profile_arn=bedrock_inference_profile_arn,
        temperature=0,
        reasoning_effort=reasoning_effort if provider == "openai" else None,
        max_tokens=bedrock_max_tokens if provider == "bedrock" else None,
    )
    return text

def search_tree(
    query,
    tree_structure,
    model="gpt-4o",
    provider="openai",
    reasoning_effort=None,
    doc_name=None,
    aws_region=None,
    aws_profile=None,
    bedrock_inference_profile_arn=None,
    bedrock_max_tokens=None,
):
    """Use LLM to find relevant nodes in the tree"""
    tree_for_search = json.dumps(tree_structure, indent=2)

    doc_context = f" from document '{doc_name}'" if doc_name else ""
    prompt = f"""You are given a question and a tree structure of a document{doc_context}.
Each node contains a node_id, title, page range (start_index, end_index), and summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{tree_for_search}

Reply in JSON format only:
{{
    "thinking": "<brief reasoning about which nodes are relevant>",
    "node_ids": ["node_id_1", "node_id_2"]
}}
"""

    result_text = call_llm(
        prompt,
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        aws_region=aws_region,
        aws_profile=aws_profile,
        bedrock_inference_profile_arn=bedrock_inference_profile_arn,
        bedrock_max_tokens=bedrock_max_tokens,
    )
    try:
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        return json.loads(result_text)
    except:
        return {"thinking": "Failed to parse", "node_ids": []}

def search_global_documents(
    query,
    documents_info,
    model="gpt-4o",
    provider="openai",
    reasoning_effort=None,
    aws_region=None,
    aws_profile=None,
    bedrock_inference_profile_arn=None,
    bedrock_max_tokens=None,
):
    """First stage: Find which documents are relevant to the query"""
    docs_summary = json.dumps(documents_info, indent=2)

    prompt = f"""You are given a question and a list of documents with their descriptions/summaries.
Your task is to identify which documents are likely to contain information relevant to the question.

Question: {query}

Available documents:
{docs_summary}

Reply in JSON format only:
{{
    "thinking": "<brief reasoning about which documents are relevant>",
    "relevant_docs": ["doc_name_1", "doc_name_2"]
}}

Select ALL documents that might be relevant. If unsure, include the document.
"""

    result_text = call_llm(
        prompt,
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        aws_region=aws_region,
        aws_profile=aws_profile,
        bedrock_inference_profile_arn=bedrock_inference_profile_arn,
        bedrock_max_tokens=bedrock_max_tokens,
    )
    try:
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        return json.loads(result_text)
    except:
        return {"thinking": "Failed to parse", "relevant_docs": []}

def generate_answer(
    query,
    context,
    model="gpt-4o",
    provider="openai",
    reasoning_effort=None,
    multi_doc=False,
    aws_region=None,
    aws_profile=None,
    bedrock_inference_profile_arn=None,
    bedrock_max_tokens=None,
):
    """Generate an answer based on retrieved context"""
    doc_note = "from multiple documents" if multi_doc else "from the document"
    prompt = f"""Answer the question based on the provided context {doc_note}.

Question: {query}

Context:
{context}

Provide a clear, concise answer based only on the context provided. If the context doesn't contain enough information, say so.
{"When citing information, mention which document it comes from." if multi_doc else ""}
"""

    return call_llm(
        prompt,
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        aws_region=aws_region,
        aws_profile=aws_profile,
        bedrock_inference_profile_arn=bedrock_inference_profile_arn,
        bedrock_max_tokens=bedrock_max_tokens,
    )

def process_pdf_with_progress(
    pdf_path,
    model="gpt-4o",
    provider="openai",
    aws_region=None,
    aws_profile=None,
    bedrock_inference_profile_arn=None,
    bedrock_max_tokens=None,
    output_dir=None,
    timeout_minutes=10,
    status_container=None,
):
    """Process a PDF using PageIndex with real-time progress updates"""
    command = [
        "python3", "-u", "run_pageindex.py",  # -u for unbuffered output
        "--pdf_path", str(pdf_path),
        "--model", model
    ]
    if provider:
        command.extend(["--provider", provider])
    if aws_region:
        command.extend(["--aws-region", aws_region])
    if aws_profile:
        command.extend(["--aws-profile", aws_profile])
    if bedrock_inference_profile_arn:
        command.extend(["--bedrock-inference-profile-arn", bedrock_inference_profile_arn])
    if bedrock_max_tokens:
        command.extend(["--bedrock-max-tokens", str(bedrock_max_tokens)])
    if output_dir:
        command.extend(["--output-dir", str(output_dir)])

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=Path(__file__).parent
    )

    output_lines = []
    start_time = __import__('time').time()
    timeout_seconds = timeout_minutes * 60

    try:
        while True:
            # Check timeout
            if __import__('time').time() - start_time > timeout_seconds:
                process.kill()
                return False, "\n".join(output_lines), f"Processing timed out after {timeout_minutes} minutes."

            # Read output line by line
            line = process.stdout.readline()
            if line:
                output_lines.append(line.strip())
                if status_container:
                    # Show last 5 lines of output
                    recent = output_lines[-5:]
                    status_container.code("\n".join(recent), language=None)

            # Check if process finished
            if process.poll() is not None:
                # Read any remaining output
                remaining = process.stdout.read()
                if remaining:
                    output_lines.extend(remaining.strip().split('\n'))
                break

        success = process.returncode == 0
        return success, "\n".join(output_lines), "" if success else "Process failed"

    except Exception as e:
        process.kill()
        return False, "\n".join(output_lines), str(e)

def get_available_trees():
    """Get list of available tree structure files"""
    return get_available_trees_for_workspace("default")


def sanitize_workspace_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or "default"


def list_workspaces() -> list:
    results_dir = Path("results")
    workspaces = set()
    if results_dir.exists():
        for entry in results_dir.iterdir():
            if entry.is_dir():
                workspaces.add(entry.name)
    workspaces.add("default")
    return sorted(workspaces)


def workspace_results_dir(workspace: str) -> Path:
    if workspace == "default":
        return Path("results")
    return Path("results") / workspace


def workspace_uploads_dir(workspace: str) -> Path:
    if workspace == "default":
        return Path("uploads")
    return Path("uploads") / workspace


def ensure_workspace_dirs(workspace: str) -> None:
    workspace_results_dir(workspace).mkdir(parents=True, exist_ok=True)
    workspace_uploads_dir(workspace).mkdir(parents=True, exist_ok=True)


def get_available_trees_for_workspace(workspace: str):
    """Get list of available tree structure files for a workspace"""
    results_dir = workspace_results_dir(workspace)
    if results_dir.exists():
        return list(results_dir.glob("*_structure.json"))
    return []

def find_pdf_for_tree(tree_path, workspace: str):
    """Try to find the original PDF for a tree structure"""
    pdf_name = tree_path.stem.replace("_structure", "")
    uploads_dir = workspace_uploads_dir(workspace)
    possible_paths = [
        Path(f"tests/pdfs/{pdf_name}.pdf"),
        Path(f"pdfs/{pdf_name}.pdf"),
        uploads_dir / f"{pdf_name}.pdf",
        Path(f"{pdf_name}.pdf"),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None

def get_doc_summary(tree_data):
    """Extract a brief summary from tree data"""
    structure = tree_data.get("structure", [])
    if structure:
        # Get root node summary or title
        root = structure[0]
        if 'summary' in root:
            return root['summary'][:200] + "..." if len(root.get('summary', '')) > 200 else root.get('summary', '')
        return root.get('title', 'No title')
    return "No summary available"

def load_all_documents(workspace: str):
    """Load all available documents and their metadata"""
    documents = {}
    for tree_path in get_available_trees_for_workspace(workspace):
        doc_name = tree_path.stem.replace("_structure", "")
        tree_data = load_tree_structure(tree_path)
        documents[doc_name] = {
            "tree_path": tree_path,
            "tree_data": tree_data,
            "structure": tree_data.get("structure", []),
            "node_map": create_node_map(tree_data.get("structure", [])),
            "pdf_path": find_pdf_for_tree(tree_path, workspace),
            "summary": get_doc_summary(tree_data),
            "node_count": len(flatten_nodes(tree_data.get("structure", [])))
        }
    return documents

# --- Streamlit App ---

st.set_page_config(page_title="Local PageIndex RAG", page_icon="📄", layout="wide")

st.title("📄 Local PageIndex RAG")
st.caption("Query your documents using reasoning-based retrieval - no external API required")

# Sidebar
with st.sidebar:
    st.header("📁 Documents")

    if "workspace_select" not in st.session_state:
        st.session_state["workspace_select"] = "default"
    pending_workspace = st.session_state.pop("workspace_select_pending", None)
    if pending_workspace:
        st.session_state["workspace_select"] = pending_workspace

    workspace_options = list_workspaces()
    create_workspace_label = "Create new..."
    if create_workspace_label not in workspace_options:
        workspace_options.append(create_workspace_label)
    selected_value = st.session_state.get("workspace_select", "default")
    default_workspace_index = workspace_options.index(selected_value) if selected_value in workspace_options else 0
    selected_workspace = st.selectbox(
        "Workspace",
        workspace_options,
        index=default_workspace_index,
        key="workspace_select",
        help="Work with a subset of documents stored under results/<workspace>/ and uploads/<workspace>/"
    )
    if selected_workspace == create_workspace_label:
        new_workspace_name = st.text_input("New workspace name", key="workspace_new_name")
        if st.button("Create workspace"):
            workspace_name = sanitize_workspace_name(new_workspace_name)
            ensure_workspace_dirs(workspace_name)
            st.session_state["workspace_select_pending"] = workspace_name
            st.rerun()
        workspace = "default"
    else:
        workspace = selected_workspace

    ensure_workspace_dirs(workspace)

    with st.expander("☁️ Bedrock Settings", expanded=False):
        bedrock_region = st.text_input(
            "AWS region",
            value=DEFAULT_BEDROCK_REGION,
            help="Bedrock region (default: eu-west-1)"
        )
        bedrock_profile = st.text_input(
            "AWS profile (SSO)",
            value=os.getenv("AWS_PROFILE", ""),
            help="Optional AWS profile name"
        )
        bedrock_inference_profile_override = st.text_input(
            "Inference profile ARN (optional override)",
            value=os.getenv("BEDROCK_INFERENCE_PROFILE_ARN", ""),
            help="Leave blank to use the selected model's default EU inference profile."
        )
        bedrock_max_tokens = st.number_input(
            "Max output tokens",
            min_value=256,
            max_value=8192,
            value=2048,
            step=256,
            help="Max output tokens for Bedrock responses"
        )

    # --- Add New Document Section ---
    with st.expander("➕ Add New Document", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDFs to process with PageIndex"
        )

        process_provider_options = ["OpenAI", "Bedrock"]
        process_provider_index = 0 if DEFAULT_PROVIDER != "bedrock" else 1
        process_provider = st.selectbox(
            "Processing provider",
            process_provider_options,
            index=process_provider_index,
            key="process_provider",
            help="Provider used for tree generation"
        )

        if process_provider == "Bedrock":
            bedrock_labels = {m["model_id"]: m["label"] for m in BEDROCK_MODELS}
            bedrock_ids = [m["model_id"] for m in BEDROCK_MODELS]
            bedrock_default_index = 0
            if DEFAULT_BEDROCK_MODEL_ID in bedrock_ids:
                bedrock_default_index = bedrock_ids.index(DEFAULT_BEDROCK_MODEL_ID)
            process_model = st.selectbox(
                "Processing model",
                bedrock_ids,
                index=bedrock_default_index,
                key="process_model",
                format_func=lambda m: bedrock_labels.get(m, m),
                help="Bedrock model used for tree generation"
            )
            process_profile_arn = next(
                (m["inference_profile_arn"] for m in BEDROCK_MODELS if m["model_id"] == process_model),
                None,
            )
        else:
            process_model = st.selectbox(
                "Processing model",
                OPENAI_MODELS,
                key="process_model",
                help="OpenAI model used for tree generation"
            )

        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")
            if st.button("🚀 Process Document(s)", type="primary"):
                uploads_dir = workspace_uploads_dir(workspace)
                uploads_dir.mkdir(parents=True, exist_ok=True)

                progress_bar = st.progress(0)
                file_status = st.empty()
                output_container = st.empty()

                success_count = 0
                for i, uploaded_file in enumerate(uploaded_files):
                    file_status.info(f"📄 Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")

                    # Save uploaded file
                    pdf_path = uploads_dir / uploaded_file.name
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Process with real-time progress
                    success, stdout, stderr = process_pdf_with_progress(
                        pdf_path,
                        process_model,
                        provider=process_provider.lower(),
                        aws_region=bedrock_region or None,
                        aws_profile=bedrock_profile or None,
                        bedrock_inference_profile_arn=bedrock_inference_profile_override or process_profile_arn,
                        bedrock_max_tokens=int(bedrock_max_tokens) if bedrock_max_tokens else None,
                        timeout_minutes=10,
                        status_container=output_container,
                        output_dir=workspace_results_dir(workspace),
                    )

                    if success:
                        success_count += 1
                        output_container.success(f"✅ {uploaded_file.name} done!")
                    else:
                        st.warning(f"⚠️ Failed: {uploaded_file.name}")
                        with st.expander(f"Error details for {uploaded_file.name}"):
                            st.code(stderr if stderr else stdout)

                    progress_bar.progress((i + 1) / len(uploaded_files))

                file_status.empty()
                output_container.empty()
                progress_bar.empty()

                if success_count == len(uploaded_files):
                    st.success(f"✅ All {success_count} document(s) processed!")
                else:
                    st.warning(f"⚠️ {success_count}/{len(uploaded_files)} documents processed successfully")
                st.rerun()

    st.divider()

    # --- Query Mode Selection ---
    available_trees = get_available_trees_for_workspace(workspace)

    if not available_trees:
        st.warning("No documents yet. Upload a PDF above to get started.")
        st.stop()

    st.subheader("Query Mode")
    query_mode = st.radio(
        "Select mode",
        ["📄 Single Document", "🌐 Global Knowledge Base"],
        label_visibility="collapsed",
        help="Single: query one document. Global: query across all documents."
    )

    is_global_mode = query_mode == "🌐 Global Knowledge Base"

    # Show document selector only in single mode
    selected_tree = None
    if not is_global_mode:
        st.subheader("Select Document")
        selected_tree = st.selectbox(
            "Available documents",
            available_trees,
            format_func=lambda x: x.stem.replace("_structure", ""),
            label_visibility="collapsed"
        )

        # Show document info
        if selected_tree:
            tree_data = load_tree_structure(selected_tree)
            structure = tree_data.get("structure", [])
            node_count = len(flatten_nodes(structure))

            pdf_path = find_pdf_for_tree(selected_tree, workspace)

            st.caption(f"📊 {node_count} nodes indexed")
            if pdf_path:
                st.caption(f"📎 PDF: {pdf_path.name}")
            else:
                st.caption("⚠️ Original PDF not found")
    else:
        # Show global stats
        total_docs = len(available_trees)
        all_docs = load_all_documents(workspace)
        total_nodes = sum(d["node_count"] for d in all_docs.values())
        st.caption(f"📚 {total_docs} documents | {total_nodes} total nodes")

        with st.expander("📋 Document List"):
            for doc_name, doc_info in all_docs.items():
                st.write(f"• **{doc_name}** ({doc_info['node_count']} nodes)")

    st.divider()

    # --- Query Settings ---
    st.subheader("Settings")
    query_provider_options = ["OpenAI", "Bedrock"]
    query_provider_index = 0 if DEFAULT_PROVIDER != "bedrock" else 1
    query_provider = st.selectbox(
        "Query provider",
        query_provider_options,
        index=query_provider_index,
        key="query_provider"
    )

    if query_provider == "Bedrock":
        bedrock_labels = {m["model_id"]: m["label"] for m in BEDROCK_MODELS}
        bedrock_ids = [m["model_id"] for m in BEDROCK_MODELS]
        bedrock_default_index = 0
        if DEFAULT_BEDROCK_MODEL_ID in bedrock_ids:
            bedrock_default_index = bedrock_ids.index(DEFAULT_BEDROCK_MODEL_ID)
        query_model = st.selectbox(
            "Query model",
            bedrock_ids,
            index=bedrock_default_index,
            key="query_model",
            format_func=lambda m: bedrock_labels.get(m, m)
        )
        query_profile_arn = next(
            (m["inference_profile_arn"] for m in BEDROCK_MODELS if m["model_id"] == query_model),
            None,
        )
    else:
        query_model = st.selectbox(
            "Query model",
            OPENAI_MODELS,
            key="query_model"
        )

    # Show reasoning effort only for supported models
    reasoning_effort = None
    if query_provider == "OpenAI" and supports_reasoning_effort(query_model):
        reasoning_effort = st.select_slider(
            "Reasoning effort",
            options=["low", "medium", "high"],
            value="medium",
            help="Higher effort = more thorough reasoning but slower/costlier"
        )

# Main content area
if is_global_mode:
    # --- Global Knowledge Base Mode ---
    st.subheader("🌐 Global Knowledge Base")
    st.caption("Query across all your documents")

    all_docs = load_all_documents(workspace)

    # Query input
    query = st.text_input(
        "Your question:",
        placeholder="e.g., What are the key findings across all documents?",
        label_visibility="collapsed",
        key="global_query"
    )

    if query:
        # Stage 1: Find relevant documents
        with st.spinner("🔍 Finding relevant documents..."):
            docs_info = [
                {"name": name, "summary": info["summary"], "node_count": info["node_count"]}
                for name, info in all_docs.items()
            ]
            doc_search_result = search_global_documents(
                query,
                docs_info,
                model=query_model,
                provider=query_provider.lower(),
                reasoning_effort=reasoning_effort,
                aws_region=bedrock_region or None,
                aws_profile=bedrock_profile or None,
                bedrock_inference_profile_arn=bedrock_inference_profile_override or (query_profile_arn if query_provider == "Bedrock" else None),
                bedrock_max_tokens=int(bedrock_max_tokens) if bedrock_max_tokens else None,
            )

        st.subheader("🔍 Document Selection")
        st.info(f"**Reasoning:** {doc_search_result.get('thinking', 'N/A')}")

        relevant_doc_names = doc_search_result.get("relevant_docs", [])

        if not relevant_doc_names:
            st.warning("No relevant documents found.")
        else:
            st.write(f"**Found {len(relevant_doc_names)} relevant document(s):**")
            for doc_name in relevant_doc_names:
                st.write(f"• {doc_name}")

            # Stage 2: Search within relevant documents
            all_context_parts = []
            all_retrieved_nodes = []

            with st.spinner("🔍 Searching within documents..."):
                for doc_name in relevant_doc_names:
                    if doc_name not in all_docs:
                        continue

                    doc_info = all_docs[doc_name]
                    search_result = search_tree(
                        query,
                        doc_info["structure"],
                        model=query_model,
                        provider=query_provider.lower(),
                        reasoning_effort=reasoning_effort,
                        doc_name=doc_name,
                        aws_region=bedrock_region or None,
                        aws_profile=bedrock_profile or None,
                        bedrock_inference_profile_arn=bedrock_inference_profile_override or (query_profile_arn if query_provider == "Bedrock" else None),
                        bedrock_max_tokens=int(bedrock_max_tokens) if bedrock_max_tokens else None,
                    )

                    retrieved_nodes = search_result.get("node_ids", [])
                    node_map = doc_info["node_map"]
                    pdf_path = doc_info["pdf_path"]

                    for node_id in retrieved_nodes:
                        if node_id in node_map:
                            node = node_map[node_id]
                            start = node.get('start_index', 1)
                            end = node.get('end_index', start)

                            all_retrieved_nodes.append({
                                "doc_name": doc_name,
                                "node_id": node_id,
                                "title": node.get('title', 'Untitled'),
                                "pages": f"{start}-{end}"
                            })

                            # Get context
                            if pdf_path:
                                text = extract_text_from_pdf(pdf_path, start, end)
                                all_context_parts.append(f"## [{doc_name}] {node.get('title', 'Section')}\n{text}")
                            elif 'summary' in node:
                                all_context_parts.append(f"## [{doc_name}] {node.get('title', 'Section')}\n{node['summary']}")

            # Show retrieved nodes
            st.subheader("📑 Retrieved Sections")
            if all_retrieved_nodes:
                for item in all_retrieved_nodes:
                    col1, col2, col3 = st.columns([2, 1, 3])
                    with col1:
                        st.write(f"**{item['doc_name']}**")
                    with col2:
                        st.write(f"p.{item['pages']}")
                    with col3:
                        st.write(item['title'])

                context = "\n\n".join(all_context_parts)

                # Show context
                with st.expander("📝 Retrieved Context", expanded=False):
                    st.text(context[:8000] + "..." if len(context) > 8000 else context)

                # Generate answer
                st.subheader("💡 Answer")
                with st.spinner("Generating answer..."):
                    answer = generate_answer(
                        query,
                        context,
                        model=query_model,
                        provider=query_provider.lower(),
                        reasoning_effort=reasoning_effort,
                        multi_doc=True,
                        aws_region=bedrock_region or None,
                        aws_profile=bedrock_profile or None,
                        bedrock_inference_profile_arn=bedrock_inference_profile_override or (query_profile_arn if query_provider == "Bedrock" else None),
                        bedrock_max_tokens=int(bedrock_max_tokens) if bedrock_max_tokens else None,
                    )
                st.markdown(answer)
            else:
                st.warning("No relevant sections found in the selected documents.")

elif selected_tree:
    # --- Single Document Mode ---
    tree_data = load_tree_structure(selected_tree)
    structure = tree_data.get("structure", [])
    node_map = create_node_map(structure)
    pdf_path = find_pdf_for_tree(selected_tree, workspace)

    doc_name = selected_tree.stem.replace("_structure", "")
    st.subheader(f"📄 {doc_name}")

    # Document structure viewer
    with st.expander("📊 Document Structure", expanded=False):
        st.json(tree_data, expanded=False)

    # Query input
    query = st.text_input(
        "Your question:",
        placeholder="e.g., What was the revenue growth?",
        label_visibility="collapsed",
        key="single_query"
    )

    if query:
        # Search
        with st.spinner("🔍 Searching document tree..."):
            search_result = search_tree(
                query,
                structure,
                model=query_model,
                provider=query_provider.lower(),
                reasoning_effort=reasoning_effort,
                aws_region=bedrock_region or None,
                aws_profile=bedrock_profile or None,
                bedrock_inference_profile_arn=bedrock_inference_profile_override or (query_profile_arn if query_provider == "Bedrock" else None),
                bedrock_max_tokens=int(bedrock_max_tokens) if bedrock_max_tokens else None,
            )

        # Show reasoning
        st.subheader("🔍 Retrieval")
        thinking = search_result.get('thinking', 'N/A')
        st.info(f"**Reasoning:** {thinking}")

        retrieved_nodes = search_result.get("node_ids", [])

        if not retrieved_nodes:
            st.warning("No relevant nodes found.")
        else:
            st.write(f"**Retrieved {len(retrieved_nodes)} node(s):**")

            context_parts = []
            for node_id in retrieved_nodes:
                if node_id in node_map:
                    node = node_map[node_id]
                    start = node.get('start_index', 1)
                    end = node.get('end_index', start)

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**{node_id}** (p.{start}-{end})")
                    with col2:
                        st.write(node.get('title', 'Untitled'))

                    # Get context
                    if pdf_path:
                        text = extract_text_from_pdf(pdf_path, start, end)
                        context_parts.append(f"## {node.get('title', 'Section')}\n{text}")
                    elif 'summary' in node:
                        context_parts.append(f"## {node.get('title', 'Section')}\n{node['summary']}")

            context = "\n\n".join(context_parts)

            # Show context
            with st.expander("📝 Retrieved Context", expanded=False):
                st.text(context[:5000] + "..." if len(context) > 5000 else context)

            # Generate answer
            st.subheader("💡 Answer")
            with st.spinner("Generating answer..."):
                answer = generate_answer(
                    query,
                    context,
                    model=query_model,
                    provider=query_provider.lower(),
                    reasoning_effort=reasoning_effort,
                    aws_region=bedrock_region or None,
                    aws_profile=bedrock_profile or None,
                    bedrock_inference_profile_arn=bedrock_inference_profile_override or (query_profile_arn if query_provider == "Bedrock" else None),
                    bedrock_max_tokens=int(bedrock_max_tokens) if bedrock_max_tokens else None,
                )
            st.markdown(answer)

# Footer
st.divider()
st.caption("Powered by PageIndex tree structures + OpenAI/Bedrock | No PageIndex SaaS required")
