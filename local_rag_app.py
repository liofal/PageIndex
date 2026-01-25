"""
Local PageIndex RAG App - No SaaS required
Uses locally generated tree structures + OpenAI for retrieval and answering
Supports single document and global knowledge base modes
"""

import streamlit as st
import json
import os
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import openai
import pymupdf

load_dotenv()

OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

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

def call_llm(prompt, model="gpt-4o", reasoning_effort=None):
    """Call LLM with optional reasoning effort"""
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Add reasoning_effort for supported models
    if reasoning_effort and supports_reasoning_effort(model):
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = 0

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

def search_tree(query, tree_structure, model="gpt-4o", reasoning_effort=None, doc_name=None):
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

    result_text = call_llm(prompt, model, reasoning_effort)
    try:
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        return json.loads(result_text)
    except:
        return {"thinking": "Failed to parse", "node_ids": []}

def search_global_documents(query, documents_info, model="gpt-4o", reasoning_effort=None):
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

    result_text = call_llm(prompt, model, reasoning_effort)
    try:
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        return json.loads(result_text)
    except:
        return {"thinking": "Failed to parse", "relevant_docs": []}

def generate_answer(query, context, model="gpt-4o", reasoning_effort=None, multi_doc=False):
    """Generate an answer based on retrieved context"""
    doc_note = "from multiple documents" if multi_doc else "from the document"
    prompt = f"""Answer the question based on the provided context {doc_note}.

Question: {query}

Context:
{context}

Provide a clear, concise answer based only on the context provided. If the context doesn't contain enough information, say so.
{"When citing information, mention which document it comes from." if multi_doc else ""}
"""

    return call_llm(prompt, model, reasoning_effort)

def process_pdf_with_progress(pdf_path, model="gpt-4o", timeout_minutes=10, status_container=None):
    """Process a PDF using PageIndex with real-time progress updates"""
    process = subprocess.Popen(
        [
            "python3", "-u", "run_pageindex.py",  # -u for unbuffered output
            "--pdf_path", str(pdf_path),
            "--model", model
        ],
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
    results_dir = Path("results")
    if results_dir.exists():
        return list(results_dir.glob("*_structure.json"))
    return []

def find_pdf_for_tree(tree_path):
    """Try to find the original PDF for a tree structure"""
    pdf_name = tree_path.stem.replace("_structure", "")
    possible_paths = [
        Path(f"tests/pdfs/{pdf_name}.pdf"),
        Path(f"pdfs/{pdf_name}.pdf"),
        Path(f"uploads/{pdf_name}.pdf"),
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

def load_all_documents():
    """Load all available documents and their metadata"""
    documents = {}
    for tree_path in get_available_trees():
        doc_name = tree_path.stem.replace("_structure", "")
        tree_data = load_tree_structure(tree_path)
        documents[doc_name] = {
            "tree_path": tree_path,
            "tree_data": tree_data,
            "structure": tree_data.get("structure", []),
            "node_map": create_node_map(tree_data.get("structure", [])),
            "pdf_path": find_pdf_for_tree(tree_path),
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

    # --- Add New Document Section ---
    with st.expander("➕ Add New Document", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDFs to process with PageIndex"
        )

        process_model = st.selectbox(
            "Processing model",
            ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o", "gpt-5.2-mini", "gpt-5.2",
             "o1-mini", "o1", "o3-mini", "o3", "o4-mini"],
            key="process_model",
            help="Model used for tree generation"
        )

        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")
            if st.button("🚀 Process Document(s)", type="primary"):
                uploads_dir = Path("uploads")
                uploads_dir.mkdir(exist_ok=True)

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
                        pdf_path, process_model,
                        timeout_minutes=10,
                        status_container=output_container
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
    available_trees = get_available_trees()

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

            pdf_path = find_pdf_for_tree(selected_tree)

            st.caption(f"📊 {node_count} nodes indexed")
            if pdf_path:
                st.caption(f"📎 PDF: {pdf_path.name}")
            else:
                st.caption("⚠️ Original PDF not found")
    else:
        # Show global stats
        total_docs = len(available_trees)
        all_docs = load_all_documents()
        total_nodes = sum(d["node_count"] for d in all_docs.values())
        st.caption(f"📚 {total_docs} documents | {total_nodes} total nodes")

        with st.expander("📋 Document List"):
            for doc_name, doc_info in all_docs.items():
                st.write(f"• **{doc_name}** ({doc_info['node_count']} nodes)")

    st.divider()

    # --- Query Settings ---
    st.subheader("Settings")
    query_model = st.selectbox(
        "Query model",
        ["gpt-5.2", "gpt-5.2-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o",
         "o1-mini", "o1", "o3-mini", "o3", "o4-mini"],
        key="query_model"
    )

    # Show reasoning effort only for supported models
    reasoning_effort = None
    if supports_reasoning_effort(query_model):
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

    all_docs = load_all_documents()

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
            doc_search_result = search_global_documents(query, docs_info, model=query_model, reasoning_effort=reasoning_effort)

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
                        query, doc_info["structure"],
                        model=query_model,
                        reasoning_effort=reasoning_effort,
                        doc_name=doc_name
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
                    answer = generate_answer(query, context, model=query_model, reasoning_effort=reasoning_effort, multi_doc=True)
                st.markdown(answer)
            else:
                st.warning("No relevant sections found in the selected documents.")

elif selected_tree:
    # --- Single Document Mode ---
    tree_data = load_tree_structure(selected_tree)
    structure = tree_data.get("structure", [])
    node_map = create_node_map(structure)
    pdf_path = find_pdf_for_tree(selected_tree)

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
            search_result = search_tree(query, structure, model=query_model, reasoning_effort=reasoning_effort)

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
                answer = generate_answer(query, context, model=query_model, reasoning_effort=reasoning_effort)
            st.markdown(answer)

# Footer
st.divider()
st.caption("Powered by PageIndex tree structures + OpenAI | No PageIndex SaaS required")
