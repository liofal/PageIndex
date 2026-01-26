PYTHON ?= python3.13
VENV ?= .venv
PIP := $(VENV)/bin/pip

.PHONY: venv deps env run run-bg stop clean

venv:
	$(PYTHON) -m venv $(VENV)

deps: venv
	$(PIP) install -r requirements.txt
	$(PIP) install streamlit

env:
	@if [ -f .env ]; then \
		echo ".env already exists"; \
	elif [ -f .env.local ]; then \
		cp .env.local .env; \
		echo "Copied .env.local to .env"; \
	else \
		echo "Create .env with CHATGPT_API_KEY=..."; \
		exit 1; \
	fi

run: deps env
	$(VENV)/bin/streamlit run local_rag_app.py

run-bg: deps env
	nohup $(VENV)/bin/streamlit run local_rag_app.py --server.port 8501 --server.address 0.0.0.0 > /tmp/streamlit.log 2>&1 & echo $$! > /tmp/streamlit.pid
	@echo "Started. PID: $$(cat /tmp/streamlit.pid). Log: /tmp/streamlit.log"

stop:
	@if [ -f /tmp/streamlit.pid ]; then \
		kill $$(cat /tmp/streamlit.pid) && rm /tmp/streamlit.pid; \
	else \
		echo "No /tmp/streamlit.pid found"; \
	fi

clean:
	rm -rf $(VENV)
