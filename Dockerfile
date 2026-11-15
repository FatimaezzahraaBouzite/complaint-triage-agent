FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# The database is built on your machine (python -m triage.data) and copied in with the code.
RUN test -f data/complaints.db || (echo "data/complaints.db is missing: run python -m triage.data first" && exit 1)
ENV PORT=8501
EXPOSE 8501
# REST API instead:  uvicorn app.api:app --host 0.0.0.0 --port $PORT
CMD streamlit run app/ui.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
