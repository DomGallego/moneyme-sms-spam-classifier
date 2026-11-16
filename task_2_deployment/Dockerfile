# Python patch tag is fixed; record the resolved image digest in the smoke-test report.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ENABLE_HOSTED=false
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY configs ./configs
COPY prompts ./prompts
COPY artifacts/release ./artifacts/release
COPY reports/catalog.json ./reports/catalog.json
COPY task_2_deployment ./task_2_deployment
RUN mkdir -p /app/runtime && chown -R app:app /app/runtime
ENV SERVICE_LEDGER=/app/runtime/service-budget.sqlite
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "task_2_deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
