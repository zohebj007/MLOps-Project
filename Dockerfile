# Dockerfile.sagemaker (recommended)
FROM 763104351884.dkr.ecr.ap-south-1.amazonaws.com/sklearn-inference:1.2-1

WORKDIR /opt/program

# Copy application code
COPY app.py /opt/program/
COPY templates /opt/program/templates

# Copy model artefacts (model.joblib or model.tar.gz)
COPY models /opt/ml/model

# If model is model.tar.gz, extract it
RUN if [ -f /opt/ml/model/model.tar.gz ]; then \
      tar -xzf /opt/ml/model/model.tar.gz -C /opt/ml/model && \
      rm -f /opt/ml/model/model.tar.gz ; \
    fi

# Install extra libraries your app needs (Flask, pandas etc.)
# NOTE: do NOT re-install scikit-learn to avoid binary mismatch
RUN pip install --no-cache-dir Flask pandas numpy joblib

EXPOSE 5000
ENV PYTHONUNBUFFERED=1

CMD ["python", "/opt/program/app.py"]