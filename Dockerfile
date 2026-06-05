FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/trustheart

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY tests ./tests
RUN pip install --no-cache-dir -e . --no-deps

ENTRYPOINT ["trustheart"]
