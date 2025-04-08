FROM python:3.12-bookworm

#RUN adduser --disabled-password --gecos "" appuser && \
#    mkdir -p /home/appuser/app && \
#    chown appuser:appuser /home/appuser/app

WORKDIR /home/appuser/app

USER root

RUN apt-get update && \
    apt-get install -y curl git wget gnupg && \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/appuser/.poetry python3 -



ENV PATH="/home/appuser/.poetry/bin:${PATH}"


COPY pyproject.toml poetry.lock ./

COPY . /home/appuser/app
COPY README.md ./
RUN poetry install


USER root

COPY entrypoint.sh /home/appuser/app/entrypoint.sh
RUN chmod +x /home/appuser/app/entrypoint.sh
#RUN chown -R appuser:appuser /home/appuser/app

#USER appuser

EXPOSE 8080

ENTRYPOINT ["/home/appuser/app/entrypoint.sh"]

CMD ["sh", "-c", "poetry run streamlit run app.py --server.port=8080 --server.address=0.0.0.0" ]