FROM --platform=linux/amd64  pytorch/pytorch

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN adduser --system --group user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app

RUN python -m pip install \
    --no-color \
    --requirement requirements.txt

COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app

COPY --chown=user:user weights /opt/app/weights
COPY --chown=user:user models /opt/app/models
COPY --chown=user:user loss /opt/app/loss
COPY --chown=user:user data_manager /opt/app/data_manager
COPY --chown=user:user data /opt/app/
COPY --chown=user:user utils.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
