FROM --platform=linux/amd64  pytorch/pytorch

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1
ENV PATH="/home/user/.local/bin:${PATH}"

RUN adduser --system --group user
USER user

WORKDIR /opt/app
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
COPY --chown=user:user requirements.txt /opt/app

RUN python -m pip install --user -U pip 
RUN pip install --user --no-cache-dir -r /opt/app/requirements.txt

COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app

COPY --chown=user:user weights /opt/app/weights
COPY --chown=user:user models /opt/app/models
COPY --chown=user:user data_manager /opt/app/data_manager
COPY --chown=user:user data /opt/app/
COPY --chown=user:user utils.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
