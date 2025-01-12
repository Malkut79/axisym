FROM dolfinx/dolfinx:v0.9.0

COPY ./src/requirements.txt ./requirements.txt

# Install python dependencies
RUN apt-get update && apt-get upgrade -y
RUN pip install  -r requirements.txt


# WORKDIR /shared
# COPY ./src/run.sh /run.sh
# RUN chmod +x /run.sh

WORKDIR /code
COPY ./app/ /code/app/

# Set the entrypoint to allow passing file path
# ENTRYPOINT ["/run.sh"]


CMD ["fastapi", "run", "app/main.py", "--port", "80"]
