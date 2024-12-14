FROM dolfinx/dolfinx:v0.9.0

WORKDIR /shared

# Create a script to read the mounted file
RUN echo '#!/bin/bash' > /run.sh && \
    echo 'python ./axisym.py' >> /run.sh && \
    chmod +x /run.sh

# Set the entrypoint to allow passing file path
ENTRYPOINT ["/run.sh"]
# Default file if no argument is provided
# CMD "./shared/run.sh"