ARG DEBIAN_FRONTEND=noninteractive

FROM compss/compss-tutorial:3.1

COPY ./framework /framework

# Install COMPSs
RUN cd /framework && \
    export EXTRAE_MPI_HEADERS=/usr/include/x86_64-linux-gnu/mpi && \
    /framework/builders/buildlocal -K -J -T -D -C -M --skip-tests /opt/COMPSs && \
    rm -rf /framework

COPY ./tests /tests
RUN cd /tests/random_forest/application && \
    mvn clean package

COPY ./launch_test /launch_test

ENTRYPOINT ["/launch_test"]