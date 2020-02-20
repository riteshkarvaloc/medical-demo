FROM manpuri1432/customtest:v1

COPY requirements.txt requirements.txt

RUN pip3 install cython
RUN pip3 install -r requirements.txt