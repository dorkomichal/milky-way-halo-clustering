FROM python:3.11

# RUN apk upgrade --update \
 #   && apk add gcc g++ gsl
RUN apt-get update && apt-get install g++ gcc libgsl0-dev 
RUN mkdir /home/docker_user
WORKDIR /home/docker_user/md2018
COPY . /home/docker_user/md2018
RUN git clone https://github.com/jobovy/extreme-deconvolution.git
RUN make install -C ./extreme-deconvolution
RUN pip install --no-cache-dir -r requirements.txt
RUN make pywrapper -C ./extreme-deconvolution
ENV PYTHONPATH=/home/docker_user/md2018/extreme-deconvolution/py
