FROM python:3.8

# Create the working directory
RUN set -ex && mkdir /translator
WORKDIR /translator

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy the relevant directories
COPY model/ ./model
COPY . ./

# Run the web server
EXPOSE 5000
ENV PYTHONPATH /translator
CMD python3 /translator/model/model.py