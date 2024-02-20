FROM toshihikoaoki/jetson-orin-llm:latest
RUN apt-get autoremove -y && apt-get clean
RUN pip install diffusers==0.26.3
RUN pip install llama-cpp-python[gpu]==0.2.44
RUN pip install uform==1.1.0
