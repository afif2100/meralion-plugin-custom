FROM vllm/vllm-openai:v0.26.0

WORKDIR /opt/meralion3-plugin
COPY plugins/vllm-plugin-meralion3-v026/ ./
RUN uv pip install --system ninja /opt/meralion3-plugin

ENV VLLM_PLUGINS=register_meralion3_v026
