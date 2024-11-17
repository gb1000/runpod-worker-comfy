# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade -r requirements.txt

# Install runpod
RUN pip3 install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories
RUN mkdir -p models/checkpoints models/vae

# Download checkpoints/vae/LoRA to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    elif [ "$MODEL_TYPE" = "sd3" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi


 RUN git clone https://github.com/rgthree/rgthree-comfy.git custom_nodes/rgthree-comfy


# RUN mkdir -p /comfyui/custom_nodes && \
    # git clone https://github.com/chrisgoringe/cg-image-picker.git custom_nodes/cg-image-picker && \
    # git clone https://github.com/chrisgoringe/cg-noisetools.git custom_nodes/cg-noisetools && \
    # git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git custom_nodes/ComfyUI-Advanced-ControlNet && \
    # git clone https://github.com/tzwm/comfyui-browser.git custom_nodes/comfyui-browser && \
    # git clone https://github.com/kijai/ComfyUI-CCSR.git custom_nodes/ComfyUI-CCSR && \
    # git clone https://github.com/crystian/ComfyUI-Crystools.git custom_nodes/ComfyUI-Crystools && \
    # git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git custom_nodes/ComfyUI-Custom-Scripts && \
    # git clone https://github.com/kijai/ComfyUI-DDColor.git custom_nodes/ComfyUI-DDColor && \
    # git clone https://github.com/kijai/ComfyUI-DynamiCrafterWrapper.git custom_nodes/ComfyUI-DynamiCrafterWrapper && \
    # git clone https://github.com/kijai/ComfyUI-Florence2.git custom_nodes/ComfyUI-Florence2 && \
    # git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git custom_nodes/ComfyUI-Frame-Interpolation && \
    # git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/ComfyUI-Impact-Pack && \
    # git clone https://github.com/Acly/comfyui-inpaint-nodes.git custom_nodes/comfyui-inpaint-nodes && \
    # git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git custom_nodes/ComfyUI-Inspire-Pack && \
    # git clone https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes && \
    # git clone https://github.com/ChrisColeTech/ComfyUI-Line-counter.git custom_nodes/ComfyUI-Line-counter && \
    # git clone https://github.com/theUpsider/ComfyUI-Logic.git custom_nodes/ComfyUI-Logic && \
    # git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager && \
    # git clone https://github.com/OliverCrosby/Comfyui-Minimap.git custom_nodes/Comfyui-Minimap && \
    # git clone https://github.com/noembryo/ComfyUI-noEmbryo.git custom_nodes/ComfyUI-noEmbryo && \
    # git clone https://github.com/royceschultz/ComfyUI-Notifications.git custom_nodes/ComfyUI-Notifications && \
    # git clone https://github.com/Zuellni/ComfyUI-PickScore-Nodes.git custom_nodes/ComfyUI-PickScore-Nodes && \
    # git clone https://github.com/EllangoK/ComfyUI-post-processing-nodes.git custom_nodes/ComfyUI-post-processing-nodes && \
    # git clone https://github.com/receyuki/comfyui-prompt-reader-node.git custom_nodes/comfyui-prompt-reader-node && \
    # git clone https://github.com/Gourieff/comfyui-reactor-node.git custom_nodes/comfyui-reactor-node && \
    # git clone https://github.com/kijai/ComfyUI-segment-anything-2.git custom_nodes/ComfyUI-segment-anything-2 && \
    # git clone https://github.com/matan1905/ComfyUI-Serving-Toolkit.git custom_nodes/ComfyUI-Serving-Toolkit && \
    # git clone https://github.com/kijai/ComfyUI-SUPIR.git custom_nodes/ComfyUI-SUPIR && \
    # git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/ComfyUI-VideoHelperSuite && \
    # git clone https://github.com/RockOfFire/ComfyUI_Comfyroll_CustomNodes.git custom_nodes/ComfyUI_Comfyroll_CustomNodes && \
    # git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git custom_nodes/comfyui_controlnet_aux && \
    # git clone https://github.com/cubiq/ComfyUI_essentials.git custom_nodes/ComfyUI_essentials && \
    # git clone https://github.com/cubiq/ComfyUI_FaceAnalysis.git custom_nodes/ComfyUI_FaceAnalysis && \
    # git clone https://github.com/cubiq/ComfyUI_InstantID.git custom_nodes/ComfyUI_InstantID && \
    # git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git custom_nodes/ComfyUI_IPAdapter_plus && \
    # git clone https://github.com/BlenderNeko/ComfyUI_Noise.git custom_nodes/ComfyUI_Noise && \
    # git clone https://github.com/dnl13/comfyui_segment_anything.git custom_nodes/comfyui_segment_anything && \
    # git clone https://github.com/TinyTerra/ComfyUI_tinyterraNodes.git custom_nodes/ComfyUI_tinyterraNodes && \
    # git clone https://github.com/melMass/comfy_mtb.git custom_nodes/comfy_mtb && \
    # git clone https://github.com/failfa-st/failfast-comfyui-extensions.git custom_nodes/failfast-comfyui-extensions && \
    # git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui.git custom_nodes/masquerade-nodes-comfyui && \
    # git clone https://github.com/glibsonoran/Plush-for-ComfyUI.git custom_nodes/Plush-for-ComfyUI && \
    # git clone https://github.com/pamparamm/sd-perturbed-attention.git custom_nodes/sd-perturbed-attention && \
    # git clone https://github.com/WASasquatch/was-node-suite-comfyui.git custom_nodes/was-node-suite-comfyui




# Stage 3: Final image
FROM base as final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models


# Start the container
CMD /start.sh
