#### 🚀 Gradio-Powered Handwriting (TrOCR + OpenVINO) to Video Generation (LTX + OpenVINO GenAI)

This Gradio app links handwriting input to video generation, using TrOCR for OCR and LTX-Video for text-to-video conversion, both powered by OpenVINO. Users write on the canvas using a mouse or touch, triggering TrOCR to recognize the handwriting and convert it into text. The recognized text is then passed to LTX-Video, which generates a short video clip based on the input text. The entire process runs on Intel hardware using OpenVINO's optimized inference for fast performance.

When a user clicks "Recognize," the handwriting is processed through TrOCR, and the recognized text is added to the prompt box. The "Generate" button then initiates the video creation process using LTX-Video. Video output is displayed as an MP4 file, with options to download.

#### 👉 Links & Resources

- [Gradio](https://www.gradio.app/)
- [microsoft/trocr-large-handwritten](https://huggingface.co/microsoft/trocr-large-handwritten)
- [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
- [Optimum Intel](https://huggingface.co/docs/optimum-intel/index)
- [OpenVINO](https://docs.openvino.ai/2026/index.html)

---

#### 🚀 Clone and Run

```bash
# Clone the repository
git clone https://github.com/Ashot72/openvino-handwriting-to-video
cd openvino-handwriting-to-video

# First-time setup
setup.bat

# Start the app
run.bat

# The app will be available at http://127.0.0.1:7860
```

#### 🛠 Debugging in VS Code

Install Microsoft’s [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy) extension.

Open the Run view (View → Run or Ctrl+Shift+D) to access the debug configuration.

#### 📺 **Video** [Watch on YouTube](https://youtu.be/trXUVk7GgC0)
