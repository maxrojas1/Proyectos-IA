# app.py
import gradio as gr
from PIL import Image
from model import predict_with_overlay

# Función de interfaz: retorna overlay y HTML dinámico coloreado
def classify(img: Image.Image):
    message, overlay = predict_with_overlay(img, alpha=0.5)
    color = "green" if "no hecha" in message.lower() else "#8B0000"
    msg_html = (
        f"<div style='color:{color} !important; "
        f"font-weight:bold; font-size:2.5rem; text-align:center; margin:20px 0;'>"
        f"{message}</div>"
    )
    return overlay, msg_html

# CSS y layout personalizados
css = """
html, body {
  background-color: #28242c !important;
  margin: 0;
  padding: 0;
  overflow: hidden !important;
}
.gradio-container {
  background-color: #28242c !important;
  border: 12px solid #383434 !important;
  display: inline-block !important;
  width: 800px !important;
  height: 900px !important;
  margin: 20px auto !important;
  padding: 20px !important;
  border-radius: 12px !important;
  text-align: center;
  overflow: hidden !important;
  overflow: hidden;
}
footer, .footer, .gradio-container footer { display: none !important; }
#title {
  font-family: 'Arial Black', sans-serif;
  font-size: 3rem !important;
  color: white !important;
  text-align: center;
  margin: 0 0 10px 0;
}
#subtitle {
  font-family: 'Georgia', serif;
  font-size: 1.8rem !important;
  color: white !important;
  text-align: center;
  margin: 0 0 20px 0;
}
#controls {
  display: flex !important;
  justify-content: space-between !important;
  align-items: center !important;
  gap: 20px !important;
  margin-bottom: 20px !important;
}
#image_input, #overlay {
  max-width: 45% !important;
  max-height: 300px !important;
}
.gradio-image-container, .gradio-image-label { display: none !important; }
#predict_button {
  font-family: 'Verdana', sans-serif;
  background-color: #383434 !important;
  color: white !important;
  border: none !important;
  padding: 0.5rem 1rem !important;
  font-size: 1rem !important;
  border-radius: 8px !important;
  width: auto !important;
  display: inline-block !important;
  margin: 20px auto 0 auto !important;
}
#predict_button:hover { background-color: #4a4a4a !important; }
#result {
  font-size: 2.5rem !important;
  color: inherit !important;
  font-weight: bold !important;
  text-align: center !important;
  margin-top: 10px !important;
}
"""

with gr.Blocks(css=css, title="NoticIA") as demo:
    gr.Markdown("<h1 id='title'>Notic<span style='color:#383434'>IA</span></h1>")
    gr.Markdown("<h3 id='subtitle'>Detección de imágenes hechas por IA</h3>")

    with gr.Row(elem_id="controls"):
        img_in = gr.Image(type="pil", show_label=False, elem_id="image_input")
        overlay = gr.Image(type="pil", show_label=False, elem_id="overlay")

    btn = gr.Button("Predecir", elem_id="predict_button")
    result = gr.HTML("", elem_id="result")

    btn.click(fn=classify, inputs=img_in, outputs=[overlay, result])

if __name__ == "__main__":
    demo.launch()
