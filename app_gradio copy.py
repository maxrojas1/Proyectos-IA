import gradio as gr
from PIL import Image
from model import predict_image

def classify(img: Image.Image) -> str:
    msg = predict_image(img)
    color = "green" if "NO hecha" in msg else "red"
    return f"<p style='text-align:center; color:{color}; font-weight:bold; font-size:1.2rem;'>{msg}</p>"

css = """
.gradio-container {
  background-color: #F8F0FF !important;    /* pastel claro */
  border: 5px solid #A78BFA !important;     /* marco lila */
  max-width: 500px !important;
  margin: 50px auto !important;
  border-radius: 12px !important;
  padding: 30px !important;
}
#title {
  text-align: center;
  font-size: 2.5rem;
  margin-bottom: 0.2rem;
  color: #333;
}
#subtitle {
  text-align: center;
  font-size: 1.2rem;
  margin-top: 0;
  margin-bottom: 1.5rem;
  color: #666;
}
#image_input {
  margin: auto !important;
}
#predict_button {
  width: 100% !important;
  background-color: #4a90e2 !important;
  color: white !important;
  border: none !important;
  padding: 0.75rem !important;
  font-size: 1rem !important;
  border-radius: 8px !important;
  margin-top: 1rem !important;
}
#predict_button:hover {
  background-color: #357abd !important;
}
#result {
  margin-top: 1.5rem;
}
"""

with gr.Blocks(css=css, title="NoticIA") as demo:
    gr.Markdown("<h1 id='title'>NoticIA</h1>")
    gr.Markdown("<h3 id='subtitle'>Detección de imágenes hechas por IA</h3>")
    img_in  = gr.Image(type="pil", label="", elem_id="image_input")
    btn     = gr.Button("Predecir", elem_id="predict_button")
    result  = gr.HTML("", elem_id="result")
    btn.click(fn=classify, inputs=img_in, outputs=result)

if __name__ == "__main__":
    demo.launch()
