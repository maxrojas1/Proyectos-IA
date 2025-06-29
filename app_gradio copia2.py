import gradio as gr
from PIL import Image
from model import predict_image

# Función de predicción

def classify(img: Image.Image) -> str:
    msg = predict_image(img)
    color = "green" if "NO hecha" in msg else "red"
    return (
        f"<p style='text-align:center; color:{color}; font-weight:bold; "
        f"font-size:1.5rem; margin:0;'>{msg}</p>"
    )

# CSS personalizado con textos más grandes y marco más grueso
css = """
.gradio-container {
  background-color: #E0D4FF !important;
  border: 12px solid #A78BFA !important;    /* marco más grueso */
  width: 85vw !important;
  max-width: none !important;
  margin: 20px auto !important;
  border-radius: 12px !important;
  padding: 20px 40px !important;
  display: flex;
  flex-direction: column;
  align-items: center;
}
/* Oculta footer de Gradio */
footer, .footer, .gradio-container footer {
  display: none !important;
}
/* Título y subtítulo con texto más grande */
#title {
  text-align: center;
  font-size: 3rem !important;               /* texto más grande */
  color: #333;
  margin-bottom: 0.2rem;
}
#subtitle {
  text-align: center;
  font-size: 1.8rem !important;             /* subtitulo más grande */
  color: #666;
  margin-top: 0;
  margin-bottom: 1.5rem;
}
/* Controles en fila */
#controls {
  display: flex !important;
  flex-direction: row !important;
  justify-content: center !important;
  align-items: center !important;
  gap: 30px !important;
  width: 100% !important;
  flex-wrap: nowrap !important;
}
/* Imagen con tamaño moderado */
#image_input {
  width: 400px !important;
  height: auto !important;
  margin: 0 !important;
}
/* Botón con texto más grande */
#predict_button {
  background-color: #4a90e2 !important;
  color: white !important;
  border: none !important;
  padding: 1rem 2rem !important;             /* botón más grande */
  font-size: 1.2rem !important;             /* texto más grande */
  border-radius: 8px !important;
}
#predict_button:hover {
  background-color: #357abd !important;
}
/* Resultado con texto más grande */
#result {
  margin-top: 1.5rem !important;
  width: 100% !important;
  font-size: 1.5rem !important;             /* resultado más grande */
}
"""

with gr.Blocks(css=css, title="NoticIA") as demo:
    gr.Markdown("<h1 id='title'>NoticIA</h1>")
    gr.Markdown("<h3 id='subtitle'>Detección de imágenes hechas por IA</h3>")

    with gr.Row(elem_id="controls"):
        img_in = gr.Image(type="pil", label="", elem_id="image_input")
        btn    = gr.Button("Predecir", elem_id="predict_button")

    result = gr.HTML("", elem_id="result")
    btn.click(fn=classify, inputs=img_in, outputs=result)

if __name__ == "__main__":
    demo.launch()
