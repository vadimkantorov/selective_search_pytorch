import gradio as gr



def predict(img, preset, algo):
    print(preset, algo)
    return img


inputs = [
    gr.Image(type = 'pil'),
    gr.Radio(label = 'preset', choices = ['fast', 'quality', 'single'], value = 'fast'),
    gr.Radio(label = 'algo', choices = ['pytorch', 'opencv', 'opencv_custom'], value = 'pytorch')
]
outputs = gr.Image(type = 'pil')

demo = gr.Interface(predict, inputs = inputs, outputs = outputs)
demo.launch()
