# Proyecto: Generación No Condicionada de Imágenes de Hongos

## Descripción del Proyecto

Este proyecto tiene como objetivo la generación no condicionada de imágenes de hongos utilizando un modelo DDPM (Denoising Diffusion Probabilistic Models) con una arquitectura UNET, con un tamaño de imagen de 128x128 píxeles. El dataset utilizado fue extraído de Kaggle y se exploraron dos enfoques diferentes: un modelo entrenado con todo el conjunto de datos y otro entrenado únicamente con dos clases específicas de hongos.

## Estructura del Proyecto

- **`src/`**: Contiene los scripts fuente del proyecto.
- **`data/`**: Almacena el dataset utilizado.
- **`ddpm-mushrooms-128`**: Guarda los modelos entrenados.
- **`ddpm-mushrooms-2class-128/samples/`**: Aquí se guardan los resultados obtenidos durante la generación de imágenes.

- **`Mushroom_Diffusionv2.ipynb`**: Es el notebook principal donde se crea el dataset, aplican transformaciones y entrena el modelo

## Configuración del Entorno

- Python 3.9.18 o superior
- Bibliotecas Python: `torch`, `numpy`, `matplotlib`, `PIL`, etc. (puedes instalarlas mediante `pip install -r requirements.txt`)

## Instrucciones de Ejecución

1. A traves de las librerias de HuggingFace:

   ```python
    from diffusers import DDPMPipeline

    model_id = "MexicanVanGogh/ddpm-mushrooms-128"

    ddpm = DDPMPipeline.from_pretrained(model_id, use_safetensors=True).to("cuda")
    image = ddpm(num_inference_steps=25).images[0]
    image


