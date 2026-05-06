from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import re


app = Flask(__name__)

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

print("Cargando modelo Qwen2-0.5B-Instruct... Esto puede tardar la primera vez.")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

model.to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Modelo cargado correctamente.")


def limpiar_texto(texto_original):
    """
    Limpia archivos HTML/TXT de la SEC.
    Elimina etiquetas HTML, espacios repetidos y caracteres innecesarios.
    """

    soup = BeautifulSoup(texto_original, "html.parser")
    texto = soup.get_text(separator=" ")

    texto = re.sub(r"\s+", " ", texto)
    texto = texto.replace("\xa0", " ")

    return texto.strip()


def descargar_archivo_sec(url, contacto):
    """
    Descarga contenido desde una URL de la SEC.
    La SEC recomienda usar User-Agent identificable.
    """

    if not contacto:
        contacto = "student@example.com"

    headers = {
        "User-Agent": f"sec-sentiment-qwen/1.0 {contacto}"
    }

    respuesta = requests.get(url, headers=headers, timeout=30)
    respuesta.raise_for_status()

    return respuesta.text


def extraer_seccion(texto, inicio, fin):
    """
    Extrae una sección del documento usando una palabra/frase inicial
    y opcionalmente una palabra/frase final.
    """

    if not inicio:
        return texto

    texto_minusculas = texto.lower()
    inicio_minusculas = inicio.lower()

    posicion_inicio = texto_minusculas.find(inicio_minusculas)

    if posicion_inicio == -1:
        return ""

    if fin:
        fin_minusculas = fin.lower()
        posicion_fin = texto_minusculas.find(fin_minusculas, posicion_inicio + len(inicio_minusculas))

        if posicion_fin != -1:
            return texto[posicion_inicio:posicion_fin]

    return texto[posicion_inicio:]


def seleccionar_muestras(texto, tamano_muestra=3500):
    """
    Como los archivos de la SEC pueden ser muy largos,
    se toman muestras del inicio, medio y final del texto.
    """

    if len(texto) <= tamano_muestra:
        return [texto]

    inicio = texto[:tamano_muestra]

    mitad_inicio = max(len(texto) // 2 - tamano_muestra // 2, 0)
    mitad = texto[mitad_inicio:mitad_inicio + tamano_muestra]

    final = texto[-tamano_muestra:]

    return [inicio, mitad, final]


def generar_respuesta_llm(mensajes, max_new_tokens=350):
    """
    Genera una respuesta usando Qwen2-0.5B-Instruct.
    """

    prompt = tokenizer.apply_chat_template(
        mensajes,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    respuesta = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return respuesta


def analizar_sentimiento_muestra(texto, numero_muestra, total_muestras):
    """
    Analiza el sentimiento financiero de una muestra del documento.
    """

    mensajes = [
        {
            "role": "system",
            "content": (
                "Eres un analista financiero. Tu tarea es analizar el sentimiento "
                "de textos de reportes de la SEC. Clasifica el sentimiento como "
                "Positivo, Negativo o Neutral. Responde en español."
            )
        },
        {
            "role": "user",
            "content": f"""
Analiza el sentimiento financiero del siguiente fragmento de un documento de la SEC.

Fragmento {numero_muestra} de {total_muestras}:

\"\"\"{texto}\"\"\"

Responde con este formato:

Sentimiento: Positivo / Negativo / Neutral
Confianza: Alta / Media / Baja
Razón principal:
Señales positivas:
Señales negativas:
"""
        }
    ]

    return generar_respuesta_llm(mensajes)


def consolidar_resultado(resultados_parciales):
    """
    Une los análisis parciales en una conclusión general.
    """

    texto_resultados = "\n\n".join(
        [f"Análisis parcial {i + 1}:\n{resultado}" for i, resultado in enumerate(resultados_parciales)]
    )

    mensajes = [
        {
            "role": "system",
            "content": (
                "Eres un analista financiero. Debes consolidar varios análisis "
                "parciales de sentimiento de un documento de la SEC."
            )
        },
        {
            "role": "user",
            "content": f"""
Con base en estos análisis parciales, da una conclusión general del sentimiento del documento.

{texto_resultados}

Responde en español con este formato:

Sentimiento general:
Confianza:
Explicación breve:
Resumen ejecutivo:
"""
        }
    ]

    return generar_respuesta_llm(mensajes, max_new_tokens=300)


@app.route("/", methods=["GET", "POST"])
def index():
    resultado_final = None
    resultados_parciales = []
    error = None
    texto_usado = ""
    fuente = ""
    longitud_texto = 0

    if request.method == "POST":
        archivo = request.files.get("archivo")
        url = request.form.get("url", "").strip()
        contacto = request.form.get("contacto", "").strip()

        tipo_analisis = request.form.get("tipo_analisis", "completo")
        inicio_seccion = request.form.get("inicio_seccion", "").strip()
        fin_seccion = request.form.get("fin_seccion", "").strip()

        try:
            contenido = ""

            if archivo and archivo.filename:
                nombre_archivo = secure_filename(archivo.filename)
                contenido = archivo.read().decode("utf-8", errors="ignore")
                fuente = f"Archivo subido: {nombre_archivo}"

            elif url:
                contenido = descargar_archivo_sec(url, contacto)
                fuente = f"URL: {url}"

            else:
                error = "Debes subir un archivo o pegar una URL de la SEC."

            if not error:
                texto_limpio = limpiar_texto(contenido)

                if tipo_analisis == "seccion":
                    texto_limpio = extraer_seccion(
                        texto_limpio,
                        inicio_seccion,
                        fin_seccion
                    )

                    if not texto_limpio:
                        error = "No se encontró la sección indicada. Prueba con otras palabras de inicio o analiza el documento completo."

                if not error:
                    longitud_texto = len(texto_limpio)
                    texto_usado = texto_limpio[:1200]

                    muestras = seleccionar_muestras(texto_limpio)

                    for i, muestra in enumerate(muestras):
                        analisis = analizar_sentimiento_muestra(
                            muestra,
                            i + 1,
                            len(muestras)
                        )
                        resultados_parciales.append(analisis)

                    resultado_final = consolidar_resultado(resultados_parciales)

        except Exception as e:
            error = f"Ocurrió un error: {str(e)}"

    return render_template(
        "index.html",
        resultado_final=resultado_final,
        resultados_parciales=resultados_parciales,
        error=error,
        texto_usado=texto_usado,
        fuente=fuente,
        longitud_texto=longitud_texto
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
