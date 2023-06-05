import tensorflow as tf
from tensorflow import keras
import pickle

# Carregando o tokenizer e o modelo
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
model = keras.models.load_model("modelo_classificador")

# Função para classificar frases
def classificar_frase(frase):
    # Pré-processamento da frase de entrada
    sequence = tokenizer.texts_to_sequences([frase])
    padded_sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen=model.input_shape[1], padding="post")
    
    # Fazendo a predição
    prediction = model.predict(padded_sequence)
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
    
    # Obtendo a classe prevista
    classes = ["horario", "clima", "lembrete", "capital"]
    predicted_class = classes[predicted_class_index]
    
    return predicted_class

# Exemplo de uso 2
frase1 = "Que horas são?"
resultado1 = classificar_frase(frase1)
print(frase1, "=> Classe:", resultado1)

# Exemplo de uso 1
frase2 = "Como está o clima?"
resultado2 = classificar_frase(frase2)
print(frase2, "=> Classe:", resultado2)
