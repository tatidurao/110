import cv2
import numpy as np

#biblioteca de aprendizado profundo
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

video = cv2.VideoCapture(0)

while True:

    check,frame = video.read()

    # Altere o dado de entrada:
     #224 pixels x 224 pixels   
    img = cv2.resize(frame,(224,224)) 

    #matriz 4D 4 dimensoes
    # 2. Converta a imagem em um array Numpy e aumente a dimensão
    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0) #3 dimensao = profundidade

    # 3. Normalize a imagem convertento todos os pixels em 0 e 1 para aumentar a precisão
    normalised_image = test_image/255.0
    # Preveja o resultado
    prediction = model.predict(normalised_image)

    print("Previsão: ", prediction)
        
    cv2.imshow("Resultado",frame)
            
    key = cv2.waitKey(1)

    if key == 32:
        print("Fechando")
        break

video.release()