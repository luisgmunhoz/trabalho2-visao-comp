import face_recognition 
import cv2
import numpy as np
import os
import pickle

known_face_encodings = []
known_face_metadata = []

def warm_up_encoding():
    '''Crie aqui a função que faz a pre inicialização dos modelos'''

    dummy_image = cv2.imread('dummy_face.jpg')  # Leia uma imagem dummy que contenha uma face
    dummy_locations = face_recognition.face_locations(dummy_image)  # Gere o BBox da face contida na imagem
    if dummy_locations:
        _ = face_recognition.face_encodings(dummy_image, dummy_locations)  # Gere os encodings da face contida na imagem
    print("Encoding model warmed up.")

def load_known_faces():
    '''Crie aqui a função que carrega as faces conhecidas contidas no arquivo .dat'''
    global known_face_encodings, known_face_metadata
    if os.path.exists('known_faces.dat'):
        with open('known_faces.dat', 'rb') as f:
            known_face_encodings, known_face_metadata = pickle.load(f)
    else:
        known_face_encodings, known_face_metadata = [], []

def lookup_known_face(face_encoding):
    '''Crie aqui a função que verifica se a pessoa é conhecida. USE 0.5 como limiar de comparação'''
    if len(known_face_encodings) == 0:
        return None

    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(distances)
    if distances[best_match_index] < 0.5:
        return known_face_metadata[best_match_index]
    return None

def main_loop():
    '''Crie a função principal aqui'''
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            metadata = lookup_known_face(face_encoding)
            if metadata:
                print(f"Known face detected: {metadata['name']}")
            else:
                print("Unknown face detected")

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    warm_up_encoding()
    load_known_faces()
    main_loop()