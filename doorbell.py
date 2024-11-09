from datetime import datetime, timedelta
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
    small_frame = cv2.resize(dummy_image, (0, 0), fx=0.25, fy=0.25)  # Diminua na mesma proporção que no código de inferencia
    dummy_locations = face_recognition.face_locations(small_frame)  # Gere o BBox da face contida na imagem
    if dummy_locations:
        _ = face_recognition.face_encodings(small_frame, dummy_locations)  # Gere os encodings da face contida na imagem
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
    metadata = None
    if len(known_face_encodings) == 0:
        return None
    now = datetime.now()

    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(distances)
    if distances[best_match_index] < 0.5:
        metadata = known_face_metadata[best_match_index]
        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1
        if datetime.now() - metadata["first_seen_this_interaction"] > timedelta(minutes=2):
            metadata['first_seen_this_interaction'] = datetime.now()
            metadata["seen_count"] += 1
    return metadata

def main_loop():
    '''Crie a função principal aqui'''
    video_capture = cv2.VideoCapture(0)

    number_of_faces_since_save = 0

    while True:
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        face_locations = face_recognition.face_locations(small_frame)

        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_labels = []
        at_home = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            metadata = lookup_known_face(face_encoding)
            if metadata is not None:
                print(f"Known face detected: {metadata['name']}")
                time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                face_label = f"{metadata['name']} {int(time_at_door.total_seconds())}s"
                at_home.append(metadata["name"])
                face_labels.append(face_label)
            else:
                face_label = "New visitor!"

                top, right, bottom, left = face_location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))
                
        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 255), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        number_of_recent_visitors = 0
        for metadata in known_face_metadata:
            if metadata["name"] in at_home:
                x_position = number_of_recent_visitors * 150
                frame[30:180, x_position:x_position + 150] = metadata["face_image"]
                number_of_recent_visitors += 1

                visits = metadata['seen_count']
                visit_label = f"{visits} visits"
                if visits == 1:
                    visit_label = "First visit"
                cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        if number_of_recent_visitors > 0:
            cv2.putText(frame, "Visitors at Door", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

                

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    warm_up_encoding()
    load_known_faces()
    main_loop()