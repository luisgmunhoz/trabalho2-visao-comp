from datetime import datetime
import face_recognition
import cv2
import os
import pickle


known_face_encodings = []
known_face_metadata = []

def load_known_faces():
    '''Crie aqui a função que carrega as faces conhecidas contidas no arquivo .dat'''
    if os.path.exists('known_faces.dat'):
        with open('known_faces.dat', 'rb') as f:
            known_faces = pickle.load(f)
    else:
        known_faces = []
    return known_faces

def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")

def register_new_face(face_encoding, face_image, name):
    known_face_encodings.append(face_encoding)
    known_face_metadata.append({
    "first_seen": datetime.now(),
    "first_seen_this_interaction": datetime.now(),
    "last_seen": datetime.now(),
    "seen_count": 1,
    "seen_frames": 1,
    "face_image": face_image,
    "name": name,
    })


def add_faces_from_gallery(gallery_path):
    '''Crie aqui a função que adiciona as faces contidas em imagens (salvas em uma pasta) no arquivo .dat '''

    known_faces = load_known_faces()

    image_files = os.listdir(gallery_path)  # liste os arquivos da pasta
    for image_file in image_files:
        image_path = os.path.join(gallery_path, image_file)  # use o os.path.join para escrever o caminho completo
        
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processando {image_path}...")

            image = cv2.imread(image_path)  # Leia a imagem a ser processada
            small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)  # Diminua na mesma proporção que no código de inferencia

            face_locations = face_recognition.face_locations(small_frame)  # Gere o BBox da face contida na imagem
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)  # Gere os encodings da face contida na imagem

            for face_encoding in face_encodings:
                name = os.path.splitext(image_file)[0]  # use o os.path.splitext para gerar o nome
                top, right, bottom, left = face_locations[0]  # defina com base na localizacao da face
                face_image = image[top*4:bottom*4, left*4:right*4]  # Use o Crop da Face
                face_image = cv2.resize(face_image, (150, 150))  # Faça o resize para (150,150)
                register_new_face(face_encoding, face_image, name)  # Registre a nova face aqui

    save_known_faces()  # Salve as faces em um arquivo .dat aqui

if __name__ == "__main__":
    gallery_path = "./gallery"  
    add_faces_from_gallery(gallery_path)