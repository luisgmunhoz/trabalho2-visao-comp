import pickle
with open("known_faces.dat", "rb") as f:
    _, meta = pickle.load(f)
    print(meta)