import face_recognition as fr
from glob import glob

KNOWN_PEOPLE = glob("people_i_know/*")
UNKNOWN_PEOPLE = glob("unknown_pictures/*")
known_faces = []
known_names = []


for person_path in KNOWN_PEOPLE:
    person_image = fr.load_image_file(person_path)
    person_image_extension = person_path.split(".")[-1]
    person_name = person_path.split(
        "/")[-1].rstrip("." + person_image_extension)

    person_image_encoding = fr.face_encodings(person_image)[0]
    known_faces.append(person_image_encoding)
    known_names.append(person_name)

for unknown_person_path in UNKNOWN_PEOPLE:
    unknown_person = fr.load_image_file(unknown_person_path)
    unknown_image_encoding = fr.face_encodings(unknown_person)[0]

    # unknown_image_extension = unknown_person_path.split(".")[-1]
    # unknown_name = unknown_person_path.split(
    #     "/")[-1].rstrip("." + unknown_image_extension)
    unknown_name = unknown_person_path.split(
        "/")[-1]

    results = fr.compare_faces(known_faces, unknown_image_encoding)

    for i, result in enumerate(results):
        if(result):
            name_of_match = known_names[i]
            print(
                f"Unknown Image = {unknown_name}: {name_of_match}")
