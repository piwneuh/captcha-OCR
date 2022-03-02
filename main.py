from process import extract_text_from_image, train_or_load_character_recognition_model
import glob
import sys
import os

# ------------------------------------------------------------------
# Ovaj fajl ne menjati, da bi automatsko ocenjivanje bilo moguce

if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[1]
else:
    VALIDATION_DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep+'validation'+os.path.sep

TRAIN_DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep+'train'+os.path.sep
VOCABULARY_PATH = '.'+os.path.sep+'dataset'+os.path.sep + "dict.txt"
# -------------------------------------------------------------------


# istrenirati model za prepoznavanje karaktera
alphabet_image_paths = [os.path.join(TRAIN_DATASET_PATH, image) for image in os.listdir(TRAIN_DATASET_PATH)]
model = train_or_load_character_recognition_model(alphabet_image_paths)

# kreiranje recnika svih poznatih reci, za korekciju Levenstein rastojanjem
vocabulary = dict()
with open(VOCABULARY_PATH, 'r', encoding='utf-8') as file:
    data = file.read()
    lines = data.split('\n')
    for index, line in enumerate(lines):
        cols = line.split()
        if len(cols) == 3:
            vocabulary[cols[1]] = cols[2]

# izvrsiti citanje teksta sa svih fotografija iz validacionog skupa podataka, koriscenjem istreniranog modela
processed_image_names = []
extracted_text = []

for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.bmp"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    extracted_text.append(extract_text_from_image(model, image_path, vocabulary))


# -----------------------------------------------------------------
# Kreiranje fajla sa rezultatima ekstrakcije za svaku sliku
result_file_contents = ""
for image_index, image_name in enumerate(processed_image_names):
    result_file_contents += "%s,%s\n" % (image_name, extracted_text[image_index])
# sacuvaj formirane rezultate u csv fajl
with open('result.csv', 'w', encoding='utf-8') as output_file:
    output_file.write(result_file_contents)

# ------------------------------------------------------------------
