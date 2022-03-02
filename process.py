# import libraries here
from __future__ import print_function
#import potrebnih biblioteka
import cv2
import collections

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation, Dropout
from keras.optimizers import SGD
from keras.models import model_from_json

#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import numpy as np
import matplotlib.pylab as plt

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def invert(image):
    return 255-image

def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def dilate(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6,1))
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized
    
def scale_to_range(image):
    return image / 255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann

def convert_output(outputs):
    return np.eye(len(outputs))

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def select_roi(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_array = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        if(h>20 and w>20):
            region = image_bin[y:y+h+1,x:x+w+1];
            regions_array.append([resize_region(region), (x,y,w,h)])
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances

def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid. -----softmax mozda ?
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(392, input_dim=784, activation='sigmoid'))
    ann.add(Dense(196))
    ann.add(Dropout(0.5, input_shape=(2,)))
    ann.add(Dense(98))
    ann.add(Dense(60, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=5500, batch_size=1, verbose = 1, shuffle=False) 
      
    return ann

def serialize_ann(ann):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open("serialization_folder/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights("serialization_folder/neuronska.h5")
    
def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('serialization_folder/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights("serialization_folder/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None

def display_result(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
        result += alphabet[winner(output)]
    return result

def train_or_load_character_recognition_model(train_image_paths):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta)

    Procedura treba da istrenira model i da ga sacuva pod proizvoljnim nazivom. Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran

    :param train_image_paths: putanje do fotografija alfabeta
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati ako je vec istreniran

    image_color = load_image(train_image_paths[0])
    img = cv2.bilateralFilter(image_color,12,75,75)
    img = dilate((cv2.Canny(image_gray(image_color), 30, 200)))
    img_dil = cv2.dilate(img[0:128, :], cv2.getStructuringElement(cv2.MORPH_RECT, (1,16)), iterations=1)
    img[0:128, :] = img_dil
    selected_regions, letters0, region_distance = select_roi(image_color.copy(), img)

    display_image(selected_regions)
    print ('Broj prepoznatih regiona:', len(letters0))

    image_color = load_image(train_image_paths[1])
    img = cv2.bilateralFilter(image_color,12,75,75)
    img = dilate((cv2.Canny(image_gray(image_color), 30, 200)))
    img_dil = cv2.dilate(img[0:128, :], cv2.getStructuringElement(cv2.MORPH_RECT, (1,16)), iterations=1)
    img[0:128, :] = img_dil
    selected_regions, letters1, region_distance = select_roi(image_color.copy(), img)

    display_image(selected_regions)
    print ('Broj prepoznatih regiona:', len(letters1))

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž',
                'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
    inputs = prepare_for_ann(letters0 + letters1)
    outputs = convert_output(alphabet)

    # probaj da ucitas prethodno istreniran model
    ann = load_trained_ann()

    # ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
    if ann == None:
        print("Traniranje modela zapoceto.")
        ann = create_ann()
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        # serijalizuj novu mrezu nakon treniranja, da se ne trenira ponovo svaki put
        serialize_ann(ann)

    return ann

def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž',
                'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']

    #image_color = load_image('dataset/validation/train63.bmp')
    image_color = load_image(image_path)
    img_rot = dilate(cv2.Canny((image_gray(image_color)), 120, 200))

    x_reg = []
    y_reg = []
    contours_heights = [] #po ovome gledamo koliko treba da se uvelica
    _, contours, hierarchy = cv2.findContours(img_rot.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours: 
        center, size, angle = cv2.minAreaRect(contour)
        if size[0] > 5 and size[1] > 5:
            x_reg.append(center[0])
            y_reg.append(center[1])
            contours_heights.append(size[1])

    x_reg = np.array(x_reg)
    y_reg = np.array(y_reg)
    x_reg = x_reg.reshape(-1, 1)
    reg = LinearRegression().fit(x_reg, y_reg)
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    (height, width) = img_rot.shape[0:2]
    half_height = height//2
    center = (width // 2, half_height)

    best_angle = reg.coef_ *180/np.pi
    print('best around: ' + str(best_angle))

    #https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
    if abs(best_angle) > 2:
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        img = cv2.warpAffine(image_color, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        if max(contours_heights) < 70:
            img = cv2.resize(img, (img.shape[1]*3, img.shape[0]*3), cv2.INTER_LINEAR) #3 puta
        else:
            img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), cv2.INTER_LINEAR) #2 puta   
        
    #https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
        img = cv2.bilateralFilter(img,12,75,75)
        img = dilate(dilate(cv2.Canny((image_gray(img)), 70, 140)))
    else:
        img = image_color
        img = cv2.resize(img, (int(img.shape[1]*1.5), int(img.shape[0]*1.5)), cv2.INTER_LINEAR)
        
        img = cv2.bilateralFilter(image_color,9,75,75)
        img = dilate(cv2.Canny((image_gray(img)), 120, 200))

    #dilacija na gornjoj polovini slike
    img_dil = cv2.dilate(img[0:half_height, :], cv2.getStructuringElement(cv2.MORPH_RECT, (1,16)), iterations=1)
    img[0:half_height, :] = img_dil

    selected_regions, letters, region_distance = select_roi(image_color.copy(), img)
    print('Broj prepoznatih regiona:', len(letters))

    #Podešavanje centara grupa K-means algoritmom
    region_distance = np.array(region_distance).reshape(len(region_distance), 1)
    #Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente
    
    if(region_distance.shape[0] * region_distance.shape[1] < 2):
        return ""
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    k_means.fit(region_distance)

    inputs = prepare_for_ann(letters)
    results = trained_model.predict(np.array(inputs, np.float32))
    extracted_text = display_result(results, alphabet, k_means)
    
    print(extracted_text)

    display_image(img)
    #plt.show()

    words = extracted_text.split(' ')
    vocabulary_words = list(vocabulary.keys())
    fuzzy_text = ''

    for w in words:
        if w in vocabulary_words: #provera da li je rec vec u vokabularu, ako jeste ne radi fuzzy
            fuzzy_text += w + ' '
        else:
            ratio_list = []
            for vw in vocabulary_words: #trazi sve reci u vokabularu 
                ratio = fuzz.ratio(w, vw)
                ratio_list.append((w, vw, ratio))
            sorted_ratio_list = sorted(ratio_list, key=lambda tup: tup[2], reverse=True) #sortiranje gledajuci treci tuple element: ratio u opadajucem red
            best_word = sorted_ratio_list[0] #tuple w, vw, ratio
            if best_word[2] > 70: #min ratio 
                fuzzy_text += best_word[1] + ' '

    print(fuzzy_text)

    return fuzzy_text