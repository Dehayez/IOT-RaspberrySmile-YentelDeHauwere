#!/usr/bin/env python3

import sys
from random import choice
import time
from sense_hat import SenseHat

sense = SenseHat()

k = [0, 0, 0]  # Blank
r = [255, 0, 0]  # Red
y = [255, 127, 0]  # Yellow
g = [0, 255, 0]  # Green

no_face = [
    r, r, r, r, r, r, r, r,
    r, k, k, k, k, k, k, r,
    r, k, k, k, k, k, k, r,
    r, k, k, k, k, k, k, r,
    r, k, k, k, k, k, k, r,
    r, k, k, k, k, k, k, r,
    r, k, k, k, k, k, k, r,
    r, r, r, r, r, r, r, r,
]

neutral_face = [
    y, y, y, y, y, y, y, y,
    y, k, k, k, k, k, k, y,
    y, k, k, k, k, k, k, y,
    y, k, k, k, k, k, k, y,
    y, k, k, k, k, k, k, y,
    y, k, k, k, k, k, k, y,
    y, k, k, k, k, k, k, y,
    y, y, y, y, y, y, y, y,
]

smile_face = [
    g, g, g, g, g, g, g, g,
    g, k, k, k, k, k, k, g,
    g, k, k, k, k, k, k, g,
    g, k, k, k, k, k, k, g,
    g, k, k, k, k, k, k, g,
    g, k, k, k, k, k, k, g,
    g, k, k, k, k, k, k, g,
    g, g, g, g, g, g, g, g
]

dad_jokes = [
    "Leven met obesitas is best wel zwaar. Haha",
    "Een kampeerwinkel die de tent moet sluiten is nooit grappig, haha",
    "Ik rook niet, ik drink niet en ik scheld niet! GODVERDOMME mijn sigaret valt in mijn bier, haha",
    "Zebras zijn eigenlijk paarden die ontsnapt zijn uit de gevangenis, haha",
    "Ik hou van haar, daarom laat ik het groeien! Haha",
    "Twee varkens stonden in een wei. Zei de ene 'knor'. Zegt de andere 'royco', haha",
    "Wat is een pater op een ei? Een broeder, haha",
    "Het zit op een paard en het heeft spijt? Een zorry, haha",
    "Hoe kan je een dommerik in de war brengen? 15, haha",
    "Wat is het toppunt van zieligheid? Gij. Haha",
    "Wat is het toppunt van schoonheid? Niet jij. Haha"
]


def check_sense_stick():

    for event in sense.stick.get_events():
        if event.action == 'pressed':
            print('STOPPING SMILE DETECTOR...')
            sense.clear()
            sys.exit()


def smile_detector(debug=False):

    import cv2

    # time for jokes
    oldTime = time.time()

    print("INITIALIZING SMILE DETECTOR...")
    sense.show_message("HAHA")

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    cam = cv2.VideoCapture(0)
    # Keeps trying to open the camera; press SENSE Hat stick to terminate
    while not cam.isOpened():
        cam.open(0)
        check_sense_stick()

    print("CAMERA IS READY TO ROLL \n")

    sense.set_pixels(no_face)

    while True:
        ret, img_color = cam.read()
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            img_gray, scaleFactor=1.05, minNeighbors=5, minSize=(45, 45))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(
                    img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
                faceimg_color = img_color[y:y + h, x:x + w]
                faceimg_gray = img_gray[y:y + h, x:x + w]

                smiles = smile_cascade.detectMultiScale(
                    faceimg_gray, scaleFactor=1.7, minNeighbors=3, minSize=(15, 15))

                if len(smiles) > 0:
                    sense.set_pixels(smile_face)

                    for (a, b, i, j) in smiles:
                        cv2.rectangle(
                            faceimg_color, (a, b), (a + i, b + j), (0, 255, 0), 1)

                else:
                    sense.set_pixels(neutral_face)

                    if oldTime + 5 < time.time():
                        oldTime = time.time()
                        print("- " + choice(dad_jokes) + "\n" )


        else:
            sense.set_pixels(no_face)

        if debug:
            cv2.imshow('Smile Detector', img_color)
            cv2.waitKey(1)

        check_sense_stick()



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        dest='debug',
        default=False,
        help='Enable debug mode with camera window (requires X)')
    debug = parser.parse_args().debug
    assert isinstance(debug, bool)

    smile_detector(debug)

    try:
        smile_detector(debug)

    except (KeyboardInterrupt, SystemExit):
        print("CLOSING PROGRAM")

    finally:
        print("CLEANING UP MATRIX")
        sense.clear()
        sys.exit(0)