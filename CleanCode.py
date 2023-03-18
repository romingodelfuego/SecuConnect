import cv2
import numpy as np


# charger le modèle HumanModel
net = cv2.dnn.readNet("HumanModelv3-tiny.weights", "HumanModelv3-tiny.cfg")
# définir les classes que le modèle peut détecter
classes = []
with open("base.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# définir la ligne fictive
line_x = 600
etatp = None
etatp2 = None

# initialiser la vidéo en direct (webcam)
cap = cv2.VideoCapture(1)
register=[0,0]

colorE=(250,128,114)
colorS=(250,128,114)

def AffichageCompteur(register,frame):
    ylabel = 60
    cv2.putText(frame, "Entrees:"+ str(register[0]), (10, ylabel), cv2.FONT_HERSHEY_SIMPLEX, 2.5, colorE, 2)
    cv2.putText(frame, "Sorties:" +str(register[1]), (10, ylabel+70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, colorS, 2)


def afficher(frame, classes, net, line_x, register):

    global etatp
    global etatp2

    AffichageCompteur(register,frame)
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes, confidences, class_ids = detecter_objets(layerOutputs, height, width)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            if (x < line_x ): #Si je suis a droite et veux aller a gauche
                color = (0, 0, 255)#bleu
                etats = False
                if(etatp != False):
                    etatp2 = True

            elif (not x < line_x): #Si je suis a gauche et veut aller a droite
                color = (255, 0, 0) #red
                etats = True
                if(etatp != True):
                    etatp2 = False
                
            else :
                color = (0, 255, 0) #green
                etats = None

            if(etats and etatp!=True):
                if(etatp2 == etats):
                    register[0]+= 1
                    register[1] = max(0, register[1]-1)
                else:
                    register[0] +=1
                    register[1] = max(0, register[1]-1)

                etatp2 = False
                etatp = True

            elif(not etats and etatp!=False):
                if(etatp2 == etats):
                    register[1] += 1
                    register[0] = max(0, register[0]-1)
                else:
                    register[1] +=1

                etatp2 = True
                etatp = False

            print(etatp)
            print(etatp2)
            print(etats)
                    
            dessiner_objet(frame, x, y, w, h, label, color)

    cv2.line(frame, (line_x, 0), (line_x, height), (255, 0, 0), 2)
    cv2.imshow("Detection @SecuConnect", frame)

def detecter_objets(layerOutputs, height, width):
    boxes, confidences, class_ids = [], [], []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    return boxes, confidences, class_ids

def dessiner_objet(frame, x, y, w, h, label, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    print("x :"+str(x),"y :"+str(y))


def start():
    # charger le modèle HumanModel
    net = cv2.dnn.readNet("HumanModelv3-tiny.weights", "HumanModelv3-tiny.cfg")
    etats = None
    print('-----START-----')
    while True:
        ret, frame = cap.read()
        afficher(frame,classes,net,line_x,register)
        # arrêter la boucle si la touche 'q' est pressée
        if cv2.waitKey(1) == ord('q'):
            break
    print('--Entrée: ' + str(register[0])+'--\n')
    print('--Sortie: ' + str(register[1])+'--\n')
    print('-----END-----')

start()