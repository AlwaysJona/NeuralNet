import neuralnet as nn 
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageDraw

model_path = "../models/new_mnist.nn"
model = nn.NeuralNetwork(model_path)
test_data = np.random.rand(784)
prediction = -1
result = np.zeros(10)

#print(model.guess(test_data))

image = Image.new("L", (280, 280), 255)
drawIm = ImageDraw.Draw(image)

root = Tk()

canvas = Canvas(root, width=280, height = 280, bg = "white")
canvas.pack()

last_x = None
last_y = None

def start_draw(event):
    global last_x, last_y
    last_x = event.x
    last_y = event.y

def draw(event):
    global last_x, last_y

    canvas.create_line(
            last_x, last_y,
            event.x, event.y,
            width = 10,
            fill="black",
            capstyle=ROUND,
            smooth=True
    )

    drawIm.line((last_x, last_y, event.x, event.y),
                fill = 0,
                width = 10)

    last_x = event.x
    last_y = event.y

def clear_canvas():
    canvas.delete("all")
    drawIm.rectangle((0,0, 280, 280), fill=255)

canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)

button_frame = Frame(root)
button_frame.pack()

clear_button = Button(button_frame, text="Clear", command=clear_canvas)
quit_button = Button(button_frame, text="Quit", command=root.destroy)
clear_button.pack(side = RIGHT, padx = 10, pady = 20)
quit_button.pack(side = RIGHT, padx = 10, pady = 20)


pred_label = Label(root, text=f"Prediction: {prediction}", font=("Arial", 16))
pred_label.pack()
val_label = Label(root, text=f"Probabilities: [ {result[0]: .4f}, {result[1]: .4f}, {result[2]: .4f}, {result[3]: .4f}, {result[4]: .4f}, {result[5]: .4f}, {result[6]: .4f}, {result[7]: .4f}, {result[8]: .4f}, {result[9]: .4f} ]")
val_label.pack()

def preprocess(img):
    
    inverted = Image.eval(img, lambda x: 255 - x);

    bbox = inverted.getbbox()

    if bbox is None:
        return None
    
    padding = 10

    bbox = (
        max(bbox[0]-padding, 0),
        max(bbox[1]-padding, 0),
        min(bbox[2]+padding, img.width),
        min(bbox[3]+padding, img.height)
    )

    digit = img.crop(bbox)

    digit.thumbnail((20,20), Image.Resampling.LANCZOS)

    new_image = Image.new("L", (28,28), 255)

    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    new_image.paste(digit, (x, y))

    return new_image

def predict():
    global prediction, result
    small_image = preprocess(image) 
    pixels = np.array(small_image, dtype=np.float32)
    pixels = 1 - (pixels / 255)
    pixels = pixels.flatten()
    result = model.guess(pixels)
    prediction = np.argmax(result)
    pred_label.config(text = f"Prediction: {prediction}")
    val_label.config(text=f"Values: [ {result[0]: .4f}, {result[1]: .4f}, {result[2]: .4f}, {result[3]: .4f}, {result[4]: .4f}, {result[5]: .4f}, {result[6]: .4f}, {result[7]: .4f}, {result[8]: .4f}, {result[9]: .4f} ]")



pred_button = Button(button_frame, text="Predict", command=predict)
pred_button.pack(side = RIGHT, padx = 10, pady = 20)


root.mainloop()
