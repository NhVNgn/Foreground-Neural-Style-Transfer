import os
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import nst
import rembg
from tkinter import ttk
import threading
import time


def select_image():
    file_path = filedialog.askopenfilename(initialdir=os.getcwd(
    ), title="Select Image File", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        return Image.open(file_path)
    else:
        return None

# Function to create the interface


def getMixedFogroundImage(mixed_image, content_image):

    foreground_only_img = rembg.remove(
        content_image).convert('RGB').resize((200, 200))
    content_image_resize = content_image.resize((200, 200))
    width, height = foreground_only_img.size

    mixed_foreground_img = mixed_image.resize((200, 200))

    for x in range(width):
        for y in range(height):
            foreground_img_color = foreground_only_img.getpixel((x, y))
            content_img_color = content_image_resize.getpixel((x, y))
            if foreground_img_color == (0, 0, 0):
                mixed_foreground_img.putpixel((x, y), content_img_color)

    return mixed_foreground_img


def create_interface():
    root = tk.Tk()
    root.title("Mixed Image Viewer")

    # Create the canvas for the images
    canvas = tk.Canvas(root, width=870, height=530)
    canvas.pack()

    # Create the "Select Images" button
    def select_images():
        nonlocal content_image, content_photo, content_label, style_image, style_photo, style_label, \
            mixed_image, mixed_photo, mixed_label, weight_image, weight_photo, weight_label, \
            mixed_foreground_img, foreground_photo, foreground_label, \
            weight_foreground_img, weight_foreground_photo, weight_foregroundforeground_label, start_button

        # Select the content image using a file dialog
        content_image = select_image()
        content_photo = getPhoto(content_image)
        content_label.config(image=content_photo)
        content_label.image = content_photo
        root.update()

        # Select the style image using a file dialog
        style_image = select_image()
        style_photo = getPhoto(style_image)
        style_label.config(image=style_photo)
        style_label.image = style_photo
        root.update()

        start_button.config(state="normal")  # enable start button

    select_images_button = tk.Button(
        root, text="Select Images", command=select_images)
    select_images_button.pack()
    select_images_button.place(x=0, y=500)

    # Create the refresh button
    def refresh_interface():
        root.destroy()
        create_interface()

    refresh_button = tk.Button(root, text="Refresh", command=refresh_interface)
    refresh_button.pack()
    refresh_button.place(x=130, y=500)

    # resize photo function
    def getPhoto(image):
        image_resize = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image_resize)
        return photo

    content_label_text = tk.Label(root, text="Content Image")
    content_label_text.pack()
    content_label_text.place(x=22, y=0)

    style_label_text = tk.Label(root, text="Style Image")
    style_label_text.pack()
    style_label_text.place(x=22, y=250)

    style_label_text = tk.Label(root, text="NST image with default weight = 0")
    style_label_text.pack()
    style_label_text.place(x=250, y=0)
    style_label_text = tk.Label(
        root, text="NST image with input content weight = ")

    style_label_text.pack()
    style_label_text.place(x=478, y=0)

    style_label_text = tk.Label(
        root, text="Foreground NST, default weight = 0")
    style_label_text.pack()
    style_label_text.place(x=250, y=250)

    style_label_text = tk.Label(
        root, text="Foreground NST with customized weight")
    style_label_text.pack()
    style_label_text.place(x=500, y=250)

    # Initialize image variables
    content_image = None
    content_photo = None
    style_image = None
    style_photo = None
    mixed_image = None
    mixed_photo = None
    weight_image = None
    weight_photo = None
    mixed_foreground_img = None
    foreground_photo = None
    weight_foreground_img = None
    weight_foreground_photo = None

    # Create the initial content image label
    content_label = tk.Label(root)
    content_label.pack()
    content_label.place(x=0, y=40)

    # Create the initial style image label
    style_label = tk.Label(root)
    style_label.pack()
    style_label.place(x=0, y=280)

    # Create the initial mixed image label
    mixed_label = tk.Label(root)
    mixed_label.pack()
    mixed_label.place(x=250, y=40)

    # Create the initial weight image label
    weight_label = tk.Label(root)
    weight_label.pack()
    weight_label.place(x=478, y=40)

    # Create the initial mixed foreground image label
    foreground_label = tk.Label(root)
    foreground_label.pack()
    foreground_label.place(x=250, y=280)

    # Create the initial weight foreground image label
    weight_foregroundforeground_label = tk.Label(root)
    weight_foregroundforeground_label.pack()
    weight_foregroundforeground_label.place(x=478, y=280)

    # Create input box
    weight_entry = tk.Entry(root)
    weight_entry.insert(0, "10")
    weight_entry.pack()
    weight_entry.place(x=700, y=0)

    # Create download button
    def download_image():
        if mixed_image and weight_image:
            mixed_image.save("mixed_image.jpg")
            weight_image.save("weight_image.jpg")

   

    # Create the "Start" button

    def start_style_transfer():
        nonlocal content_image, style_image, mixed_image, mixed_photo, mixed_label, \
            weight_image, weight_photo, weight_label, mixed_foreground_img, foreground_photo, \
            foreground_label, weight_foreground_img, weight_foreground_photo

        # read the weight value from input box
        weight = int(weight_entry.get())
        print(f"Customized weight is {weight}")
        # Perform style transfer and update the UI
        result_dict = {}

        def runGetNSTimage(content_image, style_image, result_dict, key):
            result_dict[key] = nst.getNSTimage(content_image, style_image)

        def runGetNSTimageWithWeight(content_image, style_image, weight, result_dict, key):
            result_dict[key] = nst.getNSTimageWithWeight(
                content_image, style_image, weight)

        thread = threading.Thread(target=runGetNSTimage, args=(
            content_image, style_image, result_dict, 'mixed_image'))
        thread.daemon = True
        thread2 = threading.Thread(target=runGetNSTimageWithWeight, args=(
            content_image, style_image, weight, result_dict, 'weight_image'))
        thread2.daemon = True

        thread.start()
        thread2.start()

        thread.join()
        thread2.join()

        mixed_image = result_dict['mixed_image']
        mixed_photo = getPhoto(mixed_image)
        mixed_label.config(image=mixed_photo)
        mixed_label.image = mixed_photo

        weight_image = result_dict['weight_image']
        weight_photo = getPhoto(weight_image)
        weight_label.config(image=weight_photo)
        weight_label.image = weight_photo

        mixed_foreground_img = getMixedFogroundImage(
            mixed_image, content_image)
        foreground_photo = ImageTk.PhotoImage(mixed_foreground_img)
        foreground_label.config(image=foreground_photo)
        foreground_label.image = foreground_photo

        weight_foreground_img = getMixedFogroundImage(
            weight_image, content_image)
        weight_foreground_photo = ImageTk.PhotoImage(weight_foreground_img)
        weight_foregroundforeground_label.config(image=weight_foreground_photo)
        weight_foregroundforeground_label.image = weight_foreground_photo

        download_button = tk.Button(
            root, text="Download Image", command=download_image)
        download_button.pack()
        download_button.place(x=185, y=500)

    start_button = tk.Button(
        root, text="Start", state="disabled", command=start_style_transfer)

    start_button.pack()
    start_button.place(x=90, y=500)

    root.mainloop()


create_interface()
