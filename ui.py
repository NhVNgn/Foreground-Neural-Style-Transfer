import os
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import nst
import time
import rembg


def select_image():
    file_path = filedialog.askopenfilename(initialdir=os.getcwd(
    ), title="Select Image File", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        return Image.open(file_path)
    else:
        return None

# Function to create the interface


def create_interface():
    root = tk.Tk()
    root.title("Mixed Image Viewer")

    # Create the canvas for the images
    canvas = tk.Canvas(root, width=800, height=500)
    canvas.pack()

    # Select the content image using a file dialog

    content_label_text = tk.Label(root, text="Content Image")
    content_label_text.pack()
    content_label_text.place(x=22, y=0)

    style_label_text = tk.Label(root, text="Style Image")
    style_label_text.pack()
    style_label_text.place(x=22, y=250)

    style_label_text = tk.Label(root, text="Style transfered image")
    style_label_text.pack()
    style_label_text.place(x=250, y=0)

    style_label_text = tk.Label(root, text="Foreground style transfered image")
    style_label_text.pack()
    style_label_text.place(x=250, y=250)

    content_image = select_image()

    if content_image:
        content_image_resize = content_image.resize((200, 200))
        content_photo = ImageTk.PhotoImage(content_image_resize)
        content_label = tk.Label(root, image=content_photo)
        content_label.pack()
        content_label.place(x=0, y=40)

    # Select the style image using a file dialog
    style_image = select_image()

    if style_image:
        style_image_resize = style_image.resize((200, 200))
        style_photo = ImageTk.PhotoImage(style_image_resize)
        style_label = tk.Label(root, image=style_photo)
        style_label.pack()
        style_label.place(x=0, y=280)

    root.update()

    mixed_image = nst.getNSTimage(content_image, style_image)

    if mixed_image:
        mixed_image = mixed_image.resize((200, 200))
        mixed_photo = ImageTk.PhotoImage(mixed_image)
        mixed_label = tk.Label(root, image=mixed_photo)
        mixed_label.pack()
        mixed_label.place(x=250, y=40)

    root.update()

    foreground_only_img = rembg.remove(
        content_image).convert('RGB').resize((200, 200))

    # foreground_nst_img = nst.getForegroundNSTimage(
    #     foreground_only_img, style_image)

    content_image_resize = content_image.resize((200, 200))

    width, height = foreground_only_img.size
    mixed_foreground_img = mixed_image
    for x in range(width):
        for y in range(height):
            foreground_img_color = foreground_only_img.getpixel((x, y))
            content_img_color = content_image_resize.getpixel((x, y))
            if foreground_img_color == (0, 0, 0):
                mixed_foreground_img.putpixel((x, y), content_img_color)

   

    foreground_photo = ImageTk.PhotoImage(mixed_foreground_img)
    foreground_label = tk.Label(root, image=foreground_photo)
    foreground_label.pack()
    foreground_label.place(x=250, y=280)

    # root.update()

    # transparent_image = Image.frombytes(
    #     "RGB", content_image_resize.size, foreground_nst_img.tobytes())

    # new_image.paste(transparent_image, (0, 0), transparent_image)
    # new_image = Image.alpha_composite(
    #     content_image_resize.convert("RGB"), new_image)

    # Start the main event loop
    root.mainloop()


create_interface()
