from PIL import Image
import rembg


original_image = Image.open("dog.png")

background_image = Image.open("dog.png")

output = rembg.remove(original_image)
# output.show()


new_image = Image.new("RGBA", original_image.size)


transparent_image = Image.frombytes(
    "RGBA", original_image.size, output.tobytes())

new_image.paste(transparent_image, (0, 0), transparent_image)

new_image = Image.alpha_composite(background_image.convert("RGBA"), new_image)

