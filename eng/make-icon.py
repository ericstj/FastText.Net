"""Generates the FastText.Net package icon (original artwork).

A rounded-square gradient badge containing a speech bubble (NLP/text) with a
classification label tag. Intentionally distinct from the fastText logo.
"""
from PIL import Image, ImageDraw

S = 512  # supersample, downscaled at the end
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Vertical gradient background: .NET purple -> teal
top = (88, 47, 219)      # #582FDB
bot = (33, 173, 153)     # #21AD99
for y in range(S):
    t = y / (S - 1)
    r = round(top[0] + (bot[0] - top[0]) * t)
    g = round(top[1] + (bot[1] - top[1]) * t)
    b = round(top[2] + (bot[2] - top[2]) * t)
    draw.line([(0, y), (S, y)], fill=(r, g, b, 255))

# Rounded-square mask
radius = int(S * 0.22)
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=radius, fill=255)
img.putalpha(mask)

draw = ImageDraw.Draw(img)

# Speech bubble (white) — represents text / NLP
bx0, by0, bx1, by1 = int(S * 0.16), int(S * 0.20), int(S * 0.84), int(S * 0.66)
br = int(S * 0.09)
draw.rounded_rectangle([bx0, by0, bx1, by1], radius=br, fill=(255, 255, 255, 255))
# Tail
tail = [(int(S * 0.30), by1 - 2), (int(S * 0.30), int(S * 0.80)), (int(S * 0.46), by1 - 2)]
draw.polygon(tail, fill=(255, 255, 255, 255))

# Text lines inside the bubble (decreasing length)
line_color = (88, 47, 219, 255)
lx = int(S * 0.24)
lh = int(S * 0.045)
gap = int(S * 0.075)
ly = int(S * 0.30)
for frac in (0.52, 0.44, 0.34):
    draw.rounded_rectangle([lx, ly, lx + int(S * frac), ly + lh],
                           radius=lh // 2, fill=line_color)
    ly += gap

# Classification label tag (bottom-right) — represents predicted label
tag_fill = (33, 173, 153, 255)
tx0, ty0, tx1, ty1 = int(S * 0.52), int(S * 0.62), int(S * 0.86), int(S * 0.86)
draw.rounded_rectangle([tx0, ty0, tx1, ty1], radius=int(S * 0.05), fill=tag_fill)
draw.ellipse([tx0 + int(S * 0.05) - int(S * 0.018), (ty0 + ty1) // 2 - int(S * 0.018),
              tx0 + int(S * 0.05) + int(S * 0.018), (ty0 + ty1) // 2 + int(S * 0.018)],
             fill=(255, 255, 255, 255))

out = img.resize((128, 128), Image.LANCZOS)
out.save(r"C:\src\ericstj\fastText.Net\eng\icon.png", "PNG")
print("wrote eng/icon.png")
